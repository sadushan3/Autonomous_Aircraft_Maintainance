"""
train_ppo.py

Trainer for the PPO Deep Reinforcement Learning Supervisor Agent.

Purpose
-------
Trains PPOSupervisorModel by genuine on-policy rollout collection against
TurbofanMaintenanceEnv (not supervised fitting on labelled actions -- see
models/ppo_supervisor.py), then evaluates it two ways:

1. Episode-based RL metrics (mean_episode_return, catastrophic_failure_rate,
   successful_maintenance_rate) computed by rolling the trained greedy
   policy out over held-out engine trajectories. This is the primary
   evidence for how good the policy actually is.
2. A label-agreement accuracy / macro-F1 against supervisor_labeler.py's
   rule-cascade target, computed only so PPO can be ranked against the
   classifier baselines (train_supervisor.py selects a champion by
   val_macro_f1 across heterogeneous model types). This number is
   secondary and diagnostic -- the policy was not trained to reproduce it.

Trains on RAW (pre-RobustScaler) features, not the scaled splits
-------------------------------------------------------------------
Earlier versions of this trainer loaded ``processed/.../splits/train.csv``,
which is the RobustScaler-standardized form of the data (``predicted_rul``
scaled mean ~0.04, not a cycle count). The live serving path
(``supervisor_bridge.run_supervisor_bridge`` -> ``real_output_merger``)
feeds the champion RAW physical-unit values straight from the RUL/
Failure-Risk/Anomaly exchange CSVs -- it applies no scaler. A policy
trained on scaled inputs and served raw ones sees every live input wildly
outside its training distribution and saturates to a single constant
output regardless of the engine's actual state -- confirmed: with the
reward-scale bug fixed and episode-return variance restored,
``raw_model_decision`` was *still* constant across a real 33,991-row live
run, including the 20 highest-risk rows in the dataset (all correctly
caught only by safety_policy.py's override, not by the policy itself).

This trainer now builds episodes from
``processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv`` (the
pre-scaling form, in the same physical units the live path uses),
partitioned into train/val by the exact same engine membership the
scaled splits already established (loaded from ``splits/train.csv`` /
``splits/validation.csv`` for their engine_id/fd_subset columns only, so
there is no leakage and no change to which engines are held out).
``PPOSupervisorModel`` normalizes these raw features internally (fixed
mean/std computed once from the raw training data and saved with the
model), so the network still gets well-conditioned input without the live
path ever needing to apply a scaler.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    PROCESSED_ROOT,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.models.ppo_supervisor import PPOSupervisorModel
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_TRAIN_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "train.csv"
_VAL_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "validation.csv"
_RAW_ENCODED_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_encoded_dataset.csv"
_MODEL_OUTPUT_PATH: Final[Path] = ARTIFACT_ROOT / "models" / "ppo.joblib"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "train_ppo_report.json"


def _engine_keys(path: Path) -> set[tuple]:
    """Load just the (engine_id, fd_subset) membership of a split file."""
    cols = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in (ENGINE_ID_COLUMN, SUBSET_COLUMN) if c in cols]
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if SUBSET_COLUMN in df.columns:
        return set(zip(df[ENGINE_ID_COLUMN], df[SUBSET_COLUMN]))
    return {(eid, None) for eid in df[ENGINE_ID_COLUMN]}


def _load_raw_train_val(train_path: Path, val_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the raw (pre-scaling) encoded dataset by the scaled splits' engine membership."""
    train_keys = _engine_keys(train_path)
    val_keys = _engine_keys(val_path)

    raw = pd.read_csv(_RAW_ENCODED_PATH, low_memory=False)
    if SUBSET_COLUMN in raw.columns:
        raw_keys = list(zip(raw[ENGINE_ID_COLUMN], raw[SUBSET_COLUMN]))
    else:
        raw_keys = [(eid, None) for eid in raw[ENGINE_ID_COLUMN]]
    raw_key_series = pd.Series(raw_keys, index=raw.index)

    train_df = raw.loc[raw_key_series.isin(train_keys)].reset_index(drop=True)
    val_df = raw.loc[raw_key_series.isin(val_keys)].reset_index(drop=True)
    return train_df, val_df


def train_ppo(
    train_path: Path | None = None,
    val_path: Path | None = None,
    model_output_path: Path | None = None,
) -> dict:
    train_path = Path(train_path or _TRAIN_PATH).resolve()
    val_path = Path(val_path or _VAL_PATH).resolve()
    model_output_path = Path(model_output_path or _MODEL_OUTPUT_PATH).resolve()

    section("TRAINING PPO REINFORCEMENT LEARNING SUPERVISOR STARTED")
    start_time = time.perf_counter()

    if not train_path.exists() or not val_path.exists() or not _RAW_ENCODED_PATH.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        train_df = pd.read_csv(SAMPLE_DATASET_PATH, low_memory=False)
        val_df = train_df
    else:
        train_df, val_df = _load_raw_train_val(train_path, val_path)

    # PPOSupervisorModel.fit dispatches to genuine environment-rollout
    # training whenever engine_id/cycle trajectory columns are present,
    # which they are here. val_df enables best-checkpoint-by-validation
    # selection inside fit() -- without it, training keeps whatever the
    # final epoch's policy happens to be, which measurably does not
    # generalize as well (see best_checkpoint_epoch in the report below).
    model = PPOSupervisorModel()
    model.fit(train_df, val_df=val_df)

    if not model.trained_via_environment:
        raise RuntimeError(
            "PPOSupervisorModel.fit fell back to behaviour cloning instead of "
            "training via TurbofanMaintenanceEnv -- train_df is missing the "
            "engine_id/cycle trajectory columns the environment requires."
        )

    # ------------------------------------------------------------------
    # Primary evaluation: real episode rollouts on held-out engines.
    # ------------------------------------------------------------------
    episode_metrics = model.evaluate_policy(val_df, greedy=True)

    # ------------------------------------------------------------------
    # Secondary evaluation: label agreement, for champion-selection
    # comparability against the classifier baselines only.
    # ------------------------------------------------------------------
    X_val = val_df.reindex(columns=model.feature_names, fill_value=0.0)
    X_val = X_val.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y_val = val_df[TARGET_COLUMN] if TARGET_COLUMN in val_df.columns else pd.Series(dtype=object)

    if len(y_val) > 0:
        val_preds = model.predict(X_val)
        label_acc = float(accuracy_score(y_val, val_preds))
        label_macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
    else:
        label_acc = None
        label_macro_f1 = None

    duration = time.perf_counter() - start_time
    model.save(model_output_path)

    report = {
        "status": "success",
        "model_name": "ppo",
        "model_path": str(model_output_path),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "features_count": len(model.feature_names),
        "trained_via_environment": model.trained_via_environment,
        "training_epochs": len(model.training_history),
        "final_train_epoch_stats": model.training_history[-1] if model.training_history else None,
        # Which epoch's checkpoint was actually kept (best-by-validation, not
        # necessarily the last one -- see ppo_supervisor.py's
        # _fit_via_environment) and what it scored when it was selected.
        "best_checkpoint_epoch": model.best_checkpoint_epoch,
        "best_checkpoint_val_metrics_at_selection": model.best_checkpoint_val_metrics,
        # Primary RL evidence: episode rollouts on held-out engines, using
        # whichever checkpoint fit() actually kept.
        "episode_metrics": episode_metrics,
        # Secondary, diagnostic-only: label agreement with the rule-cascade
        # target, kept so train_supervisor.py can rank PPO against the
        # classifier baselines on a common metric. Not the training
        # objective and not a claim that PPO is a classifier.
        "val_accuracy": round(label_acc, 4) if label_acc is not None else None,
        "val_macro_f1": round(label_macro_f1, 4) if label_macro_f1 is not None else None,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info(
        "PPO episode metrics: mean_return=%.2f, catastrophic_failure_rate=%.3f, "
        "successful_maintenance_rate=%.3f",
        episode_metrics["mean_episode_return"],
        episode_metrics["catastrophic_failure_rate"],
        episode_metrics["successful_maintenance_rate"],
    )
    if label_acc is not None:
        logger.info("PPO label-agreement (diagnostic): accuracy=%.4f, macro_f1=%.4f", label_acc, label_macro_f1)
    section("TRAINING PPO REINFORCEMENT LEARNING SUPERVISOR COMPLETED")
    return report


def main() -> int:
    try:
        res = train_ppo()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in train_ppo: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
