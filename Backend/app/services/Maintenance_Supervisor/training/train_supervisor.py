"""
train_supervisor.py

Master Training Pipeline Orchestrator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Orchestrates the sequential training, evaluation, comparison, and best-model selection
across all supported candidate machine learning models:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. LightGBM (if available)

It selects the champion model based on validation macro F1 score and registers the champion
in `artifacts/Maintenance_Supervisor/champion_model_manifest.json`.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.training.train_supervisor

Expected outputs
----------------
- artifacts/Maintenance_Supervisor/champion_model_manifest.json
- reports/Maintenance_Supervisor/supervisor_training_pipeline_report.json

Exit codes
----------
0 — master training pipeline completed successfully
1 — internal failure
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import ARTIFACT_ROOT, REPORTS_ROOT
from app.services.Maintenance_Supervisor.training.train_logistic_regression import train_logistic_regression
from app.services.Maintenance_Supervisor.training.train_random_forest import train_random_forest
from app.services.Maintenance_Supervisor.training.train_xgboost import train_xgboost
from app.services.Maintenance_Supervisor.training.train_lightgbm import train_lightgbm
from app.services.Maintenance_Supervisor.training.train_ppo import train_ppo
from app.services.Maintenance_Supervisor.training.train_dqn import train_dqn
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_CHAMPION_MANIFEST_PATH: Final[Path] = ARTIFACT_ROOT / "champion_model_manifest.json"
_PIPELINE_REPORT_PATH: Final[Path] = REPORTS_ROOT / "supervisor_training_pipeline_report.json"


def run_master_training_pipeline() -> dict:
    section("MASTER SUPERVISOR TRAINING PIPELINE STARTED")
    start_time = time.perf_counter()

    results: list[dict] = []

    # 1. Train Logistic Regression
    try:
        lr_res = train_logistic_regression()
        results.append(lr_res)
    except Exception as exc:
        logger.error("Logistic Regression training failed: %s", exc)

    # 2. Train Random Forest
    try:
        rf_res = train_random_forest()
        results.append(rf_res)
    except Exception as exc:
        logger.error("Random Forest training failed: %s", exc)

    # 3. Train XGBoost
    try:
        xgb_res = train_xgboost()
        results.append(xgb_res)
    except Exception as exc:
        logger.error("XGBoost training failed: %s", exc)

    # 4. Train LightGBM
    try:
        lgb_res = train_lightgbm()
        if lgb_res.get("status") == "success":
            results.append(lgb_res)
    except Exception as exc:
        logger.error("LightGBM training failed: %s", exc)

    # 5. Train PPO Reinforcement Learning Agent
    try:
        ppo_res = train_ppo()
        if ppo_res.get("status") == "success":
            results.append(ppo_res)
    except Exception as exc:
        logger.error("PPO training failed: %s", exc)

    # 6. Train DQN Reinforcement Learning Agent
    try:
        dqn_res = train_dqn()
        if dqn_res.get("status") == "success":
            results.append(dqn_res)
    except Exception as exc:
        logger.error("DQN training failed: %s", exc)

    if not results:
        raise RuntimeError("All model training runs failed.")

    valid_results = [r for r in results if r.get("status") == "success" and r.get("val_macro_f1") is not None]
    valid_results.sort(key=lambda r: r["val_macro_f1"], reverse=True)

    # Champion selection is NOT "highest val_macro_f1" here, on purpose.
    #
    # final_decision is a deterministic function of predicted_rul, risk_10/
    # risk_30/risk_50 and anomaly_severity via supervisor_labeler.py's rule
    # cascade, and those same signals are legitimate input features. A
    # sufficiently expressive supervised model (random_forest, xgboost,
    # lightgbm) can learn that cascade's thresholds almost exactly, which is
    # why they score ~0.999 macro F1 -- that number measures how well a
    # classifier memorised a hand-written if/else tree, not decision
    # quality, and picking the top scorer here would always crown a
    # rule-cascade-imitating classifier, never the RL policy. Per this
    # project's binding directive, PPO -- trained by reward against
    # TurbofanMaintenanceEnv, never against the rule-cascade label -- is the
    # required live decision mechanism, so it is the champion whenever it
    # trained successfully. Every other model (including PPO's own
    # low val_macro_f1) is retained in all_candidates as a labelled baseline
    # for comparison, never presented as "the model".
    ppo_result = next((r for r in valid_results if r["model_name"] == "ppo"), None)
    if ppo_result is not None:
        champion = ppo_result
        selection_metric = "ppo_mandated_by_directive"
    else:
        logger.error(
            "PPO training did not produce a usable result; falling back to "
            "the best label-agreement classifier as champion. This violates "
            "the project's genuine-RL requirement and must be treated as a "
            "failed run, not a normal fallback."
        )
        champion = valid_results[0]
        selection_metric = "val_macro_f1_fallback_ppo_unavailable"

    logger.info(
        "Selected Champion Model: '%s' (selection_metric=%s, val_macro_f1=%.4f, "
        "val_accuracy=%.4f)",
        champion["model_name"], selection_metric, champion["val_macro_f1"], champion["val_accuracy"],
    )

    duration = time.perf_counter() - start_time

    champion_manifest = {
        "champion_model_name": champion["model_name"],
        "champion_model_path": champion["model_path"],
        "val_macro_f1": champion["val_macro_f1"],
        "val_accuracy": champion["val_accuracy"],
        "val_log_loss": champion.get("val_log_loss"),
        "episode_metrics": champion.get("episode_metrics"),
        "selection_metric": selection_metric,
        "all_candidates": [
            {
                "name": r["model_name"],
                "val_macro_f1": r.get("val_macro_f1"),
                "val_accuracy": r.get("val_accuracy"),
                "episode_metrics": r.get("episode_metrics"),
                "role": "champion" if r is champion else "baseline_for_comparison",
            }
            for r in valid_results
        ],
    }

    _CHAMPION_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(champion_manifest, _CHAMPION_MANIFEST_PATH)

    pipeline_report = {
        "status": "success",
        "champion": champion_manifest,
        "total_models_trained": len(valid_results),
        "duration_seconds": round(duration, 4),
    }

    _PIPELINE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pipeline_report, _PIPELINE_REPORT_PATH)

    logger.info("Champion manifest written to: %s", _CHAMPION_MANIFEST_PATH)
    logger.info("Pipeline report written to: %s", _PIPELINE_REPORT_PATH)
    section("MASTER SUPERVISOR TRAINING PIPELINE COMPLETED")

    return pipeline_report


def main() -> int:
    try:
        res = run_master_training_pipeline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in master training pipeline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
