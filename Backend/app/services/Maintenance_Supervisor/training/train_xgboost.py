"""
train_xgboost.py

Standalone Trainer for XGBoost Gradient Boosted Supervisor Model.

Purpose
-------
Trains, evaluates, and serializes the XGBoost model using preprocessed
train and validation split datasets.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

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
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.models.xgboost_supervisor import XGBoostSupervisorModel
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_TRAIN_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "train.csv"
_VAL_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "validation.csv"
_MODEL_OUTPUT_PATH: Final[Path] = ARTIFACT_ROOT / "models" / "xgboost.joblib"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "train_xgboost_report.json"


def train_xgboost(
    train_path: Path | None = None,
    val_path: Path | None = None,
    model_output_path: Path | None = None,
) -> dict:
    train_path = Path(train_path or _TRAIN_PATH).resolve()
    val_path = Path(val_path or _VAL_PATH).resolve()
    model_output_path = Path(model_output_path or _MODEL_OUTPUT_PATH).resolve()

    section("TRAINING XGBOOST SUPERVISOR STARTED")
    logger.info("Train path: %s", train_path)
    logger.info("Val path  : %s", val_path)

    start_time = time.perf_counter()

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError(f"Train/Val splits not found at {train_path} / {val_path}")

    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)

    selected_features_path = ARTIFACT_ROOT / "selected_features.json"
    if selected_features_path.exists():
        with open(selected_features_path, "r", encoding="utf-8") as f:
            feat_data = json.load(f)
            feature_cols = feat_data.get("selected_features", [])
            logger.info("Using %d selected features from manifest.", len(feature_cols))
    else:
        exclude = set(_CFG.forbidden_features) | {TARGET_COLUMN, "engine_id", "fd_subset", "cycle", "schema_version"}
        feature_cols = [c for c in train_df.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_df[c])]
        logger.info("Using %d numeric features.", len(feature_cols))

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df[TARGET_COLUMN]

    model = XGBoostSupervisorModel()
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_probs = model.predict_proba(X_val)

    acc = float(accuracy_score(y_val, val_preds))
    macro_f1 = float(f1_score(y_val, val_preds, average="macro"))
    try:
        # labels= is required. Without it sklearn infers the class order from
        # y_val, while val_probs columns follow the model's own classes_
        # ordering. When those disagree -- which happens whenever y_val is
        # missing a class or orders them differently -- the probability
        # columns are silently matched to the wrong labels and the resulting
        # log loss is meaningless. That is what produced a reported
        # val_log_loss of 13.46 alongside a val_accuracy of 1.0: two numbers
        # that cannot both describe the same predictions.
        class_labels = getattr(model, "classes_", None)
        if class_labels is None:
            inner = getattr(model, "model", None)
            class_labels = getattr(inner, "classes_", None)
        if class_labels is not None:
            loss = float(log_loss(y_val, val_probs, labels=list(class_labels)))
        else:
            loss = float(log_loss(y_val, val_probs))
    except Exception:
        loss = None

    duration = time.perf_counter() - start_time

    model.save(model_output_path)

    report = {
        "status": "success",
        "model_name": "xgboost",
        "model_path": str(model_output_path),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "features_count": len(feature_cols),
        "val_accuracy": round(acc, 4),
        "val_macro_f1": round(macro_f1, 4),
        "val_log_loss": round(loss, 4) if loss is not None else None,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Val Accuracy: %.4f | Val Macro F1: %.4f", acc, macro_f1)
    section("TRAINING XGBOOST SUPERVISOR COMPLETED")
    return report


def main() -> int:
    try:
        res = train_xgboost()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in train_xgboost: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
