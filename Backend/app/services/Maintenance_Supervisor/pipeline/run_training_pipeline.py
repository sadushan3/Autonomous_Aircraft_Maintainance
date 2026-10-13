"""
run_training_pipeline.py

Master Supervisor Model Training & Baseline Benchmarking Pipeline Orchestrator.

Purpose
-------
Runs data preprocessing, dataset splitting, multi-model training, baseline benchmarking,
and champion model selection in a single automated workflow.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

import pandas as pd

from app.config.Maintenance_Supervisor.supervisor_config import (
    REPORTS_ROOT,
    PROCESSED_ROOT,
    TARGET_COLUMN,
)
from app.services.Maintenance_Supervisor.Data_preprocessing.leakage_guard import run_leakage_guard
from app.services.Maintenance_Supervisor.training.train_supervisor import run_master_training_pipeline
from app.services.Maintenance_Supervisor.evaluation.compare_baselines import run_baseline_comparison
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "run_training_pipeline_report.json"
_SCALED_DATASET_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_scaled_dataset.csv"


def _check_leakage_survived_into_features(error_columns: list[str]) -> list[str]:
    """Check whether any leakage-guard error column reached the scaled dataset.

    ``run_leakage_guard`` scans the pre-encoding engineered dataset, where
    ``final_decision`` and the other DERIVED_OUTPUT_COLUMNS are expected to
    be present (they are the labeller's output, dropped later by
    categorical_encoder.py) -- so `is_clean` is always False on that file by
    design and is not itself a training blocker. What actually matters is
    whether any of those columns (or a one-hot expansion of one, e.g.
    "priority_critical") survived into the scaled dataset that
    feature_selector.py and every training script draw candidate features
    from. TARGET_COLUMN itself is exempt: it is expected in the scaled
    dataset as the label column, just never as a candidate feature.
    """
    if not _SCALED_DATASET_PATH.exists():
        return []

    scaled_columns = pd.read_csv(_SCALED_DATASET_PATH, nrows=0).columns.tolist()
    bases = [c for c in error_columns if c != TARGET_COLUMN]
    prefixes = tuple(f"{c}_" for c in bases)

    return [
        col for col in scaled_columns
        if col in bases or col.startswith(prefixes)
    ]


def run_training_pipeline() -> dict:
    section("MASTER TRAINING PIPELINE ORCHESTRATOR STARTED")
    start_time = time.perf_counter()

    # Leakage guard is a gate, not just a report: training must not proceed
    # while a leakage-guard-flagged column (or a one-hot expansion of one)
    # is still present in the actual feature matrix. It previously produced
    # a report that nothing downstream ever consumed.
    leakage_res = run_leakage_guard()
    survived = _check_leakage_survived_into_features(leakage_res.leakage_columns_removed)
    if survived:
        raise RuntimeError(
            f"Leakage guard flagged column(s) {leakage_res.leakage_columns_removed} "
            f"on the engineered dataset, and {survived} still appear in the "
            "scaled feature dataset used for training. Training pipeline "
            "aborted; re-run categorical_encoder.py / feature_scaler.py to "
            "drop them before training."
        )

    train_res = run_master_training_pipeline()
    baseline_res = run_baseline_comparison()

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "leakage_guard": {
            "engineered_dataset_error_columns": leakage_res.leakage_columns_removed,
            "warning_count": leakage_res.warning_count,
            "safe_feature_count": len(leakage_res.safe_feature_columns),
            "leakage_survived_into_scaled_features": survived,
        },
        "training_pipeline": train_res,
        "baseline_comparison": baseline_res,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Training pipeline orchestrator completed in %.2fs.", duration)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("MASTER TRAINING PIPELINE ORCHESTRATOR COMPLETED")

    return report


def main() -> int:
    try:
        res = run_training_pipeline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in run_training_pipeline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
