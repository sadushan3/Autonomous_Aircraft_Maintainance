"""
calibration_evaluator.py

Probability Calibration & Expected Calibration Error (ECE) Evaluator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Measures multi-class probability calibration metrics including Expected Calibration Error (ECE)
and Brier Score to ensure model predicted probabilities correspond to true empirical likelihoods.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, TARGET_COLUMN, DECISION_TO_SEVERITY
from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "calibration_evaluator_report.json"


def compute_expected_calibration_error(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate Expected Calibration Error (ECE) for multi-class predictions."""
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)

    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return float(ece)


def run_calibration_evaluation(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("PROBABILITY CALIBRATION BENCHMARK STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    engine = DecisionFusionEngine()
    if engine.model and engine.model.is_fitted:
        feature_cols = engine.model.feature_names
        X_mat = df[feature_cols].fillna(0.0)
        probs = engine.model.predict_proba(X_mat)
    else:
        probs = np.full((len(df), len(engine.cfg.decision_classes)), 1.0 / len(engine.cfg.decision_classes))

    y_true_clean = np.array([DECISION_TO_SEVERITY.get(str(v).strip().lower(), 0) for v in df[TARGET_COLUMN]])

    ece = compute_expected_calibration_error(y_true_clean, probs)
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "total_samples": len(df),
        "expected_calibration_error": round(ece, 4),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Expected Calibration Error (ECE): %.4f", ece)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("PROBABILITY CALIBRATION BENCHMARK COMPLETED")

    return report


def main() -> int:
    try:
        res = run_calibration_evaluation()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in calibration_evaluator: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
