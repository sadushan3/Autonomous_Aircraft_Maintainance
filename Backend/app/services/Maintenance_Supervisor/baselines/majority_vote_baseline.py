"""
majority_vote_baseline.py

Majority Class / Mode Classifier Baseline Model for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module implements a zero-rule, naive baseline classifier that always predicts the
most frequent target class observed in the training distribution (or `continue_operation`).
It serves as the lower-bound benchmark for classification accuracy, macro F1, and log-loss.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.baselines.majority_vote_baseline

Expected outputs
----------------
- reports/Maintenance_Supervisor/majority_vote_baseline_report.json

Exit codes
----------
0 — execution completed successfully
1 — internal failure
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    REPORTS_ROOT,
    PROCESSED_ROOT,
    TARGET_COLUMN,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_SPLIT_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "majority_vote_baseline_report.json"


# ---------------------------------------------------------------------------
# Model Class
# ---------------------------------------------------------------------------

class MajorityVoteBaselineModel:
    """Naive Majority Class Baseline Classifier."""

    def __init__(self, default_class: str = "continue_operation"):
        self.majority_class = default_class

    def fit(self, df: pd.DataFrame) -> None:
        if TARGET_COLUMN in df.columns:
            counts = Counter(df[TARGET_COLUMN].astype(str).str.strip().str.lower())
            if counts:
                self.majority_class = counts.most_common(1)[0][0]
                logger.info("Fitted majority class: '%s'", self.majority_class)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(self.majority_class, index=df.index)


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

def run_majority_vote_baseline(input_path: Path | None = None) -> dict:
    if input_path is None:
        if _TEST_SPLIT_PATH.exists():
            input_path = _TEST_SPLIT_PATH
        else:
            from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
            input_path = SAMPLE_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("MAJORITY VOTE BASELINE STARTED")
    logger.info("Evaluating dataset: %s", input_path)

    start_time = time.perf_counter()
    df = pd.read_csv(input_path, low_memory=False)

    model = MajorityVoteBaselineModel()
    model.fit(df)
    preds = model.predict(df)

    duration = time.perf_counter() - start_time
    dist = dict(Counter(preds))

    acc = None
    if TARGET_COLUMN in df.columns:
        acc = float((df[TARGET_COLUMN].astype(str).str.strip().str.lower() == preds).mean())
        logger.info("Baseline Accuracy vs Target: %.4f", acc)

    report = {
        "status": "success",
        "baseline_name": "majority_vote_baseline",
        "input_path": str(input_path),
        "majority_class": model.majority_class,
        "total_samples": len(df),
        "predictions_distribution": dist,
        "accuracy": round(acc, 4) if acc is not None else None,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Majority class: '%s'", model.majority_class)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("MAJORITY VOTE BASELINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_majority_vote_baseline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in majority_vote_baseline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
