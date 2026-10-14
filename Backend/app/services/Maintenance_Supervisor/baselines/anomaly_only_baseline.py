"""
anomaly_only_baseline.py

Single-Component (Anomaly-Only) Baseline Model for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module implements a baseline decision model that evaluates ONLY the predictions
produced by the upstream Anomaly & Health Monitoring component (anomaly_score, anomaly_severity),
completely ignoring RUL and Failure Risk outputs.

Ablation Value
--------------
Evaluating this model measures the isolated decision accuracy of Anomaly signals alone,
demonstrating why anomaly detection by itself is insufficient for scheduling non-critical
preventative maintenance.

Thresholds:
----------
- anomaly_severity == "critical" OR anomaly_score >= 0.80 -> immediate_maintenance
- anomaly_severity == "high" OR anomaly_score >= 0.65 -> schedule_inspection
- anomaly_severity == "medium" OR anomaly_score >= 0.45 -> monitor_closely
- default -> continue_operation

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.baselines.anomaly_only_baseline

Expected outputs
----------------
- reports/Maintenance_Supervisor/anomaly_only_baseline_report.json

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
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()
_CFG = SupervisorConfig()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_TEST_SPLIT_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "anomaly_only_baseline_report.json"


# ---------------------------------------------------------------------------
# Model Class
# ---------------------------------------------------------------------------

class AnomalyOnlyBaselineModel:
    """Anomaly-Only Single Component Decision Baseline."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

    def predict_row(self, row: pd.Series) -> str:
        score = float(row.get("anomaly_score", 0.0))
        sev = str(row.get("anomaly_severity", "none")).strip().lower()

        if sev == "critical" or score >= self.cfg.critical_anomaly_threshold:
            return "immediate_maintenance"
        if sev == "high" or score >= self.cfg.high_anomaly_threshold:
            return "schedule_inspection"
        if sev == "medium" or score >= self.cfg.medium_anomaly_threshold:
            return "monitor_closely"

        return "continue_operation"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict_row, axis=1)


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

def run_anomaly_only_baseline(input_path: Path | None = None) -> dict:
    if input_path is None:
        if _TEST_SPLIT_PATH.exists():
            input_path = _TEST_SPLIT_PATH
        else:
            from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
            input_path = SAMPLE_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("ANOMALY-ONLY BASELINE STARTED")
    logger.info("Evaluating dataset: %s", input_path)

    start_time = time.perf_counter()
    df = pd.read_csv(input_path, low_memory=False)

    model = AnomalyOnlyBaselineModel()
    preds = model.predict(df)

    duration = time.perf_counter() - start_time
    dist = dict(Counter(preds))

    acc = None
    if TARGET_COLUMN in df.columns:
        acc = float((df[TARGET_COLUMN].astype(str).str.strip().str.lower() == preds).mean())
        logger.info("Baseline Accuracy vs Target: %.4f", acc)

    report = {
        "status": "success",
        "baseline_name": "anomaly_only_baseline",
        "input_path": str(input_path),
        "total_samples": len(df),
        "predictions_distribution": dist,
        "accuracy": round(acc, 4) if acc is not None else None,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Predictions distribution: %s", dist)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("ANOMALY-ONLY BASELINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_anomaly_only_baseline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in anomaly_only_baseline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
