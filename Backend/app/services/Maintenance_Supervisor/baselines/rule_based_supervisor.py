"""
rule_based_supervisor.py

Expert Engineering Rule-Based Baseline Model for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module implements a deterministic, expert engineering rule-based baseline
decision model. It evaluates domain-specific safety policies across upstream RUL,
Failure Risk, and Anomaly outputs to produce maintenance decisions without machine
learning models.

Decision Hierarchy (Highest Priority Wins):
-------------------------------------------
1. immediate_maintenance:
   - predicted_rul <= 10.0 OR risk_10 >= 0.82 OR anomaly_severity == "critical"
2. schedule_maintenance:
   - predicted_rul <= 30.0 OR risk_30 >= 0.72 OR predicted_fail_in_30 == True
3. schedule_inspection:
   - predicted_rul <= 50.0 OR risk_30 >= 0.55 OR anomaly_severity == "high"
4. monitor_closely:
   - predicted_rul <= 80.0 OR risk_50 >= 0.40 OR anomaly_severity == "medium"
5. continue_operation:
   - Default when no risk thresholds are triggered.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.baselines.rule_based_supervisor

Expected outputs
----------------
- reports/Maintenance_Supervisor/rule_based_supervisor_report.json

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

import numpy as np
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
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "rule_based_supervisor_report.json"


# ---------------------------------------------------------------------------
# Model Class
# ---------------------------------------------------------------------------

class RuleBasedSupervisorModel:
    """Expert Rule-Based Maintenance Decision Baseline."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

    def predict_row(self, row: pd.Series) -> str:
        rul = float(row.get("predicted_rul", 999.0))
        r10 = float(row.get("risk_10", 0.0))
        r30 = float(row.get("risk_30", 0.0))
        r50 = float(row.get("risk_50", 0.0))
        fail_30 = bool(row.get("predicted_fail_in_30", False))
        anom_sev = str(row.get("anomaly_severity", "none")).strip().lower()

        if (
            rul <= self.cfg.immediate_rul_threshold
            or r10 >= self.cfg.critical_risk_10_threshold
            or anom_sev == "critical"
        ):
            return "immediate_maintenance"

        if (
            rul <= self.cfg.maintenance_rul_threshold
            or r30 >= self.cfg.maintenance_risk_30_threshold
            or fail_30
        ):
            return "schedule_maintenance"

        if (
            rul <= self.cfg.inspection_rul_threshold
            or r30 >= self.cfg.inspection_risk_30_threshold
            or anom_sev == "high"
        ):
            return "schedule_inspection"

        if (
            rul <= self.cfg.monitoring_rul_threshold
            or r50 >= self.cfg.monitoring_risk_50_threshold
            or anom_sev == "medium"
        ):
            return "monitor_closely"

        return "continue_operation"

    def predict(self, df: pd.DataFrame) -> pd.Series:
        return df.apply(self.predict_row, axis=1)


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

def run_rule_based_baseline(input_path: Path | None = None) -> dict:
    if input_path is None:
        if _TEST_SPLIT_PATH.exists():
            input_path = _TEST_SPLIT_PATH
        else:
            from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
            input_path = SAMPLE_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("RULE-BASED SUPERVISOR BASELINE STARTED")
    logger.info("Evaluating dataset: %s", input_path)

    start_time = time.perf_counter()
    df = pd.read_csv(input_path, low_memory=False)

    model = RuleBasedSupervisorModel()
    preds = model.predict(df)

    duration = time.perf_counter() - start_time
    dist = dict(Counter(preds))

    acc = None
    if TARGET_COLUMN in df.columns:
        acc = float((df[TARGET_COLUMN].astype(str).str.strip().str.lower() == preds).mean())
        logger.info("Baseline Accuracy vs Target: %.4f", acc)

    report = {
        "status": "success",
        "baseline_name": "rule_based_supervisor",
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
    section("RULE-BASED SUPERVISOR BASELINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_rule_based_baseline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in rule_based_supervisor: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
