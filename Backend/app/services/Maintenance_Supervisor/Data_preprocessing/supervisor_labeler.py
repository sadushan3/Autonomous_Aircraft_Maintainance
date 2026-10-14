"""
supervisor_labeler.py

Rule-Based Ground-Truth Target Generator & Labeler for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module constructs deterministic, rule-based ground-truth targets (final_decision)
for synthetic training or semi-supervised labeling of integrated predictive maintenance
datasets.

Ground-Truth Decision Hierarchy (Highest Priority Wins):
--------------------------------------------------------
1. immediate_maintenance:
   - predicted_rul <= 10 OR risk_10 >= 0.82 OR anomaly_severity == "critical"

2. schedule_maintenance:
   - predicted_rul <= 30 OR risk_30 >= 0.72 OR failure_in_30 == True

3. schedule_inspection:
   - predicted_rul <= 50 OR risk_30 >= 0.55 OR anomaly_severity == "high"

4. monitor_closely:
   - predicted_rul <= 80 OR risk_50 >= 0.40 OR anomaly_severity == "medium"

5. continue_operation:
   - Default when none of the above risk thresholds are triggered.

Derived Attributes Generated:
----------------------------
- final_decision (string target)
- priority ("Low", "Medium", "High", "Critical")
- maintenance_urgency ("none", "observe_next_cycles", "within_30_cycles", etc.)
- requires_human_review (bool: triggered by high uncertainty or agent conflicts)
- supervisor_score (float [0.0, 1.0]: aggregated decision severity index)

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.supervisor_labeler

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_sanitized_input.csv (or specified path)

Expected outputs
---------------
- processed/Maintenance_Supervisor/supervisor_labeled_dataset.csv
- reports/Maintenance_Supervisor/supervisor_labeler_report.json

Exit codes
----------
0 — labeling completed successfully
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
    PROCESSED_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    DECISION_CLASSES,
    DECISION_TO_SEVERITY,
    PRIORITY_MAPPING,
    URGENCY_MAPPING,
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_csv,
)

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()
_CFG = SupervisorConfig()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_INPUT_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_sanitized_input.csv"
_OUTPUT_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_labeled_dataset.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "supervisor_labeler_report.json"


# ---------------------------------------------------------------------------
# Core Labeling Logic
# ---------------------------------------------------------------------------

def assign_rule_based_decision(row: pd.Series) -> str:
    """Evaluate decision rule hierarchy for a single dataset record."""
    rul = float(row.get("predicted_rul", 999.0))
    r10 = float(row.get("risk_10", 0.0))
    r30 = float(row.get("risk_30", 0.0))
    r50 = float(row.get("risk_50", 0.0))
    fail_30 = bool(row.get("predicted_fail_in_30", False))
    anom_sev = str(row.get("anomaly_severity", "none")).strip().lower()

    # Rule 1: Immediate Maintenance
    if (
        rul <= _CFG.immediate_rul_threshold
        or r10 >= _CFG.critical_risk_10_threshold
        or anom_sev == "critical"
    ):
        return "immediate_maintenance"

    # Rule 2: Schedule Maintenance
    if (
        rul <= _CFG.maintenance_rul_threshold
        or r30 >= _CFG.maintenance_risk_30_threshold
        or fail_30
    ):
        return "schedule_maintenance"

    # Rule 3: Schedule Inspection
    if (
        rul <= _CFG.inspection_rul_threshold
        or r30 >= _CFG.inspection_risk_30_threshold
        or anom_sev == "high"
    ):
        return "schedule_inspection"

    # Rule 4: Monitor Closely
    if (
        rul <= _CFG.monitoring_rul_threshold
        or r50 >= _CFG.monitoring_risk_50_threshold
        or anom_sev == "medium"
    ):
        return "monitor_closely"

    # Rule 5: Continue Operation
    return "continue_operation"


def compute_human_review_and_score(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Compute human review flags and continuous supervisor severity scores."""
    u_width = df.get("uncertainty_width", pd.Series(0.0, index=df.index))
    conflict = df.get("decision_conflict_score", pd.Series(0.0, index=df.index))
    c_conf = df.get("context_confidence", pd.Series(1.0, index=df.index))

    # Flag human review if uncertainty is high, conflict is high, or context confidence is low
    requires_review = (
        (u_width >= 30.0)
        | (conflict >= _CFG.high_conflict_threshold)
        | (c_conf <= _CFG.low_context_confidence_threshold)
    )

    # Compute continuous supervisor score [0.0, 1.0]
    severities = df[TARGET_COLUMN].map(DECISION_TO_SEVERITY).fillna(0)
    max_sev = float(max(DECISION_TO_SEVERITY.values()))
    base_score = severities / max_sev

    # Adjust score slightly by uncertainty and conflict
    supervisor_score = (base_score * 0.8 + conflict * 0.2).clip(0.0, 1.0)

    return requires_review, supervisor_score


def label_supervisor_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Label dataset with decisions, priorities, urgencies, and review requirements."""
    labeled_df = df.copy()

    # Assign primary target
    logger.info("Applying rule-based decision hierarchy...")
    labeled_df[TARGET_COLUMN] = labeled_df.apply(assign_rule_based_decision, axis=1)

    # Assign mapped priority and urgency
    labeled_df["priority"] = labeled_df[TARGET_COLUMN].map(PRIORITY_MAPPING)
    labeled_df["maintenance_urgency"] = labeled_df[TARGET_COLUMN].map(URGENCY_MAPPING)

    # Compute review flags and supervisor score
    requires_review, score = compute_human_review_and_score(labeled_df)
    labeled_df["requires_human_review"] = requires_review
    labeled_df["supervisor_score"] = score

    return labeled_df


# ---------------------------------------------------------------------------
# CLI & Standalone Orchestrator
# ---------------------------------------------------------------------------

def run_supervisor_labeler(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run standalone ground-truth labeler pipeline."""
    if input_path is None:
        input_path = _INPUT_PATH
    if output_path is None:
        output_path = _OUTPUT_PATH

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    section("SUPERVISOR LABELER STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Output dataset: %s", output_path)

    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        if SAMPLE_DATASET_PATH.exists():
            input_path = SAMPLE_DATASET_PATH
            logger.info("Sanitized input not found, falling back to sample input: %s", input_path)
        else:
            raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    labeled_df = label_supervisor_dataset(df)

    duration = time.perf_counter() - start_time
    class_counts = dict(Counter(labeled_df[TARGET_COLUMN]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(labeled_df, output_path)

    report = {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows_labeled": len(labeled_df),
        "class_distribution": class_counts,
        "human_review_required_count": int(labeled_df["requires_human_review"].sum()),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Labeled %d rows.", len(labeled_df))
    logger.info("Class distribution: %s", class_counts)
    logger.info("Human review required: %d rows", report["human_review_required_count"])
    logger.info("Output written to: %s", output_path)
    section("SUPERVISOR LABELER COMPLETED")

    return report


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_supervisor_labeler()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in supervisor_labeler: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
