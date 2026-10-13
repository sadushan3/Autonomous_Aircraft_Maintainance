"""
cost_evaluator.py

Economic Loss & Maintenance Cost Benefit Evaluator for the
Autonomous Maintenance Supervisor Agent.

Calculates cost-sensitive maintenance penalties comparing:
1. Champion Supervisor Agent
2. Time-Between-Overhaul (TBO) Fixed Schedule Baseline
3. Risk-Only Baseline
4. RUL-Only Baseline
5. Anomaly-Only Baseline
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

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
)
from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

# Cost Matrix Constants (USD $)
COST_CATASTROPHIC_FAILURE: Final[float] = 500000.0   # Missed critical failure (FN)
COST_PREMATURE_OVERHAUL: Final[float] = 35000.0      # False alarm overhaul (FP)
COST_PREVENTIVE_MAINT: Final[float] = 12000.0        # Timely scheduled maintenance (TP)
COST_ROUTINE_INSPECTION: Final[float] = 1500.0       # Scheduled inspection
COST_CONTINUE_OPERATION: Final[float] = 0.0          # Normal operation


def calculate_economic_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate economic cost metrics for a set of decision predictions."""
    y_true_str = np.array(y_true, dtype=str)
    y_pred_str = np.array(y_pred, dtype=str)

    total_samples = len(y_true_str)
    total_cost = 0.0

    missed_critical = 0
    false_critical = 0
    timely_maintenance = 0
    inspections = 0

    critical_labels = {"immediate_maintenance", "schedule_maintenance"}

    for true_label, pred_label in zip(y_true_str, y_pred_str):
        # Case 1: Missed Critical Failure (True is Immediate/Schedule Maint, Pred is Continue/Monitor)
        if true_label in critical_labels and pred_label not in critical_labels:
            total_cost += COST_CATASTROPHIC_FAILURE
            missed_critical += 1

        # Case 2: False Critical Alarm (True is Continue Op, Pred is Immediate Maint)
        elif true_label == "continue_operation" and pred_label == "immediate_maintenance":
            total_cost += COST_PREMATURE_OVERHAUL
            false_critical += 1

        # Case 3: Timely Maintenance (Correctly flagged critical)
        elif true_label in critical_labels and pred_label in critical_labels:
            total_cost += COST_PREVENTIVE_MAINT
            timely_maintenance += 1

        # Case 4: Routine Inspection
        elif pred_label == "schedule_inspection":
            total_cost += COST_ROUTINE_INSPECTION
            inspections += 1

    mean_cost_per_engine_cycle = total_cost / total_samples if total_samples > 0 else 0.0

    return {
        "total_cost_usd": round(total_cost, 2),
        "mean_cost_per_cycle_usd": round(mean_cost_per_engine_cycle, 2),
        "missed_critical_events": missed_critical,
        "false_critical_alarms": false_critical,
        "timely_maintenance_events": timely_maintenance,
        "inspections_performed": inspections,
    }


def run_cost_evaluation(
    test_path: Path | None = None,
) -> dict:
    """Run economic cost evaluation comparing Supervisor against traditional baselines."""
    section("ECONOMIC LOSS & MAINTENANCE COST EVALUATOR STARTED")
    start_time = time.perf_counter()

    if test_path is None:
        # Fixed bug: this used to default to splits/test.csv, the
        # RobustScaler-standardized split. compare_baselines.py already
        # documented why that breaks champion scoring: the champion is
        # trained on -- and served at inference on -- raw exchange-scale
        # values, so feeding it standardized features puts every value far
        # outside its training range (compare_baselines.py measured this
        # collapsing accuracy to 0.0567 there). supervisor_encoded_dataset.csv
        # is the pre-scaling form and matches what the model actually
        # receives; use the same input compare_baselines.py settled on.
        test_path = PROCESSED_ROOT / "supervisor_encoded_dataset.csv"

    if not test_path.exists():
        logger.error("Test dataset not found at %s", test_path)
        return {"status": "error", "message": f"Test dataset not found at {test_path}"}

    dataframe = pd.read_csv(test_path, low_memory=False)
    if TARGET_COLUMN not in dataframe.columns:
        logger.error("Target column '%s' missing from dataset.", TARGET_COLUMN)
        return {"status": "error", "message": f"Target column '{TARGET_COLUMN}' missing."}

    y_true = dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower()

    # 1. Champion Supervisor Predictions
    # Fixed bug: this used to call calculate_economic_cost(y_true, y_true) --
    # i.e. it compared the true label against itself, so the "Champion
    # Supervisor Agent" cost was fabricated to always be the theoretical
    # perfect-score minimum (every row correctly a "timely maintenance" or
    # "continue operation" case) and never touched the actual trained model.
    # Score the real champion via the same DecisionFusionEngine
    # compare_baselines.py uses, and cost its real final_decision output.
    fusion_engine = DecisionFusionEngine()
    fused_df = fusion_engine.fuse_decisions(dataframe)
    y_pred_champion = fused_df["final_decision"].astype(str).str.strip().str.lower()
    supervisor_cost = calculate_economic_cost(y_true, y_pred_champion)

    # 2. Time-Between-Overhaul (TBO) Baseline (Fixed schedule overhaul every N cycles)
    # TBO predicts Continue Operation until fixed limit then Immediate Maintenance
    tbo_preds = np.where(dataframe.get("cycle", 0) > 180, "immediate_maintenance", "continue_operation")
    tbo_cost = calculate_economic_cost(y_true, tbo_preds)

    # 3. Majority Vote Baseline (Always Continue Operation)
    majority_preds = np.full(len(y_true), "continue_operation")
    majority_cost = calculate_economic_cost(y_true, majority_preds)

    # Calculate Savings vs TBO Baseline
    cost_savings_usd = tbo_cost["total_cost_usd"] - supervisor_cost["total_cost_usd"]
    savings_pct = (cost_savings_usd / tbo_cost["total_cost_usd"] * 100.0) if tbo_cost["total_cost_usd"] > 0 else 0.0

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "test_records_evaluated": len(dataframe),
        "cost_savings_vs_tbo_usd": round(cost_savings_usd, 2),
        "cost_reduction_percentage": round(savings_pct, 2),
        "evaluations": {
            "champion_supervisor_agent": supervisor_cost,
            "time_between_overhaul_baseline": tbo_cost,
            "majority_vote_baseline": majority_cost,
        },
        "cost_matrix_usd": {
            "catastrophic_failure_fn": COST_CATASTROPHIC_FAILURE,
            "premature_overhaul_fp": COST_PREMATURE_OVERHAUL,
            "preventive_maintenance_tp": COST_PREVENTIVE_MAINT,
            "routine_inspection": COST_ROUTINE_INSPECTION,
        },
        "duration_seconds": round(duration, 4),
    }

    report_path = REPORTS_ROOT / "cost_evaluation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, report_path)

    logger.info("Cost Savings vs Fixed TBO Baseline: $%.2f (%.1f%% reduction)", cost_savings_usd, savings_pct)
    logger.info("Report written to: %s", report_path)
    section("ECONOMIC LOSS & MAINTENANCE COST EVALUATOR COMPLETED")

    return report


if __name__ == "__main__":
    res = run_cost_evaluation()
    print(json.dumps(res, indent=2))
