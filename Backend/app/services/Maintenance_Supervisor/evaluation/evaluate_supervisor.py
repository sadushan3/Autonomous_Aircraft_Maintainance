"""
evaluate_supervisor.py

Evaluation Suite for the Autonomous Maintenance Supervisor Agent.

Purpose
-------
Evaluates the champion supervisor decision fusion pipeline on the test split
and produces complete classification & domain metrics reports.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, TARGET_COLUMN
from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.services.Maintenance_Supervisor.evaluation.maintenance_metrics import compute_maintenance_metrics
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "evaluate_supervisor_report.json"


def run_supervisor_evaluation(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("CHAMPION SUPERVISOR EVALUATION STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    engine = DecisionFusionEngine()
    fused_df = engine.fuse_decisions(df)

    metrics_res = compute_maintenance_metrics(
        y_true=fused_df[TARGET_COLUMN],
        y_pred=fused_df["final_decision"],
    )

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "total_samples": len(fused_df),
        "metrics": metrics_res.to_dict(),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Accuracy: %.4f | Macro F1: %.4f | Cost Penalty: %.2f",
                metrics_res.accuracy, metrics_res.macro_f1, metrics_res.total_cost_penalty)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("CHAMPION SUPERVISOR EVALUATION COMPLETED")

    return report


def main() -> int:
    try:
        res = run_supervisor_evaluation()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in evaluate_supervisor: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
