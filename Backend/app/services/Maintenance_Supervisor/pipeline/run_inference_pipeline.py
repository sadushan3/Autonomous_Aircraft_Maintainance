"""
run_inference_pipeline.py

Batch & Streaming Real-Time Inference Pipeline Orchestrator.

Purpose
-------
Runs real-time decision fusion, safety policy enforcement, confidence scoring,
and natural language explanation generation on input datasets.
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

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, ARTIFACT_ROOT
from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.services.Maintenance_Supervisor.explanation.supervisor_explainer import SupervisorExplainer
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json, atomic_write_csv

logger = get_logger()
_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_OUTPUT_PATH: Final[Path] = ARTIFACT_ROOT / "inference_pipeline_output.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "run_inference_pipeline_report.json"


def run_inference_pipeline(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("INFERENCE PIPELINE ORCHESTRATOR STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    engine = DecisionFusionEngine()
    fused_df = engine.fuse_decisions(df)

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(fused_df, _OUTPUT_PATH)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(_OUTPUT_PATH),
        "total_samples_processed": len(fused_df),
        "mean_confidence": round(float(fused_df["confidence_score"].mean()), 4),
        "human_reviews_required": int(fused_df["requires_human_review"].sum()),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Inference pipeline completed for %d samples.", len(fused_df))
    logger.info("Output written to: %s", _OUTPUT_PATH)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("INFERENCE PIPELINE ORCHESTRATOR COMPLETED")

    return report


def main() -> int:
    try:
        res = run_inference_pipeline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in run_inference_pipeline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
