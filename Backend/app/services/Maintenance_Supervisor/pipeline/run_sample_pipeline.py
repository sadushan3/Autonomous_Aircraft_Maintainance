"""
run_sample_pipeline.py

End-to-End Sample Data Generation Pipeline Orchestrator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Runs synthetic data generation (Phase E) and immediately triggers data validation (Phase F)
and feature preprocessing (Phase D).
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

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT
from app.services.Maintenance_Supervisor.data_generation.sample_output_generator import run_sample_generator
from app.services.Maintenance_Supervisor.data_integration.upstream_output_validator import run_upstream_validator
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "run_sample_pipeline_report.json"


def run_sample_pipeline() -> dict:
    section("SAMPLE PIPELINE ORCHESTRATOR STARTED")
    start_time = time.perf_counter()

    # 1. Generate sample output
    gen_report = run_sample_generator()

    # 2. Validate sample output
    val_report = run_upstream_validator()

    duration = time.perf_counter() - start_time

    report = {
        "status": "success" if val_report.is_valid else "validation_failed",
        "sample_generation": gen_report,
        "validation_result": val_report.to_dict(),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Sample pipeline completed in %.2fs.", duration)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("SAMPLE PIPELINE ORCHESTRATOR COMPLETED")

    return report


def main() -> int:
    try:
        res = run_sample_pipeline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in run_sample_pipeline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
