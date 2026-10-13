"""
run_real_output_pipeline.py

Real Upstream Data Integration & Preprocessing Pipeline Orchestrator.

Purpose
-------
Orchestrates loading, validating, merging, and preprocessing real prediction outputs
from upstream research components.
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
from app.services.Maintenance_Supervisor.data_integration.real_output_merger import run_real_output_merger
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "run_real_output_pipeline_report.json"


def run_real_output_pipeline(
    rul_path: Path | None = None,
    risk_path: Path | None = None,
    anomaly_path: Path | None = None,
) -> dict:
    section("REAL OUTPUT PIPELINE STARTED")
    start_time = time.perf_counter()

    merge_res = run_real_output_merger(rul_path, risk_path, anomaly_path)
    duration = time.perf_counter() - start_time

    report = {
        "status": merge_res.get("status", "success"),
        "merger_report": merge_res,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Real output pipeline completed in %.2fs.", duration)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("REAL OUTPUT PIPELINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_real_output_pipeline()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in run_real_output_pipeline: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
