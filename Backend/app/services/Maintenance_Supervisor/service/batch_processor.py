"""
batch_processor.py

Offline Batch Dataset Processor & File Processing Worker.

Purpose
-------
Processes large historical dataset CSVs/Parquets in chunked batches for offline
batch inference.
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
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json, atomic_write_csv

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_BATCH_OUTPUT_PATH: Final[Path] = ARTIFACT_ROOT / "batch_processed_decisions.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "batch_processor_report.json"


def process_batch_file(
    input_path: Path | None = None,
    output_path: Path | None = None,
    chunksize: int = 5000,
) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()
    output_path = Path(output_path or _BATCH_OUTPUT_PATH).resolve()

    section("OFFLINE BATCH PROCESSOR STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    engine = DecisionFusionEngine()
    total_processed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_chunk = True

    for chunk in pd.read_csv(input_path, chunksize=chunksize, low_memory=False):
        fused_chunk = engine.fuse_decisions(chunk)
        mode = "w" if first_chunk else "a"
        header = first_chunk
        fused_chunk.to_csv(output_path, mode=mode, header=header, index=False)
        first_chunk = False
        total_processed += len(fused_chunk)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_records_processed": total_processed,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Processed %d records in batch.", total_processed)
    logger.info("Batch output written to: %s", output_path)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("OFFLINE BATCH PROCESSOR COMPLETED")

    return report


def main() -> int:
    try:
        res = process_batch_file()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in batch_processor: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
