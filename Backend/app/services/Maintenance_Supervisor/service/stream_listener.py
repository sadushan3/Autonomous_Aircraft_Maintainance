"""
stream_listener.py

Real-Time Telemetry Stream Listener & Messaging Handler.

Purpose
-------
Listens to real-time engine telemetry streaming channels and emits supervisor decisions.
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
from app.services.Maintenance_Supervisor.service.schemas import TelemetryInputSchema
from app.services.Maintenance_Supervisor.service.supervisor_service import MaintenanceSupervisorService
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "stream_listener_report.json"


def run_stream_listener_simulation(n_events: int = 10) -> dict:
    section("TELEMETRY STREAM LISTENER STARTED")
    start_time = time.perf_counter()

    service = MaintenanceSupervisorService.get_instance()
    processed_events = []

    for i in range(1, n_events + 1):
        telemetry = TelemetryInputSchema(
            engine_id=f"ENG_{i:03d}",
            cycle=150,
            fd_subset="FD001",
            predicted_rul=14.5 - i,
            risk_10=0.05 * i,
            risk_30=0.10 * i,
            risk_50=0.15 * i,
            anomaly_score=0.04 * i,
            anomaly_severity="medium" if i > 5 else "none",
        )
        dec = service.predict_single(telemetry)
        processed_events.append(dec.model_dump())

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "events_streamed": len(processed_events),
        "sample_event_output": processed_events[0],
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Stream listener simulation processed %d events.", len(processed_events))
    logger.info("Report written to: %s", _REPORT_PATH)
    section("TELEMETRY STREAM LISTENER COMPLETED")

    return report


def main() -> int:
    try:
        res = run_stream_listener_simulation()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in stream_listener: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
