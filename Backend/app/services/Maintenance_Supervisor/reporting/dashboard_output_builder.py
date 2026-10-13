"""
dashboard_output_builder.py

Frontend Dashboard JSON & Telemetry Export Builder for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Transforms inference pipeline decision outputs into structured, frontend-ready JSON
payloads formatted specifically for Web/UI dashboard visualization.
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

from app.config.Maintenance_Supervisor.supervisor_config import ARTIFACT_ROOT, REPORTS_ROOT, PROCESSED_ROOT
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_DASHBOARD_JSON_PATH: Final[Path] = ARTIFACT_ROOT / "dashboard_telemetry_export.json"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "dashboard_output_builder_report.json"


def build_dashboard_export(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("DASHBOARD OUTPUT BUILDER STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    # Structure telemetry records for frontend
    telemetry_records = []
    for idx in range(min(50, len(df))):
        row = df.iloc[idx]
        telemetry_records.append({
            "engine_id": str(row.get("engine_id", "1")),
            "cycle": int(row.get("cycle", 1)),
            "fd_subset": str(row.get("fd_subset", "FD001")),
            "predicted_rul": float(row.get("predicted_rul", 100.0)),
            "risk_30": float(row.get("risk_30", 0.0)),
            "anomaly_score": float(row.get("anomaly_score", 0.0)),
            "decision": str(row.get("final_decision", "continue_operation")),
            "priority": str(row.get("priority", "Low")),
            "urgency": str(row.get("maintenance_urgency", "none")),
            "confidence_score": float(row.get("confidence_score", 0.85)),
            "requires_human_review": bool(row.get("requires_human_review", False)),
        })

    summary_stats = {
        "total_fleet_engines": int(df["engine_id"].nunique()) if "engine_id" in df.columns else 1,
        "total_cycles_monitored": len(df),
        "critical_alerts_count": int(sum(1 for r in telemetry_records if r["priority"] == "Critical")),
        "high_priority_count": int(sum(1 for r in telemetry_records if r["priority"] == "High")),
    }

    export_payload = {
        "generated_at": time.time(),
        "summary": summary_stats,
        "fleet_telemetry": telemetry_records,
    }

    _DASHBOARD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(export_payload, _DASHBOARD_JSON_PATH)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "export_path": str(_DASHBOARD_JSON_PATH),
        "telemetry_records_exported": len(telemetry_records),
        "summary_stats": summary_stats,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Dashboard export written to: %s", _DASHBOARD_JSON_PATH)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("DASHBOARD OUTPUT BUILDER COMPLETED")

    return report


def main() -> int:
    try:
        res = build_dashboard_export()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in dashboard_output_builder: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
