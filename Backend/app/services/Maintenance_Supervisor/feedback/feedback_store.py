"""
feedback_store.py

Human Engineering Feedback Store & Audit Logger for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Persists human engineer overrides, corrections, and review feedback on supervisor decisions
to an auditable JSONL store.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import ARTIFACT_ROOT, REPORTS_ROOT
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_STORE_PATH: Final[Path] = ARTIFACT_ROOT / "feedback_store.jsonl"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "feedback_store_report.json"


@dataclass
class FeedbackEntry:
    sample_id: str
    original_decision: str
    engineer_decision: str
    engineer_id: str
    comments: str
    timestamp: float


class SupervisorFeedbackStore:
    """Manages persistent feedback logging for continuous policy learning."""

    def __init__(self, store_path: Path | str | None = None):
        self.store_path = Path(store_path or _STORE_PATH).resolve()

    def log_feedback(
        self,
        sample_id: str,
        original_decision: str,
        engineer_decision: str,
        engineer_id: str = "ENG_001",
        comments: str = "",
    ) -> FeedbackEntry:
        entry = FeedbackEntry(
            sample_id=str(sample_id),
            original_decision=str(original_decision),
            engineer_decision=str(engineer_decision),
            engineer_id=str(engineer_id),
            comments=str(comments),
            timestamp=time.time(),
        )

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.__dict__) + "\n")

        logger.info("Logged feedback for sample %s by %s.", sample_id, engineer_id)
        return entry

    def load_all_feedback(self) -> list[dict]:
        if not self.store_path.exists():
            return []

        entries = []
        with open(self.store_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries


def run_feedback_store() -> dict:
    section("FEEDBACK STORE TEST STARTED")
    start_time = time.perf_counter()

    store = SupervisorFeedbackStore()
    entry = store.log_feedback(
        sample_id="ENG_101_CYCLE_150",
        original_decision="schedule_maintenance",
        engineer_decision="immediate_maintenance",
        engineer_id="CHIEF_ENG_ALPHA",
        comments="Vibration harmonics indicate immediate bearing failure imminent.",
    )

    all_entries = store.load_all_feedback()
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "store_path": str(store.store_path),
        "total_feedback_entries": len(all_entries),
        "latest_logged_entry": entry.__dict__,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Total Feedback Entries Stored: %d", len(all_entries))
    logger.info("Report written to: %s", _REPORT_PATH)
    section("FEEDBACK STORE TEST COMPLETED")

    return report


def main() -> int:
    try:
        res = run_feedback_store()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in feedback_store: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
