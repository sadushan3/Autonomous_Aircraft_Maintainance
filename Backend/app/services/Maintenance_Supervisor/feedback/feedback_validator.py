"""
feedback_validator.py

Feedback Integrity & Malicious Input Validator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Validates human feedback entries to prevent corrupted or malicious overrides from
altering the agent's decision policies.
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

from app.config.Maintenance_Supervisor.supervisor_config import DECISION_CLASSES, REPORTS_ROOT
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "feedback_validator_report.json"


def validate_feedback_entry(entry: dict) -> tuple[bool, str]:
    """Validate a single human feedback dictionary entry."""
    req_keys = ["sample_id", "original_decision", "engineer_decision", "engineer_id"]
    for k in req_keys:
        if k not in entry or not str(entry[k]).strip():
            return False, f"Missing or empty required field '{k}'."

    eng_decision = str(entry["engineer_decision"]).strip().lower()
    if eng_decision not in DECISION_CLASSES:
        return False, f"Invalid engineer decision class '{eng_decision}'."

    return True, "Valid"


def run_feedback_validator() -> dict:
    section("FEEDBACK VALIDATOR TEST STARTED")
    start_time = time.perf_counter()

    sample_valid = {
        "sample_id": "S_001",
        "original_decision": "continue_operation",
        "engineer_decision": "schedule_inspection",
        "engineer_id": "ENG_42",
    }
    sample_invalid = {
        "sample_id": "S_002",
        "original_decision": "continue_operation",
        "engineer_decision": "invalid_action_class",
        "engineer_id": "ENG_42",
    }

    ok1, msg1 = validate_feedback_entry(sample_valid)
    ok2, msg2 = validate_feedback_entry(sample_invalid)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "valid_sample_test": {"is_valid": ok1, "message": msg1},
        "invalid_sample_test": {"is_valid": ok2, "message": msg2},
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Feedback validator tested: valid_test=%s, invalid_test=%s", ok1, ok2)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("FEEDBACK VALIDATOR TEST COMPLETED")

    return report


def main() -> int:
    try:
        res = run_feedback_validator()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in feedback_validator: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
