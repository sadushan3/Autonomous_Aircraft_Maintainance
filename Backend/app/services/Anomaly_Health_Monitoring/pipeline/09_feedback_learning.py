"""
Pipeline stage 09: feedback learning.

Ingests freshly reasoned anomalies into the alert memory, recomputes the
adaptive anomaly threshold from the full operator-feedback history, and
reports the current feedback/threshold state.

    learning_updater -> feedback_updates.csv, residual_thresholds.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config  # noqa: E402
from app.services.Anomaly_Health_Monitoring.feedback.learning_updater import (  # noqa: E402
    LearningUpdater,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "09_feedback_learning"


@stage_guard(STAGE_NAME)
def run_stage(config: Config) -> Dict[str, Any]:
    """Execute feedback learning: memory ingestion + threshold adaptation.

    Args:
        config: Active configuration.

    Returns:
        The feedback-learning report.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    updater = LearningUpdater(config)
    learning_report = updater.run()

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        **learning_report,
    }
    logger.info(
        "%s complete: %d feedback record(s), threshold=%.4f",
        STAGE_NAME,
        learning_report.get("total_feedback_records", 0),
        learning_report.get("threshold_state", {}).get("fusion_threshold", 0.0),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    return argparse.ArgumentParser(description="Stage 09: feedback learning").parse_args(argv)


def main() -> None:
    """Entry point for standalone execution."""
    parse_args()
    config = get_config()
    try:
        run_stage(config)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
