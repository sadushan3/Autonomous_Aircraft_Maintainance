"""
Pipeline stage 08: uncertainty estimation.

Combines twin disagreement, detector disagreement and data quality into the
confidence / reliability / uncertainty triple every prediction carries.

    confidence_estimator -> confidence_scores.csv
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
from app.services.Anomaly_Health_Monitoring.uncertainty.confidence_estimator import (  # noqa: E402
    ConfidenceEstimator,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "08_uncertainty"


@stage_guard(STAGE_NAME)
def run_stage(config: Config) -> Dict[str, Any]:
    """Execute uncertainty estimation.

    Args:
        config: Active configuration.

    Returns:
        The confidence-estimation report.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    estimator = ConfidenceEstimator(config)
    _, confidence_report = estimator.run()

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        **confidence_report,
    }
    logger.info(
        "%s complete: mean_confidence=%.2f",
        STAGE_NAME,
        confidence_report.get("mean_confidence", 0.0),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    return argparse.ArgumentParser(description="Stage 08: uncertainty").parse_args(argv)


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
