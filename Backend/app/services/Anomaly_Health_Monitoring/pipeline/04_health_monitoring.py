"""
Pipeline stage 04: health index, score, state and alert derivation.

Runs, in order:

    health_index_calculator -> health_index.csv
    health_score_engine     -> health_scores.csv (index + trend + score)
    health_state_classifier -> health_states.csv (+ alert_level from health_alert_engine)
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
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_alert_engine import (  # noqa: E402
    HealthAlertEngine,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import (  # noqa: E402
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import (  # noqa: E402
    HealthScoreEngine,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import (  # noqa: E402
    HealthStateClassifier,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    atomic_write_csv,
    get_logger,
    stage_guard,
)

STAGE_NAME = "04_health_monitoring"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, recalibrate: bool = True) -> Dict[str, Any]:
    """Execute health monitoring: index, score, state, alert.

    Args:
        config: Active configuration.
        recalibrate: Re-fit the health index's healthy-behaviour reference
            from this data (training). False reuses the persisted reference,
            which is required whenever this stage runs over data that is not
            the authoritative training set -- see
            :meth:`HealthIndexCalculator.fit_calibration` for why that
            reference has to stay fixed across batches.

    Returns:
        A combined report of every sub-stage.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    index_calculator = HealthIndexCalculator(config)
    health_index, index_report = index_calculator.run(recalibrate=recalibrate)

    score_engine = HealthScoreEngine(config)
    health_scores, score_report = score_engine.run(health_index)

    state_classifier = HealthStateClassifier(config)
    health_states, state_report = state_classifier.run(health_scores)

    alert_engine = HealthAlertEngine(config)
    alerted, alert_report = alert_engine.run(health_states)

    # Fold the alert level back into health_states.csv so downstream stages
    # (dashboard, exchange) see one enriched file rather than a fourth
    # near-duplicate carry-forward CSV.
    output = atomic_write_csv(
        alerted, config.paths.processed_path(config.paths.health_states_csv), config=config
    )

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "health_index": index_report,
        "health_score": score_report,
        "health_state": state_report,
        "health_alert": alert_report,
        "rows_out": int(len(alerted)),
        "output_path": str(output),
    }
    logger.info(
        "%s complete: mean_health_index=%.2f, state_distribution=%s",
        STAGE_NAME,
        index_report.get("mean_health_index", 0.0),
        state_report.get("state_distribution"),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Stage 04: health monitoring")
    parser.add_argument(
        "--no-recalibrate",
        action="store_true",
        help=(
            "Reuse the persisted healthy-behaviour reference instead of "
            "re-fitting it from this data."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    config = get_config()
    try:
        run_stage(config, recalibrate=not args.no_recalibrate)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
