"""
Pipeline stage 02: operating-mode discovery, transition marking and per-mode scaling.

Runs, in order:

    operating_mode_detector -> context_clusters.csv
    context_drift           -> drift report (metrics/context_drift.json)
    scaler                  -> scaled_features.csv

This is the stage that makes FD002/FD004 tractable: everything downstream
assumes sensors have already been normalised per operating regime.
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
from app.services.Anomaly_Health_Monitoring.context_modeling.context_drift import (  # noqa: E402
    ContextDriftDetector,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import (  # noqa: E402
    OperatingModeDetector,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import (  # noqa: E402
    PerModeScaler,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "02_context_modeling"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, retrain: bool = True) -> Dict[str, Any]:
    """Execute context modelling: regime discovery, drift check, scaling.

    Args:
        config: Active configuration.
        retrain: Fit fresh context/scaling models, or reload saved ones.

    Returns:
        A combined report of every sub-stage.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    detector = OperatingModeDetector(config)
    labelled_frame, context_report = detector.run(refit=retrain)

    drift_detector = ContextDriftDetector(config)
    drift_report = drift_detector.run(labelled_frame, capture_reference=retrain)

    scaler = PerModeScaler(config)
    scaled_frame, scaler_report = scaler.run(labelled_frame, refit=retrain)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "context": context_report,
        "drift": drift_report,
        "scaling": scaler_report,
        "rows_out": int(len(scaled_frame)),
    }
    logger.info(
        "%s complete: n_modes=%s, drift_detected=%s, rows=%d",
        STAGE_NAME,
        context_report.get("n_modes"),
        drift_report.get("drift_detected"),
        report["rows_out"],
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Stage 02: context modelling")
    parser.add_argument("--no-retrain", action="store_true", help="Reload saved models instead of refitting.")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    config = get_config()
    try:
        run_stage(config, retrain=not args.no_retrain)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
