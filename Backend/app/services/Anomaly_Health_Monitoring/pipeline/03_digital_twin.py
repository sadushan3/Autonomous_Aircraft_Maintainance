"""
Pipeline stage 03: Digital Twin training, residual calculation and validation.

Runs, in order:

    ensemble_twin (RF, XGBoost, LightGBM, Keras + fusion) -> twin_predictions.csv
    residual_calculator                                    -> residuals.csv
    twin_comparator                                          (row/engine summaries)
    validate_residuals                                       (acceptance gate)
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
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import (  # noqa: E402
    EnsembleTwin,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (  # noqa: E402
    ResidualCalculator,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.twin_comparator import (  # noqa: E402
    TwinComparator,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.validate_residuals import (  # noqa: E402
    ResidualValidator,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "03_digital_twin"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, retrain: bool = True) -> Dict[str, Any]:
    """Execute the Digital Twin stage: train/predict, residuals, validation.

    Args:
        config: Active configuration.
        retrain: Train fresh twins, or reload saved ones.

    Returns:
        A combined report of every sub-stage.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    ensemble = EnsembleTwin(config)
    predictions, twin_report = ensemble.run(retrain=retrain)

    calculator = ResidualCalculator(config)
    residuals, residual_report = calculator.run(recalibrate=retrain)

    comparator = TwinComparator(config)
    _, comparison_report = comparator.run(residuals)

    validator = ResidualValidator(config)
    validation_report = validator.run(residuals)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "twin": twin_report,
        "residuals": residual_report,
        "comparison": comparison_report,
        "validation": validation_report,
        "rows_out": int(len(residuals)),
    }
    logger.info(
        "%s complete: twins=%s, mean_abs_residual=%.4f, validation_passed=%s",
        STAGE_NAME,
        twin_report.get("twins"),
        residual_report.get("mean_abs_residual", 0.0),
        validation_report.get("passed"),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Stage 03: digital twin")
    parser.add_argument("--no-retrain", action="store_true")
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
