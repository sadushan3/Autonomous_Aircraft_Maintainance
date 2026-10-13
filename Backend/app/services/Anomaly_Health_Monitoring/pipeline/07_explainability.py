"""
Pipeline stage 07: explanation generation.

Turns the root-cause records into human-readable narratives (WHY / WHICH
sensors / WHAT pattern / HOW severe), optionally including local grouped SHAP
against the calibrated monitoring ensemble.

    explanation_generator -> explanation_reports.csv
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
from app.services.Anomaly_Health_Monitoring.explainability.explanation_generator import (  # noqa: E402
    ExplanationGenerator,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "07_explainability"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, include_shap: bool = True) -> Dict[str, Any]:
    """Execute explanation generation.

    Args:
        config: Active configuration.
        include_shap: Whether to compute local SHAP for a bounded event sample.

    Returns:
        The explanation-generation report.
    """
    logger = get_logger(__name__, config)
    started = time.perf_counter()

    generator = ExplanationGenerator(config)
    _, generation_report = generator.run(include_shap=include_shap)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        **generation_report,
    }
    logger.info(
        "%s complete: %d explanation(s) generated",
        STAGE_NAME,
        generation_report.get("rows_explained", 0),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Stage 07: explainability")
    parser.add_argument("--no-shap", action="store_true", help="Skip local SHAP computation.")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    config = get_config()
    try:
        run_stage(config, include_shap=not args.no_shap)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
