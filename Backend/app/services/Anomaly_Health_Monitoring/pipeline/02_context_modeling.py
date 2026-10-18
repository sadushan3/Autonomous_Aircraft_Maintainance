from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.context_modeling.context_drift import (
    ContextDriftDetector,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import (
    OperatingModeDetector,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import (
    PerModeScaler,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "02_context_modeling"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, retrain: bool = True) -> Dict[str, Any]:


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


    parser = argparse.ArgumentParser(description="Stage 02: context modelling")
    parser.add_argument("--no-retrain", action="store_true", help="Reload saved models instead of refitting.")
    return parser.parse_args(argv)


def main() -> None:

    args = parse_args()
    config = get_config()
    try:
        run_stage(config, retrain=not args.no_retrain)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
