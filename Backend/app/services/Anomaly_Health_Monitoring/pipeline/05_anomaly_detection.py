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
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import (
    AnomalyFusion,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "05_anomaly_detection"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, retrain: bool = True) -> Dict[str, Any]:


    logger = get_logger(__name__, config)
    started = time.perf_counter()

    fusion = AnomalyFusion(config)
    _, fusion_report = fusion.run(retrain=retrain)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        **fusion_report,
    }
    logger.info(
        "%s complete: %d fault(s), %d operational, threshold=%.4f",
        STAGE_NAME,
        fusion_report.get("fault_anomalies", 0),
        fusion_report.get("operational_anomalies", 0),
        fusion_report.get("fusion_threshold", 0.0),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:


    parser = argparse.ArgumentParser(description="Stage 05: anomaly detection")
    parser.add_argument("--no-retrain", action="store_true")
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
