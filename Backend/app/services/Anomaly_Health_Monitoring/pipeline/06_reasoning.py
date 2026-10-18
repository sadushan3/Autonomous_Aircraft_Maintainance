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
from app.services.Anomaly_Health_Monitoring.reasoning.root_cause_analyzer import (
    RootCauseAnalyzer,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "06_reasoning"


@stage_guard(STAGE_NAME)
def run_stage(config: Config, only_anomalies: bool = True) -> Dict[str, Any]:


    logger = get_logger(__name__, config)
    started = time.perf_counter()

    analyzer = RootCauseAnalyzer(config)
    _, analysis_report = analyzer.run(only_anomalies=only_anomalies)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        **analysis_report,
    }
    logger.info(
        "%s complete: reasoned %d/%d row(s)",
        STAGE_NAME,
        analysis_report.get("rows_reasoned", 0),
        analysis_report.get("rows_input", 0),
    )
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:


    parser = argparse.ArgumentParser(description="Stage 06: reasoning")
    parser.add_argument("--all-rows", action="store_true", help="Reason about every row, not just anomalies.")
    return parser.parse_args(argv)


def main() -> None:

    args = parse_args()
    config = get_config()
    try:
        run_stage(config, only_anomalies=not args.all_rows)
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
