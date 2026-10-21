from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import (
    DataCleaner,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.data_loader import (
    DataLoader,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.feature_engineering import (
    FeatureEngineer,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    StageExecutionError,
    get_logger,
    stage_guard,
)

STAGE_NAME = "01_preprocessing"


@stage_guard(STAGE_NAME)
def run_stage(
    config: Config,
    subsets: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    source_csv: Optional[str] = None,
    max_engines: Optional[int] = None,
    fit_policy: bool = True,
) -> Dict[str, Any]:


    logger = get_logger(__name__, config)
    started = time.perf_counter()

    loader = DataLoader(config)
    raw_frame, load_report = loader.run(subsets, split, source_csv, max_engines)

    cleaner = DataCleaner(config)
    cleaned_frame, clean_report = cleaner.run(raw_frame, fit_policy=fit_policy)

    engineer = FeatureEngineer(config)
    engineered_frame, feature_report = engineer.run(cleaned_frame, fit_policy=fit_policy)

    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "duration_seconds": round(time.perf_counter() - started, 3),
        "load": load_report,
        "clean": clean_report,
        "feature_engineering": feature_report,
        "rows_out": int(len(engineered_frame)),
        "engines_out": int(engineered_frame["unit"].nunique()),
    }
    logger.info("%s complete: %s", STAGE_NAME, {k: v for k, v in report.items() if k not in ("load", "clean", "feature_engineering")})
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:


    parser = argparse.ArgumentParser(description="Stage 01: preprocessing")
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--source-csv", type=str, default=None)
    parser.add_argument("--max-engines", type=int, default=None)
    parser.add_argument(
        "--no-fit-policy",
        action="store_true",
        help="Apply the persisted sensor/feature policy instead of fitting a new one.",
    )
    return parser.parse_args(argv)


def main() -> None:

    args = parse_args()
    config = get_config()
    try:
        run_stage(
            config,
            args.subsets,
            args.split,
            args.source_csv,
            args.max_engines,
            fit_policy=not args.no_fit_policy,
        )
    except StageExecutionError as exc:
        get_logger(__name__, config).error("Stage failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
