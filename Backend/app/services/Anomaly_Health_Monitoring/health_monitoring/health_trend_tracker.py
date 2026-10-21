from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import (
    HEALTH_INDEX_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    get_logger,
    read_csv,
    rolling_slope,
    safe_divide,
)


HEALTH_TREND_COLUMN = "health_trend"


DETERIORATION_RATE_COLUMN = "deterioration_rate"


REMAINING_HEALTH_PCT_COLUMN = "remaining_health_pct"


_BEST_HEALTH_SO_FAR_COLUMN = "_best_health_so_far"


class HealthTrendTracker:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def compute(self, health_index: pd.DataFrame) -> pd.DataFrame:


        result = health_index.sort_values(get_identity_columns()).reset_index(drop=True)
        window = int(self.config.health.trend_window)

        result[HEALTH_TREND_COLUMN] = (
            result.groupby("unit", sort=False)[HEALTH_INDEX_COLUMN]
            .transform(lambda series: rolling_slope(series, window))
        )
        result[DETERIORATION_RATE_COLUMN] = np.clip(-result[HEALTH_TREND_COLUMN], 0.0, None)

        best_so_far = result.groupby("unit", sort=False)[HEALTH_INDEX_COLUMN].cummax()
        result[_BEST_HEALTH_SO_FAR_COLUMN] = best_so_far.clip(lower=1.0)
        result[REMAINING_HEALTH_PCT_COLUMN] = np.clip(
            safe_divide(
                result[HEALTH_INDEX_COLUMN].to_numpy(),
                result[_BEST_HEALTH_SO_FAR_COLUMN].to_numpy(),
                fallback=1.0,
            )
            * 100.0,
            0.0,
            100.0,
        )
        result = result.drop(columns=[_BEST_HEALTH_SO_FAR_COLUMN])

        self.logger.info(
            "Trend computed: mean slope=%.4f, mean deterioration rate=%.4f, "
            "mean remaining health=%.1f%%",
            float(result[HEALTH_TREND_COLUMN].mean()),
            float(result[DETERIORATION_RATE_COLUMN].mean()),
            float(result[REMAINING_HEALTH_PCT_COLUMN].mean()),
        )
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        verify_identity_columns(frame, context="health_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_trend_tracker"
        )

    def run(self, health_index: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = health_index
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.health_index_csv),
                required_columns=get_identity_columns() + [HEALTH_INDEX_COLUMN],
                stage="health_trend_tracker",
            )

        scored = self.compute(source)
        output = self.save(scored)

        report: Dict[str, Any] = {
            "rows": int(len(scored)),
            "engines": int(scored["unit"].nunique()),
            "mean_trend": float(scored[HEALTH_TREND_COLUMN].mean()),
            "mean_deterioration_rate": float(scored[DETERIORATION_RATE_COLUMN].mean()),
            "mean_remaining_health_pct": float(scored[REMAINING_HEALTH_PCT_COLUMN].mean()),
            "output_path": str(output),
        }
        self.logger.info("Health trend report: %s", report)
        return scored, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    tracker = HealthTrendTracker(config)
    scored, report = tracker.run()
    logger.info("health_scores.csv written: %d rows", report["rows"])
    print(scored.head(10).to_string())


if __name__ == "__main__":
    main()
