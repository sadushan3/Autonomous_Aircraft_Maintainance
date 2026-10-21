from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import (
    HEALTH_INDEX_COLUMN,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_trend_tracker import (
    HEALTH_TREND_COLUMN,
    HealthTrendTracker,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    get_logger,
    read_csv,
)


HEALTH_SCORE_COLUMN = "health_score"


_TREND_PROJECTION_HORIZON = 3


_TREND_WEIGHT = 0.5


class HealthScoreEngine:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.trend_tracker = HealthTrendTracker(self.config)


    def compute(self, health_index: pd.DataFrame) -> pd.DataFrame:


        with_trend = self.trend_tracker.compute(health_index)

        projected_change = (
            _TREND_WEIGHT * with_trend[HEALTH_TREND_COLUMN] * _TREND_PROJECTION_HORIZON
        )
        blended = with_trend[HEALTH_INDEX_COLUMN] + projected_change
        with_trend[HEALTH_SCORE_COLUMN] = np.clip(blended, 0.0, 100.0)

        self.logger.info(
            "Health score computed: mean=%.2f (index mean=%.2f)",
            float(with_trend[HEALTH_SCORE_COLUMN].mean()),
            float(with_trend[HEALTH_INDEX_COLUMN].mean()),
        )
        return with_trend


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        verify_identity_columns(frame, context="health_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_score_engine"
        )

    def run(self, health_index: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = health_index
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.health_index_csv),
                required_columns=get_identity_columns() + [HEALTH_INDEX_COLUMN],
                stage="health_score_engine",
            )

        scored = self.compute(source)
        output = self.save(scored)

        report: Dict[str, Any] = {
            "rows": int(len(scored)),
            "engines": int(scored["unit"].nunique()),
            "mean_health_score": float(scored[HEALTH_SCORE_COLUMN].mean()),
            "min_health_score": float(scored[HEALTH_SCORE_COLUMN].min()),
            "output_path": str(output),
        }
        self.logger.info("Health score report: %s", report)
        return scored, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    engine = HealthScoreEngine(config)
    scored, report = engine.run()
    logger.info("health_scores.csv written: %d rows", report["rows"])
    print(scored.head(10).to_string())


if __name__ == "__main__":
    main()
