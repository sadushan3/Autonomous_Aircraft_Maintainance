"""
Health trend tracking: slope, deterioration rate and remaining-health percentage.

The health index alone answers "how healthy is this engine right now"; this
module answers "which way is it heading, and how fast".  Three derived signals
come out of a trailing linear fit over the configured trend window, computed
per engine so no trend ever crosses an engine boundary:

* ``health_trend`` -- the raw slope (health points per cycle). Negative means
  worsening.
* ``deterioration_rate`` -- ``max(-health_trend, 0)``, the rate of loss with
  improvement clipped away, since a positive trend does not mean "gaining
  health" so much as "recovering from noise".
* ``remaining_health_pct`` -- the current health index expressed as a
  percentage of the engine's own best observed health, so a unit whose sensors
  started slightly rough (never quite reaching 100) is not penalised for
  something that was never a fault to begin with.
"""

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

#: Slope of the health index, health points per cycle.
HEALTH_TREND_COLUMN = "health_trend"

#: Non-negative rate of health loss.
DETERIORATION_RATE_COLUMN = "deterioration_rate"

#: Current health as a percentage of this engine's own best observed health.
REMAINING_HEALTH_PCT_COLUMN = "remaining_health_pct"

#: Running best (maximum) health index observed so far for the engine.
_BEST_HEALTH_SO_FAR_COLUMN = "_best_health_so_far"


class HealthTrendTracker:
    """Derives trend, deterioration rate and remaining-health percentage."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the tracker.

        Args:
            config: Configuration supplying the trend window.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def compute(self, health_index: pd.DataFrame) -> pd.DataFrame:
        """Attach trend, deterioration rate and remaining-health percentage.

        Args:
            health_index: Frame carrying identity keys and ``health_index``.

        Returns:
            The frame with the three trend columns added.
        """
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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``health_scores.csv`` atomically.

        Args:
            frame: Data with trend columns attached.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        verify_identity_columns(frame, context="health_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``health_scores.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(self.config.paths.health_scores_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_trend_tracker"
        )

    def run(self, health_index: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the trend-tracking stage.

        Args:
            health_index: Optional in-memory input; otherwise
                ``health_index.csv`` is read from disk.

        Returns:
            A ``(scored_frame, report)`` tuple.
        """
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
    """Run trend tracking standalone against ``health_index.csv``."""
    config = get_config()
    logger = get_logger(__name__, config)
    tracker = HealthTrendTracker(config)
    scored, report = tracker.run()
    logger.info("health_scores.csv written: %d rows", report["rows"])
    print(scored.head(10).to_string())


if __name__ == "__main__":
    main()
