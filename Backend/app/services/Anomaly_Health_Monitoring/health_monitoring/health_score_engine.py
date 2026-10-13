"""
Health score engine: blends the health index with its own trend.

The raw health index (:mod:`health_index_calculator`) reacts immediately to
whatever the current cycle's residuals look like, which makes it a little
jumpy on noisy sensors.  The health *score* produced here folds in the trend
computed by :mod:`health_trend_tracker` -- a sustained decline pulls the score
below the index, and a flat trend leaves it close to the index -- giving a
number that reflects trajectory as well as instantaneous state.

    health_score = clip(health_index + trend_weight * health_trend * horizon, 0, 100)

``health_trend`` is health points per cycle (negative when worsening), so
projecting it forward by a short horizon anticipates where the index is
heading rather than only reporting where it has been.  The horizon is kept
short (a few cycles) specifically so this stays a same-cycle adjustment, not a
forecast -- true forecasting is out of scope for this component.

This module owns writing ``health_scores.csv``, the single carry-forward file
that combines index, trend and score.
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

#: Final blended 0-100 health score, the column the state classifier consumes.
HEALTH_SCORE_COLUMN = "health_score"

#: How many cycles ahead the trend is projected when blending.
_TREND_PROJECTION_HORIZON = 3

#: How strongly the projected trend pulls the score away from the raw index.
_TREND_WEIGHT = 0.5


class HealthScoreEngine:
    """Blends health index and trend into the final health score."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the engine.

        Args:
            config: Configuration passed through to the trend tracker.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.trend_tracker = HealthTrendTracker(self.config)

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def compute(self, health_index: pd.DataFrame) -> pd.DataFrame:
        """Compute trend and the blended health score.

        Args:
            health_index: Frame carrying identity keys and ``health_index``.

        Returns:
            The frame with trend columns plus ``health_score`` added.
        """
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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``health_scores.csv`` atomically.

        Args:
            frame: Data with the health score attached.
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
            target, required_columns=get_identity_columns(), stage="health_score_engine"
        )

    def run(self, health_index: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the health-score stage.

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
    """Run the health-score engine standalone against ``health_index.csv``."""
    config = get_config()
    logger = get_logger(__name__, config)
    engine = HealthScoreEngine(config)
    scored, report = engine.run()
    logger.info("health_scores.csv written: %d rows", report["rows"])
    print(scored.head(10).to_string())


if __name__ == "__main__":
    main()
