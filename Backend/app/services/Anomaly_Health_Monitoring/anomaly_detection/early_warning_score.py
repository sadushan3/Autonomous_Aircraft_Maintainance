"""
Early-warning score: likelihood of reaching a critical state soon.

Severity says how bad things are *right now*.  This module answers a
forward-looking question instead: given the current anomaly score and how fast
it and the health index are trending, how likely is this engine to cross into
Critical / Failure Risk territory within the configured early-warning horizon?

The estimate is a lightweight blend, not a trained forecaster (forecasting is
out of scope for this component):

    early_warning = clip(anomaly_score + trend_pressure, 0, 1)
    trend_pressure = w * horizon * max(-health_trend, 0) / 100

A flat or improving health trend contributes no pressure; a fast decline adds
meaningfully to the current anomaly score, projecting it forward over the
horizon.  This gives operators a single number to sort engines by for
prioritising inspection queues, without pretending to be a survival model.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, merge_on_keys
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

#: Fused anomaly score column, mirrored from anomaly_fusion.py.
#: Kept as a local string constant (rather than imported) to avoid a circular
#: import: anomaly_fusion.py is the orchestrator that calls this module.
ANOMALY_SCORE_COLUMN = "anomaly_score"

#: Column name added by this module.
EARLY_WARNING_SCORE_COLUMN = "early_warning_score"

#: Weight applied to the health-trend pressure term.
_TREND_WEIGHT = 0.6


class EarlyWarningScorer:
    """Blends current anomaly score with health-trend pressure."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the scorer.

        Args:
            config: Configuration supplying the early-warning horizon.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.horizon = int(self.config.anomaly.early_warning_horizon)

    def compute(
        self,
        fused: pd.DataFrame,
        health_trend: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute the early-warning score for every row.

        Args:
            fused: Frame carrying ``anomaly_score`` and identity keys.
            health_trend: Optional frame carrying ``health_trend`` (health
                points per cycle); when omitted, only the current anomaly
                score is used.

        Returns:
            The frame with ``early_warning_score`` added.
        """
        result = fused.copy()

        if health_trend is not None and "health_trend" in health_trend.columns:
            merged = merge_on_keys(
                result,
                health_trend[get_identity_columns() + ["health_trend"]],
                keys=get_identity_columns(),
                how="left",
            )
            decline_rate = np.clip(-merged["health_trend"].fillna(0.0), 0.0, None)
        else:
            merged = result
            decline_rate = pd.Series(np.zeros(len(result)), index=result.index)

        pressure = _TREND_WEIGHT * self.horizon * decline_rate / 100.0
        merged[EARLY_WARNING_SCORE_COLUMN] = np.clip(
            merged[ANOMALY_SCORE_COLUMN].astype(float) + pressure.to_numpy(), 0.0, 1.0
        )

        self.logger.info(
            "Early-warning score computed (horizon=%d cycles): mean=%.4f, max=%.4f",
            self.horizon,
            float(merged[EARLY_WARNING_SCORE_COLUMN].mean()),
            float(merged[EARLY_WARNING_SCORE_COLUMN].max()),
        )
        return merged

    def run(
        self, fused: pd.DataFrame, health_trend: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Execute early-warning scoring as an in-memory transform.

        Args:
            fused: Fused-anomaly frame.
            health_trend: Optional health-trend frame.

        Returns:
            The frame with the early-warning score added.
        """
        return self.compute(fused, health_trend)


def main() -> None:
    """Run early-warning scoring standalone."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    fused = read_csv(
        config.paths.processed_path(config.paths.anomaly_fusion_csv),
        required_columns=get_identity_columns() + [ANOMALY_SCORE_COLUMN],
        stage="early_warning_score",
    )
    scores_path = config.paths.processed_path(config.paths.health_scores_csv)
    trend = read_csv(scores_path, stage="early_warning_score") if scores_path.is_file() else None

    scorer = EarlyWarningScorer(config)
    result = scorer.run(fused, trend)
    logger.info("Mean early-warning score: %.4f", float(result[EARLY_WARNING_SCORE_COLUMN].mean()))
    print(result.head(10).to_string())


if __name__ == "__main__":
    main()
