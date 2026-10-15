"""
Model agreement: turning ensemble disagreement into a usable signal.

Two independent kinds of disagreement exist in this pipeline and both matter:

* **Twin disagreement** (``twin_spread_mean``, produced by
  :mod:`digital_twin.ensemble_twin`) -- how much the four Digital Twins
  disagreed about what a sensor *should* read. High disagreement means the
  residual computed from their fused prediction is itself on shaky ground.
* **Detector disagreement** -- how much the three anomaly detectors
  (Isolation Forest, Mahalanobis, LSTM-AE) disagree about whether a row is
  anomalous. If they agree, the anomaly call is corroborated from multiple
  independent angles; if they scatter, the fused score is averaging over real
  disagreement rather than confirming a shared signal.

Both are normalised into [0, 1] "agreement" scores (1 = full agreement) so the
confidence estimator can combine them without unit conversion.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, minmax_normalize

#: Column produced for twin-based agreement.
TWIN_AGREEMENT_COLUMN = "model_agreement"

#: Column produced for detector-based agreement.
DETECTOR_AGREEMENT_COLUMN = "detector_agreement"

_DETECTOR_SCORE_COLUMNS = [
    "isolation_forest_score",
    "mahalanobis_score",
    "lstm_ae_score",
    "residual_rule_score",
    "condition_consensus_score",
]


class ModelAgreementScorer:
    """Computes twin-disagreement and detector-disagreement agreement scores."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the scorer.

        Args:
            config: Configuration supplying the spread saturation point.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.saturation = float(self.config.uncertainty.spread_saturation)

    # ------------------------------------------------------------------ #
    # Twin agreement
    # ------------------------------------------------------------------ #
    def twin_agreement(self, frame: pd.DataFrame) -> pd.Series:
        """Convert twin-spread into an agreement score in [0, 1].

        Args:
            frame: Frame carrying ``twin_spread_mean``.

        Returns:
            A series of agreement scores; 1 means the twins fully agreed.
        """
        if "twin_spread_mean" not in frame.columns:
            return pd.Series(np.ones(len(frame)), index=frame.index)

        spread = frame["twin_spread_mean"].astype(float).clip(lower=0.0)
        normalized_spread = np.clip(spread / max(self.saturation, 1e-9), 0.0, 1.0)
        return 1.0 - normalized_spread

    # ------------------------------------------------------------------ #
    # Detector agreement
    # ------------------------------------------------------------------ #
    def detector_agreement(self, frame: pd.DataFrame) -> pd.Series:
        """Compute pairwise-disagreement-derived agreement across detectors.

        Uses the mean pairwise absolute difference between detector scores as
        the disagreement measure, since with only 3-4 detectors this is more
        interpretable than a variance-based measure and behaves identically at
        the extremes (all agree -> 0 disagreement; maximally split -> high
        disagreement).

        Args:
            frame: Frame carrying the per-detector score columns.

        Returns:
            A series of agreement scores in [0, 1].
        """
        present = [c for c in _DETECTOR_SCORE_COLUMNS if c in frame.columns]
        if len(present) < 2:
            return pd.Series(np.ones(len(frame)), index=frame.index)

        values = frame[present].astype(float).to_numpy()
        n = values.shape[1]
        pairwise_sum = np.zeros(values.shape[0], dtype=float)
        n_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_sum += np.abs(values[:, i] - values[:, j])
                n_pairs += 1

        mean_pairwise_disagreement = pairwise_sum / max(n_pairs, 1)
        return pd.Series(1.0 - np.clip(mean_pairwise_disagreement, 0.0, 1.0), index=frame.index)

    # ------------------------------------------------------------------ #
    # Combined
    # ------------------------------------------------------------------ #
    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach both agreement columns to a frame.

        Args:
            frame: Frame carrying twin-spread and/or detector score columns.

        Returns:
            The frame with ``model_agreement`` and ``detector_agreement`` added.
        """
        result = frame.copy()
        result[TWIN_AGREEMENT_COLUMN] = self.twin_agreement(frame)
        result[DETECTOR_AGREEMENT_COLUMN] = self.detector_agreement(frame)

        self.logger.info(
            "Model agreement computed: mean twin_agreement=%.4f, mean detector_agreement=%.4f",
            float(result[TWIN_AGREEMENT_COLUMN].mean()),
            float(result[DETECTOR_AGREEMENT_COLUMN].mean()),
        )
        return result


def main() -> None:
    """Demonstrate model agreement scoring against the residuals + scores."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv, merge_on_keys
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    residuals = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="model_agreement",
    )
    scores_path = config.paths.processed_path(config.paths.anomaly_scores_csv)
    if scores_path.is_file():
        scores = read_csv(scores_path, stage="model_agreement")
        merged = merge_on_keys(residuals, scores, keys=get_identity_columns(), how="left")
    else:
        merged = residuals

    scorer = ModelAgreementScorer(config)
    result = scorer.compute(merged)
    logger.info("Sample rows:\n%s", result[[TWIN_AGREEMENT_COLUMN, DETECTOR_AGREEMENT_COLUMN]].head())


if __name__ == "__main__":
    main()
