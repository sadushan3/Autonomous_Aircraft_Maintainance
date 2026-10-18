from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, minmax_normalize


TWIN_AGREEMENT_COLUMN = "model_agreement"


DETECTOR_AGREEMENT_COLUMN = "detector_agreement"

_DETECTOR_SCORE_COLUMNS = [
    "isolation_forest_score",
    "mahalanobis_score",
    "lstm_ae_score",
    "residual_rule_score",
    "condition_consensus_score",
]


class ModelAgreementScorer:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.saturation = float(self.config.uncertainty.spread_saturation)


    def twin_agreement(self, frame: pd.DataFrame) -> pd.Series:


        if "twin_spread_mean" not in frame.columns:
            return pd.Series(np.ones(len(frame)), index=frame.index)

        spread = frame["twin_spread_mean"].astype(float).clip(lower=0.0)
        normalized_spread = np.clip(spread / max(self.saturation, 1e-9), 0.0, 1.0)
        return 1.0 - normalized_spread


    def detector_agreement(self, frame: pd.DataFrame) -> pd.Series:


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


    def compute(self, frame: pd.DataFrame) -> pd.DataFrame:


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
