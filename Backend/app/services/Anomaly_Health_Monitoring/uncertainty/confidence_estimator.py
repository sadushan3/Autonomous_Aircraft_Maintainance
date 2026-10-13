"""
Confidence estimation: the uncertainty-layer orchestrator.

Combines model agreement (:mod:`model_agreement`) with data quality
(from the cleaning stage) into the single ``confidence_score`` every prediction
in this component carries, following the weighted formula specified for the
research objective:

    confidence = 100 * (w_agree * model_agreement
                       + w_rel   * detector_agreement
                       + w_qual  * data_quality_score)

``reliability_score`` is reported separately as the detector-agreement term
alone (a caller who wants "how much did the detectors corroborate each other,
independent of twin quality" can read it directly), and
``uncertainty_score = 1 - model_agreement`` is reported as its own column
because "how much does the expected value itself vary across twins" is exactly
what a maintenance engineer needs when deciding whether to trust a flagged
reading or ask for a second sensor check.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import (
    DATA_QUALITY_COLUMN,
)
from app.services.Anomaly_Health_Monitoring.uncertainty.model_agreement import (
    DETECTOR_AGREEMENT_COLUMN,
    TWIN_AGREEMENT_COLUMN,
    ModelAgreementScorer,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    merge_on_keys,
    read_csv,
)

#: Final 0-100 confidence score.
CONFIDENCE_SCORE_COLUMN = "confidence_score"

#: 0-1 reliability, from detector agreement.
RELIABILITY_SCORE_COLUMN = "reliability_score"

#: 0-1 uncertainty, from twin disagreement.
UNCERTAINTY_SCORE_COLUMN = "uncertainty_score"


class ConfidenceEstimator:
    """Blends model agreement and data quality into a single confidence score."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the estimator.

        Args:
            config: Configuration supplying the blend weights.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.agreement_scorer = ModelAgreementScorer(self.config)

    # ------------------------------------------------------------------ #
    # Computation
    # ------------------------------------------------------------------ #
    def compute(
        self,
        frame: pd.DataFrame,
        weight_agreement: Optional[float] = None,
        weight_reliability: Optional[float] = None,
        weight_data_quality: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compute the confidence triple for every row.

        Args:
            frame: Frame carrying twin-spread, detector scores and (optionally)
                ``data_quality_score``.
            weight_agreement: Optional override of the agreement weight.
            weight_reliability: Optional override of the reliability weight.
            weight_data_quality: Optional override of the data-quality weight.

        Returns:
            The frame with ``confidence_score``, ``reliability_score``,
            ``uncertainty_score``, ``model_agreement`` and
            ``detector_agreement`` added.
        """
        result = self.agreement_scorer.compute(frame)

        if DATA_QUALITY_COLUMN not in result.columns:
            result[DATA_QUALITY_COLUMN] = 1.0
            self.logger.info(
                "No data_quality_score found in input; defaulting to 1.0 "
                "(the cleaning stage was not merged in)."
            )

        w_agree = float(weight_agreement if weight_agreement is not None else self.config.uncertainty.w_agreement)
        w_rel = float(weight_reliability if weight_reliability is not None else self.config.uncertainty.w_reliability)
        w_qual = float(weight_data_quality if weight_data_quality is not None else self.config.uncertainty.w_data_quality)
        total_weight = max(w_agree + w_rel + w_qual, 1e-9)
        w_agree, w_rel, w_qual = w_agree / total_weight, w_rel / total_weight, w_qual / total_weight

        confidence = (
            w_agree * result[TWIN_AGREEMENT_COLUMN]
            + w_rel * result[DETECTOR_AGREEMENT_COLUMN]
            + w_qual * result[DATA_QUALITY_COLUMN].astype(float)
        )

        result[CONFIDENCE_SCORE_COLUMN] = np.clip(
            100.0 * confidence,
            self.config.uncertainty.min_confidence,
            self.config.uncertainty.max_confidence,
        )
        result[RELIABILITY_SCORE_COLUMN] = result[DETECTOR_AGREEMENT_COLUMN]
        result[UNCERTAINTY_SCORE_COLUMN] = 1.0 - result[TWIN_AGREEMENT_COLUMN]

        self.logger.info(
            "Confidence computed: mean=%.2f, min=%.2f (weights: agree=%.2f, rel=%.2f, qual=%.2f)",
            float(result[CONFIDENCE_SCORE_COLUMN].mean()),
            float(result[CONFIDENCE_SCORE_COLUMN].min()),
            w_agree,
            w_rel,
            w_qual,
        )
        return result

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``confidence_scores.csv`` atomically.

        Args:
            frame: Frame with the confidence triple attached.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.confidence_scores_csv
        )
        verify_identity_columns(frame, context="confidence_scores output")
        return atomic_write_csv(frame, target, config=self.config)

    def run(
        self,
        anomaly_scores: Optional[pd.DataFrame] = None,
        residuals: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the uncertainty stage.

        Args:
            anomaly_scores: Optional in-memory input carrying per-detector
                scores and twin spread; otherwise ``anomaly_fusion.csv`` (which
                carries ``anomaly_score``) merged with ``twin_predictions.csv``
                (which carries ``twin_spread_mean``) is used.
            residuals: Optional in-memory residuals carrying
                ``data_quality_score``; otherwise ``cleaned_data.csv`` is
                consulted for it.

        Returns:
            A ``(confidence_frame, report)`` tuple.
        """
        self.config.ensure_directories()
        source = anomaly_scores
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.anomaly_fusion_csv),
                required_columns=get_identity_columns(),
                stage="confidence_estimator",
            )
            twin_path = self.config.paths.processed_path(self.config.paths.twin_predictions_csv)
            if twin_path.is_file():
                twin_predictions = read_csv(twin_path, stage="confidence_estimator")
                spread_columns = [c for c in twin_predictions.columns if c == "twin_spread_mean"]
                if spread_columns:
                    source = merge_on_keys(
                        source,
                        twin_predictions[get_identity_columns() + spread_columns],
                        keys=get_identity_columns(),
                        how="left",
                    )

        if DATA_QUALITY_COLUMN not in source.columns:
            cleaned_path = self.config.paths.processed_path(self.config.paths.cleaned_data_csv)
            if cleaned_path.is_file():
                cleaned = read_csv(cleaned_path, stage="confidence_estimator")
                if DATA_QUALITY_COLUMN in cleaned.columns:
                    source = merge_on_keys(
                        source,
                        cleaned[get_identity_columns() + [DATA_QUALITY_COLUMN]],
                        keys=get_identity_columns(),
                        how="left",
                    )

        computed = self.compute(source)
        output = self.save(computed)

        report: Dict[str, Any] = {
            "rows": int(len(computed)),
            "engines": int(computed["unit"].nunique()),
            "mean_confidence": float(computed[CONFIDENCE_SCORE_COLUMN].mean()),
            "mean_reliability": float(computed[RELIABILITY_SCORE_COLUMN].mean()),
            "mean_uncertainty": float(computed[UNCERTAINTY_SCORE_COLUMN].mean()),
            "low_confidence_rows": int((computed[CONFIDENCE_SCORE_COLUMN] < 50.0).sum()),
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "uncertainty.json")
        self.logger.info("Confidence estimation report: %s", report)
        return computed, report


def main() -> None:
    """Run confidence estimation standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    estimator = ConfidenceEstimator(config)
    result, report = estimator.run()
    logger.info("confidence_scores.csv written: %d rows", report["rows"])
    print(result[["unit", "cycle", CONFIDENCE_SCORE_COLUMN, RELIABILITY_SCORE_COLUMN, UNCERTAINTY_SCORE_COLUMN]].head(10).to_string())


if __name__ == "__main__":
    main()
