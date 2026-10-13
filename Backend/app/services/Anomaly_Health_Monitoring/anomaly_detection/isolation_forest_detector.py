"""
Isolation Forest anomaly detector.

Fits ``sklearn.ensemble.IsolationForest`` on the healthy-region residual
features.  Isolation Forest scores a point by how few random splits it takes
to separate it from the rest of the data; anomalous points isolate quickly
because they sit in sparse regions of the residual space, which makes the
method effective without requiring any distributional assumption -- unlike the
Mahalanobis detector, which assumes (approximately) Gaussian residuals.

The raw ``decision_function`` output is unbounded and lower-is-more-anomalous,
which is awkward to fuse with the other detectors.  This module inverts and
min-max normalises it into [0, 1] where higher means more anomalous, and
calibrates the flagging threshold as a percentile of the *healthy* region's own
scores -- so the threshold means "more anomalous than nearly all healthy
behaviour", not an arbitrary constant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_identity_columns,
    present_residual_z_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_json,
    empirical_percentile_normalize,
    get_logger,
    load_model,
    percentile_threshold,
    save_model,
)

#: Column name this detector contributes to anomaly_scores.csv.
SCORE_COLUMN = "isolation_forest_score"


class IsolationForestDetector:
    """Wraps IsolationForest with healthy-region fitting and score calibration."""

    ARTIFACT_NAME = "isolation_forest_detector.pkl"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the detector.

        Args:
            config: Configuration supplying hyper-parameters and threshold
                percentile.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.model: Optional[IsolationForest] = None
        self.feature_columns: List[str] = []
        self.threshold: float = 1.0
        self._healthy_scores_sorted: np.ndarray = np.array([], dtype=float)

    # ------------------------------------------------------------------ #
    # Healthy-region selection (mirrors BaseTwin's rule for consistency)
    # ------------------------------------------------------------------ #
    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Select the same healthy region the digital twin trained on.

        Args:
            frame: Residual features carrying RUL and cycle when available.

        Returns:
            A boolean mask.
        """
        settings = self.config.healthy_region
        if RUL_COLUMN in frame.columns:
            rul_ok = frame[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=frame.index)
        max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = frame["cycle"].astype(float) <= (float(settings.cycle_fraction) * max_cycle)
        return rul_ok & cycle_ok

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "IsolationForestDetector":
        """Fit the forest on the healthy region and calibrate its threshold.

        Args:
            frame: Residual feature frame.
            feature_columns: Columns to use as the anomaly-detection feature
                space, typically the residual z-score columns.

        Returns:
            This fitted instance.

        Raises:
            AnomalyHealthError: If no feature columns are available.
        """
        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for IsolationForest.")
        self.feature_columns = present

        healthy = frame.loc[self._healthy_mask(frame), present]
        if healthy.empty:
            healthy = frame[present]

        values = healthy.astype(float).to_numpy()
        params = dict(self.config.anomaly.isolation_forest)
        self.model = IsolationForest(**params).fit(values)

        healthy_scores = -self.model.decision_function(values)
        self._healthy_scores_sorted = np.sort(healthy_scores)
        normalized_healthy = empirical_percentile_normalize(
            self._healthy_scores_sorted, healthy_scores
        )
        self.threshold = percentile_threshold(
            normalized_healthy, self.config.anomaly.iforest_threshold_percentile
        )

        self.logger.info(
            "IsolationForest fitted on %d healthy row(s), %d feature(s); "
            "threshold=%.4f (p%.0f)",
            len(healthy),
            len(present),
            self.threshold,
            self.config.anomaly.iforest_threshold_percentile,
        )
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score every row and normalise into [0, 1], higher = more anomalous.

        Args:
            frame: Residual feature frame.

        Returns:
            An array of normalised anomaly scores.

        Raises:
            AnomalyHealthError: If called before fitting or loading.
        """
        if self.model is None:
            raise AnomalyHealthError("IsolationForestDetector used before fit/load.")
        values = frame[self.feature_columns].astype(float).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        raw = -self.model.decision_function(values)
        return empirical_percentile_normalize(self._healthy_scores_sorted, raw)

    def predict_flags(self, frame: pd.DataFrame) -> np.ndarray:
        """Return boolean anomaly flags using the calibrated threshold.

        Args:
            frame: Residual feature frame.

        Returns:
            A boolean array.
        """
        return self.score(frame) > self.threshold

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self) -> Path:
        """Persist the fitted forest and calibration metadata.

        Returns:
            The artefact path written.

        Raises:
            AnomalyHealthError: If nothing has been fitted.
        """
        if self.model is None:
            raise AnomalyHealthError("Nothing to save; fit the detector first.")
        bundle = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
            "healthy_scores_sorted": self._healthy_scores_sorted,
        }
        return save_model(bundle, self.config.paths.model_path(self.ARTIFACT_NAME), config=self.config)

    def load(self) -> "IsolationForestDetector":
        """Reload a previously persisted detector.

        Returns:
            This instance, ready to score.
        """
        bundle = load_model(self.config.paths.model_path(self.ARTIFACT_NAME))
        self.model = bundle["model"]
        self.feature_columns = bundle["feature_columns"]
        self.threshold = float(bundle["threshold"])
        self._healthy_scores_sorted = np.asarray(bundle["healthy_scores_sorted"], dtype=float)
        return self


def main() -> None:
    """Fit and score the Isolation Forest detector standalone."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="isolation_forest_detector",
    )
    feature_columns = present_residual_z_columns(frame)
    detector = IsolationForestDetector(config).fit(frame, feature_columns)
    detector.save()
    scores = detector.score(frame)
    logger.info("Mean score=%.4f, flagged=%d/%d", float(scores.mean()), int((scores > detector.threshold).sum()), len(frame))


if __name__ == "__main__":
    main()
