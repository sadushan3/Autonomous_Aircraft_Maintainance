"""
Mahalanobis-distance anomaly detector.

Fits a covariance matrix over the healthy-region residual features and scores
every row by its Mahalanobis distance from the healthy mean:

    D^2(x) = (x - mu)^T * Sigma^-1 * (x - mu)

Two things make this more robust than a textbook implementation:

**Ledoit-Wolf shrinkage.**  21 residual channels is a lot of dimensions
relative to the number of independent healthy samples available for many
engines, and the sample covariance matrix in that regime is poorly
conditioned -- small eigenvalues get wildly overweighted by the inverse,
producing distances that are dominated by estimation noise rather than genuine
correlation structure.  Shrinkage blends the sample covariance toward a scaled
identity matrix, which is exactly the regularisation this needs.

**Distributional grounding, not an arbitrary cutoff.**  Under the working
assumption that healthy residuals are approximately multivariate normal,
``D^2`` follows a chi-squared distribution with degrees of freedom equal to the
feature count.  The detector reports both that chi-squared-implied threshold
and the empirical percentile threshold from the healthy region, and uses the
empirical one for flagging -- more robust when the normality assumption is
only approximate, which it always is for real sensor data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import LedoitWolf, EmpiricalCovariance

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    present_residual_z_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    empirical_percentile_normalize,
    get_logger,
    load_model,
    percentile_threshold,
    save_model,
)

#: Column name this detector contributes to anomaly_scores.csv.
SCORE_COLUMN = "mahalanobis_score"


class MahalanobisDetector:
    """Mahalanobis-distance detector with shrinkage-regularised covariance."""

    ARTIFACT_NAME = "mahalanobis_detector.pkl"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the detector.

        Args:
            config: Configuration supplying shrinkage settings and threshold
                percentile.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.covariance_estimator: Optional[Any] = None
        self.mean_: Optional[np.ndarray] = None
        self.feature_columns: List[str] = []
        self.threshold: float = 1.0
        self._distance_p99: float = 1.0
        self.chi2_threshold: float = 1.0
        self._healthy_distances_sorted: np.ndarray = np.array([], dtype=float)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:
        """Select the same healthy region the digital twin trained on.

        Args:
            frame: Residual feature frame.

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
    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "MahalanobisDetector":
        """Fit the covariance estimator on the healthy region.

        Args:
            frame: Residual feature frame.
            feature_columns: Columns forming the multivariate feature space.

        Returns:
            This fitted instance.

        Raises:
            AnomalyHealthError: If no feature columns are available.
        """
        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for Mahalanobis.")
        self.feature_columns = present

        healthy = frame.loc[self._healthy_mask(frame), present]
        if healthy.empty:
            healthy = frame[present]
        values = healthy.astype(float).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        use_shrinkage = bool(self.config.anomaly.mahalanobis.get("shrinkage", True))
        reg = float(self.config.anomaly.mahalanobis.get("regularization", 1e-6))

        if use_shrinkage:
            self.covariance_estimator = LedoitWolf().fit(values)
        else:
            self.covariance_estimator = EmpiricalCovariance().fit(values)
            # Manual regularisation for the non-shrinkage path.
            self.covariance_estimator.covariance_ += reg * np.eye(values.shape[1])
            self.covariance_estimator.precision_ = np.linalg.pinv(
                self.covariance_estimator.covariance_
            )

        self.mean_ = values.mean(axis=0)

        distances = self.covariance_estimator.mahalanobis(values)
        self._distance_p99 = percentile_threshold(
            distances, self.config.anomaly.mahalanobis_threshold_percentile
        )
        self.chi2_threshold = float(stats.chi2.ppf(0.99, df=values.shape[1]))

        self._healthy_distances_sorted = np.sort(distances)
        normalized_healthy = empirical_percentile_normalize(
            self._healthy_distances_sorted, distances
        )
        self.threshold = percentile_threshold(
            normalized_healthy, self.config.anomaly.mahalanobis_threshold_percentile
        )

        self.logger.info(
            "Mahalanobis fitted on %d healthy row(s), %d dim(s); "
            "empirical D^2 p%.0f=%.3f, chi2(df=%d) p99=%.3f, shrinkage=%s",
            len(healthy),
            len(present),
            self.config.anomaly.mahalanobis_threshold_percentile,
            self._distance_p99,
            values.shape[1],
            self.chi2_threshold,
            use_shrinkage,
        )
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def raw_distance(self, frame: pd.DataFrame) -> np.ndarray:
        """Compute the Mahalanobis distance for every row.

        Args:
            frame: Residual feature frame.

        Returns:
            An array of distances.

        Raises:
            AnomalyHealthError: If called before fitting or loading.
        """
        if self.covariance_estimator is None:
            raise AnomalyHealthError("MahalanobisDetector used before fit/load.")
        values = frame[self.feature_columns].astype(float).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return self.covariance_estimator.mahalanobis(values)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Normalise the Mahalanobis distance into [0, 1].

        Args:
            frame: Residual feature frame.

        Returns:
            An array of normalised anomaly scores.
        """
        distances = self.raw_distance(frame)
        return empirical_percentile_normalize(self._healthy_distances_sorted, distances)

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
        """Persist the fitted covariance estimator and calibration.

        Returns:
            The artefact path written.

        Raises:
            AnomalyHealthError: If nothing has been fitted.
        """
        if self.covariance_estimator is None:
            raise AnomalyHealthError("Nothing to save; fit the detector first.")
        bundle = {
            "covariance_estimator": self.covariance_estimator,
            "mean": self.mean_,
            "feature_columns": self.feature_columns,
            "threshold": self.threshold,
            "distance_p99": self._distance_p99,
            "chi2_threshold": self.chi2_threshold,
            "healthy_distances_sorted": self._healthy_distances_sorted,
        }
        return save_model(bundle, self.config.paths.model_path(self.ARTIFACT_NAME), config=self.config)

    def load(self) -> "MahalanobisDetector":
        """Reload a previously persisted detector.

        Returns:
            This instance, ready to score.
        """
        bundle = load_model(self.config.paths.model_path(self.ARTIFACT_NAME))
        self.covariance_estimator = bundle["covariance_estimator"]
        self.mean_ = bundle["mean"]
        self.feature_columns = bundle["feature_columns"]
        self.threshold = float(bundle["threshold"])
        self._distance_p99 = float(bundle["distance_p99"])
        self.chi2_threshold = float(bundle["chi2_threshold"])
        self._healthy_distances_sorted = np.asarray(
            bundle["healthy_distances_sorted"], dtype=float
        )
        return self


def main() -> None:
    """Fit and score the Mahalanobis detector standalone."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="mahalanobis_detector",
    )
    feature_columns = present_residual_z_columns(frame)
    detector = MahalanobisDetector(config).fit(frame, feature_columns)
    detector.save()
    scores = detector.score(frame)
    logger.info(
        "Mean score=%.4f, flagged=%d/%d", float(scores.mean()),
        int((scores > detector.threshold).sum()), len(frame),
    )


if __name__ == "__main__":
    main()
