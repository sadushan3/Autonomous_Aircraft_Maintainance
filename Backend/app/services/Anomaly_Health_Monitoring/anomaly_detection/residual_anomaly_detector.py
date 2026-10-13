"""
Threshold-based residual anomaly scoring.

The three model-based detectors (Isolation Forest, Mahalanobis, LSTM-AE) each
learn a notion of "unusual" from the data.  This module is the simple,
transparent complement: a row is scored directly from how far its worst
sensors have drifted, in robust z-score units, with no fitting involved beyond
the residual calculation that already happened upstream.

Its purpose in the fusion is to act as a floor.  A model-based detector can, in
principle, be fooled by a fault pattern that happens to resemble the healthy
manifold in its particular feature space; a rule that says "three sensors are
each more than 4 standard deviations off" cannot be fooled that way, because it
does not depend on any learned structure.  It also gives the operator a
detector whose behaviour they can verify by hand.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    empirical_percentile_normalize,
    get_logger,
    percentile_threshold,
    atomic_write_json,
    read_json,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    present_residual_z_columns,
)

#: Column name this detector contributes to anomaly_scores.csv.
SCORE_COLUMN = "residual_rule_score"

#: |z| above which a single sensor counts as a rule violation.
_SENSOR_Z_TRIGGER = 3.0


class ResidualAnomalyDetector:
    """Rule-based anomaly score from robust residual z-scores directly."""

    META_NAME = "residual_rule_meta.json"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the detector.

        Args:
            config: Configuration supplying the threshold percentile.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.feature_columns: List[str] = []
        self.threshold: float = 1.0
        self._healthy_raw_sorted: np.ndarray = np.array([], dtype=float)

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
    # Raw scoring rule
    # ------------------------------------------------------------------ #
    def _raw_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Combine per-sensor violation count and magnitude into a raw score.

        The raw score is ``mean(|z|) + 0.5 * count(|z| > trigger)`` -- a blend
        of "how far off, on average" and "how many sensors are clearly wrong",
        so a single extreme sensor and many mildly-off sensors can both surface.

        Args:
            frame: Residual feature frame.

        Returns:
            An array of raw (unnormalised) scores.
        """
        values = frame[self.feature_columns].astype(float).to_numpy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean_abs_z = np.abs(values).mean(axis=1)
        violation_count = (np.abs(values) > _SENSOR_Z_TRIGGER).sum(axis=1)
        return mean_abs_z + 0.5 * violation_count

    # ------------------------------------------------------------------ #
    # Fitting (calibration only; no learned parameters)
    # ------------------------------------------------------------------ #
    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "ResidualAnomalyDetector":
        """Calibrate the flagging threshold against the healthy region.

        Args:
            frame: Residual feature frame.
            feature_columns: Residual z-score columns.

        Returns:
            This fitted instance.

        Raises:
            AnomalyHealthError: If no feature columns are available.
        """
        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for the rule detector.")
        self.feature_columns = present

        healthy = frame.loc[self._healthy_mask(frame)]
        if healthy.empty:
            healthy = frame

        raw = self._raw_score(healthy)
        self._healthy_raw_sorted = np.sort(raw)
        normalized_healthy = empirical_percentile_normalize(self._healthy_raw_sorted, raw)
        self.threshold = percentile_threshold(
            normalized_healthy, self.config.anomaly.residual_threshold_percentile
        )
        self.logger.info(
            "Residual rule detector calibrated on %d healthy row(s); threshold=%.4f (p%.0f)",
            len(healthy),
            self.threshold,
            self.config.anomaly.residual_threshold_percentile,
        )
        return self

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def score(self, frame: pd.DataFrame) -> np.ndarray:
        """Score every row and normalise into [0, 1].

        Args:
            frame: Residual feature frame.

        Returns:
            An array of normalised anomaly scores.

        Raises:
            AnomalyHealthError: If called before fitting or loading.
        """
        if not self.feature_columns:
            raise AnomalyHealthError("ResidualAnomalyDetector used before fit/load.")
        raw = self._raw_score(frame)
        return empirical_percentile_normalize(self._healthy_raw_sorted, raw)

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
    def save(self) -> None:
        """Persist the calibration (there is no fitted model to save)."""
        atomic_write_json(
            {
                "feature_columns": self.feature_columns,
                "threshold": self.threshold,
                "healthy_raw_sorted": self._healthy_raw_sorted.tolist(),
            },
            self.config.paths.model_path(self.META_NAME),
        )

    def load(self) -> "ResidualAnomalyDetector":
        """Reload persisted calibration.

        Returns:
            This instance, ready to score.

        Raises:
            AnomalyHealthError: If no metadata is present.
        """
        meta = read_json(self.config.paths.model_path(self.META_NAME), default=None)
        if not meta:
            raise AnomalyHealthError("No residual rule metadata found; calibrate it first.")
        self.feature_columns = list(meta["feature_columns"])
        self.threshold = float(meta["threshold"])
        self._healthy_raw_sorted = np.asarray(meta["healthy_raw_sorted"], dtype=float)
        return self


def main() -> None:
    """Fit and score the residual rule detector standalone."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="residual_anomaly_detector",
    )
    feature_columns = present_residual_z_columns(frame)
    detector = ResidualAnomalyDetector(config).fit(frame, feature_columns)
    detector.save()
    scores = detector.score(frame)
    logger.info(
        "Mean score=%.4f, flagged=%d/%d", float(scores.mean()),
        int((scores > detector.threshold).sum()), len(frame),
    )


if __name__ == "__main__":
    main()
