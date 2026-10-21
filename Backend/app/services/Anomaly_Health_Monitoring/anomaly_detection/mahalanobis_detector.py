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


SCORE_COLUMN = "mahalanobis_score"


class MahalanobisDetector:


    ARTIFACT_NAME = "mahalanobis_detector.pkl"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.covariance_estimator: Optional[Any] = None
        self.mean_: Optional[np.ndarray] = None
        self.feature_columns: List[str] = []
        self.threshold: float = 1.0
        self._distance_p99: float = 1.0
        self.chi2_threshold: float = 1.0
        self._healthy_distances_sorted: np.ndarray = np.array([], dtype=float)
        self.temporal_window = max(int(self.config.anomaly.mahalanobis.get("temporal_window", 1)), 1)
        self.temporal_method = str(self.config.anomaly.mahalanobis.get("temporal_method", "rolling"))
        self.score_normalization = str(self.config.anomaly.mahalanobis.get("score_normalization", "empirical"))

    def _feature_values(self, frame: pd.DataFrame) -> np.ndarray:

        ordered = frame.reset_index(drop=True).sort_values(["unit", "cycle"])
        values = ordered[self.feature_columns].astype(float)
        if ordered.duplicated(["unit", "cycle"]).any() or ordered[["unit", "cycle"]].isna().any().any():
            raise AnomalyHealthError("Anomaly input needs unique, non-missing unit/cycle identities.")
        if not np.isfinite(values.to_numpy()).all():
            raise AnomalyHealthError("Anomaly input contains missing or non-finite residuals.")
        if self.temporal_method not in ("rolling", "ewm"):
            raise AnomalyHealthError("Unsupported Mahalanobis temporal method.")
        if self.temporal_window > 1:

            gap = ordered.groupby("unit", sort=False)["cycle"].diff().ne(1)
            segment = gap.cumsum()
            if self.temporal_method == "rolling" and self.temporal_window <= 20 and not values.empty:


                matrix = values.to_numpy()
                smoothed = np.empty_like(matrix)
                starts = np.flatnonzero(gap.to_numpy())
                ends = np.r_[starts[1:], len(matrix)]
                for start, end in zip(starts, ends):
                    block = matrix[start:end]
                    width = min(self.temporal_window, len(block))
                    for count in range(1, width):
                        smoothed[start + count - 1] = block[:count].mean(axis=0)
                    windows = np.lib.stride_tricks.sliding_window_view(block, width, axis=0)
                    smoothed[start + width - 1:end] = windows.mean(axis=-1)
                values = pd.DataFrame(smoothed, index=values.index, columns=values.columns)
            else:
                groups = values.groupby([ordered["unit"], segment], sort=False)
                if self.temporal_method == "ewm":
                    values = groups.transform(lambda s: s.ewm(span=self.temporal_window, adjust=False).mean())
                else:
                    values = groups.transform(lambda s: s.rolling(self.temporal_window, min_periods=1).mean())
        result = np.zeros((len(frame), len(self.feature_columns)), dtype=float)
        result[ordered.index.to_numpy()] = values.to_numpy()
        return result

    def _normalize(self, distances: np.ndarray) -> np.ndarray:
        if self.score_normalization == "rational":


            return distances / (distances + max(self._distance_p99, 1e-12))
        return empirical_percentile_normalize(self._healthy_distances_sorted, distances)

    def _healthy_mask(self, frame: pd.DataFrame) -> pd.Series:


        settings = self.config.healthy_region
        if self.temporal_window > 1 and RUL_COLUMN in frame:
            return (frame[RUL_COLUMN].astype(float) >= settings.rul_threshold) & (
                frame["cycle"] > self.config.twin.commissioning_cycles
            )
        if RUL_COLUMN in frame.columns:
            rul_ok = frame[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=frame.index)
        max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = frame["cycle"].astype(float) <= (float(settings.cycle_fraction) * max_cycle)
        return rul_ok & cycle_ok


    def fit(self, frame: pd.DataFrame, feature_columns: List[str]) -> "MahalanobisDetector":


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
        frame = training_scope(frame, self.config)
        record_training_scope("covariance_reference", frame, self.config)
        present = [c for c in feature_columns if c in frame.columns]
        if not present:
            raise AnomalyHealthError("No residual feature columns available for Mahalanobis.")
        self.feature_columns = present

        healthy_mask = self._healthy_mask(frame).to_numpy()
        if not healthy_mask.any():
            raise AnomalyHealthError("No healthy post-commissioning rows available for covariance fitting.")

        values = self._feature_values(frame)[healthy_mask]
        healthy = frame.loc[healthy_mask]

        use_shrinkage = bool(self.config.anomaly.mahalanobis.get("shrinkage", True))
        reg = float(self.config.anomaly.mahalanobis.get("regularization", 1e-6))

        if use_shrinkage:
            self.covariance_estimator = LedoitWolf().fit(values)
        else:
            self.covariance_estimator = EmpiricalCovariance().fit(values)

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
        normalized_healthy = self._normalize(distances)
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


    def raw_distance(self, frame: pd.DataFrame) -> np.ndarray:


        if self.covariance_estimator is None:
            raise AnomalyHealthError("MahalanobisDetector used before fit/load.")
        values = self._feature_values(frame)
        return self.covariance_estimator.mahalanobis(values)

    def score(self, frame: pd.DataFrame) -> np.ndarray:


        distances = self.raw_distance(frame)
        return self._normalize(distances)

    def predict_flags(self, frame: pd.DataFrame) -> np.ndarray:


        return self.score(frame) > self.threshold


    def save(self) -> Path:


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
            "temporal_window": self.temporal_window,
            "temporal_method": self.temporal_method,
            "score_normalization": self.score_normalization,
        }
        return save_model(bundle, self.config.paths.model_path(self.ARTIFACT_NAME), config=self.config)

    def load(self) -> "MahalanobisDetector":


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

        self.temporal_window = int(bundle.get("temporal_window", 1))
        self.temporal_method = str(bundle.get("temporal_method", "rolling"))
        self.score_normalization = str(bundle.get("score_normalization", "empirical"))
        return self


def main() -> None:

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
