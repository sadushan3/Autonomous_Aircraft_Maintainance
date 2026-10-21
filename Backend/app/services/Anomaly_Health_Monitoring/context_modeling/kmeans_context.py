from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import get_op_setting_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    get_logger,
    load_model,
    save_model,
)


class KMeansContextModel:


    ARTIFACT_NAME = "context_kmeans.pkl"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.columns: List[str] = get_op_setting_columns()
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[KMeans] = None
        self.n_modes: int = 0
        self.silhouette: Optional[float] = None


    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:


        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(
                f"Operating settings missing from input: {missing}"
            )
        values = frame[self.columns].astype(float).to_numpy()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def select_k(self, values: np.ndarray, raw: Optional[np.ndarray] = None) -> Tuple[int, Optional[float]]:


        configured = int(self.config.context.n_modes)
        if configured > 0:
            self.logger.info("Using configured operating-mode count k=%d", configured)
            return configured, None


        reference = raw if raw is not None else values
        column_range = np.ptp(reference, axis=0)
        single_regime_band = np.array([1.0, 0.05, 2.0])[: column_range.shape[0]]
        if np.all(column_range < single_regime_band):
            self.logger.info(
                "Operating settings range %s is within single-regime noise "
                "band %s -> a single operating regime.",
                np.round(column_range, 4).tolist(),
                single_regime_band.tolist(),
            )
            return 1, None

        sample = values
        limit = int(self.config.context.silhouette_sample_size)
        if values.shape[0] > limit:
            rng = np.random.default_rng(self.config.data.random_state)
            sample = values[rng.choice(values.shape[0], size=limit, replace=False)]

        best_k = 1
        best_score = -1.0
        scores: Dict[int, float] = {}

        for candidate in self.config.context.auto_k_range:
            k = int(candidate)
            if k < 2 or k >= sample.shape[0]:
                continue
            model = KMeans(
                n_clusters=k,
                n_init=10,
                random_state=self.config.data.random_state,
            )
            labels = model.fit_predict(sample)
            if len(np.unique(labels)) < 2:
                continue
            score = float(silhouette_score(sample, labels))
            scores[k] = score
            if score > best_score:
                best_score = score
                best_k = k

        if not scores:
            self.logger.info("Silhouette search found no valid k>=2; using k=1.")
            return 1, None

        self.logger.info("Silhouette scores by k: %s", {k: round(v, 4) for k, v in scores.items()})
        self.logger.info("Selected k=%d (silhouette=%.4f)", best_k, best_score)
        return best_k, best_score

    def fit(self, frame: pd.DataFrame) -> "KMeansContextModel":


        raw = self._matrix(frame)
        self.scaler = StandardScaler().fit(raw)
        scaled = self.scaler.transform(raw)

        self.n_modes, self.silhouette = self.select_k(scaled, raw=raw)
        self.model = KMeans(
            n_clusters=max(self.n_modes, 1),
            n_init=10,
            random_state=self.config.data.random_state,
        ).fit(scaled)

        counts = np.bincount(self.model.labels_, minlength=self.n_modes)
        self.logger.info(
            "KMeans fitted: %d regime(s), rows per regime %s",
            self.n_modes,
            counts.tolist(),
        )
        return self


    def predict(self, frame: pd.DataFrame) -> np.ndarray:


        if self.model is None or self.scaler is None:
            raise AnomalyHealthError("KMeansContextModel used before fit/load.")
        scaled = self.scaler.transform(self._matrix(frame))
        return self.model.predict(scaled).astype(int)

    def confidence(self, frame: pd.DataFrame) -> np.ndarray:


        if self.model is None or self.scaler is None:
            raise AnomalyHealthError("KMeansContextModel used before fit/load.")
        if self.n_modes < 2:
            return np.ones(len(frame), dtype=float)

        scaled = self.scaler.transform(self._matrix(frame))
        distances = self.model.transform(scaled)
        ordered = np.sort(distances, axis=1)
        nearest = ordered[:, 0]
        second = ordered[:, 1]
        total = nearest + second
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = np.where(total > 0, (second - nearest) / total, 1.0)
        return np.clip(np.nan_to_num(margin, nan=1.0), 0.0, 1.0)

    def fit_predict(self, frame: pd.DataFrame) -> np.ndarray:


        return self.fit(frame).predict(frame)


    def save(self, path: Optional[Path] = None) -> Path:


        if self.model is None:
            raise AnomalyHealthError("Nothing to save; fit the model first.")
        target = path or self.config.paths.model_path(self.ARTIFACT_NAME)
        bundle = {
            "scaler": self.scaler,
            "model": self.model,
            "columns": self.columns,
            "n_modes": self.n_modes,
            "silhouette": self.silhouette,
        }
        return save_model(bundle, target, config=self.config)

    def load(self, path: Optional[Path] = None) -> "KMeansContextModel":


        target = path or self.config.paths.model_path(self.ARTIFACT_NAME)
        bundle = load_model(target)
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.columns = bundle["columns"]
        self.n_modes = int(bundle["n_modes"])
        self.silhouette = bundle.get("silhouette")
        self.logger.info("Loaded KMeans context model with %d regime(s)", self.n_modes)
        return self

    def summary(self) -> Dict[str, Any]:


        centroids: List[List[float]] = []
        if self.model is not None and self.scaler is not None:
            centroids = self.scaler.inverse_transform(
                self.model.cluster_centers_
            ).tolist()
        return {
            "algorithm": "kmeans",
            "n_modes": self.n_modes,
            "silhouette_score": self.silhouette,
            "columns": self.columns,
            "centroids_in_original_units": centroids,
        }


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.engineered_features_csv),
        required_columns=get_op_setting_columns(),
        stage="kmeans_context",
    )
    model = KMeansContextModel(config)
    labels = model.fit_predict(frame)
    model.save()
    logger.info("Regime distribution: %s", pd.Series(labels).value_counts().sort_index().to_dict())
    logger.info("Summary: %s", model.summary())


if __name__ == "__main__":
    main()
