"""
Gaussian-Mixture-based operating-regime discovery.

The GMM is the probabilistic alternative to KMeans for the same job.  Its
advantage here is that it yields a genuine posterior over regimes rather than a
hard assignment, and that posterior is exactly what the uncertainty layer wants:
a row sitting between two regimes gets a low ``mode_confidence`` and therefore a
lower final confidence score, instead of being silently forced into whichever
regime happened to be marginally closer.

Model order is selected by BIC rather than silhouette.  BIC is the natural
criterion for a likelihood-based model and, unlike the silhouette, it is
defined for a single component, so a one-regime subset (FD001, FD003) can be
identified honestly instead of being forced to k>=2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import get_op_setting_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    get_logger,
    load_model,
    save_model,
)


class GMMContextModel:
    """Discovers operating regimes as a Gaussian mixture over the settings."""

    #: Filename used for the persisted model bundle.
    ARTIFACT_NAME = "context_gmm.pkl"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the model.

        Args:
            config: Configuration supplying the component count and range.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.columns: List[str] = get_op_setting_columns()
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[GaussianMixture] = None
        self.n_modes: int = 0
        self.bic: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Fitting
    # ------------------------------------------------------------------ #
    def _matrix(self, frame: pd.DataFrame) -> np.ndarray:
        """Extract and sanitise the settings matrix.

        Args:
            frame: Data containing the operating settings.

        Returns:
            An ``(n_rows, 3)`` float array.

        Raises:
            AnomalyHealthError: If a settings column is absent.
        """
        missing = [c for c in self.columns if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(f"Operating settings missing from input: {missing}")
        values = frame[self.columns].astype(float).to_numpy()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def select_components(
        self, values: np.ndarray, raw: Optional[np.ndarray] = None
    ) -> Tuple[int, Optional[float]]:
        """Choose the number of mixture components by minimising BIC.

        Args:
            values: Scaled settings matrix.
            raw: Pre-scaling settings matrix, used to detect the single-regime
                case (see :meth:`KMeansContextModel.select_k` for why the
                check must run on raw, not scaled, values).

        Returns:
            A ``(n_components, bic)`` tuple.
        """
        configured = int(self.config.context.n_modes)
        if configured > 0:
            self.logger.info("Using configured component count k=%d", configured)
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
        best_bic = float("inf")
        scores: Dict[int, float] = {}

        for candidate in self.config.context.auto_k_range:
            k = int(candidate)
            if k < 1 or k >= sample.shape[0]:
                continue
            try:
                model = GaussianMixture(
                    n_components=k,
                    covariance_type="full",
                    reg_covar=1e-6,
                    n_init=3,
                    random_state=self.config.data.random_state,
                ).fit(sample)
            except ValueError as exc:
                self.logger.debug("GMM with k=%d failed to converge: %s", k, exc)
                continue
            value = float(model.bic(sample))
            scores[k] = value
            if value < best_bic:
                best_bic = value
                best_k = k

        if not scores:
            return 1, None

        self.logger.info("BIC by k: %s", {k: round(v, 1) for k, v in scores.items()})
        self.logger.info("Selected k=%d (BIC=%.1f)", best_k, best_bic)
        return best_k, best_bic

    def fit(self, frame: pd.DataFrame) -> "GMMContextModel":
        """Fit the scaler and Gaussian mixture on the operating settings.

        Args:
            frame: Data containing the operating settings.

        Returns:
            This fitted instance.
        """
        raw = self._matrix(frame)
        self.scaler = StandardScaler().fit(raw)
        scaled = self.scaler.transform(raw)

        self.n_modes, self.bic = self.select_components(scaled, raw=raw)
        self.model = GaussianMixture(
            n_components=max(self.n_modes, 1),
            covariance_type="full",
            reg_covar=1e-6,
            n_init=3,
            random_state=self.config.data.random_state,
        ).fit(scaled)

        labels = self.model.predict(scaled)
        counts = np.bincount(labels, minlength=self.n_modes)
        self.logger.info(
            "GMM fitted: %d regime(s), rows per regime %s, converged=%s",
            self.n_modes,
            counts.tolist(),
            self.model.converged_,
        )
        return self

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Assign the most likely operating regime to every row.

        Args:
            frame: Data containing the operating settings.

        Returns:
            An integer array of regime labels.

        Raises:
            AnomalyHealthError: If called before fitting or loading.
        """
        if self.model is None or self.scaler is None:
            raise AnomalyHealthError("GMMContextModel used before fit/load.")
        scaled = self.scaler.transform(self._matrix(frame))
        return self.model.predict(scaled).astype(int)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the full posterior over regimes for every row.

        Args:
            frame: Data containing the operating settings.

        Returns:
            An ``(n_rows, n_modes)`` array of posterior probabilities.

        Raises:
            AnomalyHealthError: If called before fitting or loading.
        """
        if self.model is None or self.scaler is None:
            raise AnomalyHealthError("GMMContextModel used before fit/load.")
        scaled = self.scaler.transform(self._matrix(frame))
        return self.model.predict_proba(scaled)

    def confidence(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the posterior probability of each row's assigned regime.

        This is the honest version of regime confidence: it is a probability,
        not a distance heuristic.

        Args:
            frame: Data containing the operating settings.

        Returns:
            A float array in [0, 1].
        """
        if self.n_modes < 2:
            return np.ones(len(frame), dtype=float)
        posterior = self.predict_proba(frame)
        return np.clip(posterior.max(axis=1), 0.0, 1.0)

    def fit_predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Fit the mixture then label the same data.

        Args:
            frame: Data containing the operating settings.

        Returns:
            An integer array of regime labels.
        """
        return self.fit(frame).predict(frame)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Optional[Path] = None) -> Path:
        """Persist the fitted scaler and mixture together.

        Args:
            path: Optional explicit destination.

        Returns:
            The path written.

        Raises:
            AnomalyHealthError: If nothing has been fitted.
        """
        if self.model is None:
            raise AnomalyHealthError("Nothing to save; fit the model first.")
        target = path or self.config.paths.model_path(self.ARTIFACT_NAME)
        bundle = {
            "scaler": self.scaler,
            "model": self.model,
            "columns": self.columns,
            "n_modes": self.n_modes,
            "bic": self.bic,
        }
        return save_model(bundle, target, config=self.config)

    def load(self, path: Optional[Path] = None) -> "GMMContextModel":
        """Reload a previously persisted model bundle.

        Args:
            path: Optional explicit source.

        Returns:
            This instance, ready to predict.
        """
        target = path or self.config.paths.model_path(self.ARTIFACT_NAME)
        bundle = load_model(target)
        self.scaler = bundle["scaler"]
        self.model = bundle["model"]
        self.columns = bundle["columns"]
        self.n_modes = int(bundle["n_modes"])
        self.bic = bundle.get("bic")
        self.logger.info("Loaded GMM context model with %d regime(s)", self.n_modes)
        return self

    def summary(self) -> Dict[str, Any]:
        """Return a description of the fitted mixture.

        Returns:
            A mapping of component count, BIC, weights and means.
        """
        means: List[List[float]] = []
        weights: List[float] = []
        if self.model is not None and self.scaler is not None:
            means = self.scaler.inverse_transform(self.model.means_).tolist()
            weights = self.model.weights_.tolist()
        return {
            "algorithm": "gmm",
            "n_modes": self.n_modes,
            "bic": self.bic,
            "columns": self.columns,
            "weights": weights,
            "means_in_original_units": means,
        }


def main() -> None:
    """Fit GMM regimes against ``engineered_features.csv``."""
    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.engineered_features_csv),
        required_columns=get_op_setting_columns(),
        stage="gmm_context",
    )
    model = GMMContextModel(config)
    labels = model.fit_predict(frame)
    model.save()
    logger.info(
        "Regime distribution: %s", pd.Series(labels).value_counts().sort_index().to_dict()
    )
    logger.info("Mean posterior confidence: %.4f", float(model.confidence(frame).mean()))
    logger.info("Summary: %s", model.summary())


if __name__ == "__main__":
    main()
