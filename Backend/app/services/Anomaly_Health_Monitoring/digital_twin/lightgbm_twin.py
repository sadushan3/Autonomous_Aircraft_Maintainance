"""
LightGBM Digital Twin.

``LGBMRegressor`` is single-output, so as with XGBoost it is wrapped in
``MultiOutputRegressor`` to produce one leaf-wise boosted ensemble per sensor.

LightGBM grows trees leaf-wise rather than level-wise, which lets it capture the
sharp regime boundaries in FD002/FD004 -- where a single throttle step moves a
sensor by many standard deviations -- with far fewer splits than a depth-wise
learner needs.  On the one-regime subsets it behaves very similarly to XGBoost,
which is useful: when two independently-implemented boosters agree closely, the
ensemble spread is genuinely small and the uncertainty layer says so.

Verbosity is pinned to silent in config because a 21-booster fit otherwise
emits several hundred lines of per-iteration output that drowns the pipeline log.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class LightGBMTwin(BaseTwin):
    """Digital Twin backed by per-sensor leaf-wise gradient boosting."""

    name = "lightgbm"
    artifact_suffix = ".pkl"

    def _build_estimator(self, n_outputs: int) -> MultiOutputRegressor:
        """Construct a multi-output wrapper around LGBMRegressor.

        Args:
            n_outputs: Number of sensor channels, one booster each.

        Returns:
            An unfitted :class:`MultiOutputRegressor`.

        Raises:
            AnomalyHealthError: If lightgbm is not installed.
        """
        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AnomalyHealthError(
                "lightgbm is required for the LightGBM twin. Install it with "
                "'pip install lightgbm'."
            ) from exc

        params = dict(self.config.twin.lightgbm)
        self.logger.info(
            "Building LightGBM twin: %d booster(s), params=%s", n_outputs, params
        )
        base = LGBMRegressor(**params)
        return MultiOutputRegressor(base, n_jobs=1)

    def _fit_estimator(
        self,
        estimator: MultiOutputRegressor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> MultiOutputRegressor:
        """Fit one booster per sensor channel.

        Args:
            estimator: The unfitted wrapper.
            x_train: Training design matrix.
            y_train: Training targets.
            x_val: Unused; the wrapper cannot forward eval sets per output.
            y_val: Unused.

        Returns:
            The fitted wrapper.
        """
        estimator.fit(x_train, y_train)
        return estimator

    def _predict_estimator(
        self, estimator: MultiOutputRegressor, x: np.ndarray
    ) -> np.ndarray:
        """Predict expected sensor values.

        Args:
            estimator: The fitted wrapper.
            x: Design matrix.

        Returns:
            An ``(n_rows, n_targets)`` prediction array.
        """
        predictions = estimator.predict(x)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def feature_importances(self) -> Dict[str, float]:
        """Average split-count importance across every per-sensor booster.

        Args:
            None.

        Returns:
            A mapping of feature name to mean importance, highest first.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return {}
        stacked: List[np.ndarray] = []
        for booster in self.model.estimators_:
            if hasattr(booster, "feature_importances_"):
                stacked.append(np.asarray(booster.feature_importances_, dtype=float))
        if not stacked:
            return {}
        mean_importance = np.mean(np.vstack(stacked), axis=0)
        total = float(mean_importance.sum())
        if total > 0:
            mean_importance = mean_importance / total
        paired = sorted(
            zip(self.feature_names, mean_importance.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: float(value) for name, value in paired}

    def per_sensor_models(self) -> Dict[str, Any]:
        """Expose the individual boosters keyed by sensor name.

        Returns:
            A mapping of sensor column name to its fitted booster.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return {}
        return dict(zip(self.target_names, self.model.estimators_))


def main() -> None:
    """Train the LightGBM twin standalone and report accuracy."""
    config = get_config()
    logger = get_logger(__name__, config)
    twin = LightGBMTwin(config)
    predictions, metrics = twin.run()
    logger.info("LightGBM twin metrics: %s", metrics)
    logger.info(
        "Top feature importances: %s", dict(list(twin.feature_importances().items())[:8])
    )
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
