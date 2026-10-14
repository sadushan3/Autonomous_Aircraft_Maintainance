"""
XGBoost Digital Twin.

``XGBRegressor`` is single-output, so it is wrapped in scikit-learn's
``MultiOutputRegressor``, which fits one boosted ensemble per sensor channel.
That is more expensive than the random forest's shared-tree approach, but it
lets each sensor get a model tuned to its own noise level and response curve --
sensor 9 and sensor 16 have wildly different dynamic ranges, and a shared
structure compromises between them.

Gradient boosting is the sharpest of the four twins on the healthy region,
which makes it the highest-weighted member of the ensemble under inverse-RMSE
weighting, and also the one most prone to confidently extrapolating nonsense
outside the training envelope.  That tension is precisely why the component
fuses four twins and reports their disagreement as uncertainty rather than
trusting any single one.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class XGBoostTwin(BaseTwin):
    """Digital Twin backed by per-sensor gradient-boosted trees."""

    name = "xgboost"
    artifact_suffix = ".pkl"

    def _build_estimator(self, n_outputs: int) -> MultiOutputRegressor:
        """Construct a multi-output wrapper around XGBRegressor.

        Args:
            n_outputs: Number of sensor channels, one booster each.

        Returns:
            An unfitted :class:`MultiOutputRegressor`.

        Raises:
            AnomalyHealthError: If xgboost is not installed.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise AnomalyHealthError(
                "xgboost is required for the XGBoost twin. Install it with "
                "'pip install xgboost'."
            ) from exc

        params = dict(self.config.twin.xgboost)
        self.logger.info(
            "Building XGBoost twin: %d booster(s), params=%s", n_outputs, params
        )
        base = XGBRegressor(**params)
        # n_jobs=1 on the wrapper: the boosters already parallelise internally,
        # and nesting the two thread pools oversubscribes the CPU badly.
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
            x_val: Unused; MultiOutputRegressor cannot forward eval sets.
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
        """Average gain-based importance across every per-sensor booster.

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
        paired = sorted(
            zip(self.feature_names, mean_importance.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: float(value) for name, value in paired}

    def per_sensor_models(self) -> Dict[str, Any]:
        """Expose the individual boosters keyed by sensor name.

        The SHAP explainer uses this to build a ``TreeExplainer`` against the
        specific booster for the sensor under investigation, rather than trying
        to explain all 21 outputs at once.

        Returns:
            A mapping of sensor column name to its fitted booster.
        """
        if self.model is None or not hasattr(self.model, "estimators_"):
            return {}
        return dict(zip(self.target_names, self.model.estimators_))


def main() -> None:
    """Train the XGBoost twin standalone and report accuracy."""
    config = get_config()
    logger = get_logger(__name__, config)
    twin = XGBoostTwin(config)
    predictions, metrics = twin.run()
    logger.info("XGBoost twin metrics: %s", metrics)
    logger.info(
        "Top feature importances: %s", dict(list(twin.feature_importances().items())[:8])
    )
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
