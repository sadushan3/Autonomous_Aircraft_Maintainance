"""
Random Forest Digital Twin.

``RandomForestRegressor`` supports multi-output regression natively -- each leaf
stores a vector of sensor means rather than a scalar -- so all 21 channels are
predicted by one model without any wrapper.  That matters for more than
convenience: the outputs share the same tree structure, so sensors that respond
to the same operating condition are predicted consistently with one another
instead of drifting apart the way independently fitted models can.

The forest is the most robust of the four twins.  It extrapolates poorly beyond
the training envelope, which is exactly the behaviour wanted here: when a
degraded engine leaves the healthy region the forest keeps predicting healthy
values, and the residual grows.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


class RandomForestTwin(BaseTwin):
    """Digital Twin backed by a natively multi-output random forest."""

    name = "random_forest"
    artifact_suffix = ".pkl"

    def _build_estimator(self, n_outputs: int) -> RandomForestRegressor:
        """Construct the forest from configured hyper-parameters.

        Args:
            n_outputs: Number of sensor channels; informational only, since the
                forest infers output width from the target matrix.

        Returns:
            An unfitted :class:`RandomForestRegressor`.
        """
        params = dict(self.config.twin.random_forest)
        self.logger.info(
            "Building random forest twin for %d output(s): %s", n_outputs, params
        )
        return RandomForestRegressor(**params)

    def _fit_estimator(
        self,
        estimator: RandomForestRegressor,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> RandomForestRegressor:
        """Fit the forest.

        The forest has no early-stopping mechanism, so the validation arrays are
        accepted for interface symmetry and used only by the caller for scoring.

        Args:
            estimator: The unfitted forest.
            x_train: Training design matrix.
            y_train: Training targets.
            x_val: Unused.
            y_val: Unused.

        Returns:
            The fitted forest.
        """
        estimator.fit(x_train, y_train)
        return estimator

    def _predict_estimator(
        self, estimator: RandomForestRegressor, x: np.ndarray
    ) -> np.ndarray:
        """Predict expected sensor values.

        Args:
            estimator: The fitted forest.
            x: Design matrix.

        Returns:
            An ``(n_rows, n_targets)`` prediction array.
        """
        predictions = estimator.predict(x)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def feature_importances(self) -> Dict[str, float]:
        """Return mean impurity-decrease importance per input feature.

        Useful for confirming the twin is actually keying off operating context
        rather than latching onto raw cycle count.

        Returns:
            A mapping of feature name to importance, highest first.
        """
        if self.model is None or not hasattr(self.model, "feature_importances_"):
            return {}
        importances = np.asarray(self.model.feature_importances_, dtype=float)
        paired = sorted(
            zip(self.feature_names, importances.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return {name: float(value) for name, value in paired}

    def prediction_spread(self, frame: pd.DataFrame) -> np.ndarray:
        """Estimate epistemic uncertainty from disagreement between trees.

        Each tree in the forest is an independent estimate of healthy
        behaviour; where they disagree, the expected value is itself uncertain.
        This complements the cross-twin spread computed by the ensemble.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            An ``(n_rows, n_targets)`` array of per-sensor standard deviations.
        """
        if self.model is None:
            return np.zeros((len(frame), len(self.target_names)), dtype=float)
        matrix, _ = self.build_design_matrix(frame, n_modes=self.n_modes)
        per_tree = np.stack(
            [tree.predict(matrix) for tree in self.model.estimators_], axis=0
        )
        return np.std(per_tree, axis=0)


def main() -> None:
    """Train the random forest twin standalone and report accuracy."""
    config = get_config()
    logger = get_logger(__name__, config)
    twin = RandomForestTwin(config)
    predictions, metrics = twin.run()
    logger.info("Random forest twin metrics: %s", metrics)
    logger.info("Top feature importances: %s", dict(list(twin.feature_importances().items())[:8]))
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
