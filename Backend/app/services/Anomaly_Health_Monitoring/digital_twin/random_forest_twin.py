from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


class RandomForestTwin(BaseTwin):


    name = "random_forest"
    artifact_suffix = ".pkl"

    def _build_estimator(self, n_outputs: int) -> RandomForestRegressor:


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


        estimator.fit(x_train, y_train)
        return estimator

    def _predict_estimator(
        self, estimator: RandomForestRegressor, x: np.ndarray
    ) -> np.ndarray:


        predictions = estimator.predict(x)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def feature_importances(self) -> Dict[str, float]:


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


        if self.model is None:
            return np.zeros((len(frame), len(self.target_names)), dtype=float)
        matrix, _ = self.build_design_matrix(frame, n_modes=self.n_modes)
        per_tree = np.stack(
            [tree.predict(matrix) for tree in self.model.estimators_], axis=0
        )
        return np.std(per_tree, axis=0)


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    twin = RandomForestTwin(config)
    predictions, metrics = twin.run()
    logger.info("Random forest twin metrics: %s", metrics)
    logger.info("Top feature importances: %s", dict(list(twin.feature_importances().items())[:8]))
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
