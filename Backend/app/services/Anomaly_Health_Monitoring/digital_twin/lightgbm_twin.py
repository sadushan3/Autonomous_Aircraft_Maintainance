from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger


class LightGBMTwin(BaseTwin):


    name = "lightgbm"
    artifact_suffix = ".pkl"

    def _build_estimator(self, n_outputs: int) -> MultiOutputRegressor:


        try:
            from lightgbm import LGBMRegressor
        except ImportError as exc:
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


        estimator.fit(x_train, y_train)
        return estimator

    def _predict_estimator(
        self, estimator: MultiOutputRegressor, x: np.ndarray
    ) -> np.ndarray:


        predictions = estimator.predict(x)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        return predictions

    def feature_importances(self) -> Dict[str, float]:


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


        if self.model is None or not hasattr(self.model, "estimators_"):
            return {}
        return dict(zip(self.target_names, self.model.estimators_))


def main() -> None:

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
