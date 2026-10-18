from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.lightgbm_twin import LightGBMTwin
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_csv,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns


class LightGBMTwinMetrics:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def evaluate(self, twin: LightGBMTwin, frame: pd.DataFrame) -> Dict[str, Any]:


        healthy = twin.select_healthy_region(frame)
        matrix, _ = twin.build_design_matrix(healthy, n_modes=twin.n_modes)
        targets, sensors = twin.build_targets(healthy)
        predictions = twin._predict_estimator(twin.model, matrix)

        per_sensor = twin.per_sensor_metrics(targets, predictions)
        worst = sorted(per_sensor.items(), key=lambda item: item[1]["rmse"], reverse=True)[:5]
        best = sorted(per_sensor.items(), key=lambda item: item[1]["rmse"])[:5]

        report: Dict[str, Any] = {
            "twin": twin.name,
            "rows_evaluated": int(len(healthy)),
            "per_sensor_metrics": per_sensor,
            "worst_5_sensors": {name: metrics for name, metrics in worst},
            "best_5_sensors": {name: metrics for name, metrics in best},
            "feature_importances_top10": dict(
                list(twin.feature_importances().items())[:10]
            ),
        }
        self.logger.info(
            "LightGBM per-sensor metrics computed for %d sensor(s); worst RMSE=%s",
            len(sensors),
            worst[0] if worst else None,
        )
        return report

    def run(self, frame: Optional[pd.DataFrame] = None) -> Dict[str, Any]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.scaled_features_csv),
                required_columns=get_identity_columns(),
                stage="lightgbm_twin_metrics",
            )
        twin = LightGBMTwin(self.config).load()
        report = self.evaluate(twin, source)
        atomic_write_json(report, self.config.paths.metrics / "lightgbm_twin_metrics.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    reporter = LightGBMTwinMetrics(config)
    report = reporter.run()
    logger.info("Worst 5 sensors: %s", report["worst_5_sensors"])
    logger.info("Best 5 sensors: %s", report["best_5_sensors"])


if __name__ == "__main__":
    main()
