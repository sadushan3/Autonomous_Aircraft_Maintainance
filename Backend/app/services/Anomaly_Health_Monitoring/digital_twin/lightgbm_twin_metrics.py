"""
Per-sensor diagnostic metrics for the LightGBM twin.

The aggregate RMSE/MAE/R2 reported by :meth:`BaseTwin.train` hides which
specific sensors the twin models well and which it struggles with.  A sensor
with poor twin accuracy will produce noisy residuals regardless of the
engine's actual health, which inflates its anomaly false-alarm rate.  This
module surfaces that per-sensor breakdown so it can be inspected (and, if
needed, that sensor's contribution can be down-weighted in the fusion layers)
before it becomes a silent source of false positives.
"""

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
    """Computes and reports per-sensor accuracy for a trained LightGBM twin."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the metrics reporter.

        Args:
            config: Configuration supplying output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def evaluate(self, twin: LightGBMTwin, frame: pd.DataFrame) -> Dict[str, Any]:
        """Score a trained twin per sensor on the given data.

        Args:
            twin: A trained (or loaded) :class:`LightGBMTwin`.
            frame: Scaled, context-labelled evaluation data, typically the
                validation engines or the healthy region.

        Returns:
            A report with global and per-sensor accuracy plus feature
            importances.
        """
        healthy = twin.select_healthy_region(frame)
        matrix, _ = twin.build_design_matrix(healthy, n_modes=twin.n_modes)
        targets, sensors = twin.build_targets(healthy)
        predictions = twin._predict_estimator(twin.model, matrix)  # noqa: SLF001

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
        """Load the trained LightGBM twin and evaluate it per sensor.

        Args:
            frame: Optional in-memory input; otherwise ``scaled_features.csv``
                is read from disk.

        Returns:
            The metrics report.
        """
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
    """Report per-sensor LightGBM twin accuracy standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    reporter = LightGBMTwinMetrics(config)
    report = reporter.run()
    logger.info("Worst 5 sensors: %s", report["worst_5_sensors"])
    logger.info("Best 5 sensors: %s", report["best_5_sensors"])


if __name__ == "__main__":
    main()
