from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import (
    SUBSYSTEM_DESCRIPTIONS,
    SensorDependencyGraph,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_residual_z_columns,
    sensor_from_residual,
    sensors_from_residual_frame,
)
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


_ELEVATED_THRESHOLD = 1.5
_DEGRADED_THRESHOLD = 3.0


class SubsystemExplainer:


    def __init__(
        self,
        config: Optional[Config] = None,
        graph: Optional[SensorDependencyGraph] = None,
    ) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.graph = graph or SensorDependencyGraph(self.config)

    def _status(self, mean_abs_z: float) -> str:


        if mean_abs_z >= _DEGRADED_THRESHOLD:
            return "degraded"
        if mean_abs_z >= _ELEVATED_THRESHOLD:
            return "elevated"
        return "nominal"

    def explain_row(self, row: pd.Series, sensors: List[str]) -> List[Dict[str, object]]:


        z_columns = get_residual_z_columns(sensors)
        by_subsystem: Dict[str, List[float]] = {}
        for col in z_columns:
            if col not in row.index or pd.isna(row[col]) or not np.isfinite(float(row[col])):
                continue
            sensor = sensor_from_residual(col)
            subsystem = self.graph.subsystem_of(sensor)
            by_subsystem.setdefault(subsystem, []).append(float(row[col]))

        total_abs = sum(abs(v) for values in by_subsystem.values() for v in values) or 1e-9

        results: List[Dict[str, object]] = []
        for subsystem, values in by_subsystem.items():
            mean_z = float(np.mean(values))
            mean_abs_z = float(np.mean(np.abs(values)))
            contribution = sum(abs(v) for v in values) / total_abs
            results.append(
                {
                    "subsystem": subsystem,
                    "subsystem_label": SUBSYSTEM_DESCRIPTIONS.get(subsystem, subsystem),
                    "contribution": round(contribution, 5),
                    "sensors": self.graph.sensors_in_subsystem(subsystem),
                    "mean_residual_z": round(mean_z, 5),
                    "status": self._status(mean_abs_z),
                }
            )
        results.sort(key=lambda item: item["contribution"], reverse=True)
        return results


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="subsystem_explainer",
    )
    sensors = sensors_from_residual_frame(frame)
    explainer = SubsystemExplainer(config)
    result = explainer.explain_row(frame.iloc[0], sensors)
    logger.info("Subsystem attribution for row 0: %s", result)


if __name__ == "__main__":
    main()
