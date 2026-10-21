from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_residual_z_columns,
    sensor_from_residual,
    sensors_from_residual_frame,
)
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


class SensorResidualRanker:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def rank_row(
        self, row: pd.Series, sensors: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, float]]:


        z_columns = get_residual_z_columns(sensors)
        pairs = [
            (sensor_from_residual(col), float(row.get(col, 0.0)))
            for col in z_columns
            if col in row.index and pd.notna(row[col]) and np.isfinite(float(row[col]))
        ]
        total = sum(abs(value) for _, value in pairs) or 1e-9
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)

        limit = int(top_k or self.config.reasoning.top_k_sensors)
        return [
            {
                "sensor": name,
                "residual_z": round(value, 5),
                "contribution": round(abs(value) / total, 5),
                "direction": "rising" if value > 0 else ("falling" if value < 0 else "stable"),
            }
            for name, value in pairs[:limit]
        ]

    def rank_frame(
        self, frame: pd.DataFrame, sensors: List[str], top_k: Optional[int] = None
    ) -> pd.DataFrame:


        z_columns = [c for c in get_residual_z_columns(sensors) if c in frame.columns]
        limit = int(top_k or self.config.reasoning.top_k_sensors)
        if not z_columns:
            return pd.DataFrame({"top_sensors": [[] for _ in range(len(frame))]}, index=frame.index)

        values = frame[z_columns].to_numpy()
        abs_values = np.abs(values)
        order = np.argsort(-abs_values, axis=1)[:, :limit]
        sensor_names = np.array([sensor_from_residual(c) for c in z_columns])

        top_sensors = [sensor_names[order[i]].tolist() for i in range(order.shape[0])]
        return pd.DataFrame({"top_sensors": top_sensors}, index=frame.index)


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv
    from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns

    config = get_config()
    logger = get_logger(__name__, config)
    frame = read_csv(
        config.paths.processed_path(config.paths.residuals_csv),
        required_columns=get_identity_columns(),
        stage="sensor_residual_ranking",
    )
    sensors = sensors_from_residual_frame(frame)
    ranker = SensorResidualRanker(config)
    ranked = ranker.rank_row(frame.iloc[0], sensors)
    logger.info("Top sensors for row 0: %s", ranked)


if __name__ == "__main__":
    main()
