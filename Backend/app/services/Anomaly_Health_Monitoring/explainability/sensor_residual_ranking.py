"""
Sensor residual ranking: answering "which sensors contributed" cheaply.

SHAP explains the twin's *prediction*; this module explains the *deviation*
directly from the residuals already computed, with none of SHAP's
computational cost.  It is what the reasoning layer and the dashboard use for
every row (SHAP is reserved for the smaller sample the explainability stage
processes on request), and it is deliberately simple enough that an operator
can recompute it by hand from the residual columns: rank sensors by absolute
robust z-score, done.
"""

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
    """Ranks sensors by contribution to a row's total residual deviation."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the ranker.

        Args:
            config: Configuration supplying the default top-k count.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)

    def rank_row(
        self, row: pd.Series, sensors: List[str], top_k: Optional[int] = None
    ) -> List[Dict[str, float]]:
        """Rank one row's sensors by |residual z-score|.

        Args:
            row: A row carrying ``residual_z_sensor_*`` columns.
            sensors: Active sensor names to consider.
            top_k: Cap on how many to return; defaults to config.

        Returns:
            A list of ``{'sensor', 'residual_z', 'contribution'}`` mappings,
            highest contribution first. ``contribution`` is each sensor's
            share of the row's total absolute z-score.
        """
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
        """Compute the top contributing sensor names for every row at once.

        Vectorised over the residual z-score matrix for speed on the full
        dataset, rather than looping via :meth:`rank_row`.

        Args:
            frame: Frame carrying ``residual_z_sensor_*`` columns.
            sensors: Active sensor names to consider.
            top_k: Cap on how many sensors to keep per row.

        Returns:
            A frame with one column, ``top_sensors``, holding a Python list of
            sensor names per row, ranked by |z|.
        """
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
    """Demonstrate ranking against ``residuals.csv``."""
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
