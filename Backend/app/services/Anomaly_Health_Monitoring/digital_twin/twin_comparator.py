from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
    RESIDUAL_ABS_MEAN_COLUMN,
    RESIDUAL_ABS_Z_MEAN_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    get_residual_z_columns,
    present_residual_z_columns,
    sensor_from_residual,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_csv,
)


N_DEVIATING_SENSORS_COLUMN = "n_deviating_sensors"


DEVIATION_PATTERN_COLUMN = "deviation_pattern"


DEVIATION_Z_THRESHOLD = 2.0


BROAD_PATTERN_FRACTION = 0.35


class TwinComparator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def summarize_rows(self, residuals: pd.DataFrame) -> pd.DataFrame:


        z_columns = present_residual_z_columns(residuals)

        result = residuals.copy()
        if not z_columns:
            result[N_DEVIATING_SENSORS_COLUMN] = 0
            result[DEVIATION_PATTERN_COLUMN] = "none"
            return result

        deviating = (result[z_columns].abs() > DEVIATION_Z_THRESHOLD).sum(axis=1)
        result[N_DEVIATING_SENSORS_COLUMN] = deviating.astype(int)

        fraction = deviating / max(len(z_columns), 1)
        pattern = np.where(
            deviating == 0,
            "none",
            np.where(fraction >= BROAD_PATTERN_FRACTION, "broad", "narrow"),
        )
        result[DEVIATION_PATTERN_COLUMN] = pattern
        return result


    def summarize_engines(self, residuals: pd.DataFrame) -> pd.DataFrame:


        z_columns = present_residual_z_columns(residuals)
        rows: List[Dict[str, Any]] = []

        for unit, group in residuals.groupby("unit", sort=True):
            worst_index = group[RESIDUAL_ABS_Z_MEAN_COLUMN].idxmax()
            worst_row = group.loc[worst_index]
            dominant_sensor = "unknown"
            if z_columns:
                dominant_column = worst_row[z_columns].abs().idxmax()
                dominant_sensor = sensor_from_residual(dominant_column)

            rows.append(
                {
                    "unit": int(unit),
                    "last_cycle": int(group["cycle"].max()),
                    "mean_abs_residual": float(group[RESIDUAL_ABS_MEAN_COLUMN].mean()),
                    "max_abs_residual": float(group[RESIDUAL_ABS_MEAN_COLUMN].max()),
                    "mean_deviating_sensors": float(group[N_DEVIATING_SENSORS_COLUMN].mean()),
                    "worst_cycle": int(worst_row["cycle"]),
                    "dominant_sensor_at_worst": dominant_sensor,
                    "dominant_pattern": str(
                        group[DEVIATION_PATTERN_COLUMN].mode().iloc[0]
                        if not group[DEVIATION_PATTERN_COLUMN].mode().empty
                        else "none"
                    ),
                }
            )
        return pd.DataFrame(rows)


    def run(self, residuals: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = residuals
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="twin_comparator",
            )

        row_summary = self.summarize_rows(source)
        engine_summary = self.summarize_engines(row_summary)

        pattern_counts = row_summary[DEVIATION_PATTERN_COLUMN].value_counts().to_dict()
        report: Dict[str, Any] = {
            "rows": int(len(row_summary)),
            "engines": int(row_summary["unit"].nunique()),
            "pattern_distribution": {str(k): int(v) for k, v in pattern_counts.items()},
            "mean_deviating_sensors": float(row_summary[N_DEVIATING_SENSORS_COLUMN].mean()),
            "engine_summary": engine_summary.to_dict(orient="records"),
        }
        atomic_write_json(report, self.config.paths.metrics / "twin_comparison.json")
        self.logger.info(
            "Twin comparison: pattern distribution %s, mean deviating sensors %.2f",
            report["pattern_distribution"],
            report["mean_deviating_sensors"],
        )
        return row_summary, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    comparator = TwinComparator(config)
    row_summary, report = comparator.run()
    logger.info(
        "Compared %d rows across %d engines; pattern distribution %s",
        report["rows"],
        report["engines"],
        report["pattern_distribution"],
    )
    print(row_summary[[
        "unit", "cycle", N_DEVIATING_SENSORS_COLUMN, DEVIATION_PATTERN_COLUMN
    ]].head(10).to_string())


if __name__ == "__main__":
    main()
