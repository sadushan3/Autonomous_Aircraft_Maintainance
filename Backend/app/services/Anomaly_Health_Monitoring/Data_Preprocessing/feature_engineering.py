from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import (
    get_active_sensors,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    FEATURE_PREFIX,
    get_identity_columns,
    get_op_setting_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    rolling_slope,
)


CYCLE_NORM_COLUMN = "cycle_norm"


MAX_CYCLE_COLUMN = "engine_max_cycle"


class FeatureEngineer:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.active_sensors: List[str] = get_active_sensors(self.config)
        self.feature_names: List[str] = []


    def add_cycle_features(self, frame: pd.DataFrame) -> pd.DataFrame:


        result = frame.copy()
        reference = float(self.config.twin.age_reference_cycles)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("A positive fixed age scale is required; future trajectory length is forbidden.")
        result = result.drop(columns=[MAX_CYCLE_COLUMN], errors="ignore")
        result[CYCLE_NORM_COLUMN] = (result["cycle"].astype(float) / reference).clip(0.0, 1.0)
        return result

    def add_rolling_features(
        self, frame: pd.DataFrame, columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:


        result = frame.copy()
        created: List[str] = []
        new_columns: Dict[str, np.ndarray] = {}
        grouped = result.groupby("unit", sort=False)

        for window in self.config.features.rolling_windows:
            if window < 2:
                continue
            rolled = grouped[columns].rolling(window=window, min_periods=1)

            if self.config.features.enable_rolling_mean:
                stats = rolled.mean().reset_index(level=0, drop=True)
                for column in columns:
                    name = f"{FEATURE_PREFIX}{column}_rmean_{window}"
                    new_columns[name] = stats[column].to_numpy()
                    created.append(name)

            if self.config.features.enable_rolling_std:
                stats = rolled.std().reset_index(level=0, drop=True)
                for column in columns:
                    name = f"{FEATURE_PREFIX}{column}_rstd_{window}"
                    new_columns[name] = np.nan_to_num(stats[column].to_numpy(), nan=0.0)
                    created.append(name)

            if self.config.features.enable_rolling_min_max:
                min_stats = rolled.min().reset_index(level=0, drop=True)
                max_stats = rolled.max().reset_index(level=0, drop=True)
                for column in columns:
                    min_name = f"{FEATURE_PREFIX}{column}_rmin_{window}"
                    max_name = f"{FEATURE_PREFIX}{column}_rmax_{window}"
                    new_columns[min_name] = min_stats[column].to_numpy()
                    new_columns[max_name] = max_stats[column].to_numpy()
                    created.extend([min_name, max_name])

        if new_columns:
            result = pd.concat(
                [result, pd.DataFrame(new_columns, index=result.index)], axis=1
            )
        return result, created

    def add_delta_features(
        self, frame: pd.DataFrame, columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:


        result = frame.copy()
        created: List[str] = []
        if not self.config.features.enable_delta:
            return result, created

        deltas = result.groupby("unit", sort=False)[columns].diff().fillna(0.0)
        for column in columns:
            name = f"{FEATURE_PREFIX}{column}_delta"
            result[name] = deltas[column].to_numpy()
            created.append(name)
        return result, created

    def add_slope_features(
        self, frame: pd.DataFrame, columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:


        result = frame.copy()
        created: List[str] = []
        if not self.config.features.enable_slope:
            return result, created

        window = max(self.config.features.rolling_windows)
        for column in columns:
            name = f"{FEATURE_PREFIX}{column}_slope_{window}"
            result[name] = (
                result.groupby("unit", sort=False)[column]
                .transform(lambda series: rolling_slope(series, window))
                .to_numpy()
            )
            created.append(name)
        return result, created

    def add_ewma_features(
        self, frame: pd.DataFrame, columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:


        result = frame.copy()
        created: List[str] = []
        if not self.config.features.enable_ewma:
            return result, created

        alpha = float(self.config.features.ewma_alpha)
        smoothed = (
            result.groupby("unit", sort=False)[columns]
            .transform(lambda series: series.ewm(alpha=alpha, adjust=False).mean())
        )
        for column in columns:
            name = f"{FEATURE_PREFIX}{column}_ewma"
            result[name] = smoothed[column].to_numpy()
            created.append(name)
        return result, created


    def transform(
        self, frame: pd.DataFrame, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        result = frame.sort_values(get_identity_columns()).reset_index(drop=True)


        channels = [c for c in self.active_sensors if c in result.columns]
        settings = [c for c in get_op_setting_columns() if c in result.columns]
        summarised = channels + settings

        result = self.add_cycle_features(result)

        created: List[str] = []
        result, names = self.add_rolling_features(result, summarised)
        created.extend(names)
        result, names = self.add_delta_features(result, summarised)
        created.extend(names)
        result, names = self.add_slope_features(result, channels)
        created.extend(names)
        result, names = self.add_ewma_features(result, channels)
        created.extend(names)

        result[created] = result[created].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        self.feature_names = created

        report: Dict[str, Any] = {
            "rows": int(len(result)),
            "engines": int(result["unit"].nunique()),
            "base_channels": len(summarised),
            "features_created": len(created),
            "total_columns": int(result.shape[1]),
            "windows": list(self.config.features.rolling_windows),
        }
        self.logger.info("Feature engineering report: %s", report)
        if fit_policy:
            self._persist_feature_names(created)
        return result, report

    def _persist_feature_names(self, names: List[str]) -> Path:


        payload = {
            "engineered_features": names,
            "cycle_features": [CYCLE_NORM_COLUMN],
            "age_reference_cycles": float(self.config.twin.age_reference_cycles),
            "active_sensors": self.active_sensors,
            "windows": list(self.config.features.rolling_windows),
        }
        return atomic_write_json(
            payload, self.config.paths.models / "engineered_feature_names.json"
        )


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.engineered_features_csv
        )
        verify_identity_columns(frame, context="engineered_features output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(
            self.config.paths.engineered_features_csv
        )
        return read_csv(
            target, required_columns=get_identity_columns(), stage="feature_engineering"
        )

    def run(
        self, frame: Optional[pd.DataFrame] = None, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.cleaned_data_csv),
                required_columns=get_identity_columns(),
                stage="feature_engineering",
            )
        engineered, report = self.transform(source, fit_policy=fit_policy)
        report["output_path"] = str(self.save(engineered))
        return engineered, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    engineer = FeatureEngineer(config)
    engineered, report = engineer.run()
    logger.info(
        "engineered_features.csv written: %d rows, %d features",
        report["rows"],
        report["features_created"],
    )
    print(engineered.iloc[:5, :12].to_string())


if __name__ == "__main__":
    main()
