"""
Feature engineering over engine trajectories.

Every feature here is **causal**: it is computed from the current cycle and
earlier cycles of the *same* engine only.  Two rules make that true:

* All rolling operations are grouped by ``unit``, so no engine ever sees
  another engine's history.
* All windows are trailing (``rolling(window)`` with no centring and no
  negative shift), so no future cycle leaks backwards into the present.

That discipline matters more here than in most pipelines: the label is derived
from how far the engine is from its own failure, so a single non-causal feature
would let a model read the answer off the input.

Feature families produced, all prefixed ``feat_``:

* rolling mean / std / min / max over the configured windows
* first difference (delta) against the previous cycle
* trailing least-squares slope
* exponentially weighted moving average
* normalised age (``cycle_norm``) and raw cycle
"""

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

#: Normalised position within the engine's observed life, in [0, 1].
CYCLE_NORM_COLUMN = "cycle_norm"

#: Total observed cycles for the engine, broadcast to every row.
MAX_CYCLE_COLUMN = "engine_max_cycle"


class FeatureEngineer:
    """Builds causal, per-engine temporal features from cleaned sensor data."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the feature engineer.

        Args:
            config: Configuration supplying window sizes and feature toggles.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.active_sensors: List[str] = get_active_sensors(self.config)
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------ #
    # Individual feature families
    # ------------------------------------------------------------------ #
    def add_cycle_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add raw and normalised engine age.

        ``cycle_norm`` is the engine's position within its own observed life.
        It is the single most important context variable for the digital twin,
        which must know how far into a trajectory it is being asked to predict.

        Args:
            frame: Cleaned data.

        Returns:
            The frame with age features added.
        """
        result = frame.copy()
        max_cycle = result.groupby("unit")["cycle"].transform("max").astype(float)
        result[MAX_CYCLE_COLUMN] = max_cycle
        result[CYCLE_NORM_COLUMN] = (
            result["cycle"].astype(float) / max_cycle.replace(0.0, np.nan)
        ).fillna(0.0).clip(0.0, 1.0)
        return result

    def add_rolling_features(
        self, frame: pd.DataFrame, columns: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Add trailing rolling statistics per engine.

        Args:
            frame: Data sorted by ``(unit, cycle)``.
            columns: Channels to summarise.

        Returns:
            A ``(frame, new_feature_names)`` tuple.
        """
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
        """Add per-engine first differences.

        Args:
            frame: Data sorted by ``(unit, cycle)``.
            columns: Channels to difference.

        Returns:
            A ``(frame, new_feature_names)`` tuple.
        """
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
        """Add trailing least-squares slopes over the longest window.

        The slope is what distinguishes a genuine degradation trend from a
        one-off sensor excursion, and it feeds the temporal reasoning layer
        directly.

        Args:
            frame: Data sorted by ``(unit, cycle)``.
            columns: Channels to fit.

        Returns:
            A ``(frame, new_feature_names)`` tuple.
        """
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
        """Add exponentially weighted moving averages per engine.

        Args:
            frame: Data sorted by ``(unit, cycle)``.
            columns: Channels to smooth.

        Returns:
            A ``(frame, new_feature_names)`` tuple.
        """
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

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def transform(
        self, frame: pd.DataFrame, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Build every enabled feature family.

        Args:
            frame: Cleaned data.
            fit_policy: Persist the engineered-feature manifest (training).
                False leaves ``engineered_feature_names.json`` untouched --
                the feature set is already pinned by ``self.active_sensors``
                (loaded from the persisted cleaner policy at construction),
                so this only guards against an unnecessary rewrite during
                inference.

        Returns:
            A ``(engineered_frame, report)`` tuple.
        """
        result = frame.sort_values(get_identity_columns()).reset_index(drop=True)

        # Sensors plus operating settings: the settings also drift, and their
        # rolling statistics are what expose a regime transition.
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
        """Record the engineered feature list for downstream alignment.

        Args:
            names: Feature column names created.

        Returns:
            The path written.
        """
        payload = {
            "engineered_features": names,
            "cycle_features": [CYCLE_NORM_COLUMN, MAX_CYCLE_COLUMN],
            "active_sensors": self.active_sensors,
            "windows": list(self.config.features.rolling_windows),
        }
        return atomic_write_json(
            payload, self.config.paths.models / "engineered_feature_names.json"
        )

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``engineered_features.csv`` atomically.

        Args:
            frame: Engineered data.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.engineered_features_csv
        )
        verify_identity_columns(frame, context="engineered_features output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back ``engineered_features.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(
            self.config.paths.engineered_features_csv
        )
        return read_csv(
            target, required_columns=get_identity_columns(), stage="feature_engineering"
        )

    def run(
        self, frame: Optional[pd.DataFrame] = None, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the feature-engineering stage.

        Args:
            frame: Optional in-memory input; otherwise ``cleaned_data.csv`` is
                read from disk.
            fit_policy: See :meth:`transform`. False for inference over new
                data.

        Returns:
            A ``(engineered_frame, report)`` tuple.
        """
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
    """Run feature engineering standalone against ``cleaned_data.csv``."""
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
