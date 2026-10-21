from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
    RESIDUAL_ABS_Z_MEAN_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_identity_columns,
    get_residual_z_columns,
    present_residual_columns,
    sensor_from_residual,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    merge_on_keys,
    read_csv,
    read_json,
)


HEALTH_INDEX_COLUMN = "health_index"


HEALTH_INDEX_RAW_COLUMN = "health_index_raw"


HEALTH_INDEX_TARGET_COLUMN = "health_index_target"


class HealthIndexCalculator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def _informative_sensors(self, residuals: pd.DataFrame) -> List[str]:


        candidates = present_residual_columns(residuals)
        if not candidates:
            return []
        spread = residuals[candidates].std(numeric_only=True)
        informative = [c for c in candidates if float(spread.get(c, 0.0) or 0.0) > 1e-8]
        return informative or candidates

    def _deviation(
        self,
        residuals: pd.DataFrame,
        columns: List[str],
        sensor_weights: Optional[Dict[str, float]] = None,
        residual_window: int = 1,
    ) -> np.ndarray:

        values = np.abs(self.sensor_values(residuals, columns, residual_window))
        if not columns or not np.isfinite(values).all():
            raise AnomalyHealthError("Health calculation requires finite residuals for the calibrated sensor channels.")
        if not sensor_weights:
            return np.nanmean(values, axis=1)
        weights = np.array(
            [sensor_weights.get(sensor_from_residual(c), 1.0) for c in columns],
            dtype=float,
        )
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise AnomalyHealthError("Health sensor weights must be finite and non-negative.")
        if weights.sum() <= 0:


            return np.nanmean(values, axis=1)
        weights = weights / max(weights.sum(), 1e-9)
        return values @ weights

    @staticmethod
    def sensor_values(residuals: pd.DataFrame, columns: List[str], window: int = 1) -> np.ndarray:

        values = residuals[columns].to_numpy(float)
        if not np.isfinite(values).all():
            raise AnomalyHealthError("Health sensor residuals must be finite.")
        if window < 1:
            raise AnomalyHealthError("Health residual window must be a positive integer.")
        if window == 1:
            return values
        ordered = residuals.reset_index(drop=True).sort_values(["unit", "cycle"])
        gaps = ordered.groupby("unit", sort=False).cycle.diff().ne(1).cumsum()
        smoothed = ordered[columns].groupby([ordered.unit, gaps], sort=False).transform(
            lambda s: s.rolling(window, min_periods=1).mean())
        result = np.empty_like(values)
        result[ordered.index.to_numpy()] = smoothed.to_numpy()
        return result


    def calibration_path(self) -> Path:


        return self.config.paths.model_path("health_index_calibration.json")

    def fit_calibration(self, residuals: pd.DataFrame) -> Dict[str, Any]:


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
        training = training_scope(residuals, self.config)
        record_training_scope("health_reference", training, self.config)
        if training.empty:
            raise AnomalyHealthError("Health reference calibration requires training rows.")
        columns = self._informative_sensors(training)
        if RUL_COLUMN in training:
            early = (training[RUL_COLUMN] >= self.config.anomaly.calibration_healthy_rul) & (
                training["cycle"] > self.config.twin.commissioning_cycles)
        else:


            early = training["cycle"].between(1, self.config.twin.commissioning_cycles)
        if not early.any():
            raise AnomalyHealthError("No healthy training rows available for health reference calibration.")
        previous = read_json(self.calibration_path(), default={})
        deviation = self._deviation(training, columns, previous.get("sensor_weights"),
                                    int(previous.get("residual_window", 1)))
        healthy_deviation = deviation[early.to_numpy()]
        if healthy_deviation.size == 0:
            healthy_deviation = deviation
        reference = float(np.median(healthy_deviation))


        overall = float(np.median(deviation)) if deviation.size else 0.0
        reference = max(reference, 0.05 * overall, 1e-9)

        calibration = {
            "informative_columns": columns,
            "healthy_reference_deviation": reference,
            "n_rows_calibrated": int(early.sum()),
        }
        if previous.get("sensor_weights"):
            calibration["sensor_weights"] = previous["sensor_weights"]
            calibration["weight_selection"] = previous.get("weight_selection", "persisted policy")
        for key in ("residual_window", "ewma_alpha", "decay_lambda"):
            if key in previous:
                calibration[key] = previous[key]
        atomic_write_json(calibration, self.calibration_path())
        self.logger.info(
            "Calibrated health index on %d healthy row(s): reference deviation "
            "%.5f across %d informative sensor(s)",
            calibration["n_rows_calibrated"],
            calibration["healthy_reference_deviation"],
            len(columns),
        )
        return calibration

    def load_calibration(self, residuals: pd.DataFrame) -> Dict[str, Any]:


        stored = read_json(self.calibration_path(), default=None)
        if not stored or "healthy_reference_deviation" not in stored:
            raise AnomalyHealthError("No health reference is available; explicitly calibrate on healthy training data first.")
        reference = float(stored["healthy_reference_deviation"])
        if not np.isfinite(reference) or reference <= 0:
            raise AnomalyHealthError("Stored health reference must be finite and positive.")

        available = set(residuals.columns)
        required = [c for c in stored.get("informative_columns", [])
                    if stored.get("sensor_weights", {}).get(sensor_from_residual(c), 1.0) > 0]
        missing = sorted(set(required) - available)
        if missing:
            raise AnomalyHealthError(f"Missing calibrated health sensor channels: {missing}")
        columns = [c for c in stored.get("informative_columns", []) if c in available]
        if not columns:
            raise AnomalyHealthError("No calibrated health sensor channels are present; inference cannot recalibrate.")
        stored["informative_columns"] = columns
        return stored

    def compute(
        self,
        residuals: pd.DataFrame,
        sensor_weights: Optional[Dict[str, float]] = None,
        recalibrate: bool = False,
    ) -> pd.DataFrame:


        ordered = residuals.sort_values(get_identity_columns()).reset_index(drop=True)
        if ordered.duplicated(get_identity_columns()).any() or ordered[get_identity_columns()].isna().any().any():
            raise AnomalyHealthError("Health input needs unique, non-missing unit/cycle identities.")

        calibration = (
            self.fit_calibration(ordered) if recalibrate else self.load_calibration(ordered)
        )
        columns = calibration["informative_columns"]
        reference = float(calibration["healthy_reference_deviation"])
        if sensor_weights is None:


            sensor_weights = calibration.get("sensor_weights")

        window = int(calibration.get("residual_window", 1))
        deviation = self._deviation(ordered, columns, sensor_weights, window)


        excess = np.clip(deviation / max(reference, 1e-9) - 1.0, 0.0, None)

        decay = float(calibration.get("decay_lambda", self.config.health.decay_lambda))
        alpha = float(calibration.get("ewma_alpha", self.config.health.ewma_alpha))
        if not np.isfinite([decay, alpha]).all() or decay <= 0 or not 0 < alpha <= 1:
            raise AnomalyHealthError("Health decay must be positive and EWMA alpha must be within (0, 1].")
        h_raw = np.exp(-decay * excess)

        result = ordered[get_identity_columns()].copy()
        result[HEALTH_INDEX_RAW_COLUMN] = 100.0 * np.clip(h_raw, 0.0, 1.0)

        smoothed = (
            pd.Series(h_raw, index=ordered.index)
            .groupby(ordered["unit"], sort=False)
            .transform(lambda series: series.ewm(alpha=alpha, adjust=False).mean())
        )
        result[HEALTH_INDEX_COLUMN] = 100.0 * np.clip(smoothed, 0.0, 1.0)

        if RUL_COLUMN in ordered.columns:
            cap = float(self.config.data.rul_cap)
            target = 100.0 * np.clip(ordered[RUL_COLUMN].astype(float), 0.0, cap) / cap
            result[HEALTH_INDEX_TARGET_COLUMN] = target

        self.logger.info(
            "Health index computed: mean=%.2f, min=%.2f (decay=%.3f, ewma_alpha=%.3f)",
            float(result[HEALTH_INDEX_COLUMN].mean()),
            float(result[HEALTH_INDEX_COLUMN].min()),
            decay,
            alpha,
        )
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.health_index_csv)
        verify_identity_columns(frame, context="health_index output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.health_index_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="health_index_calculator"
        )

    def run(
        self,
        residuals: Optional[pd.DataFrame] = None,
        recalibrate: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = residuals
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="health_index_calculator",
            )

        raw_path = self.config.paths.processed_path(self.config.paths.raw_data_csv)
        if raw_path.is_file():
            raw = read_csv(raw_path, required_columns=get_identity_columns(), stage="health_index_calculator")
            extras = [c for c in (RUL_COLUMN, "split", "subset_id") if c in raw and c not in source]
            if extras:
                source = merge_on_keys(
                    source,
                    raw[get_identity_columns() + extras],
                    keys=get_identity_columns(),
                    how="left",
                )

        if recalibrate:
            from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
            fitting = training_scope(source, self.config)
            record_training_scope("health_reference", fitting, self.config)
            self.fit_calibration(fitting)
        health_index = self.compute(source, recalibrate=False)
        output = self.save(health_index)

        report: Dict[str, Any] = {
            "rows": int(len(health_index)),
            "engines": int(health_index["unit"].nunique()),
            "mean_health_index": float(health_index[HEALTH_INDEX_COLUMN].mean()),
            "min_health_index": float(health_index[HEALTH_INDEX_COLUMN].min()),
            "has_supervised_target": HEALTH_INDEX_TARGET_COLUMN in health_index.columns,
            "output_path": str(output),
        }
        self.logger.info("Health index report: %s", report)
        return health_index, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    calculator = HealthIndexCalculator(config)
    health_index, report = calculator.run()
    logger.info("health_index.csv written: %d rows", report["rows"])
    print(health_index.head(10).to_string())


if __name__ == "__main__":
    main()
