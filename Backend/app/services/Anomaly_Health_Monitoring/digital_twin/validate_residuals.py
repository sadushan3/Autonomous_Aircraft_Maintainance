from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
    RESIDUAL_ABS_MEAN_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_identity_columns,
    get_residual_columns,
    get_residual_z_columns,
    sensor_from_residual,
    sensors_from_residual_frame,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_csv,
)


class ResidualValidator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)


    def _healthy_mask(self, residuals: pd.DataFrame) -> pd.Series:


        settings = self.config.healthy_region
        if RUL_COLUMN in residuals.columns:
            rul_ok = residuals[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            rul_ok = pd.Series(True, index=residuals.index)

        max_cycle = residuals.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = residuals["cycle"].astype(float) <= (
            float(settings.cycle_fraction) * max_cycle
        )
        return rul_ok & cycle_ok


    def check_centring(self, healthy: pd.DataFrame, sensors: List[str]) -> Dict[str, Any]:


        residual_columns = get_residual_columns(sensors)
        means = {
            sensor: float(healthy[col].mean())
            for sensor, col in zip(sensors, residual_columns)
            if col in healthy.columns
        }
        mean_abs_bias = float(np.mean(np.abs(list(means.values())))) if means else 0.0
        return {"per_sensor_mean": means, "mean_absolute_bias": mean_abs_bias}

    def check_spread(self, healthy: pd.DataFrame, sensors: List[str]) -> Dict[str, Any]:


        z_columns = get_residual_z_columns(sensors)
        present = [c for c in z_columns if c in healthy.columns]
        ceiling = float(self.config.twin.max_acceptable_healthy_residual_z)

        if not present or healthy.empty:
            return {"exceedance_fraction": 0.0, "ceiling": ceiling, "per_sensor_p99_abs_z": {}}

        abs_z = healthy[present].abs()
        exceed = (abs_z > ceiling).to_numpy()
        p99 = {
            sensor_from_residual(col): float(np.percentile(abs_z[col], 99))
            for col in present
        }
        return {
            "exceedance_fraction": float(exceed.mean()),
            "ceiling": ceiling,
            "per_sensor_p99_abs_z": p99,
        }

    def check_coverage(self, residuals: pd.DataFrame, sensors: List[str]) -> Dict[str, Any]:


        residual_columns = get_residual_columns(sensors)
        missing = [
            sensor
            for sensor, col in zip(sensors, residual_columns)
            if col not in residuals.columns
        ]
        return {"expected_sensors": len(sensors), "missing_sensors": missing}

    def check_growth(self, residuals: pd.DataFrame) -> Dict[str, Any]:


        rising = 0
        total = 0
        for _, group in residuals.groupby("unit", sort=False):
            if len(group) < 5:
                continue
            max_cycle = group["cycle"].max()
            if max_cycle <= 0:
                continue
            position = group["cycle"].astype(float) / float(max_cycle)
            correlation = np.corrcoef(position, group[RESIDUAL_ABS_MEAN_COLUMN].astype(float))[0, 1]
            if np.isfinite(correlation):
                total += 1
                if correlation > 0:
                    rising += 1
        fraction = float(rising / total) if total else 0.0
        return {"engines_checked": total, "engines_with_rising_residual": rising, "fraction_rising": fraction}


    def validate(
        self, residuals: pd.DataFrame, sensors: Optional[List[str]] = None
    ) -> Dict[str, Any]:


        active_sensors = sensors or sensors_from_residual_frame(residuals)

        healthy_mask = self._healthy_mask(residuals)
        healthy = residuals.loc[healthy_mask]
        if healthy.empty:
            self.logger.warning(
                "No healthy-region rows found for residual validation; "
                "falling back to the full dataset."
            )
            healthy = residuals

        coverage = self.check_coverage(residuals, active_sensors)
        centring = self.check_centring(healthy, active_sensors)
        spread = self.check_spread(healthy, active_sensors)
        growth = self.check_growth(residuals)

        passed = (
            not coverage["missing_sensors"]
            and centring["mean_absolute_bias"] < 1.0
            and spread["exceedance_fraction"] < 0.05
        )

        report: Dict[str, Any] = {
            "passed": bool(passed),
            "healthy_rows_checked": int(len(healthy)),
            "total_rows": int(len(residuals)),
            "coverage": coverage,
            "centring": centring,
            "spread": spread,
            "growth": growth,
        }

        if not passed:
            self.logger.warning(
                "Residual validation FAILED: bias=%.4f, exceedance=%.4f%%, "
                "missing_sensors=%s. Downstream thresholds may be miscalibrated.",
                centring["mean_absolute_bias"],
                100.0 * spread["exceedance_fraction"],
                coverage["missing_sensors"],
            )
        else:
            self.logger.info(
                "Residual validation passed: bias=%.4f, exceedance=%.4f%%, "
                "%.0f%% of engines show rising residual trend.",
                centring["mean_absolute_bias"],
                100.0 * spread["exceedance_fraction"],
                100.0 * growth["fraction_rising"],
            )
        return report

    def run(self, residuals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:


        self.config.ensure_directories()
        source = residuals
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.residuals_csv),
                required_columns=get_identity_columns(),
                stage="validate_residuals",
            )
        report = self.validate(source)
        atomic_write_json(report, self.config.paths.metrics / "residual_validation.json")
        return report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    validator = ResidualValidator(config)
    report = validator.run()
    logger.info("Validation passed: %s", report["passed"])
    logger.info("Centring: %s", report["centring"]["mean_absolute_bias"])
    logger.info("Spread exceedance: %.4f%%", 100.0 * report["spread"]["exceedance_fraction"])


if __name__ == "__main__":
    main()
