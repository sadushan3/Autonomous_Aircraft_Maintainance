from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_expected_columns,
    get_identity_columns,
    get_residual_columns,
    get_residual_z_columns,
    get_twin_spread_columns,
    present_expected_columns,
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


RESIDUAL_ABS_MEAN_COLUMN = "residual_abs_mean"


RESIDUAL_ABS_Z_MEAN_COLUMN = "residual_abs_z_mean"


Z_CALIBRATION_FILE = "residual_z_calibration.json"


class ResidualCalculator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.sensor_columns: List[str] = []
        self._z_reference: Optional[Dict[str, Tuple[float, float]]] = None


    def compute(
        self,
        scaled: pd.DataFrame,
        predictions: pd.DataFrame,
        recalibrate: bool = False,
    ) -> pd.DataFrame:


        merged = merge_on_keys(
            scaled, predictions, keys=get_identity_columns(), how="inner"
        )

        expected_columns = present_expected_columns(merged)
        self.sensor_columns = [sensor_from_residual(c) for c in expected_columns]
        self.sensor_columns = [s for s in self.sensor_columns if s in merged.columns]
        if not self.sensor_columns:
            raise AnomalyHealthError(
                "No sensor could be matched between actual and expected columns; "
                "check that the twin stage ran on the same active-sensor set."
            )

        result = merged[get_identity_columns()].copy()
        for metadata in ("RUL", "split", "subset_id"):
            if metadata in merged:
                result[metadata] = merged[metadata].to_numpy()

        residual_columns = get_residual_columns(self.sensor_columns)
        expected_matched = get_expected_columns(self.sensor_columns)
        actual = merged[self.sensor_columns].astype(float).to_numpy()
        expected = merged[expected_matched].astype(float).to_numpy()
        reference_columns = ["reference_expected_" + s for s in self.sensor_columns]
        if any(c in merged for c in reference_columns):
            if not all(c in merged for c in reference_columns):
                raise AnomalyHealthError("Monitoring reference must cover every predicted sensor.")
            prediction_residuals = actual - expected
            for index, sensor in enumerate(self.sensor_columns):
                result[f"prediction_residual_{sensor}"] = prediction_residuals[:, index]
            expected = merged[reference_columns].astype(float).to_numpy()
            result["residual_reference"] = "context_ensemble"
        residuals = actual - expected

        for index, column in enumerate(residual_columns):
            result[column] = residuals[:, index]


        if self.config.twin.commissioning_cycles <= 0:


            result["calibration_ready"] = True
        else:
            limit = int(self.config.twin.commissioning_cycles)
            counts = scaled["cycle"].between(1, limit).groupby(scaled["unit"]).sum()
            ready = scaled["unit"].map(counts).ge(limit) & scaled["cycle"].gt(limit)
            readiness = scaled[get_identity_columns()].copy()
            readiness["calibration_ready"] = ready.to_numpy()
            result = result.merge(readiness, on=get_identity_columns(), how="left", validate="one_to_one")

        z_scores = self._zscore(result, residual_columns, recalibrate=recalibrate)
        z_columns = get_residual_z_columns(self.sensor_columns)
        for original, z_name in zip(residual_columns, z_columns):
            result[z_name] = z_scores[original].to_numpy()

        result[RESIDUAL_ABS_MEAN_COLUMN] = np.abs(residuals).mean(axis=1)
        result[RESIDUAL_ABS_Z_MEAN_COLUMN] = (
            result[z_columns].abs().mean(axis=1).to_numpy()
        )

        spread_columns = get_twin_spread_columns(self.sensor_columns)
        for column in spread_columns:
            if column in merged.columns:
                result[column] = merged[column].to_numpy()
        if "twin_spread_mean" in merged.columns:
            result["twin_spread_mean"] = merged["twin_spread_mean"].to_numpy()

        self.logger.info(
            "Computed residuals for %d sensor(s) over %d row(s); mean |residual|=%.5f",
            len(self.sensor_columns),
            len(result),
            float(result[RESIDUAL_ABS_MEAN_COLUMN].mean()),
        )
        return result

    @property
    def z_calibration_path(self) -> Path:

        return self.config.paths.models / Z_CALIBRATION_FILE

    def fit_z_calibration(
        self, result: pd.DataFrame, residual_columns: List[str]
    ) -> Dict[str, Any]:


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope
        result = training_scope(result, self.config)
        record_training_scope("residual_standardization", result, self.config)
        values = result.loc[:, residual_columns].astype(float)
        median = values.median()
        mad = (values - median).abs().median()
        scale = (mad * 1.4826).replace(0.0, np.nan)
        scale = scale.fillna(values.std().replace(0.0, 1.0)).replace(0.0, 1.0)

        payload: Dict[str, Any] = {
            "median": {c: float(median[c]) for c in residual_columns},
            "scale": {c: float(scale[c]) for c in residual_columns},
            "n_rows": int(len(result)),
            "n_engines": int(result["unit"].nunique()),
        }
        atomic_write_json(payload, self.z_calibration_path)
        self.logger.info(
            "Fitted residual z calibration on %d row(s) / %d engine(s); "
            "median scale=%.5f",
            payload["n_rows"],
            payload["n_engines"],
            float(np.median(list(payload["scale"].values()))),
        )
        return payload

    def load_z_calibration(
        self, residual_columns: List[str]
    ) -> Optional[Dict[str, Any]]:


        payload = read_json(self.z_calibration_path, default=None)
        if not payload or "median" not in payload or "scale" not in payload:
            return None
        missing = [c for c in residual_columns if c not in payload["median"]
                   or c not in payload["scale"]]
        if missing:
            self.logger.warning(
                "Residual z calibration is missing %d column(s) (e.g. %s); "
                "falling back to batch-relative z-scores.",
                len(missing),
                missing[:3],
            )
            return None
        return payload

    def _zscore(
        self,
        result: pd.DataFrame,
        residual_columns: List[str],
        recalibrate: bool = False,
    ) -> pd.DataFrame:


        calibration = (
            self.fit_z_calibration(result, residual_columns)
            if recalibrate
            else self.load_z_calibration(residual_columns)
        )
        if calibration is None:
            raise AnomalyHealthError("No frozen residual z calibration; fit it on training engines before inference.")

        values = result.loc[:, residual_columns].astype(float)
        median = pd.Series(calibration["median"], dtype=float).reindex(residual_columns)
        scale = pd.Series(calibration["scale"], dtype=float).reindex(residual_columns)
        scale = scale.replace(0.0, 1.0).fillna(1.0)
        return (values - median) / scale


    def top_contributors(
        self, row: pd.Series, top_k: int = 5
    ) -> List[Tuple[str, float]]:


        z_columns = get_residual_z_columns(self.sensor_columns)
        pairs = [
            (sensor_from_residual(col), float(row.get(col, 0.0))) for col in z_columns
        ]
        pairs.sort(key=lambda item: abs(item[1]), reverse=True)
        return pairs[: max(top_k, 0)]


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.residuals_csv)
        verify_identity_columns(frame, context="residuals output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.residuals_csv)
        return read_csv(
            target, required_columns=get_identity_columns(), stage="residual_calculator"
        )

    def run(
        self,
        scaled: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.DataFrame] = None,
        recalibrate: bool = False,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        scaled_source = scaled
        if scaled_source is None:
            scaled_source = read_csv(
                self.config.paths.processed_path(self.config.paths.scaled_features_csv),
                required_columns=get_identity_columns(),
                stage="residual_calculator",
            )
        prediction_source = predictions
        if prediction_source is None:
            prediction_source = read_csv(
                self.config.paths.processed_path(self.config.paths.twin_predictions_csv),
                required_columns=get_identity_columns(),
                stage="residual_calculator",
            )

        residuals = self.compute(
            scaled_source, prediction_source, recalibrate=recalibrate
        )
        output = self.save(residuals)

        report: Dict[str, Any] = {
            "rows": int(len(residuals)),
            "engines": int(residuals["unit"].nunique()),
            "sensors": self.sensor_columns,
            "mean_abs_residual": float(residuals[RESIDUAL_ABS_MEAN_COLUMN].mean()),
            "max_abs_residual": float(residuals[RESIDUAL_ABS_MEAN_COLUMN].max()),
            "output_path": str(output),
        }
        self.logger.info("Residual calculation report: %s", {k: v for k, v in report.items() if k != "sensors"})
        return residuals, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    calculator = ResidualCalculator(config)
    residuals, report = calculator.run()
    logger.info("residuals.csv written: %d rows, %d sensors", report["rows"], len(report["sensors"]))
    print(residuals.iloc[:5, :10].to_string())


if __name__ == "__main__":
    main()
