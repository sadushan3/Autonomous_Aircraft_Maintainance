from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    detect_constant_sensors,
    get_feature_columns,
    get_identity_columns,
    get_op_setting_columns,
    get_sensor_columns,
    resolve_active_sensors,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
)
from app.utils.Anomaly_Health_Monitoring.training_split import prepare_training_split, training_scope, record_training_scope


DATA_QUALITY_COLUMN = "data_quality_score"


IMPUTED_COUNT_COLUMN = "imputed_value_count"


class DataCleaner:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.dropped_sensors: List[str] = []
        self.active_sensors: List[str] = get_sensor_columns()


    def deduplicate(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, int]:


        keys = get_identity_columns()
        before = len(frame)
        result = frame.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
        removed = before - len(result)
        if removed:
            self.logger.warning("Removed %d duplicate (unit, cycle) row(s)", removed)
        return result, removed

    def find_constant_sensors(self, frame: pd.DataFrame) -> List[str]:


        return detect_constant_sensors(
            frame,
            std_epsilon=self.config.data.constant_sensor_std_eps,
            sensors=get_sensor_columns(),
        )

    def repair_values(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:


        result = frame.copy()
        feature_columns = [c for c in get_feature_columns() if c in result.columns]

        non_finite_mask = ~np.isfinite(result[feature_columns].to_numpy(dtype=float))
        n_non_finite = int(non_finite_mask.sum())
        if n_non_finite:
            self.logger.warning("Replacing %d non-finite feature value(s)", n_non_finite)
            result[feature_columns] = result[feature_columns].replace(
                [np.inf, -np.inf], np.nan
            )

        missing_before = result[feature_columns].isna()
        n_missing = int(missing_before.to_numpy().sum())


        result[IMPUTED_COUNT_COLUMN] = missing_before.sum(axis=1).astype(int)

        if n_missing:
            self.logger.warning(
                "Imputing %d missing feature value(s) within engine trajectories",
                n_missing,
            )
            ordered = result.sort_values(["unit", "cycle"])
            filled = ordered.groupby("unit", sort=False)[feature_columns].ffill()
            result.loc[ordered.index, feature_columns] = filled.to_numpy()
            still_missing = result[feature_columns].isna().to_numpy().sum()
            if still_missing:
                payload = read_json(self.config.paths.models / "imputation_reference.json", default={})
                medians = pd.Series(payload.get("medians", {}), dtype=float)
                result[feature_columns] = result[feature_columns].fillna(medians)
                if result[feature_columns].isna().any().any():
                    from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError
                    raise AnomalyHealthError("Missing leading sensor evidence and no frozen training imputation reference.")

        return result, {"non_finite": n_non_finite, "missing": n_missing}

    def apply_constant_sensor_policy(
        self, frame: pd.DataFrame, constant_sensors: List[str]
    ) -> pd.DataFrame:


        if self.config.data.drop_constant_sensors and constant_sensors:
            self.dropped_sensors = list(constant_sensors)
            result = frame.drop(columns=constant_sensors, errors="ignore")
            self.logger.info(
                "Dropped %d constant sensor(s): %s",
                len(constant_sensors),
                constant_sensors,
            )
        else:
            self.dropped_sensors = []
            result = frame
            if constant_sensors:
                self.logger.info(
                    "Retaining %d constant sensor(s) because drop_constant_sensors "
                    "is disabled: %s",
                    len(constant_sensors),
                    constant_sensors,
                )

        self.active_sensors = resolve_active_sensors(self.dropped_sensors)
        self._persist_sensor_policy(constant_sensors)
        return result

    def apply_persisted_sensor_policy(self, frame: pd.DataFrame) -> pd.DataFrame:


        active = self.load_sensor_policy()
        all_sensors = get_sensor_columns()
        self.dropped_sensors = [s for s in all_sensors if s not in active]
        self.active_sensors = active
        result = frame.drop(columns=self.dropped_sensors, errors="ignore")
        if self.dropped_sensors:
            self.logger.info(
                "Applied persisted sensor policy (inference): dropped %d "
                "sensor(s) fit at training time: %s",
                len(self.dropped_sensors),
                self.dropped_sensors,
            )
        return result

    def _persist_sensor_policy(self, constant_sensors: List[str]) -> Path:


        payload = {
            "detected_constant_sensors": constant_sensors,
            "dropped_sensors": self.dropped_sensors,
            "active_sensors": self.active_sensors,
            "drop_policy_enabled": bool(self.config.data.drop_constant_sensors),
            "std_epsilon": self.config.data.constant_sensor_std_eps,
        }
        return atomic_write_json(
            payload, self.config.paths.models / self.config.paths.dropped_sensors_json
        )

    def score_quality(self, frame: pd.DataFrame) -> pd.DataFrame:


        result = frame.copy()
        n_features = max(
            len([c for c in get_feature_columns() if c in result.columns]), 1
        )
        imputed = result.get(
            IMPUTED_COUNT_COLUMN, pd.Series(np.zeros(len(result)), index=result.index)
        )
        penalty = (imputed.astype(float) / float(n_features)).clip(0.0, 1.0)
        result[DATA_QUALITY_COLUMN] = (1.0 - penalty).clip(0.0, 1.0)
        return result


    def clean(
        self, frame: pd.DataFrame, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        rows_in = len(frame)
        result = frame.sort_values(get_identity_columns()).reset_index(drop=True)

        result, duplicates_removed = self.deduplicate(result)
        if fit_policy:
            prepare_training_split(result, self.config)
            fitting = training_scope(result, self.config, required=True)
            columns = [c for c in get_feature_columns() if c in fitting]
            medians = fitting[columns].replace([np.inf, -np.inf], np.nan).median()
            if not np.isfinite(medians.to_numpy()).all():
                from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError
                raise AnomalyHealthError("Training has a sensor with no finite imputation evidence.")
            atomic_write_json({"medians": medians.to_dict(), "units": sorted(map(int, fitting.unit.unique()))},
                              self.config.paths.models / "imputation_reference.json")
            record_training_scope("cleaning", fitting, self.config)
        result, repair_counts = self.repair_values(result)
        if fit_policy:
            constant_sensors = self.find_constant_sensors(training_scope(result, self.config, required=True))
            result = self.apply_constant_sensor_policy(result, constant_sensors)
        else:
            constant_sensors = self.find_constant_sensors(result)
            result = self.apply_persisted_sensor_policy(result)
        result = self.score_quality(result)

        result = self._clip_rul(result)

        report: Dict[str, Any] = {
            "rows_in": int(rows_in),
            "rows_out": int(len(result)),
            "duplicate_rows_removed": int(duplicates_removed),
            "missing_values_imputed": int(repair_counts["missing"]),
            "non_finite_values_fixed": int(repair_counts["non_finite"]),
            "constant_sensors_detected": constant_sensors,
            "constant_sensors_dropped": self.dropped_sensors,
            "active_sensors": self.active_sensors,
            "mean_data_quality_score": float(result[DATA_QUALITY_COLUMN].mean()),
            "engines": int(result["unit"].nunique()),
        }
        self.logger.info("Cleaning report: %s", report)
        return result, report

    def _clip_rul(self, frame: pd.DataFrame) -> pd.DataFrame:


        if RUL_COLUMN not in frame.columns:
            return frame
        result = frame.copy()
        cap = int(self.config.data.rul_cap)
        result["RUL_true"] = pd.to_numeric(result[RUL_COLUMN], errors="coerce")
        result[RUL_COLUMN] = result["RUL_true"].clip(upper=cap)
        self.logger.info("Clipped %s at %d cycles (RUL_true retains the raw value)", RUL_COLUMN, cap)
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.cleaned_data_csv
        )
        verify_identity_columns(frame, context="cleaned_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(
            self.config.paths.cleaned_data_csv
        )
        return read_csv(target, required_columns=get_identity_columns(), stage="cleaner")

    def load_sensor_policy(self) -> List[str]:


        payload = read_json(
            self.config.paths.models / self.config.paths.dropped_sensors_json,
            default=None,
        )
        if not payload:
            return get_sensor_columns()
        return resolve_active_sensors(payload.get("dropped_sensors", []))

    def run(
        self, frame: Optional[pd.DataFrame] = None, fit_policy: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.raw_data_csv),
                required_columns=get_identity_columns(),
                stage="cleaner",
            )
        cleaned, report = self.clean(source, fit_policy=fit_policy)
        report["output_path"] = str(self.save(cleaned))
        return cleaned, report


def get_active_sensors(config: Optional[Config] = None) -> List[str]:


    return DataCleaner(config).load_sensor_policy()


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    cleaner = DataCleaner(config)
    cleaned, report = cleaner.run()
    logger.info("cleaned_data.csv written: %d rows", report["rows_out"])
    logger.info("Active sensors (%d): %s", len(report["active_sensors"]), report["active_sensors"])
    print(cleaned.head(8).to_string())


if __name__ == "__main__":
    main()
