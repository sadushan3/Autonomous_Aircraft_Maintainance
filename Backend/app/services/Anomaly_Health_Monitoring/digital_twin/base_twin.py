from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import (
    get_active_sensors,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.feature_engineering import (
    CYCLE_NORM_COLUMN,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    get_expected_columns,
    get_identity_columns,
    get_op_setting_columns,
    get_operating_mode_column,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    ExperimentTracker,
    ModelRegistry,
    engine_wise_split,
    get_logger,
    load_model,
    read_csv,
    save_model,
)
from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, training_scope, record_training_scope


MAX_MODE_COLUMNS = 8


class BaseTwin(ABC):


    name: str = "base"


    artifact_suffix: str = ".pkl"

    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.{self.name}", self.config)
        self.model: Any = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        self.n_modes: int = 1
        self.metrics: Dict[str, Any] = {}


    @abstractmethod
    def _build_estimator(self, n_outputs: int) -> Any:
        pass


    @abstractmethod
    def _fit_estimator(
        self,
        estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Any:
        pass


    @abstractmethod
    def _predict_estimator(self, estimator: Any, x: np.ndarray) -> np.ndarray:
        pass


    def select_healthy_region(self, frame: pd.DataFrame) -> pd.DataFrame:


        settings = self.config.healthy_region
        result = frame.copy()

        if RUL_COLUMN in result.columns:
            rul_ok = result[RUL_COLUMN].astype(float) >= float(settings.rul_threshold)
        else:
            self.logger.warning(
                "No %s column; healthy-region selection falls back to the "
                "cycle-fraction rule alone.",
                RUL_COLUMN,
            )
            rul_ok = pd.Series(True, index=result.index)

        max_cycle = result.groupby("unit")["cycle"].transform("max").astype(float)
        cycle_ok = result["cycle"].astype(float) <= (
            float(settings.cycle_fraction) * max_cycle
        )

        healthy_mask = rul_ok & cycle_ok


        rank = result.groupby("unit")["cycle"].rank(method="first")
        floor_mask = rank <= float(settings.min_rows_per_engine)
        healthy_mask = healthy_mask | floor_mask

        healthy = result.loc[healthy_mask].copy()
        if healthy.empty:
            raise AnomalyHealthError(
                "Healthy-region selection produced no rows. Loosen "
                "healthy_region.rul_threshold or healthy_region.cycle_fraction."
            )

        self.logger.info(
            "Healthy region: %d of %d rows (%.1f%%) across %d engine(s) "
            "[RUL >= %d and cycle <= %.0f%% of life]",
            len(healthy),
            len(result),
            100.0 * len(healthy) / max(len(result), 1),
            healthy["unit"].nunique(),
            settings.rul_threshold,
            100.0 * settings.cycle_fraction,
        )
        return healthy


    def build_design_matrix(
        self, frame: pd.DataFrame, n_modes: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:


        mode_column = get_operating_mode_column()
        settings = get_op_setting_columns()

        missing = [c for c in settings if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(f"Digital twin input missing settings: {missing}")
        if mode_column not in frame.columns:
            raise AnomalyHealthError(
                f"'{mode_column}' is missing. The digital twin needs the "
                f"operating context; run the context-modelling stage first."
            )

        blocks: List[np.ndarray] = [frame[settings].astype(float).to_numpy()]
        names: List[str] = list(settings)

        blocks.append(frame["cycle"].astype(float).to_numpy().reshape(-1, 1))
        names.append("cycle")

        reference = float(self.config.twin.age_reference_cycles)
        if not np.isfinite(reference) or reference <= 0:
            raise AnomalyHealthError("Twin prediction requires a positive fixed age scale.")
        cycle_norm = np.minimum(frame["cycle"].to_numpy(float) / reference, 1.0)
        blocks.append(cycle_norm.reshape(-1, 1))
        names.append(CYCLE_NORM_COLUMN)

        mode_count = int(n_modes if n_modes is not None else self.n_modes)
        mode_count = max(min(mode_count, MAX_MODE_COLUMNS), 1)
        modes = frame[mode_column].astype(int).to_numpy()
        one_hot = np.zeros((len(frame), mode_count), dtype=float)
        valid = (modes >= 0) & (modes < mode_count)
        one_hot[np.arange(len(frame))[valid], modes[valid]] = 1.0
        blocks.append(one_hot)
        names.extend([f"mode_{i}" for i in range(mode_count)])

        matrix = np.hstack(blocks).astype(float)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return matrix, names

    def calibrate_predictions(self, frame: pd.DataFrame, predictions: np.ndarray) -> np.ndarray:


        limit = int(self.config.twin.commissioning_cycles)
        if limit <= 0 or frame.empty:
            return predictions
        if any(column not in frame for column in self.target_names):
            return predictions
        order = np.lexsort((frame["cycle"].to_numpy(), frame["unit"].to_numpy()))
        ordered = frame.iloc[order].reset_index(drop=True)
        residual = ordered[self.target_names].to_numpy(float) - predictions[order]
        starts_at_one = ordered.groupby("unit", sort=False)["cycle"].transform("min").eq(1)
        commissioning = ordered["cycle"].between(1, limit) & starts_at_one
        observed = pd.DataFrame(residual).mul(commissioning.to_numpy(), axis=0)
        past_sum = observed.groupby(ordered["unit"], sort=False).cumsum() - observed
        past_count = commissioning.astype(int).groupby(ordered["unit"], sort=False).cumsum()
        past_count -= commissioning.astype(int)
        offset = past_sum.to_numpy() / np.maximum(past_count.to_numpy()[:, None], 1)
        calibrated = np.asarray(predictions, dtype=float).copy()
        calibrated[order] += offset
        return calibrated

    def build_targets(self, frame: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:


        sensors = [c for c in get_active_sensors(self.config) if c in frame.columns]
        if not sensors:
            raise AnomalyHealthError(
                "No active sensor columns found; cannot train a digital twin."
            )
        values = frame[sensors].astype(float).to_numpy()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), sensors


    def train(self, frame: pd.DataFrame) -> Dict[str, Any]:


        started = time.perf_counter()
        mode_column = get_operating_mode_column()
        self.n_modes = (
            int(frame[mode_column].max()) + 1 if mode_column in frame.columns else 1
        )
        self.n_modes = max(min(self.n_modes, MAX_MODE_COLUMNS), 1)

        plan = load_training_split(self.config)
        if plan:
            fitting = training_scope(frame, self.config, required=True)
            validation = training_scope(frame, self.config, ("selection",), required=True)
            train_frame = self.select_healthy_region(fitting)
            val_frame = self.select_healthy_region(validation)
            record_training_scope("twin_" + self.name, fitting, self.config)
        else:

            healthy = self.select_healthy_region(training_scope(frame, self.config))
            train_frame, val_frame = engine_wise_split(
                healthy, unit_column="unit",
                validation_fraction=self.config.data.validation_engine_fraction,
                random_state=self.config.data.random_state,
            )
        if val_frame.empty:
            self.logger.warning(
                "Validation split is empty (too few engines); scoring on the "
                "training rows instead."
            )
            val_frame = train_frame

        x_train, self.feature_names = self.build_design_matrix(train_frame)
        y_train, self.target_names = self.build_targets(train_frame)
        x_val, _ = self.build_design_matrix(val_frame)
        y_val, _ = self.build_targets(val_frame)

        self.logger.info(
            "Training %s twin: X%s -> Y%s", self.name, x_train.shape, y_train.shape
        )

        estimator = self._build_estimator(n_outputs=y_train.shape[1])
        self.model = self._fit_estimator(estimator, x_train, y_train, x_val, y_val)

        predictions = self._predict_estimator(self.model, x_val)
        population_metrics = self._score(y_val, predictions)
        scored_rows = len(y_val)
        self.metrics = dict(population_metrics)
        if self.config.twin.commissioning_cycles > 0:
            context = frame.loc[frame["unit"].isin(val_frame["unit"].unique())]
            context_x, _ = self.build_design_matrix(context)
            calibrated = self.calibrate_predictions(
                context, self._predict_estimator(self.model, context_x)
            )
            eligible = context.index.isin(val_frame.index) & (
                context["cycle"].to_numpy() > self.config.twin.commissioning_cycles
            )
            if eligible.any():
                actual = context.loc[eligible, self.target_names].to_numpy(float)
                self.metrics = self._score(actual, calibrated[eligible])
                scored_rows = int(eligible.sum())
                self.metrics["population_validation_metrics"] = population_metrics
                self.metrics["commissioning_cycles"] = int(self.config.twin.commissioning_cycles)
        self.metrics.update(
            {
                "name": self.name,
                "train_rows": int(x_train.shape[0]),
                "val_rows": scored_rows,
                "n_features": int(x_train.shape[1]),
                "n_targets": int(y_train.shape[1]),
                "n_modes": self.n_modes,
                "fit_seconds": round(time.perf_counter() - started, 3),
                "age_reference_cycles": float(self.config.twin.age_reference_cycles),
                "training_plan_id": plan["plan_id"] if plan else None,
                "validation_role": "selection" if plan else "inner_development",
            }
        )
        self.logger.info(
            "%s twin trained: RMSE=%.5f MAE=%.5f R2=%.4f (%.1fs)",
            self.name,
            self.metrics["rmse"],
            self.metrics["mae"],
            self.metrics["r2"],
            self.metrics["fit_seconds"],
        )
        return self.metrics

    @staticmethod
    def _score(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:


        actual = np.nan_to_num(np.asarray(actual, dtype=float))
        predicted = np.nan_to_num(np.asarray(predicted, dtype=float))
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))

        try:
            r2_all = float(r2_score(actual, predicted, multioutput="uniform_average"))
        except ValueError:
            r2_all = 0.0

        informative = actual.var(axis=0) > 1e-8
        if informative.any():
            try:
                r2 = float(
                    r2_score(
                        actual[:, informative],
                        predicted[:, informative],
                        multioutput="uniform_average",
                    )
                )
            except ValueError:
                r2 = r2_all
        else:
            r2 = r2_all

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "r2_all_targets": r2_all,
            "n_informative_targets": int(informative.sum()),
        }

    def per_sensor_metrics(
        self, actual: np.ndarray, predicted: np.ndarray
    ) -> Dict[str, Dict[str, float]]:


        results: Dict[str, Dict[str, float]] = {}
        for index, sensor in enumerate(self.target_names):
            column_actual = actual[:, index]
            column_pred = predicted[:, index]
            try:
                r2 = float(r2_score(column_actual, column_pred))
            except ValueError:
                r2 = 0.0
            results[sensor] = {
                "rmse": float(np.sqrt(mean_squared_error(column_actual, column_pred))),
                "mae": float(mean_absolute_error(column_actual, column_pred)),
                "r2": r2,
            }
        return results


    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:


        if self.model is None:
            raise AnomalyHealthError(f"{self.name} twin used before train/load.")

        matrix, _ = self.build_design_matrix(frame, n_modes=self.n_modes)
        predictions = self._predict_estimator(self.model, matrix)
        predictions = np.nan_to_num(np.asarray(predictions, dtype=float))
        predictions = self.calibrate_predictions(frame, predictions)

        expected_names = get_expected_columns(self.target_names)
        result = frame[get_identity_columns()].copy().reset_index(drop=True)
        for index, column in enumerate(expected_names):
            result[column] = predictions[:, index]
        return result


    def artifact_path(self) -> Path:


        return self.config.paths.model_path(f"{self.name}_twin{self.artifact_suffix}")

    def metadata_path(self) -> Path:


        return self.config.paths.model_path(f"{self.name}_twin_meta.json")

    def save(self) -> Path:


        if self.model is None:
            raise AnomalyHealthError(f"Nothing to save; train the {self.name} twin first.")

        from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json

        path = save_model(self.model, self.artifact_path(), config=self.config)
        atomic_write_json(
            {
                "name": self.name,
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "n_modes": self.n_modes,
                "metrics": self.metrics,
            },
            self.metadata_path(),
        )
        ModelRegistry(self.config).register(
            name=f"{self.name}_twin",
            artifact_path=path,
            metrics=self.metrics,
            params=self._hyperparameters(),
            extra={
                "feature_names": self.feature_names,
                "target_names": self.target_names,
                "n_modes": self.n_modes,
            },
        )
        return path

    def load(self) -> "BaseTwin":


        from app.utils.Anomaly_Health_Monitoring.utils import read_json

        meta = read_json(self.metadata_path(), default=None)
        if not meta:
            raise AnomalyHealthError(
                f"No metadata for the {self.name} twin. Train it before inference."
            )
        self.model = load_model(self.artifact_path())
        self.feature_names = list(meta.get("feature_names", []))
        self.target_names = list(meta.get("target_names", []))
        self.n_modes = int(meta.get("n_modes", 1))
        self.metrics = meta.get("metrics", {})
        self.logger.info(
            "Loaded %s twin (%d features -> %d sensors)",
            self.name,
            len(self.feature_names),
            len(self.target_names),
        )
        return self

    def _hyperparameters(self) -> Dict[str, Any]:


        return dict(getattr(self.config.twin, self.name, {}) or {})


    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(
                    self.config.paths.scaled_features_csv
                ),
                required_columns=get_identity_columns(),
                stage=f"{self.name}_twin",
            )

        tracker = ExperimentTracker(self.config)
        if retrain:
            with tracker.run(f"{self.name}_twin") as run:
                run.log_params(self._hyperparameters())
                metrics = self.train(source)
                run.log_metrics(
                    {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                )
            self.save()
        else:
            self.load()
            metrics = dict(self.metrics)

        predictions = self.predict(source)
        return predictions, metrics
