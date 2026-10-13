"""
Shared contract for every Digital Twin variant.

A Digital Twin here answers one question: *given the operating context and how
far into its life this engine is, what should its sensors be reading if it were
healthy?*  The gap between that prediction and reality is the residual, and
every downstream layer -- health index, anomaly detection, root-cause
reasoning, explainability, uncertainty -- is built on it.

Two design decisions are load-bearing and are enforced here rather than left to
each variant:

**Twins train on the healthy region only.**  Every C-MAPSS training engine runs
to failure, so a twin fitted on the whole trajectory would learn *average*
behaviour including the degraded tail, and residuals would stay flat as the
engine dies.  The healthy region is defined once, in
:meth:`BaseTwin.select_healthy_region`: high remaining life *and* early in the
trajectory.  Twins then predict across all rows, so residuals grow as
degradation proceeds. That growth is the entire signal.

**The design matrix is context, never sensors.**  Inputs are the three
operating settings, engine age, and the one-hot operating mode.  Sensors appear
only as targets.  Feeding sensor values in as inputs would let the twin
reconstruct a degraded reading from its neighbours and hide the fault it is
supposed to expose.

After prediction, commissioning calibration can remove a unit's manufacturing
offset using only its first configured cycles. It excludes the current row and
freezes after commissioning, so it cannot track or reconstruct a later fault.

Classic C-MAPSS has no virtual sensors, so nothing here references ``X_v`` or
any other N-CMAPSS variable group.
"""

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

#: Maximum number of one-hot operating-mode columns. Six covers FD002/FD004.
MAX_MODE_COLUMNS = 8


class BaseTwin(ABC):
    """Abstract base for the multi-output expected-sensor regressors.

    Subclasses implement three hooks -- :meth:`_build_estimator`,
    :meth:`_fit_estimator` and :meth:`_predict_estimator` -- and inherit the
    design matrix, healthy-region selection, evaluation, persistence and
    registry integration.
    """

    #: Logical name used for artefacts, registry entries and ensemble weights.
    name: str = "base"

    #: File extension for the persisted model. Keras twins override this.
    artifact_suffix: str = ".pkl"

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the twin.

        Args:
            config: Configuration supplying hyper-parameters and paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(f"{__name__}.{self.name}", self.config)
        self.model: Any = None
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        self.n_modes: int = 1
        self.metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _build_estimator(self, n_outputs: int) -> Any:
        """Construct an unfitted estimator for this variant.

        Args:
            n_outputs: Number of sensor channels to predict.

        Returns:
            The unfitted model object.
        """

    @abstractmethod
    def _fit_estimator(
        self,
        estimator: Any,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> Any:
        """Fit the estimator.

        Args:
            estimator: The object returned by :meth:`_build_estimator`.
            x_train: Training design matrix.
            y_train: Training targets.
            x_val: Optional validation design matrix.
            y_val: Optional validation targets.

        Returns:
            The fitted estimator.
        """

    @abstractmethod
    def _predict_estimator(self, estimator: Any, x: np.ndarray) -> np.ndarray:
        """Predict expected sensor values.

        Args:
            estimator: The fitted model.
            x: Design matrix.

        Returns:
            An ``(n_rows, n_targets)`` prediction array.
        """

    # ------------------------------------------------------------------ #
    # Healthy-region selection
    # ------------------------------------------------------------------ #
    def select_healthy_region(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return the rows a twin is allowed to learn 'normal' from.

        A row qualifies when it has plenty of remaining life *and* sits early in
        its trajectory.  Requiring both guards against two different mistakes:
        the RUL test alone would admit long-lived engines' late cycles, and the
        cycle-fraction test alone would admit early cycles of an engine that was
        already unhealthy when observation began.

        Every engine contributes at least ``min_rows_per_engine`` rows so that
        an unusually short trajectory is not excluded entirely.

        Args:
            frame: Scaled data carrying RUL and identity keys.

        Returns:
            The healthy subset.

        Raises:
            AnomalyHealthError: If the selection is empty.
        """
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

        # Guarantee a minimum contribution from every engine.
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

    # ------------------------------------------------------------------ #
    # Design matrix
    # ------------------------------------------------------------------ #
    def build_design_matrix(
        self, frame: pd.DataFrame, n_modes: Optional[int] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Build the context-only input matrix.

        Columns, in fixed order: the three operating settings, raw cycle,
        normalised cycle, then one-hot operating-mode indicators.  The order is
        fixed so a saved model always meets the same layout at inference.

        Args:
            frame: Data carrying settings, cycle and operating mode.
            n_modes: Number of one-hot mode columns to emit.  Defaults to the
                count learned at training time.

        Returns:
            A ``(matrix, feature_names)`` tuple.

        Raises:
            AnomalyHealthError: If a required context column is absent.
        """
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
        if reference > 0:
            cycle_norm = np.minimum(frame["cycle"].to_numpy(float) / reference, 1.0)
        elif CYCLE_NORM_COLUMN in frame.columns:
            cycle_norm = frame[CYCLE_NORM_COLUMN].astype(float).to_numpy()
        else:
            max_cycle = frame.groupby("unit")["cycle"].transform("max").astype(float)
            cycle_norm = (
                frame["cycle"].astype(float) / max_cycle.replace(0.0, np.nan)
            ).fillna(0.0).to_numpy()
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
        """Apply a past-only commissioning offset, frozen after N cycles.

        Supply history from cycle 1. A partial trajectory starting later uses
        the population prediction because it has no valid commissioning prefix.
        Sorting internally makes the result independent of caller row order.
        """
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
        """Build the multi-output sensor target matrix.

        Args:
            frame: Scaled data carrying the active sensor columns.

        Returns:
            A ``(matrix, sensor_names)`` tuple.

        Raises:
            AnomalyHealthError: If no active sensor column is present.
        """
        sensors = [c for c in get_active_sensors(self.config) if c in frame.columns]
        if not sensors:
            raise AnomalyHealthError(
                "No active sensor columns found; cannot train a digital twin."
            )
        values = frame[sensors].astype(float).to_numpy()
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), sensors

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Fit the twin on the healthy region and score it on held-out engines.

        The validation split is engine-wise, so accuracy is measured on engines
        the twin has never seen rather than on later cycles of engines it has.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            A metrics dictionary including RMSE, MAE and R-squared.
        """
        started = time.perf_counter()
        mode_column = get_operating_mode_column()
        self.n_modes = (
            int(frame[mode_column].max()) + 1 if mode_column in frame.columns else 1
        )
        self.n_modes = max(min(self.n_modes, MAX_MODE_COLUMNS), 1)

        healthy = self.select_healthy_region(frame)
        train_frame, val_frame = engine_wise_split(
            healthy,
            unit_column="unit",
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
        """Compute multi-output regression metrics.

        ``r2`` is averaged over *informative* sensors only -- those whose
        target actually varies. Several C-MAPSS channels are constant within
        an operating mode, and R-squared is undefined on a constant target
        (its denominator is the target variance, which is zero). scikit-learn
        does not error on that case; it silently returns 1.0 when the model
        happens to reproduce the constant and 0.0 when it does not. Averaging
        those pseudo-scores in badly distorts the result: on the pooled
        FD001-4 fleet the four constant sensors handed the tree twins four
        free 1.0s, inflating their reported R-squared by roughly 4/21 ~ 0.19
        and reversing the true ranking against the Keras twin, which does not
        reproduce the constants exactly and was being penalised for it.

        ``rmse`` and ``mae`` stay pooled across all targets: unlike
        R-squared they are well defined on a constant target, so excluding
        those columns would change their meaning for no benefit.

        Args:
            actual: Ground-truth sensor matrix.
            predicted: Predicted sensor matrix.

        Returns:
            A mapping with ``rmse``, ``mae``, ``r2`` (informative targets
            only), ``r2_all_targets`` (the legacy figure, kept so older runs
            remain comparable) and ``n_informative_targets``.
        """
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
        """Break accuracy down by sensor channel.

        Args:
            actual: Ground-truth sensor matrix.
            predicted: Predicted sensor matrix.

        Returns:
            A mapping of sensor name to its own RMSE / MAE / R-squared.
        """
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

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #
    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict expected sensor values for every row.

        Args:
            frame: Scaled, context-labelled data.

        Returns:
            A frame of identity keys plus ``expected_sensor_*`` columns.

        Raises:
            AnomalyHealthError: If the twin has not been trained or loaded.
        """
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

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def artifact_path(self) -> Path:
        """Return where this twin's model file lives.

        Returns:
            The artefact path.
        """
        return self.config.paths.model_path(f"{self.name}_twin{self.artifact_suffix}")

    def metadata_path(self) -> Path:
        """Return where this twin's metadata document lives.

        Returns:
            The metadata path.
        """
        return self.config.paths.model_path(f"{self.name}_twin_meta.json")

    def save(self) -> Path:
        """Persist the fitted model and its column metadata.

        Args:
            None.

        Returns:
            The artefact path written.

        Raises:
            AnomalyHealthError: If nothing has been trained.
        """
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
        """Reload a previously persisted twin and its column metadata.

        Returns:
            This instance, ready to predict.

        Raises:
            AnomalyHealthError: If the metadata document is missing.
        """
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
        """Return this variant's configured hyper-parameters.

        Returns:
            The hyper-parameter mapping, or an empty dict when the variant has
            no configured block.
        """
        return dict(getattr(self.config.twin, self.name, {}) or {})

    # ------------------------------------------------------------------ #
    # Stage entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Train (or load) the twin and produce expected-sensor predictions.

        Args:
            frame: Optional in-memory input; otherwise ``scaled_features.csv``
                is read from disk.
            retrain: Fit a new model, or reload the saved one.

        Returns:
            A ``(predictions, metrics)`` tuple.
        """
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
