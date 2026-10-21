from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.digital_twin.base_twin import BaseTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.lightgbm_twin import LightGBMTwin
from app.services.Anomaly_Health_Monitoring.digital_twin.random_forest_twin import (
    RandomForestTwin,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.tensorflow_twin import (
    TensorFlowTwin,
)
from app.services.Anomaly_Health_Monitoring.digital_twin.xgboost_twin import XGBoostTwin
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_expected_columns,
    get_identity_columns,
    get_twin_spread_columns,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    read_csv,
    read_json,
    load_model,
    save_model,
    engine_wise_split,
)


TWIN_SPREAD_MEAN_COLUMN = "twin_spread_mean"
STARTUP_PROFILE_ARTIFACT = "startup_profile_twin.pkl"


TWIN_REGISTRY: Dict[str, type] = {
    "random_forest": RandomForestTwin,
    "xgboost": XGBoostTwin,
    "lightgbm": LightGBMTwin,
    "tensorflow": TensorFlowTwin,
}


class EnsembleTwin:


    def __init__(
        self,
        config: Optional[Config] = None,
        twins: Optional[List[BaseTwin]] = None,
    ) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.twins: List[BaseTwin] = twins if twins is not None else self._build_twins()
        self.weights: Dict[str, float] = {}
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.target_names: List[str] = []
        self.startup_profile: Optional[Dict[str, Any]] = None

    @staticmethod
    def _profile_context(frame: pd.DataFrame, bundle: Dict[str, Any], age: float = 0) -> np.ndarray:

        modes = frame["operating_mode"].to_numpy(int)
        one_hot = np.zeros((len(frame), int(bundle["n_modes"])))
        valid = (modes >= 0) & (modes < one_hot.shape[1])
        one_hot[np.arange(len(frame))[valid], modes[valid]] = 1.
        subsets = np.column_stack([(frame["subset_id"] == name).to_numpy(float)
                                   for name in bundle["subsets"]])
        groups = (subsets[:, :, None] * one_hot[:, None, :]).reshape(len(frame), -1)
        cycle = np.minimum(frame["cycle"].to_numpy(float), age) / age if age > 0 else np.zeros(len(frame))
        return np.column_stack([frame[[f"op_setting_{i}" for i in (1, 2, 3)]].to_numpy(float),
                                cycle, one_hot, subsets, groups])

    @classmethod
    def _profile_features(cls, frame: pd.DataFrame, population: np.ndarray,
                          bundle: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:


        limit = int(bundle["commissioning_cycles"])
        startup = frame["cycle"].between(1, limit)
        values = pd.DataFrame(frame[bundle["target_names"]].to_numpy(float) - population)
        values["unit"] = frame["unit"].to_numpy()
        values["operating_mode"] = frame["operating_mode"].to_numpy()
        values["cycle"] = frame["cycle"].to_numpy()
        values = values.loc[startup.to_numpy()]
        targets = list(range(population.shape[1]))
        global_means = values.groupby("unit")[targets].mean()
        global_profile = global_means.reindex(frame["unit"]).fillna(0.).to_numpy()
        grouped = values.groupby(["unit", "operating_mode"])[targets]
        mode_means = grouped.mean()
        keys = pd.MultiIndex.from_frame(frame[["unit", "operating_mode"]])
        mode_profile = mode_means.reindex(keys).fillna(0.).to_numpy()
        counts = grouped.size().reindex(keys).fillna(0.).to_numpy()[:, None]
        mode_profile *= counts / (counts + float(bundle["mode_prior"]))
        context = cls._profile_context(frame, bundle, float(bundle["age_reference_cycles"]))
        features = np.column_stack([context, global_profile, mode_profile, context[:, 3:4] * global_profile])
        distinct = values.groupby("unit")["cycle"].nunique()
        complete = frame["unit"].map(distinct).fillna(0).eq(limit)
        ready = (complete & frame["cycle"].gt(limit)).to_numpy()
        schema = bundle.get("feature_schema")
        if schema is not None:
            if schema == "causal_trend_history_v1":
                from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import trend_features
                return trend_features(frame, population, bundle, base_features=features, ready=ready)
            if schema == "causal_fast_modes_v1":
                from app.services.Anomaly_Health_Monitoring.digital_twin.fast_history import fast_mode_features
                return fast_mode_features(frame, population, bundle, base_features=features, ready=ready)
            if schema in ("causal_sensor_history_v1", "causal_multimode_history_v1"):
                from app.services.Anomaly_Health_Monitoring.digital_twin.causal_forecast import prior_sensor_features
                return prior_sensor_features(frame, population, bundle, base_features=features, ready=ready,
                    multimode=schema == "causal_multimode_history_v1")
            if schema != "commissioning_statistics_v1":
                raise AnomalyHealthError("Unsupported commissioning feature schema")
            from app.services.Anomaly_Health_Monitoring.digital_twin.profile_models import commissioning_statistics
            return commissioning_statistics(frame, population, bundle, base_features=features, ready=ready)
        return features, ready

    def _profile_prediction(self, frame: pd.DataFrame, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        ready = np.zeros(len(frame), dtype=bool)
        bundle = self.startup_profile
        if not self.config.twin.startup_profile_enabled or not bundle:
            return reference, ready
        required = ["unit", "cycle", "operating_mode", "subset_id"] + [f"op_setting_{i}" for i in (1, 2, 3)]
        if any(name not in frame for name in required + bundle["target_names"]):
            self.logger.warning("Startup-profile context/history is incomplete; retaining the context twin.")
            return reference, ready
        if list(bundle["target_names"]) != self.target_names:
            self.logger.warning("Startup-profile targets differ from the active twins; retaining the context twin.")
            return reference, ready
        if int(bundle["commissioning_cycles"]) != int(self.config.twin.commissioning_cycles):
            self.logger.warning("Startup-profile commissioning policy differs; retaining the context twin.")
            return reference, ready
        population = bundle["population"].predict(self._profile_context(frame, bundle))
        if bundle.get("feature_schema") == "causal_trend_history_v1" and frame.unit.nunique() > 24:
            from app.services.Anomaly_Health_Monitoring.digital_twin.trend_forecast import engine_batches
            result = reference.copy()
            for rows in engine_batches(frame):
                batch = frame.iloc[rows].reset_index(drop=True)
                features, batch_ready = self._profile_features(batch, population[rows], bundle)
                batch_ready &= batch["subset_id"].isin(bundle["subsets"]).to_numpy()
                batch_ready &= batch["operating_mode"].between(0, int(bundle["n_modes"]) - 1).to_numpy()
                ready[rows] = batch_ready
                if batch_ready.any():
                    selected = rows[batch_ready]
                    enhanced = population[selected] + self._predict_profile_correction(
                        bundle["correction"], features, row_mask=batch_ready)
                    if not np.isfinite(enhanced).all():
                        raise AnomalyHealthError("Startup-profile twin produced non-finite predictions.")
                    result[selected] = enhanced
            return result, ready
        features, ready = self._profile_features(frame, population, bundle)
        ready &= frame["subset_id"].isin(bundle["subsets"]).to_numpy()
        ready &= frame["operating_mode"].between(0, int(bundle["n_modes"]) - 1).to_numpy()
        result = reference.copy()
        if ready.any():
            enhanced = population[ready] + self._predict_profile_correction(bundle["correction"], features, row_mask=ready)
            if not np.isfinite(enhanced).all():
                raise AnomalyHealthError("Startup-profile twin produced non-finite predictions.")
            result[ready] = enhanced
        return result, ready

    @staticmethod
    def _predict_profile_correction(model: Any, features: np.ndarray,
                                    row_mask: Optional[np.ndarray] = None) -> np.ndarray:

        if row_mask is None:
            batches = (features[start:start + 1024] for start in range(0,len(features),1024))
        else:
            mask = np.asarray(row_mask,dtype=bool)
            if mask.shape != (len(features),):
                raise ValueError("Forecast row mask must match the feature matrix")
            rows = np.flatnonzero(mask)
            batches = (features[rows[start:start + 1024]] for start in range(0,len(rows),1024))
        return np.concatenate([model.predict(batch) for batch in batches],axis=0)

    def _fit_startup_profile(self, frame: pd.DataFrame) -> None:

        from sklearn.linear_model import Ridge
        from sklearn.kernel_ridge import KernelRidge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.startup_profile = None
        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope, record_training_scope, load_training_split
        source_frame = frame
        frame = training_scope(frame, self.config)
        record_training_scope("startup_profile", frame, self.config)
        required = ["subset_id", "operating_mode", "RUL"] + self.target_names
        if not self.config.twin.startup_profile_enabled or any(c not in frame for c in required):
            return
        if int(self.config.twin.commissioning_cycles) != 30:
            return
        train, validation = engine_wise_split(frame,
            validation_fraction=self.config.data.validation_engine_fraction,
            random_state=self.config.data.random_state)
        healthy = (frame["RUL"] >= 130).to_numpy()
        train_mask = frame["unit"].isin(train["unit"].unique()).to_numpy() & healthy
        fit = train_mask & (frame["cycle"] > 30).to_numpy()
        score_mask = frame["unit"].isin(validation["unit"].unique()).to_numpy()
        score_mask &= (frame["RUL"] >= 125).to_numpy() & (frame["cycle"] > 30).to_numpy()
        if fit.sum() < 100 or score_mask.sum() < 2:
            return
        bundle: Dict[str, Any] = {"version": 1, "target_names": list(self.target_names), "n_modes": 6,
            "subsets": ["FD001", "FD002", "FD003", "FD004"], "commissioning_cycles": 30,
            "age_reference_cycles": 400., "mode_prior": 5., "aggregation_bins": 20}
        y = frame[self.target_names].to_numpy(float)
        context = self._profile_context(frame, bundle)
        population = make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(context[train_mask], y[train_mask])
        p = population.predict(context)
        x, ready = self._profile_features(frame, p, bundle)
        fit &= ready
        score_mask &= ready
        if fit.sum() < 100 or score_mask.sum() < 2:
            return
        keys = [frame.loc[fit, "unit"].reset_index(drop=True),
                frame.loc[fit, "operating_mode"].reset_index(drop=True),
                (frame.loc[fit, "cycle"] // 20).reset_index(drop=True)]
        xa = pd.DataFrame(x[fit]).groupby(keys, sort=True).mean().to_numpy()
        ya = pd.DataFrame((y - p)[fit]).groupby(keys, sort=True).mean().to_numpy()
        correction = make_pipeline(StandardScaler(), Ridge(alpha=100)).fit(xa, ya)

        reference = self.predict(frame)
        base = reference[get_expected_columns(self.target_names)].to_numpy()
        candidate = p[score_mask] + self._predict_profile_correction(correction, x[score_mask])
        before = BaseTwin._score(y[score_mask], base[score_mask])
        after = BaseTwin._score(y[score_mask], candidate)
        if after["r2"] <= before["r2"] or after["rmse"] >= before["rmse"]:
            self.logger.warning("Startup-profile validation did not improve both R2 and RMSE; not activated.")
            return
        ridge_score = dict(after)
        recipe = "ridge_alpha100"


        try:
            nonlinear = make_pipeline(StandardScaler(), KernelRidge(alpha=1., kernel="rbf", gamma=.001)).fit(xa, ya)
            prediction = p[score_mask] + self._predict_profile_correction(nonlinear, x[score_mask])
            score = BaseTwin._score(y[score_mask], prediction)
            if score["r2"] > after["r2"] and score["rmse"] < after["rmse"]:
                correction, after, recipe = nonlinear, score, "kernel_rbf_0.001_alpha1"
            else:
                self.logger.info("Kernel correction did not improve both metrics; retaining validated ridge.")
        except Exception as exc:
            self.logger.warning("Optional kernel fit failed; retaining validated ridge: %s", exc)


        extra = {}
        validated = read_json(self.config.paths.models / "startup_profile_recipe.json")
        if validated and validated.get("family") in ("full_fit_profile", "full_fit_sensor_mixture",
                                                    "full_fit_causal_forecast", "full_fit_causal_sensor_mixture",
                                                    "full_fit_forecast_target_mixture", "full_fit_context_sensor_mixture"):
            try:
                from app.services.Anomaly_Health_Monitoring.digital_twin.validated_profile import refit_validated_profile
                result = refit_validated_profile(source_frame, self.config,
                    dict(bundle, population=population), correction, validated)
                accepted_recipe = validated if result["accepted"] else validated["baseline_recipe"]
                profile_extra = {"validated_profile_recipe":accepted_recipe, "validation_role":"selection",
                    "prediction_role":result.get("prediction_role","frozen_commissioning_expectation"),
                    "prediction_objective":accepted_recipe.get("objective","healthy_sensor_forecasts"),
                    "full_trajectory_validation_metrics":result.get("full_trajectory_metrics"),
                    "reference_validation_model":"previous_startup_profile_recipe",
                    "fit_unit_ids":result["fit_unit_ids"], "selection_unit_ids":result["selection_unit_ids"],
                    "training_plan_id":load_training_split(self.config)["plan_id"],
                    "ridge_validation_role":"inner_development"}
                correction, after = result["model"], result["metrics"]
                before = result["baseline_metrics"]
                recipe = str(validated["name"] if result["accepted"] else validated["baseline_recipe"]["name"])
                if result["feature_schema"]:
                    bundle["feature_schema"] = result["feature_schema"]
                extra = profile_extra
                if not result["accepted"]:
                    self.logger.info("Validated forecast did not pass its engine-held-out gates; retaining its previous recipe.")
            except Exception as exc:
                self.logger.warning("Full profile refit failed; retaining the validated baseline: %s", exc)
        elif validated:
            try:
                from app.services.Anomaly_Health_Monitoring.digital_twin.startup_correction import (
                    StartupCorrectionRegressor, BlendedStartupCorrection, fit_sensor_blend,
                )
                spec = dict(validated["spec"])
                bins = int(spec.pop("bins"))
                weight = float(validated["weight"])
                if bins <= 0 or not 0. < weight <= 1.:
                    raise ValueError("Invalid validated aggregation/blend policy")
                validated_keys = [frame.loc[fit, "unit"].reset_index(drop=True),
                    frame.loc[fit, "operating_mode"].reset_index(drop=True),
                    (frame.loc[fit, "cycle"] // bins).reset_index(drop=True)]
                vx = pd.DataFrame(x[fit]).groupby(validated_keys, sort=True).mean().to_numpy()
                vy = pd.DataFrame((y - p)[fit]).groupby(validated_keys, sort=True).mean().to_numpy()
                candidate = (fit_sensor_blend(vx, vy, correction, spec) if spec["family"] == "sensor_blend"
                             else StartupCorrectionRegressor(**spec).fit(vx, vy))
                blended = BlendedStartupCorrection(correction, candidate, weight)
                predicted = p[score_mask] + self._predict_profile_correction(blended, x[score_mask])
                score = BaseTwin._score(y[score_mask], predicted)
                if score["r2"] > after["r2"] and score["rmse"] < after["rmse"]:
                    correction, after = blended, score
                    recipe = str(validated.get("name", "validated_" + spec["family"]))
                    extra = {"validated_correction_spec": validated["spec"],
                             "validated_blend_weight": weight, "report": validated.get("report")}
                else:
                    self.logger.info("Saved correction did not improve both metrics; retaining the validated baseline.")
            except Exception as exc:
                self.logger.warning("Saved correction fit failed; retaining the validated baseline: %s", exc)
        bundle.update(population=population, correction=correction,
                      validation_metrics=after, reference_validation_metrics=before,
                      ridge_validation_metrics=ridge_score, correction_recipe=recipe,
                      validation_rows=int(after.get("rows", score_mask.sum())), reference_preserved_for_monitoring=True, **extra)
        self.startup_profile = bundle

    def _persist_startup_profile(self) -> None:


        save_model(self.startup_profile, self.config.paths.model_path(STARTUP_PROFILE_ARTIFACT), config=self.config)

    def _build_twins(self) -> List[BaseTwin]:


        built: List[BaseTwin] = []
        for name in self.config.twin.enabled_twins:
            twin_class = TWIN_REGISTRY.get(str(name).lower().strip())
            if twin_class is None:
                self.logger.warning("Unknown twin variant '%s' ignored", name)
                continue
            built.append(twin_class(self.config))
        if not built:
            raise AnomalyHealthError(
                f"No valid twins enabled. Choose from {sorted(TWIN_REGISTRY)}."
            )
        self.logger.info("Ensemble twins: %s", [t.name for t in built])
        return built


    def train(self, frame: pd.DataFrame) -> Dict[str, Dict[str, Any]]:


        self.metrics = {}
        trained: List[BaseTwin] = []

        for twin in self.twins:
            try:
                metrics = twin.train(frame)
                twin.save()
                self.metrics[twin.name] = metrics
                trained.append(twin)
                if not self.target_names:
                    self.target_names = list(twin.target_names)
            except Exception as exc:
                self.logger.error(
                    "Twin '%s' failed to train and is excluded from the ensemble: %s",
                    twin.name,
                    exc,
                )

        if not trained:
            raise AnomalyHealthError(
                "Every digital twin failed to train; the pipeline cannot continue."
            )

        self.twins = trained
        self.weights = self.compute_weights()
        self._fit_startup_profile(frame)
        self._persist_startup_profile()
        self._persist_weights()
        return self.metrics

    def compute_weights(self) -> Dict[str, float]:

        strategy = str(self.config.twin.ensemble_weighting).lower().strip()
        names = [twin.name for twin in self.twins]

        if strategy == "uniform" or not self.metrics:
            weight = 1.0 / max(len(names), 1)
            return {name: weight for name in names}

        raw: Dict[str, float] = {}
        if strategy == "r2":
            for name in names:
                r2 = float(self.metrics.get(name, {}).get("r2", 0.0) or 0.0)
                raw[name] = max(r2, 0.0) if np.isfinite(r2) else 0.0
        else:
            for name in names:
                rmse = float(self.metrics.get(name, {}).get("rmse", 0.0) or 0.0)
                raw[name] = 1.0 / rmse if np.isfinite(rmse) and rmse > 1e-12 else 0.0

        total = sum(raw.values())
        if total <= 0:
            weight = 1.0 / max(len(names), 1)
            self.logger.warning(
                "All twin %s scores were degenerate; using uniform weights.",
                "R-squared" if strategy == "r2" else "RMSE",
            )
            return {name: weight for name in names}

        weights = {name: value / total for name, value in raw.items()}
        self.logger.info(
            "Ensemble weights (%s): %s",
            "mean per-target R-squared" if strategy == "r2" else "inverse RMSE",
            {k: round(v, 4) for k, v in weights.items()},
        )
        return weights

    def _persist_weights(self) -> Path:


        payload = {
            "weighting_strategy": self.config.twin.ensemble_weighting,
            "weights": self.weights,
            "metrics": self.metrics,
            "target_names": self.target_names,
            "twins": [twin.name for twin in self.twins],
            "startup_profile": {
                "active": self.startup_profile is not None,
                "validation_metrics": (self.startup_profile or {}).get("validation_metrics"),
                "prediction_role": (self.startup_profile or {}).get("prediction_role", "frozen_commissioning_expectation"),
                "prediction_objective": (self.startup_profile or {}).get("prediction_objective", "healthy_sensor_forecasts"),
                "full_trajectory_validation_metrics": (self.startup_profile or {}).get("full_trajectory_validation_metrics"),
                "monitoring_reference": "unchanged_context_ensemble",
            },
        }
        return atomic_write_json(
            payload, self.config.paths.model_path(self.config.paths.twin_weights_json)
        )

    def load(self) -> "EnsembleTwin":


        payload = read_json(
            self.config.paths.model_path(self.config.paths.twin_weights_json),
            default=None,
        )
        loaded: List[BaseTwin] = []
        for twin in self.twins:
            try:
                twin.load()
                loaded.append(twin)
                if not self.target_names:
                    self.target_names = list(twin.target_names)
            except Exception as exc:
                self.logger.warning("Could not reload twin '%s': %s", twin.name, exc)

        if not loaded:
            raise AnomalyHealthError(
                "No saved digital twins could be loaded. Train the twin stage first."
            )
        self.twins = loaded

        if payload:
            self.weights = {
                name: float(value)
                for name, value in (payload.get("weights") or {}).items()
                if name in {t.name for t in loaded}
            }
            self.metrics = payload.get("metrics", {})
        if not self.weights:
            self.weights = {twin.name: 1.0 / len(loaded) for twin in loaded}


        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        self.startup_profile = None
        profile_path = self.config.paths.model_path(STARTUP_PROFILE_ARTIFACT)
        if self.config.twin.startup_profile_enabled and profile_path.is_file():
            try:
                candidate = load_model(profile_path)
                required = {"version", "population", "correction", "target_names", "subsets", "n_modes",
                            "commissioning_cycles", "age_reference_cycles", "mode_prior"}
                if candidate is not None and (not isinstance(candidate, dict) or
                        not required.issubset(candidate) or candidate["version"] != 1):
                    raise ValueError("Unsupported startup-profile artifact schema")
                self.startup_profile = candidate
            except Exception as exc:
                self.logger.warning("Could not load startup-profile twin; retaining context predictions: %s", exc)
        return self


    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:


        if not self.twins:
            raise AnomalyHealthError("Ensemble has no trained twins.")

        stacked: List[np.ndarray] = []
        used_weights: List[float] = []
        used_names: List[str] = []

        for twin in self.twins:
            try:
                prediction = twin.predict(frame)
            except Exception as exc:
                self.logger.error("Twin '%s' failed during inference: %s", twin.name, exc)
                continue
            expected_columns = get_expected_columns(twin.target_names)
            stacked.append(prediction[expected_columns].to_numpy(dtype=float))
            used_weights.append(float(self.weights.get(twin.name, 0.0)))
            used_names.append(twin.name)
            if not self.target_names:
                self.target_names = list(twin.target_names)

        if not stacked:
            raise AnomalyHealthError("No twin produced predictions.")


        weight_array = np.asarray(used_weights, dtype=float)
        if weight_array.sum() <= 0:
            weight_array = np.ones_like(weight_array) / len(weight_array)
        else:
            weight_array = weight_array / weight_array.sum()

        tensor = np.stack(stacked, axis=0)
        fused = np.tensordot(weight_array, tensor, axes=(0, 0))
        spread = (
            np.std(tensor, axis=0)
            if tensor.shape[0] > 1
            else np.zeros_like(fused)
        )

        self.logger.info(
            "Fused %d twin(s) %s with weights %s",
            len(used_names),
            used_names,
            np.round(weight_array, 4).tolist(),
        )

        result = frame[get_identity_columns()].copy().reset_index(drop=True)
        reference = fused
        try:
            fused, profile_ready = self._profile_prediction(frame, reference)
        except Exception as exc:
            self.logger.warning("Startup-profile inference failed; retaining context predictions: %s", exc)
            fused, profile_ready = reference, np.zeros(len(frame), dtype=bool)
        for index, column in enumerate(get_expected_columns(self.target_names)):
            result[column] = fused[:, index]
        if self.startup_profile is not None:
            for index, sensor in enumerate(self.target_names):
                result[f"reference_expected_{sensor}"] = reference[:, index]
            result["startup_profile_ready"] = profile_ready
        for index, column in enumerate(get_twin_spread_columns(self.target_names)):
            result[column] = spread[:, index]
        result[TWIN_SPREAD_MEAN_COLUMN] = spread.mean(axis=1)

        self.logger.info(
            "Twin disagreement: mean spread %.5f, max %.5f",
            float(result[TWIN_SPREAD_MEAN_COLUMN].mean()),
            float(result[TWIN_SPREAD_MEAN_COLUMN].max()),
        )
        return result


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.twin_predictions_csv
        )
        verify_identity_columns(frame, context="twin_predictions output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(
            self.config.paths.twin_predictions_csv
        )
        return read_csv(
            target, required_columns=get_identity_columns(), stage="ensemble_twin"
        )

    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        retrain: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:

        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.scaled_features_csv),
                required_columns=get_identity_columns(),
                stage="ensemble_twin",
            )

        if retrain:
            self.train(source)
        else:
            self.load()

        predictions = self.predict(source)
        output = self.save(predictions)

        ensemble_rmse = float(
            np.average(
                [self.metrics.get(t.name, {}).get("rmse", 0.0) for t in self.twins],
                weights=[max(self.weights.get(t.name, 0.0), 1e-9) for t in self.twins],
            )
        ) if self.metrics else 0.0

        report: Dict[str, Any] = {
            "rows": int(len(predictions)),
            "engines": int(predictions["unit"].nunique()),
            "twins": [t.name for t in self.twins],
            "weights": self.weights,
            "per_twin_metrics": self.metrics,
            "weighted_mean_rmse": round(ensemble_rmse, 6),
            "mean_twin_spread": float(predictions[TWIN_SPREAD_MEAN_COLUMN].mean()),
            "n_sensors_modelled": len(self.target_names),
            "output_path": str(output),
            "startup_profile": {
                "active": self.startup_profile is not None,
                "validation_metrics": (self.startup_profile or {}).get("validation_metrics"),
                "prediction_role": (self.startup_profile or {}).get("prediction_role", "frozen_commissioning_expectation"),
                "prediction_objective": (self.startup_profile or {}).get("prediction_objective", "healthy_sensor_forecasts"),
                "full_trajectory_validation_metrics": (self.startup_profile or {}).get("full_trajectory_validation_metrics"),
                "monitoring_reference": "unchanged_context_ensemble",
                "spread_describes": "monitoring_reference_ensemble",
            },
        }
        atomic_write_json(report, self.config.paths.metrics / "digital_twin.json")
        self.logger.info(
            "Digital twin stage complete: %d rows, %d twin(s), weighted RMSE %.5f",
            report["rows"],
            len(self.twins),
            ensemble_rmse,
        )
        return predictions, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    ensemble = EnsembleTwin(config)
    predictions, report = ensemble.run()
    logger.info("Ensemble report: %s", {k: v for k, v in report.items() if k != "per_twin_metrics"})
    print(predictions.head(8).to_string())


if __name__ == "__main__":
    main()
