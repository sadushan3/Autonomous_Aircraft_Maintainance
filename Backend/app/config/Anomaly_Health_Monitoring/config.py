from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parents[3]


@dataclass
class PathConfig:


    backend_root: Path = BACKEND_ROOT

    data_raw: Path = BACKEND_ROOT / "data" / "raw"
    data_exchange: Path = BACKEND_ROOT / "data" / "exchange"
    processed: Path = BACKEND_ROOT / "processed"
    outputs: Path = BACKEND_ROOT / "outputs"
    models: Path = BACKEND_ROOT / "models"
    scalers: Path = BACKEND_ROOT / "models" / "scalers"
    logs: Path = BACKEND_ROOT / "logs"
    mlruns: Path = BACKEND_ROOT / "mlruns"
    experiments: Path = BACKEND_ROOT / "experiments"
    metrics: Path = BACKEND_ROOT / "metrics"
    reports: Path = BACKEND_ROOT / "reports"
    uploads: Path = BACKEND_ROOT / "data" / "uploads"


    raw_data_csv: str = "raw_data.csv"
    cleaned_data_csv: str = "cleaned_data.csv"
    engineered_features_csv: str = "engineered_features.csv"
    context_clusters_csv: str = "context_clusters.csv"
    scaled_features_csv: str = "scaled_features.csv"
    twin_predictions_csv: str = "twin_predictions.csv"
    residuals_csv: str = "residuals.csv"
    health_index_csv: str = "health_index.csv"
    health_scores_csv: str = "health_scores.csv"
    health_states_csv: str = "health_states.csv"
    anomaly_scores_csv: str = "anomaly_scores.csv"
    anomaly_fusion_csv: str = "anomaly_fusion.csv"
    root_cause_analysis_csv: str = "root_cause_analysis.csv"
    explanation_reports_csv: str = "explanation_reports.csv"
    confidence_scores_csv: str = "confidence_scores.csv"
    feedback_updates_csv: str = "feedback_updates.csv"
    dashboard_data_csv: str = "dashboard_data.csv"


    exchange_predictions_csv: str = "anomaly_health_predictions.csv"


    dropped_sensors_json: str = "dropped_sensors.json"
    residual_thresholds_json: str = "residual_thresholds.json"
    health_state_thresholds_json: str = "health_state_thresholds.json"
    twin_weights_json: str = "twin_ensemble_weights.json"
    feedback_store_json: str = "feedback_store.json"
    alert_memory_json: str = "alert_memory.json"
    model_registry_json: str = "model_registry.json"

    def processed_path(self, name: str) -> Path:

        return self.processed / name

    def model_path(self, name: str) -> Path:

        return self.models / name

    def all_directories(self) -> List[Path]:

        return [
            self.data_raw,
            self.data_exchange,
            self.processed,
            self.outputs,
            self.models,
            self.scalers,
            self.logs,
            self.mlruns,
            self.experiments,
            self.metrics,
            self.reports,
            self.uploads,
        ]


@dataclass
class DataConfig:


    subsets: List[str] = field(
        default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"]
    )


    split: str = "train"


    rul_cap: int = 130


    unit_id_offset_step: int = 10_000


    validation_engine_fraction: float = 0.2


    random_state: int = 42


    drop_constant_sensors: bool = True
    constant_sensor_std_eps: float = 1e-6


    max_engines: int = 0


@dataclass
class HealthyRegionConfig:


    rul_threshold: int = 130
    cycle_fraction: float = 0.30


    min_rows_per_engine: int = 10


@dataclass
class FeatureConfig:


    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    enable_rolling_mean: bool = True
    enable_rolling_std: bool = True
    enable_rolling_min_max: bool = True
    enable_delta: bool = True
    enable_slope: bool = True
    enable_ewma: bool = True
    ewma_alpha: float = 0.3

    sequence_length: int = 30
    sequence_stride: int = 1


@dataclass
class ContextConfig:


    algorithm: str = "kmeans"

    n_modes: int = 0
    auto_k_range: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    silhouette_sample_size: int = 20_000

    mode_transition_guard: int = 3

    drift_psi_threshold: float = 0.2
    scaler_kind: str = "standard"


@dataclass
class TwinConfig:


    enabled_twins: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "lightgbm", "tensorflow"]
    )

    random_forest: Dict[str, Any] = field(
        default_factory=lambda: {


            "n_estimators": 200,
            "max_depth": 14,
            "min_samples_leaf": 4,
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    xgboost: Dict[str, Any] = field(
        default_factory=lambda: {


            "n_estimators": 250,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "random_state": 42,
            "tree_method": "hist",
        }
    )
    lightgbm: Dict[str, Any] = field(
        default_factory=lambda: {

            "n_estimators": 300,
            "num_leaves": 31,
            "learning_rate": 0.06,
            "min_child_samples": 20,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        }
    )
    tensorflow: Dict[str, Any] = field(
        default_factory=lambda: {

            "hidden_units": [128, 64, 32],
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "batch_size": 256,
            "epochs": 150,
            "early_stopping_patience": 20,
            "reduce_lr_patience": 6,
            "reduce_lr_factor": 0.5,
            "min_lr": 1e-6,
            "validation_split": 0.15,
            "random_state": 42,
        }
    )


    ensemble_weighting: str = "inverse_rmse"


    age_reference_cycles: float = 300.0


    commissioning_cycles: int = 30


    startup_profile_enabled: bool = True

    max_acceptable_healthy_residual_z: float = 3.5


@dataclass
class HealthConfig:


    decay_lambda: float = 0.25
    ewma_alpha: float = 0.2

    trend_window: int = 20

    state_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "healthy": 80.0,
            "degrading": 60.0,
            "warning": 40.0,
            "critical": 20.0,
        }
    )
    state_order: List[str] = field(
        default_factory=lambda: [
            "Healthy",
            "Degrading",
            "Warning",
            "Critical",
            "Failure Risk",
        ]
    )
    alert_levels: Dict[str, str] = field(
        default_factory=lambda: {
            "Healthy": "none",
            "Degrading": "info",
            "Warning": "warning",
            "Critical": "high",
            "Failure Risk": "critical",
        }
    )

    min_monotonicity_score: float = 0.7


@dataclass
class AnomalyConfig:


    isolation_forest: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_samples": "auto",
            "contamination": "auto",
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    mahalanobis: Dict[str, Any] = field(
        default_factory=lambda: {
            "shrinkage": True,
            "regularization": 1e-6,
            "temporal_window": 5,
            "score_normalization": "rational",
        }
    )
    condition_consensus: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 220,
            "max_depth": 18,
            "min_samples_leaf": 10,
            "seeds": [42, 123, 2026],


            "margin": 1.0,
            "max_false_alarm_increase": 0.001,
        }
    )

    feature_space: str = "raw"

    calibration_healthy_rul: int = 125
    lstm_autoencoder: Dict[str, Any] = field(
        default_factory=lambda: {
            "latent_units": 32,
            "encoder_units": 64,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "batch_size": 128,
            "epochs": 40,
            "early_stopping_patience": 6,
            "validation_split": 0.15,
            "random_state": 42,
        }
    )

    iforest_threshold_percentile: float = 99.0
    mahalanobis_threshold_percentile: float = 99.0
    lstm_ae_threshold_percentile: float = 99.0
    residual_threshold_percentile: float = 99.0


    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "isolation_forest": 0.25,
            "mahalanobis": 0.25,
            "lstm_autoencoder": 0.25,
            "residual_rule": 0.25,
        }
    )


    fusion_threshold: float = 0.5
    fusion_threshold_percentile: float = 99.0


    persistence_min_cycles: int = 3
    mode_transition_guard: int = 3

    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.5, "medium": 0.65, "high": 0.8, "critical": 0.9}
    )
    early_warning_horizon: int = 30


@dataclass
class ReasoningConfig:


    top_k_sensors: int = 5

    trend_slope_eps: float = 1e-3

    evolution_window: int = 20

    memory_match_threshold: float = 0.75

    hard_causal_claim: bool = False


@dataclass
class ExplainabilityConfig:


    explainer_kind: str = "auto"
    background_sample_size: int = 100
    kmeans_background_clusters: int = 50
    max_explain_rows: int = 2000
    top_k_features: int = 8
    permutation_max_evals: int = 200

    local_shap_max_rows: int = 128
    reference_background_rows: int = 32
    fidelity_tolerance: float = 1e-5


@dataclass
class UncertaintyConfig:


    w_agreement: float = 0.45
    w_reliability: float = 0.35
    w_data_quality: float = 0.20

    spread_saturation: float = 1.0
    min_confidence: float = 0.0
    max_confidence: float = 100.0


@dataclass
class FeedbackConfig:


    false_alarm_step: float = 1.05
    missed_alarm_step: float = 0.95
    accepted_step: float = 1.0
    min_threshold_scale: float = 0.5
    max_threshold_scale: float = 2.0

    min_feedback_records: int = 1
    memory_capacity: int = 5000


@dataclass
class ApiConfig:


    title: str = "Explainable Health Monitoring & Anomaly Reasoning Agent"
    version: str = "1.0.0"
    router_prefix: str = "/anomaly-health"
    tags: List[str] = field(default_factory=lambda: ["Anomaly Health Monitoring"])
    max_upload_mb: int = 512
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MlopsConfig:


    log_level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "anomaly_health_monitoring.log"
    enable_mlflow: bool = True
    mlflow_experiment_name: str = "explainable_health_monitoring"
    model_version_tag: str = "v1"
    csv_float_format: str = "%.6f"


@dataclass
class Config:


    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    healthy_region: HealthyRegionConfig = field(default_factory=HealthyRegionConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    twin: TwinConfig = field(default_factory=TwinConfig)
    health: HealthConfig = field(default_factory=HealthConfig)
    anomaly: AnomalyConfig = field(default_factory=AnomalyConfig)
    reasoning: ReasoningConfig = field(default_factory=ReasoningConfig)
    explainability: ExplainabilityConfig = field(default_factory=ExplainabilityConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    mlops: MlopsConfig = field(default_factory=MlopsConfig)


    @classmethod
    def load(cls, override_file: Optional[str | Path] = None) -> "Config":


        cfg = cls()

        path = override_file or os.environ.get("AHM_CONFIG_FILE")
        if path:
            file_path = Path(path)
            if file_path.is_file():
                try:
                    with file_path.open("r", encoding="utf-8") as handle:
                        overrides = json.load(handle)
                    cfg.apply_overrides(overrides)
                    logger.info("Applied config overrides from %s", file_path)
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning("Could not read config override %s: %s", file_path, exc)
            else:
                logger.warning("Config override file %s does not exist", file_path)

        cfg.apply_env_overrides()
        return cfg

    def apply_overrides(self, overrides: Dict[str, Any]) -> None:


        for section_name, section_values in overrides.items():
            if not hasattr(self, section_name):
                logger.warning("Unknown config section '%s' ignored", section_name)
                continue
            section = getattr(self, section_name)
            if not isinstance(section_values, dict):
                logger.warning("Config section '%s' must be an object", section_name)
                continue
            valid_names = {f.name for f in fields(section)}
            for key, value in section_values.items():
                if key not in valid_names:
                    logger.warning("Unknown config key '%s.%s' ignored", section_name, key)
                    continue
                current = getattr(section, key)
                if isinstance(current, Path):
                    value = Path(value)
                setattr(section, key, value)

    def apply_env_overrides(self) -> None:


        for env_key, env_value in os.environ.items():
            if not env_key.startswith("AHM_") or "__" not in env_key:
                continue
            _, remainder = env_key.split("AHM_", 1)
            section_name, _, key_name = remainder.partition("__")
            section_name = section_name.lower()
            key_name = key_name.lower()
            if not hasattr(self, section_name):
                continue
            section = getattr(self, section_name)
            if not any(f.name == key_name for f in fields(section)):
                continue
            try:
                parsed: Any = json.loads(env_value)
            except json.JSONDecodeError:
                parsed = env_value
            current = getattr(section, key_name)
            if isinstance(current, Path):
                parsed = Path(parsed)
            setattr(section, key_name, parsed)
            logger.info("Config override from env: %s.%s", section_name, key_name)

    def ensure_directories(self) -> None:


        for directory in self.paths.all_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:


        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_convert(v) for v in value]
            return value

        return {k: _convert(v) for k, v in asdict(self).items()}

    def snapshot(self, destination: Optional[Path] = None) -> Path:


        self.ensure_directories()
        target = destination or (self.paths.experiments / "active_config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        os.replace(tmp, target)
        return target


CONFIG: Config = Config.load()


def get_config() -> Config:


    return CONFIG


def reload_config(override_file: Optional[str | Path] = None) -> Config:


    global CONFIG
    CONFIG = Config.load(override_file)
    return CONFIG


def main() -> None:

    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    cfg.ensure_directories()
    snapshot_path = cfg.snapshot()
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"\nSnapshot written to: {snapshot_path}")


if __name__ == "__main__":
    main()
