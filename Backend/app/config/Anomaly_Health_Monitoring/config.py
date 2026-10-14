"""
Central configuration for the Explainable Health Monitoring & Anomaly Reasoning Agent.

Every tunable value in the component lives here.  No module may hardcode a path,
a hyper-parameter, or a threshold; they all read from the ``CONFIG`` singleton
exported at the bottom of this file.

The configuration is layered:

1. Dataclass defaults (declared below).
2. Optional JSON override file pointed at by ``AHM_CONFIG_FILE``.
3. Environment variables prefixed with ``AHM_`` (highest precedence).

This ordering lets the same code run unchanged in a laptop fast-path
(``FD001`` only) and in a full pooled ``FD001..FD004`` research run.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Path anchoring
# --------------------------------------------------------------------------- #
# config.py lives at: Backend/app/config/Anomaly_Health_Monitoring/config.py
# The Backend root is therefore four parents up.
_THIS_FILE = Path(__file__).resolve()
BACKEND_ROOT = _THIS_FILE.parents[3]


@dataclass
class PathConfig:
    """Filesystem layout for the component.

    All paths are absolute and derived from :data:`BACKEND_ROOT` so the
    component behaves identically regardless of the current working directory.
    """

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

    # ---- carry-forward CSV names (stage -> filename) ---------------------- #
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

    # ---- integration handoff --------------------------------------------- #
    exchange_predictions_csv: str = "anomaly_health_predictions.csv"

    # ---- persisted JSON state -------------------------------------------- #
    dropped_sensors_json: str = "dropped_sensors.json"
    residual_thresholds_json: str = "residual_thresholds.json"
    health_state_thresholds_json: str = "health_state_thresholds.json"
    twin_weights_json: str = "twin_ensemble_weights.json"
    feedback_store_json: str = "feedback_store.json"
    alert_memory_json: str = "alert_memory.json"
    model_registry_json: str = "model_registry.json"

    def processed_path(self, name: str) -> Path:
        """Return the absolute path of a carry-forward CSV inside ``processed/``."""
        return self.processed / name

    def model_path(self, name: str) -> Path:
        """Return the absolute path of an artefact inside ``models/``."""
        return self.models / name

    def all_directories(self) -> List[Path]:
        """Every directory the component expects to exist at runtime."""
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
    """C-MAPSS ingestion behaviour."""

    #: Which subsets to load. Any of FD001..FD004.
    #:
    #: Defaults to the full pooled fleet, because that is what the persisted
    #: artifacts are fitted on and the sensor/feature policy is derived from.
    #: This used to default to ``["FD001"]`` as a "fast path", which is a trap:
    #: the CLI pipeline retrains by default, so anyone running it without an
    #: explicit override silently refitted every twin on one single-condition
    #: subset. Measured cost of doing exactly that on 2026-08-24 -- twin
    #: training rows 35,684 -> 4,609, operating modes 6 -> 1, sensor targets
    #: 21 -> 15, and all four twin R-squared values from +0.231/+0.218/+0.182/
    #: +0.086 to NEGATIVE (worse than predicting the mean). The health index
    #: still improved on that run, so downstream metrics did not catch it.
    #:
    #: For a fast development loop, narrow it explicitly and deliberately:
    #:     AHM_DATA__SUBSETS='["FD001"]' python ...10_full_pipeline.py --no-retrain
    #: Use ``--no-retrain`` when narrowing, so a quick run scores against the
    #: pooled artifacts instead of replacing them.
    subsets: List[str] = field(
        default_factory=lambda: ["FD001", "FD002", "FD003", "FD004"]
    )

    #: Which split to run the pipeline over ("train" | "test" | "both").
    split: str = "train"

    #: Standard C-MAPSS RUL clipping ceiling (Heimes 2008 convention).
    rul_cap: int = 130

    #: When pooling multiple subsets, offset unit ids so they never collide.
    unit_id_offset_step: int = 10_000

    #: Fraction of *engines* (never rows) held out for validation.
    validation_engine_fraction: float = 0.2

    #: Seed for the engine-wise split and every stochastic model.
    random_state: int = 42

    #: Drop sensors whose std is below ``constant_sensor_std_eps``.
    drop_constant_sensors: bool = True
    constant_sensor_std_eps: float = 1e-6

    #: Sub-sample engines for a quick smoke run (0 = use all).
    max_engines: int = 0


@dataclass
class HealthyRegionConfig:
    """Definition of the 'healthy' region used to train the Digital Twin.

    A row is healthy iff it satisfies BOTH conditions.  Training the twin on
    healthy rows only is what makes residuals grow with degradation.
    """

    rul_threshold: int = 130
    cycle_fraction: float = 0.30
    #: Guarantee at least this many healthy rows per engine even if the
    #: two rules above are unusually strict for a short trajectory.
    min_rows_per_engine: int = 10


@dataclass
class FeatureConfig:
    """Feature engineering windows and toggles."""

    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20])
    enable_rolling_mean: bool = True
    enable_rolling_std: bool = True
    enable_rolling_min_max: bool = True
    enable_delta: bool = True
    enable_slope: bool = True
    enable_ewma: bool = True
    ewma_alpha: float = 0.3
    #: Sequence window used by the LSTM auto-encoder.
    sequence_length: int = 30
    sequence_stride: int = 1


@dataclass
class ContextConfig:
    """Operating-mode discovery over ``op_setting_1..3``."""

    #: "kmeans" | "gmm"
    algorithm: str = "kmeans"
    #: Explicit cluster count. 0 => auto-select via silhouette search.
    n_modes: int = 0
    auto_k_range: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    silhouette_sample_size: int = 20_000
    #: A row within this many cycles of a mode change is a transition row.
    mode_transition_guard: int = 3
    #: Population-stability-index level above which drift is declared.
    drift_psi_threshold: float = 0.2
    scaler_kind: str = "standard"  # "standard" | "robust"


@dataclass
class TwinConfig:
    """Digital Twin hyper-parameters (one block per twin variant)."""

    enabled_twins: List[str] = field(
        default_factory=lambda: ["random_forest", "xgboost", "lightgbm", "tensorflow"]
    )

    random_forest: Dict[str, Any] = field(
        default_factory=lambda: {
            # Reverted to these values after an experiment: bumping to
            # 400/18/2 measurably *lowered* held-out engine R2 (0.231 ->
            # 0.192). More/deeper trees overfit the healthy-region training
            # engines rather than buying real generalisation -- the ceiling
            # here is set by how much of real C-MAPSS sensor noise is
            # explainable from operating context alone, not by tree capacity.
            "n_estimators": 200,
            "max_depth": 14,
            "min_samples_leaf": 4,
            "n_jobs": -1,
            "random_state": 42,
        }
    )
    xgboost: Dict[str, Any] = field(
        default_factory=lambda: {
            # Reverted after the same experiment: 500/7/0.05 lowered held-out
            # R2 (0.218 -> 0.206) versus these settings.
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
            # Reverted after the same experiment: 500/63/0.05 lowered held-out
            # R2 the most of the three tree twins (0.182 -> 0.146).
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
            # The MLP badly underperformed the tree twins (R2 0.07 vs
            # 0.18-0.23) at [256,128,64]/dropout=0.2: that architecture is
            # over-parameterised for an 11-24 dim input dominated by one-hot
            # categoricals, and 0.2 dropout on every layer was too aggressive
            # regularisation for a feature space this small and structured.
            # Smaller network, lighter dropout, and enough epochs/patience to
            # actually reach its plateau (it stopped at epoch 26 of 60 last
            # run) fixes the underfitting without touching any other twin.
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

    #: Ensemble weighting: "inverse_rmse" | "uniform".
    ensemble_weighting: str = "inverse_rmse"
    #: Residual acceptance gate used by validate_residuals.py.
    max_acceptable_healthy_residual_z: float = 3.5


@dataclass
class HealthConfig:
    """Health index / score / state parameters."""

    #: Decay rate lambda in HI = 100 * exp(-lambda * excess), where excess is
    #: the row's mean absolute centred residual expressed as a multiple above
    #: the healthy reference level. Swept over {0.25, 0.35, 0.50} against true
    #: RUL on the pooled FD001-4 fleet: all three score |rho| ~ 0.836, but 0.25
    #: is the only one reaching 1.000 monotonicity (vs 0.9986), so it wins on
    #: the tie-break without costing correlation.
    decay_lambda: float = 0.25
    ewma_alpha: float = 0.2
    #: Trend window (cycles) for deterioration rate.
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
    #: Monotonicity acceptance gate reported by evaluate_health.py.
    min_monotonicity_score: float = 0.7


@dataclass
class AnomalyConfig:
    """Anomaly detector hyper-parameters, thresholds and fusion weights."""

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
            "shrinkage": True,  # Ledoit-Wolf for 21-dim stability
            "regularization": 1e-6,
        }
    )
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

    #: Per-detector fusion weights.
    #:
    #: Isolation Forest and Mahalanobis correlate at r=0.980 on real output --
    #: both are distance-in-feature-space methods -- so together they hold 60%
    #: of the vote for what looks like one signal, while the least-correlated
    #: detector (residual_rule, r=0.614 against Isolation Forest) is weighted
    #: least. That looks like double-counting, and it was measured on
    #: 2026-08-24 against ground truth (Spearman of the fused score against
    #: true RUL, which comes from cycle counts and is therefore independent of
    #: the residuals every detector consumes):
    #:
    #:     current  IF.30 MH.30 LSTM.25 RR.15   rho = -0.8716   <- best
    #:     family-normalised (redundancy-aware) rho = -0.8621
    #:     uniform 0.25 each                    rho = -0.8685
    #:     drop mahalanobis entirely            rho = -0.8616
    #:
    #: Every "fix" for the redundancy is worse. Correlation between detectors
    #: is not the same as redundancy of *useful* signal: the distance-based
    #: pair carries the most degradation information here, and the detector
    #: that is most statistically independent is also the least predictive.
    #: Leave these weights alone unless a measurement on held-out ground truth
    #: says otherwise. Do not re-derive them from inter-detector correlation.
    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "isolation_forest": 0.3,
            "mahalanobis": 0.3,
            "lstm_autoencoder": 0.25,
            "residual_rule": 0.15,
        }
    )
    #: Fallback fusion threshold, used only when the fusion stage has no
    #: healthy region to calibrate against yet. In normal operation the
    #: fusion threshold is calibrated from the healthy population's own fused
    #: score at ``fusion_threshold_percentile``, the same way every individual
    #: detector calibrates its own threshold -- a flat constant here would be
    #: tied to whatever score scale happened to be in use when it was chosen
    #: and go stale the moment that scale changes.
    fusion_threshold: float = 0.5
    fusion_threshold_percentile: float = 99.0

    #: Operational-vs-fault gate.
    persistence_min_cycles: int = 3
    mode_transition_guard: int = 3

    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"low": 0.5, "medium": 0.65, "high": 0.8, "critical": 0.9}
    )
    early_warning_horizon: int = 30


@dataclass
class ReasoningConfig:
    """Root-cause reasoning parameters."""

    top_k_sensors: int = 5
    #: Slope magnitude above which a residual trend counts as "rising".
    trend_slope_eps: float = 1e-3
    #: Window used to characterise how an anomaly evolved.
    evolution_window: int = 20
    #: Similarity threshold for matching against remembered anomalies.
    memory_match_threshold: float = 0.75
    #: The component never asserts proven causation.
    hard_causal_claim: bool = False


@dataclass
class ExplainabilityConfig:
    """SHAP and narrative-generation settings."""

    #: "auto" | "tree" | "permutation"
    explainer_kind: str = "auto"
    background_sample_size: int = 100
    kmeans_background_clusters: int = 50
    max_explain_rows: int = 2000
    top_k_features: int = 8
    permutation_max_evals: int = 200


@dataclass
class UncertaintyConfig:
    """Confidence / reliability / uncertainty weighting."""

    w_agreement: float = 0.45
    w_reliability: float = 0.35
    w_data_quality: float = 0.20
    #: Spread above which twin disagreement is considered maximal.
    spread_saturation: float = 1.0
    min_confidence: float = 0.0
    max_confidence: float = 100.0


@dataclass
class FeedbackConfig:
    """Operator-feedback driven threshold adaptation."""

    #: Multiplicative nudges applied to residual thresholds.
    false_alarm_step: float = 1.05   # operator rejected -> raise threshold
    missed_alarm_step: float = 0.95  # operator reported miss -> lower threshold
    accepted_step: float = 1.0       # confirmed -> keep
    min_threshold_scale: float = 0.5
    max_threshold_scale: float = 2.0
    #: Ignore adaptation until this many feedback records exist.
    min_feedback_records: int = 1
    memory_capacity: int = 5000


@dataclass
class ApiConfig:
    """FastAPI surface configuration."""

    title: str = "Explainable Health Monitoring & Anomaly Reasoning Agent"
    version: str = "1.0.0"
    router_prefix: str = "/anomaly-health"
    tags: List[str] = field(default_factory=lambda: ["Anomaly Health Monitoring"])
    max_upload_mb: int = 512
    cors_origins: List[str] = field(default_factory=lambda: ["*"])


@dataclass
class MlopsConfig:
    """Logging, experiment tracking and model versioning."""

    log_level: str = "INFO"
    log_to_file: bool = True
    log_filename: str = "anomaly_health_monitoring.log"
    enable_mlflow: bool = True
    mlflow_experiment_name: str = "explainable_health_monitoring"
    model_version_tag: str = "v1"
    csv_float_format: str = "%.6f"


@dataclass
class Config:
    """Root configuration object aggregating every sub-config."""

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

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, override_file: Optional[str | Path] = None) -> "Config":
        """Build a config from defaults, then a JSON file, then env vars.

        Args:
            override_file: Optional path to a JSON file whose top-level keys
                match the sub-config names (``data``, ``twin``, ...).  When
                omitted the ``AHM_CONFIG_FILE`` environment variable is used.

        Returns:
            A fully materialised :class:`Config`.
        """
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
        """Recursively apply a nested override mapping onto this config.

        Args:
            overrides: Mapping such as ``{"data": {"subsets": ["FD001","FD002"]}}``.
        """
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
        """Apply ``AHM_<SECTION>__<KEY>`` environment variables.

        Values are parsed as JSON when possible so lists and numbers work,
        falling back to the raw string.  Example::

            set AHM_DATA__SUBSETS=["FD001","FD002"]
            set AHM_TWIN__ENSEMBLE_WEIGHTING=uniform
        """
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
        """Create every runtime directory if it does not already exist.

        Never deletes or clears anything -- this is safe to call at the start
        of every stage.
        """
        for directory in self.paths.all_directories():
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view of the whole configuration."""

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
        """Persist the active configuration next to the experiment artefacts.

        Args:
            destination: Optional explicit output path.

        Returns:
            The path the snapshot was written to.
        """
        self.ensure_directories()
        target = destination or (self.paths.experiments / "active_config.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
        os.replace(tmp, target)
        return target


#: Module-level singleton consumed by every other module in the component.
CONFIG: Config = Config.load()


def get_config() -> Config:
    """Return the active configuration singleton.

    Modules should depend on this accessor (or accept a ``Config`` via their
    constructor) rather than importing ``CONFIG`` directly, which keeps the
    dependency-injection story intact for tests.

    Returns:
        The process-wide :class:`Config` instance.
    """
    return CONFIG


def reload_config(override_file: Optional[str | Path] = None) -> Config:
    """Rebuild the singleton, e.g. after an API caller supplies new settings.

    Args:
        override_file: Optional JSON override path.

    Returns:
        The freshly built configuration, which also replaces ``CONFIG``.
    """
    global CONFIG
    CONFIG = Config.load(override_file)
    return CONFIG


def main() -> None:
    """Print the resolved configuration -- useful for verifying overrides."""
    logging.basicConfig(level=logging.INFO)
    cfg = get_config()
    cfg.ensure_directories()
    snapshot_path = cfg.snapshot()
    print(json.dumps(cfg.to_dict(), indent=2))
    print(f"\nSnapshot written to: {snapshot_path}")


if __name__ == "__main__":
    main()
