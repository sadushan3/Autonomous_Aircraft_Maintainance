"""
supervisor_config.py

Central configuration for the Autonomous Maintenance Supervisor Agent.

This file defines:

1. Project directories
2. Dataset paths
3. Processed-data paths
4. Model artifact paths
5. Output and report paths
6. Feature groups
7. Maintenance decision classes
8. Dataset split settings
9. Baseline and machine-learning parameters
10. Confidence and safety thresholds

No model should contain hard-coded paths or thresholds. Other modules must
import their settings from this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Current file:
# Backend/app/config/Maintenance_Supervisor/supervisor_config.py
#
# parents[0] -> Maintenance_Supervisor
# parents[1] -> config
# parents[2] -> app
# parents[3] -> Backend
BACKEND_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

APP_ROOT: Final[Path] = BACKEND_ROOT / "app"

COMPONENT_NAME: Final[str] = "Maintenance_Supervisor"


# =============================================================================
# MAIN DIRECTORIES
# =============================================================================

DATA_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "data"
    / COMPONENT_NAME
)

SAMPLE_DATA_ROOT: Final[Path] = (
    DATA_ROOT
    / "sample"
)

RAW_DATA_ROOT: Final[Path] = (
    DATA_ROOT
    / "raw"
)

UPSTREAM_DATA_ROOT: Final[Path] = (
    DATA_ROOT
    / "upstream"
)

FEEDBACK_DATA_ROOT: Final[Path] = (
    DATA_ROOT
    / "feedback"
)

PROCESSED_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "processed"
    / COMPONENT_NAME
)

OUTPUT_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "outputs"
    / COMPONENT_NAME
)

MODEL_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "models"
    / COMPONENT_NAME
)

ARTIFACT_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "artifacts"
    / COMPONENT_NAME
)

METRICS_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "metrics"
    / COMPONENT_NAME
)

REPORTS_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "reports"
    / COMPONENT_NAME
)

DATA_VALIDATION_REPORT_ROOT: Final[Path] = (
    REPORTS_ROOT
    / "data_validation"
)

FIGURES_ROOT: Final[Path] = (
    REPORTS_ROOT
    / "figures"
)

TABLES_ROOT: Final[Path] = (
    REPORTS_ROOT
    / "tables"
)

EXPERIMENTS_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "experiments"
    / COMPONENT_NAME
)

LOGS_ROOT: Final[Path] = (
    BACKEND_ROOT
    / "logs"
    / COMPONENT_NAME
)


# =============================================================================
# INPUT DATASET PATHS
# =============================================================================

SAMPLE_DATASET_PATH: Final[Path] = (
    SAMPLE_DATA_ROOT
    / "supervisor_sample_dataset.csv"
)

RUL_AGENT_OUTPUT_PATH: Final[Path] = (
    UPSTREAM_DATA_ROOT
    / "rul_agent_outputs.csv"
)

RISK_AGENT_OUTPUT_PATH: Final[Path] = (
    UPSTREAM_DATA_ROOT
    / "failure_risk_outputs.csv"
)

ANOMALY_AGENT_OUTPUT_PATH: Final[Path] = (
    UPSTREAM_DATA_ROOT
    / "anomaly_agent_outputs.csv"
)

MERGED_REAL_AGENT_OUTPUT_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "merged_real_agent_outputs.csv"
)


# =============================================================================
# PROCESSED DATASET PATHS
# =============================================================================

CLEANED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "supervisor_cleaned_dataset.csv"
)

ENGINEERED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "supervisor_engineered_dataset.csv"
)

TRAIN_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "train_dataset.parquet"
)

VALIDATION_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "validation_dataset.parquet"
)

TEST_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "test_dataset.parquet"
)

SPLIT_MANIFEST_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "split_manifest.csv"
)

SPLIT_SUMMARY_PATH: Final[Path] = (
    PROCESSED_ROOT
    / "split_summary.json"
)


# =============================================================================
# MODEL PATHS
# =============================================================================

LOGISTIC_REGRESSION_MODEL_PATH: Final[Path] = (
    MODEL_ROOT
    / "logistic_regression_supervisor.joblib"
)

RANDOM_FOREST_MODEL_PATH: Final[Path] = (
    MODEL_ROOT
    / "random_forest_supervisor.joblib"
)

XGBOOST_MODEL_PATH: Final[Path] = (
    MODEL_ROOT
    / "xgboost_supervisor.joblib"
)

LIGHTGBM_MODEL_PATH: Final[Path] = (
    MODEL_ROOT
    / "lightgbm_supervisor.joblib"
)

SELECTED_MODEL_PATH: Final[Path] = (
    MODEL_ROOT
    / "selected_supervisor_model.joblib"
)

PREPROCESSOR_PATH: Final[Path] = (
    ARTIFACT_ROOT
    / "supervisor_preprocessor.joblib"
)

LABEL_ENCODER_PATH: Final[Path] = (
    ARTIFACT_ROOT
    / "decision_label_encoder.joblib"
)

FEATURE_MANIFEST_PATH: Final[Path] = (
    ARTIFACT_ROOT
    / "feature_manifest.json"
)

MODEL_METADATA_PATH: Final[Path] = (
    ARTIFACT_ROOT
    / "selected_model_metadata.json"
)


# =============================================================================
# PREDICTION OUTPUT PATHS
# =============================================================================

RULE_BASED_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "rule_based_predictions.csv"
)

RUL_ONLY_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "rul_only_predictions.csv"
)

RISK_ONLY_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "risk_only_predictions.csv"
)

ANOMALY_ONLY_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "anomaly_only_predictions.csv"
)

MAJORITY_VOTE_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "majority_vote_predictions.csv"
)

LOGISTIC_REGRESSION_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "logistic_regression_predictions.csv"
)

RANDOM_FOREST_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "random_forest_predictions.csv"
)

XGBOOST_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "xgboost_predictions.csv"
)

LIGHTGBM_PREDICTIONS_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "lightgbm_predictions.csv"
)

FINAL_SUPERVISOR_OUTPUT_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "final_supervisor_decisions.csv"
)

CONFIDENCE_OUTPUT_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "confidence_results.csv"
)

EXPLANATION_OUTPUT_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "explanation_results.csv"
)

DASHBOARD_OUTPUT_PATH: Final[Path] = (
    OUTPUT_ROOT
    / "dashboard_data.csv"
)


# =============================================================================
# DATA VALIDATION AND PROCESSING REPORT PATHS
# =============================================================================

DATA_VALIDATION_REPORT_PATH: Final[Path] = (
    DATA_VALIDATION_REPORT_ROOT
    / "dataset_validation_report.json"
)

DATA_VALIDATION_ISSUES_PATH: Final[Path] = (
    DATA_VALIDATION_REPORT_ROOT
    / "dataset_validation_issues.csv"
)

CLEANING_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT
    / "dataset_cleaning_report.json"
)

FEATURE_ENGINEERING_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT
    / "feature_engineering_report.json"
)


# =============================================================================
# METRICS AND REPORT PATHS
# =============================================================================

DATA_QUALITY_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT
    / "data_quality_report.json"
)

DATA_QUALITY_MARKDOWN_PATH: Final[Path] = (
    REPORTS_ROOT
    / "data_quality_report.md"
)

EDA_SUMMARY_PATH: Final[Path] = (
    REPORTS_ROOT
    / "eda_summary.json"
)

BASELINE_METRICS_PATH: Final[Path] = (
    METRICS_ROOT
    / "baseline_metrics.csv"
)

MODEL_COMPARISON_PATH: Final[Path] = (
    METRICS_ROOT
    / "model_comparison.csv"
)

CONFUSION_MATRIX_PATH: Final[Path] = (
    METRICS_ROOT
    / "confusion_matrix.csv"
)

CLASSIFICATION_REPORT_PATH: Final[Path] = (
    METRICS_ROOT
    / "classification_report.json"
)

CONFIDENCE_METRICS_PATH: Final[Path] = (
    METRICS_ROOT
    / "confidence_metrics.json"
)

LATENCY_METRICS_PATH: Final[Path] = (
    METRICS_ROOT
    / "inference_latency.json"
)

SHAP_VALUES_PATH: Final[Path] = (
    REPORTS_ROOT
    / "shap_values.csv"
)

FEATURE_IMPORTANCE_PATH: Final[Path] = (
    REPORTS_ROOT
    / "feature_importance.csv"
)


# =============================================================================
# BASIC DATASET COLUMNS
# =============================================================================

SCHEMA_VERSION_COLUMN: Final[str] = "schema_version"
ENGINE_ID_COLUMN: Final[str] = "engine_id"
SUBSET_COLUMN: Final[str] = "fd_subset"
CYCLE_COLUMN: Final[str] = "cycle"
TARGET_COLUMN: Final[str] = "final_decision"

UNIQUE_KEY_COLUMNS: Final[list[str]] = [
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
]

METADATA_COLUMNS: Final[list[str]] = [
    SCHEMA_VERSION_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
]


# =============================================================================
# RUL AGENT FEATURES
# =============================================================================

RUL_NUMERICAL_FEATURES: Final[list[str]] = [
    "predicted_rul",
    "lower_bound",
    "upper_bound",
    "uncertainty_width",
    "degradation_speed",
    "degradation_acceleration",
]

RUL_CATEGORICAL_FEATURES: Final[list[str]] = [
    "lifecycle_state",
]

RUL_BOOLEAN_FEATURES: Final[list[str]] = [
    "confidence_collapse_flag",
    "sudden_health_drop_flag",
    "instability_flag",
]

RUL_FEATURES: Final[list[str]] = (
    RUL_NUMERICAL_FEATURES
    + RUL_CATEGORICAL_FEATURES
    + RUL_BOOLEAN_FEATURES
)


# =============================================================================
# FAILURE-RISK AGENT FEATURES
# =============================================================================

RISK_NUMERICAL_FEATURES: Final[list[str]] = [
    "risk_10",
    "risk_30",
    "risk_50",
    "uncertainty_10",
    "uncertainty_30",
    "uncertainty_50",
    "threshold_10",
    "threshold_30",
    "threshold_50",
    "control_risk",
    "risk_velocity",
    "risk_acceleration",
]

RISK_CATEGORICAL_FEATURES: Final[list[str]] = [
    "risk_state",
    "action_hint_for_supervisor",
]

RISK_BOOLEAN_FEATURES: Final[list[str]] = [
    "predicted_fail_in_10",
    "predicted_fail_in_30",
    "predicted_fail_in_50",
]

RISK_FEATURES: Final[list[str]] = (
    RISK_NUMERICAL_FEATURES
    + RISK_CATEGORICAL_FEATURES
    + RISK_BOOLEAN_FEATURES
)


# =============================================================================
# ANOMALY AGENT FEATURES
# =============================================================================

ANOMALY_NUMERICAL_FEATURES: Final[list[str]] = [
    "anomaly_score",
    "anomaly_threshold",
    "residual_anomaly_score",
    "isolation_forest_score",
    "mahalanobis_score",
    "top_sensor_1_score",
    "top_sensor_2_score",
    "top_sensor_3_score",
    "context_confidence",
]

ANOMALY_CATEGORICAL_FEATURES: Final[list[str]] = [
    "anomaly_severity",
    "top_sensor_1",
    "top_sensor_2",
    "top_sensor_3",
]

ANOMALY_BOOLEAN_FEATURES: Final[list[str]] = [
    "is_anomalous",
    "context_drift_flag",
]

ANOMALY_FEATURES: Final[list[str]] = (
    ANOMALY_NUMERICAL_FEATURES
    + ANOMALY_CATEGORICAL_FEATURES
    + ANOMALY_BOOLEAN_FEATURES
)


# =============================================================================
# MODEL TRACE COLUMNS
# =============================================================================

MODEL_TRACE_COLUMNS: Final[list[str]] = [
    "rul_model_used",
    "risk_model_used",
    "anomaly_model_used",
]


# =============================================================================
# COLUMNS GENERATED AFTER SUPERVISOR DECISION
# =============================================================================

DERIVED_OUTPUT_COLUMNS: Final[list[str]] = [
    "priority",
    "maintenance_urgency",
    "requires_human_review",
    "supervisor_score",
]


# =============================================================================
# FEATURE GROUPS
# =============================================================================

BASE_INPUT_FEATURES: Final[list[str]] = (
    RUL_FEATURES
    + RISK_FEATURES
    + ANOMALY_FEATURES
)

NUMERICAL_FEATURES: Final[list[str]] = (
    RUL_NUMERICAL_FEATURES
    + RISK_NUMERICAL_FEATURES
    + ANOMALY_NUMERICAL_FEATURES
)

CATEGORICAL_FEATURES: Final[list[str]] = (
    RUL_CATEGORICAL_FEATURES
    + RISK_CATEGORICAL_FEATURES
    + ANOMALY_CATEGORICAL_FEATURES
)

BOOLEAN_FEATURES: Final[list[str]] = (
    RUL_BOOLEAN_FEATURES
    + RISK_BOOLEAN_FEATURES
    + ANOMALY_BOOLEAN_FEATURES
)


# =============================================================================
# ENGINEERED FEATURES
# =============================================================================

ENGINEERED_FEATURES: Final[list[str]] = [
    "rul_severity",
    "lower_bound_severity",
    "risk_severity",
    "short_horizon_risk_pressure",
    "medium_horizon_risk_pressure",
    "long_horizon_risk_pressure",
    "anomaly_severity_numeric",
    "uncertainty_severity",
    "trend_severity",
    "failure_window_pressure",
    "agent_agreement_score",
    "decision_conflict_score",
    "risk_anomaly_interaction",
    "rul_risk_interaction",
    "rul_anomaly_interaction",
    "uncertainty_risk_interaction",
    "uncertainty_anomaly_interaction",
    "confidence_penalty",
    "instability_count",
    "critical_flag_count",
    "risk_threshold_exceedance_count",
]


# =============================================================================
# LEAKAGE PROTECTION
# =============================================================================

FORBIDDEN_FEATURE_COLUMNS: Final[list[str]] = [
    "true_rul",
    "actual_rul",
    "remaining_cycles",
    "remaining_useful_life",
    "future_failure_cycle",
    "failure_cycle",
    "max_cycle",
    "maximum_cycle",
    "engine_lifetime",
    "failure_timestamp",
    "future_anomaly_score",
    "future_risk",
    "future_sensor_value",
    "maintenance_label",
    "rule_label",
    "pseudo_decision_code",
    "generated_target_score",
    TARGET_COLUMN,
    "priority",
    "maintenance_urgency",
    "requires_human_review",
    "supervisor_score",
]


# =============================================================================
# DECISION CLASSES
# =============================================================================

DECISION_CLASSES: Final[list[str]] = [
    "continue_operation",
    "monitor_closely",
    "schedule_inspection",
    "schedule_maintenance",
    "immediate_maintenance",
]

DECISION_TO_SEVERITY: Final[dict[str, int]] = {
    "continue_operation": 0,
    "monitor_closely": 1,
    "schedule_inspection": 2,
    "schedule_maintenance": 3,
    "immediate_maintenance": 4,
}

SEVERITY_TO_DECISION: Final[dict[int, str]] = {
    severity: decision
    for decision, severity in DECISION_TO_SEVERITY.items()
}

PRIORITY_MAPPING: Final[dict[str, str]] = {
    "continue_operation": "Low",
    "monitor_closely": "Medium",
    "schedule_inspection": "High",
    "schedule_maintenance": "High",
    "immediate_maintenance": "Critical",
}

URGENCY_MAPPING: Final[dict[str, str]] = {
    "continue_operation": "none",
    "monitor_closely": "observe_next_cycles",
    "schedule_inspection": "within_30_cycles",
    "schedule_maintenance": "within_10_to_30_cycles",
    "immediate_maintenance": "immediate",
}

ACTION_MAPPING: Final[dict[str, str]] = {
    "continue_operation": (
        "Continue normal operation and routine monitoring."
    ),
    "monitor_closely": (
        "Continue operation with increased monitoring frequency."
    ),
    "schedule_inspection": (
        "Schedule an engineering inspection and prepare a maintenance plan."
    ),
    "schedule_maintenance": (
        "Schedule maintenance within the recommended maintenance window."
    ),
    "immediate_maintenance": (
        "Stop operation when safely possible and perform immediate maintenance."
    ),
}


# =============================================================================
# MAIN CONFIGURATION
# =============================================================================

@dataclass(frozen=True, slots=True)
class SupervisorConfig:
    """Runtime configuration for training, evaluation and inference."""

    project_name: str = "Autonomous Maintenance Supervisor Agent"
    component_name: str = COMPONENT_NAME

    random_seed: int = 42

    # -------------------------------------------------------------------------
    # Dataset split
    # -------------------------------------------------------------------------

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    # -------------------------------------------------------------------------
    # Main columns
    # -------------------------------------------------------------------------

    engine_id_column: str = ENGINE_ID_COLUMN
    subset_column: str = SUBSET_COLUMN
    cycle_column: str = CYCLE_COLUMN
    target_column: str = TARGET_COLUMN

    # -------------------------------------------------------------------------
    # Confidence thresholds
    # -------------------------------------------------------------------------

    low_confidence_threshold: float = 0.60
    medium_confidence_threshold: float = 0.80
    high_confidence_threshold: float = 0.90

    high_uncertainty_threshold: float = 0.70
    high_conflict_threshold: float = 0.30
    low_context_confidence_threshold: float = 0.60

    # Fixed modelling gap: confidence_engine.py previously had no way to
    # distinguish a row with a genuine, freshly-computed RUL/Failure-Risk
    # score from one carrying a forward-filled value from auto-stride
    # thinning (see rul_failure_batch_service.py) -- both looked identical
    # to the confidence formula, so ~97% of rows in a live run shared one
    # confidence_score. This penalty (subtracted from confidence whenever
    # model_used is null/empty, i.e. the row's score is stale) makes that
    # distinction real. Kept below the smallest existing penalty tier
    # (u_penalty/drift_penalty/collapse_penalty each cap near 0.2-0.3) so it
    # nudges confidence and the resulting human-review rate without making
    # every thinned row look catastrophically uncertain on its own.
    stale_score_confidence_penalty: float = 0.20

    # -------------------------------------------------------------------------
    # Rule-based RUL thresholds
    # -------------------------------------------------------------------------

    immediate_rul_threshold: float = 10.0
    maintenance_rul_threshold: float = 30.0
    inspection_rul_threshold: float = 50.0
    monitoring_rul_threshold: float = 80.0

    # -------------------------------------------------------------------------
    # Rule-based risk thresholds
    # -------------------------------------------------------------------------

    critical_risk_10_threshold: float = 0.82
    maintenance_risk_30_threshold: float = 0.72
    inspection_risk_30_threshold: float = 0.55
    monitoring_risk_50_threshold: float = 0.42

    # -------------------------------------------------------------------------
    # Rule-based anomaly thresholds
    # -------------------------------------------------------------------------

    critical_anomaly_threshold: float = 0.82
    high_anomaly_threshold: float = 0.60
    medium_anomaly_threshold: float = 0.38

    # -------------------------------------------------------------------------
    # Logistic Regression
    # -------------------------------------------------------------------------

    logistic_max_iter: int = 2_000
    logistic_solver: str = "lbfgs"
    logistic_class_weight: str = "balanced"

    # -------------------------------------------------------------------------
    # Random Forest
    # -------------------------------------------------------------------------

    random_forest_n_estimators: int = 300
    random_forest_max_depth: int | None = 20
    random_forest_min_samples_split: int = 5
    random_forest_min_samples_leaf: int = 2
    random_forest_class_weight: str = "balanced_subsample"
    random_forest_n_jobs: int = -1

    # -------------------------------------------------------------------------
    # XGBoost
    # -------------------------------------------------------------------------

    xgboost_n_estimators: int = 400
    xgboost_max_depth: int = 6
    xgboost_learning_rate: float = 0.05
    xgboost_subsample: float = 0.85
    xgboost_colsample_bytree: float = 0.85
    xgboost_min_child_weight: float = 2.0
    xgboost_reg_alpha: float = 0.05
    xgboost_reg_lambda: float = 1.0

    # -------------------------------------------------------------------------
    # LightGBM
    # -------------------------------------------------------------------------

    lightgbm_n_estimators: int = 400
    lightgbm_learning_rate: float = 0.05
    lightgbm_num_leaves: int = 31
    lightgbm_max_depth: int = -1
    lightgbm_subsample: float = 0.85
    lightgbm_colsample_bytree: float = 0.85
    lightgbm_reg_alpha: float = 0.05
    lightgbm_reg_lambda: float = 1.0

    # -------------------------------------------------------------------------
    # Reinforcement Learning (PPO & DQN) Parameters
    # -------------------------------------------------------------------------

    rl_epochs: int = 80
    rl_batch_size: int = 64
    rl_gamma: float = 0.99
    rl_learning_rate: float = 3e-4

    # PPO Parameters
    ppo_clip_ratio: float = 0.2
    ppo_gae_lambda: float = 0.95
    ppo_entropy_coef: float = 0.01
    ppo_value_coef: float = 0.5
    ppo_update_epochs: int = 5

    # DQN Parameters
    dqn_buffer_size: int = 10_000
    dqn_epsilon_start: float = 1.0
    dqn_epsilon_min: float = 0.05
    dqn_epsilon_decay: float = 0.995
    dqn_target_update_freq: int = 10

    # Reward Function Costs (in currency / penalty units)
    cost_unplanned_failure: float = 5000.0
    cost_immediate_maintenance: float = 1200.0
    cost_scheduled_maintenance: float = 500.0
    cost_inspection: float = 150.0
    cost_monitoring: float = 30.0
    cost_continue_operation: float = 0.0

    # Fixed modelling bug (delayed-only failure signal): the "wait" actions
    # (continue_operation, monitor_closely, schedule_inspection) previously
    # cost either 0 or a small flat fee every step, no matter how close the
    # engine actually was to failure, with the entire cost_unplanned_failure
    # penalty landing only once, on the terminal step. Verified against a
    # retrained policy (2026-09-16): that reward shape let the agent learn
    # "continuing is free" and run engines to genuine failure in 84% of
    # held-out evaluation episodes, missing 80% of the engines the Anomaly &
    # Health component itself flagged Critical/Failure Risk. This weight
    # adds a dense, risk-proportional cost to every "wait" step (see
    # rl_environment.py step()), so the policy gets gradient signal well
    # before the single delayed terminal penalty, instead of only at
    # failure. 0.15 means a step at risk_10=1.0 (near-certain failure within
    # 10 cycles) costs 0.15 * cost_unplanned_failure = 750 that step alone;
    # at risk_10=0.1 it costs 75 -- negligible next to cost_monitoring/
    # cost_inspection until risk is genuinely elevated. Retune if the
    # retrained policy over- or under-reacts.
    rl_wait_risk_penalty_weight: float = 0.15

    # -------------------------------------------------------------------------
    # Federated Learning Parameters
    # -------------------------------------------------------------------------

    fl_num_clients: int = 4
    fl_num_rounds: int = 5
    fl_local_epochs: int = 3
    fl_fraction_clients: float = 1.0


    # -------------------------------------------------------------------------
    # Processing
    # -------------------------------------------------------------------------

    csv_chunk_size: int = 100_000
    parquet_compression: str = "snappy"

    @property
    def decision_classes(self) -> tuple[str, ...]:
        """Return maintenance decision classes in severity order."""

        return tuple(DECISION_CLASSES)

    @property
    def base_input_features(self) -> tuple[str, ...]:
        """Return all raw supervisor input features."""

        return tuple(BASE_INPUT_FEATURES)

    @property
    def engineered_features(self) -> tuple[str, ...]:
        """Return all feature-engineering output columns."""

        return tuple(ENGINEERED_FEATURES)

    @property
    def all_model_features(self) -> tuple[str, ...]:
        """Return raw and engineered features used for full models."""

        return tuple(
            BASE_INPUT_FEATURES
            + ENGINEERED_FEATURES
        )

    @property
    def numerical_features(self) -> tuple[str, ...]:
        """Return numerical raw feature columns."""

        return tuple(NUMERICAL_FEATURES)

    @property
    def categorical_features(self) -> tuple[str, ...]:
        """Return categorical raw feature columns."""

        return tuple(CATEGORICAL_FEATURES)

    @property
    def boolean_features(self) -> tuple[str, ...]:
        """Return Boolean raw feature columns."""

        return tuple(BOOLEAN_FEATURES)

    @property
    def forbidden_features(self) -> tuple[str, ...]:
        """Return columns prohibited from model training."""

        return tuple(FORBIDDEN_FEATURE_COLUMNS)

    def validate(self) -> None:
        """Validate configuration consistency."""

        self._validate_dataset_split()
        self._validate_confidence_thresholds()
        self._validate_rul_thresholds()
        self._validate_risk_thresholds()
        self._validate_anomaly_thresholds()
        self._validate_model_parameters()
        self._validate_decision_configuration()
        self._validate_feature_configuration()
        self._validate_processing_configuration()

    def _validate_dataset_split(self) -> None:
        """Validate train, validation and test fractions."""

        split_values = [
            self.train_fraction,
            self.validation_fraction,
            self.test_fraction,
        ]

        for value in split_values:
            if not 0.0 < value < 1.0:
                raise ValueError(
                    "Each dataset split fraction must be greater than "
                    "zero and less than one."
                )

        split_total = sum(split_values)

        if abs(split_total - 1.0) > 1e-9:
            raise ValueError(
                "Train, validation and test fractions must total 1.0. "
                f"Received: {split_total:.6f}"
            )

        if self.random_seed < 0:
            raise ValueError(
                "random_seed must be zero or a positive integer."
            )

    def _validate_confidence_thresholds(self) -> None:
        """Validate confidence, uncertainty and conflict thresholds."""

        confidence_values = {
            "low_confidence_threshold": self.low_confidence_threshold,
            "medium_confidence_threshold": self.medium_confidence_threshold,
            "high_confidence_threshold": self.high_confidence_threshold,
            "high_uncertainty_threshold": self.high_uncertainty_threshold,
            "high_conflict_threshold": self.high_conflict_threshold,
            "low_context_confidence_threshold": (
                self.low_context_confidence_threshold
            ),
        }

        for name, value in confidence_values.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1. Received: {value}"
                )

        if not (
            self.low_confidence_threshold
            < self.medium_confidence_threshold
            < self.high_confidence_threshold
        ):
            raise ValueError(
                "Confidence thresholds must satisfy: "
                "low < medium < high."
            )

    def _validate_rul_thresholds(self) -> None:
        """Validate RUL decision-threshold ordering."""

        rul_values = [
            self.immediate_rul_threshold,
            self.maintenance_rul_threshold,
            self.inspection_rul_threshold,
            self.monitoring_rul_threshold,
        ]

        if any(value < 0 for value in rul_values):
            raise ValueError(
                "RUL thresholds cannot be negative."
            )

        if not (
            self.immediate_rul_threshold
            < self.maintenance_rul_threshold
            < self.inspection_rul_threshold
            < self.monitoring_rul_threshold
        ):
            raise ValueError(
                "RUL thresholds must satisfy: "
                "immediate < maintenance < inspection < monitoring."
            )

    def _validate_risk_thresholds(self) -> None:
        """Validate failure-risk thresholds."""

        risk_thresholds = {
            "critical_risk_10_threshold": (
                self.critical_risk_10_threshold
            ),
            "maintenance_risk_30_threshold": (
                self.maintenance_risk_30_threshold
            ),
            "inspection_risk_30_threshold": (
                self.inspection_risk_30_threshold
            ),
            "monitoring_risk_50_threshold": (
                self.monitoring_risk_50_threshold
            ),
        }

        for name, value in risk_thresholds.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1. Received: {value}"
                )

        if not (
            self.critical_risk_10_threshold
            > self.maintenance_risk_30_threshold
            > self.inspection_risk_30_threshold
            > self.monitoring_risk_50_threshold
        ):
            raise ValueError(
                "Risk thresholds must decrease from critical short-horizon "
                "risk to long-horizon monitoring risk."
            )

    def _validate_anomaly_thresholds(self) -> None:
        """Validate anomaly severity thresholds."""

        anomaly_thresholds = [
            self.medium_anomaly_threshold,
            self.high_anomaly_threshold,
            self.critical_anomaly_threshold,
        ]

        for value in anomaly_thresholds:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    "Anomaly thresholds must be between 0 and 1."
                )

        if not (
            self.medium_anomaly_threshold
            < self.high_anomaly_threshold
            < self.critical_anomaly_threshold
        ):
            raise ValueError(
                "Anomaly thresholds must satisfy: "
                "medium < high < critical."
            )

    def _validate_model_parameters(self) -> None:
        """Validate model hyperparameters."""

        if self.logistic_max_iter <= 0:
            raise ValueError(
                "logistic_max_iter must be greater than zero."
            )

        if not self.logistic_solver.strip():
            raise ValueError(
                "logistic_solver cannot be empty."
            )

        if self.random_forest_n_estimators <= 0:
            raise ValueError(
                "random_forest_n_estimators must be greater than zero."
            )

        if (
            self.random_forest_max_depth is not None
            and self.random_forest_max_depth <= 0
        ):
            raise ValueError(
                "random_forest_max_depth must be positive or None."
            )

        if self.random_forest_min_samples_split < 2:
            raise ValueError(
                "random_forest_min_samples_split must be at least 2."
            )

        if self.random_forest_min_samples_leaf < 1:
            raise ValueError(
                "random_forest_min_samples_leaf must be at least 1."
            )

        if self.random_forest_n_jobs == 0:
            raise ValueError(
                "random_forest_n_jobs cannot be zero."
            )

        if self.xgboost_n_estimators <= 0:
            raise ValueError(
                "xgboost_n_estimators must be greater than zero."
            )

        if self.xgboost_max_depth <= 0:
            raise ValueError(
                "xgboost_max_depth must be greater than zero."
            )

        if not 0.0 < self.xgboost_learning_rate <= 1.0:
            raise ValueError(
                "xgboost_learning_rate must be greater than zero "
                "and at most one."
            )

        if not 0.0 < self.xgboost_subsample <= 1.0:
            raise ValueError(
                "xgboost_subsample must be greater than zero "
                "and at most one."
            )

        if not 0.0 < self.xgboost_colsample_bytree <= 1.0:
            raise ValueError(
                "xgboost_colsample_bytree must be greater than zero "
                "and at most one."
            )

        if self.xgboost_min_child_weight < 0:
            raise ValueError(
                "xgboost_min_child_weight cannot be negative."
            )

        if self.xgboost_reg_alpha < 0:
            raise ValueError(
                "xgboost_reg_alpha cannot be negative."
            )

        if self.xgboost_reg_lambda < 0:
            raise ValueError(
                "xgboost_reg_lambda cannot be negative."
            )

        if self.lightgbm_n_estimators <= 0:
            raise ValueError(
                "lightgbm_n_estimators must be greater than zero."
            )

        if not 0.0 < self.lightgbm_learning_rate <= 1.0:
            raise ValueError(
                "lightgbm_learning_rate must be greater than zero "
                "and at most one."
            )

        if self.lightgbm_num_leaves < 2:
            raise ValueError(
                "lightgbm_num_leaves must be at least 2."
            )

        if self.lightgbm_max_depth == 0 or self.lightgbm_max_depth < -1:
            raise ValueError(
                "lightgbm_max_depth must be -1 or a positive integer."
            )

        if not 0.0 < self.lightgbm_subsample <= 1.0:
            raise ValueError(
                "lightgbm_subsample must be greater than zero "
                "and at most one."
            )

        if not 0.0 < self.lightgbm_colsample_bytree <= 1.0:
            raise ValueError(
                "lightgbm_colsample_bytree must be greater than zero "
                "and at most one."
            )

        if self.lightgbm_reg_alpha < 0:
            raise ValueError(
                "lightgbm_reg_alpha cannot be negative."
            )

        if self.lightgbm_reg_lambda < 0:
            raise ValueError(
                "lightgbm_reg_lambda cannot be negative."
            )

    def _validate_decision_configuration(self) -> None:
        """Validate decision classes and related mappings."""

        if len(DECISION_CLASSES) != len(set(DECISION_CLASSES)):
            raise ValueError(
                "Decision class definitions contain duplicate values."
            )

        decision_set = set(DECISION_CLASSES)

        if decision_set != set(DECISION_TO_SEVERITY):
            raise ValueError(
                "Decision classes and severity mapping are inconsistent."
            )

        if decision_set != set(PRIORITY_MAPPING):
            raise ValueError(
                "Decision classes and priority mapping are inconsistent."
            )

        if decision_set != set(URGENCY_MAPPING):
            raise ValueError(
                "Decision classes and urgency mapping are inconsistent."
            )

        if decision_set != set(ACTION_MAPPING):
            raise ValueError(
                "Decision classes and action mapping are inconsistent."
            )

        expected_severities = set(range(len(DECISION_CLASSES)))

        if set(DECISION_TO_SEVERITY.values()) != expected_severities:
            raise ValueError(
                "Decision severity values must form a continuous sequence "
                "starting from zero."
            )

        if set(SEVERITY_TO_DECISION) != expected_severities:
            raise ValueError(
                "Reverse decision severity mapping is incomplete."
            )

    def _validate_feature_configuration(self) -> None:
        """Validate feature groups and leakage protection."""

        duplicated_features = (
            set(RUL_FEATURES).intersection(RISK_FEATURES)
            | set(RUL_FEATURES).intersection(ANOMALY_FEATURES)
            | set(RISK_FEATURES).intersection(ANOMALY_FEATURES)
        )

        if duplicated_features:
            raise ValueError(
                "Features are duplicated across agent feature groups: "
                f"{sorted(duplicated_features)}"
            )

        if len(BASE_INPUT_FEATURES) != len(set(BASE_INPUT_FEATURES)):
            raise ValueError(
                "BASE_INPUT_FEATURES contains duplicate columns."
            )

        if len(ENGINEERED_FEATURES) != len(set(ENGINEERED_FEATURES)):
            raise ValueError(
                "ENGINEERED_FEATURES contains duplicate columns."
            )

        overlap = set(BASE_INPUT_FEATURES).intersection(
            ENGINEERED_FEATURES
        )

        if overlap:
            raise ValueError(
                "Raw and engineered feature groups overlap: "
                f"{sorted(overlap)}"
            )

        leaked_inputs = set(BASE_INPUT_FEATURES).intersection(
            FORBIDDEN_FEATURE_COLUMNS
        )

        if leaked_inputs:
            raise ValueError(
                "Forbidden leakage columns appear in model inputs: "
                f"{sorted(leaked_inputs)}"
            )

        leaked_engineered_features = set(
            ENGINEERED_FEATURES
        ).intersection(
            FORBIDDEN_FEATURE_COLUMNS
        )

        if leaked_engineered_features:
            raise ValueError(
                "Forbidden leakage columns appear in engineered features: "
                f"{sorted(leaked_engineered_features)}"
            )

        if TARGET_COLUMN not in FORBIDDEN_FEATURE_COLUMNS:
            raise ValueError(
                "The target column must be included in "
                "FORBIDDEN_FEATURE_COLUMNS."
            )

    def _validate_processing_configuration(self) -> None:
        """Validate data-processing settings."""

        if self.csv_chunk_size <= 0:
            raise ValueError(
                "csv_chunk_size must be greater than zero."
            )

        if not self.parquet_compression.strip():
            raise ValueError(
                "parquet_compression cannot be empty."
            )

    def create_directories(self) -> None:
        """
        Create all required project directories.

        Existing directories and files are not deleted or overwritten.
        """

        directories = [
            DATA_ROOT,
            SAMPLE_DATA_ROOT,
            RAW_DATA_ROOT,
            UPSTREAM_DATA_ROOT,
            FEEDBACK_DATA_ROOT,
            PROCESSED_ROOT,
            OUTPUT_ROOT,
            MODEL_ROOT,
            ARTIFACT_ROOT,
            METRICS_ROOT,
            REPORTS_ROOT,
            DATA_VALIDATION_REPORT_ROOT,
            FIGURES_ROOT,
            TABLES_ROOT,
            EXPERIMENTS_ROOT,
            LOGS_ROOT,
        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )


CONFIG: Final[SupervisorConfig] = SupervisorConfig()


def initialize_supervisor_environment() -> SupervisorConfig:
    """
    Validate configuration and create required directories.

    Returns
    -------
    SupervisorConfig
        Validated configuration instance.
    """

    CONFIG.validate()
    CONFIG.create_directories()

    return CONFIG


def print_configuration_summary(
    config: SupervisorConfig,
) -> None:
    """Print a readable configuration summary."""

    print("=" * 78)
    print("AUTONOMOUS MAINTENANCE SUPERVISOR CONFIGURATION")
    print("=" * 78)

    print(f"Project                  : {config.project_name}")
    print(f"Component                : {config.component_name}")
    print(f"Backend root             : {BACKEND_ROOT}")
    print(f"Application root         : {APP_ROOT}")
    print(f"Sample dataset           : {SAMPLE_DATASET_PATH}")
    print(f"Sample dataset exists    : {SAMPLE_DATASET_PATH.exists()}")
    print(f"Processed data directory : {PROCESSED_ROOT}")
    print(f"Model directory          : {MODEL_ROOT}")
    print(f"Artifact directory       : {ARTIFACT_ROOT}")
    print(f"Output directory         : {OUTPUT_ROOT}")
    print(f"Metrics directory        : {METRICS_ROOT}")
    print(f"Reports directory        : {REPORTS_ROOT}")
    print(f"Validation reports       : {DATA_VALIDATION_REPORT_ROOT}")
    print(f"Logs directory           : {LOGS_ROOT}")

    print("-" * 78)

    print(
        "Dataset split            : "
        f"train={config.train_fraction:.0%}, "
        f"validation={config.validation_fraction:.0%}, "
        f"test={config.test_fraction:.0%}"
    )

    print(
        "Unique key columns       : "
        f"{UNIQUE_KEY_COLUMNS}"
    )

    print(
        "Target column            : "
        f"{config.target_column}"
    )

    print(
        "Decision classes         : "
        f"{list(config.decision_classes)}"
    )

    print(
        "Raw input feature count  : "
        f"{len(config.base_input_features)}"
    )

    print(
        "Engineered feature count : "
        f"{len(config.engineered_features)}"
    )

    print(
        "Total feature count      : "
        f"{len(config.all_model_features)}"
    )

    print(
        "Forbidden feature count  : "
        f"{len(config.forbidden_features)}"
    )

    print("Configuration valid      : True")
    print("=" * 78)


if __name__ == "__main__":
    supervisor_config = initialize_supervisor_environment()
    print_configuration_summary(supervisor_config)