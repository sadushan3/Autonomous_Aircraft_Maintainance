"""
feature_engineering.py

Feature-engineering service for the Autonomous Maintenance Supervisor Agent.

This module transforms the cleaned integrated dataset into a model-ready
feature dataset.

Main responsibilities
---------------------
1. Load the cleaned supervisor dataset.
2. Validate essential metadata and target columns.
3. Sort records chronologically by engine and cycle.
4. Create RUL-derived features.
5. Create failure-risk aggregation and trend features.
6. Create anomaly aggregation and persistence features.
7. Create cross-component agreement and conflict features.
8. Create temporal rolling and lag features safely within each engine.
9. Create confidence and uncertainty features.
10. Preserve metadata and target labels.
11. Prevent direct target leakage.
12. Save the engineered dataset and a JSON report atomically.

Recommended execution
---------------------
Run from the Backend directory:

    "C:\\Python313\\python.exe" -m \
    app.services.Maintenance_Supervisor.Data_preprocessing.feature_engineering
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATH BOOTSTRAP
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# parents[0] -> Data_preprocessing
# parents[1] -> Maintenance_Supervisor
# parents[2] -> services
# parents[3] -> app
# parents[4] -> Backend
BACKEND_ROOT = CURRENT_FILE.parents[4]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# =============================================================================
# CONFIG IMPORT
# =============================================================================

try:
    from app.config.Maintenance_Supervisor import supervisor_config as config
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Could not import app.config.Maintenance_Supervisor.supervisor_config. "
        f"Expected Backend root: {BACKEND_ROOT}"
    ) from exc


# =============================================================================
# LOGGER
# =============================================================================

try:
    from app.utils.Maintenance_Supervisor.logger import get_logger

    try:
        logger = get_logger(__name__)
    except TypeError:
        logger = get_logger()

except (ImportError, ModuleNotFoundError, AttributeError):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION WITH FALLBACKS
# =============================================================================

def config_value(name: str, default: Any) -> Any:
    """Read a config value safely."""

    return getattr(config, name, default)


ENGINE_ID_COLUMN = str(
    config_value(
        "ENGINE_ID_COLUMN",
        "engine_id",
    )
)

SUBSET_COLUMN = str(
    config_value(
        "SUBSET_COLUMN",
        "fd_subset",
    )
)

TARGET_COLUMN = str(
    config_value(
        "TARGET_COLUMN",
        "final_decision",
    )
)

SCHEMA_VERSION_COLUMN = str(
    config_value(
        "SCHEMA_VERSION_COLUMN",
        "schema_version",
    )
)

UNIQUE_KEY_COLUMNS = tuple(
    config_value(
        "UNIQUE_KEY_COLUMNS",
        (
            ENGINE_ID_COLUMN,
            SUBSET_COLUMN,
            "cycle",
        ),
    )
)

CLEANED_DATASET_PATH = Path(
    config_value(
        "CLEANED_DATASET_PATH",
        BACKEND_ROOT
        / "processed"
        / "Maintenance_Supervisor"
        / "supervisor_cleaned_dataset.csv",
    )
)

ENGINEERED_DATASET_PATH = Path(
    config_value(
        "ENGINEERED_DATASET_PATH",
        BACKEND_ROOT
        / "processed"
        / "Maintenance_Supervisor"
        / "supervisor_engineered_dataset.csv",
    )
)

FEATURE_ENGINEERING_REPORT_PATH = Path(
    config_value(
        "FEATURE_ENGINEERING_REPORT_PATH",
        BACKEND_ROOT
        / "reports"
        / "Maintenance_Supervisor"
        / "feature_engineering_report.json",
    )
)

ROLLING_WINDOWS = tuple(
    config_value(
        "FEATURE_ROLLING_WINDOWS",
        (3, 5, 10),
    )
)

LAG_PERIODS = tuple(
    config_value(
        "FEATURE_LAG_PERIODS",
        (1, 3, 5),
    )
)

EPSILON = 1e-8


# =============================================================================
# FEATURE GROUPS
# =============================================================================

RISK_COLUMNS: tuple[str, ...] = (
    "risk_10",
    "risk_30",
    "risk_50",
)

RISK_UNCERTAINTY_COLUMNS: tuple[str, ...] = (
    "uncertainty_10",
    "uncertainty_30",
    "uncertainty_50",
)

RISK_THRESHOLD_COLUMNS: tuple[str, ...] = (
    "threshold_10",
    "threshold_30",
    "threshold_50",
)

ANOMALY_SCORE_COLUMNS: tuple[str, ...] = (
    "anomaly_score",
    "residual_anomaly_score",
    "isolation_forest_score",
    "mahalanobis_score",
)

ANOMALY_BOOLEAN_COLUMNS: tuple[str, ...] = (
    "is_anomaly",
    "residual_anomaly",
    "isolation_forest_anomaly",
    "mahalanobis_anomaly",
)

RUL_COLUMNS: tuple[str, ...] = (
    "predicted_rul",
    "lower_bound",
    "upper_bound",
    "uncertainty_width",
    "degradation_speed",
    "degradation_acceleration",
)

TEMPORAL_SOURCE_COLUMNS: tuple[str, ...] = (
    "predicted_rul",
    "uncertainty_width",
    "risk_10",
    "risk_30",
    "risk_50",
    "control_risk",
    "anomaly_score",
    "residual_anomaly_score",
    "isolation_forest_score",
    "mahalanobis_score",
    "context_confidence",
)

PROTECTED_COLUMNS: frozenset[str] = frozenset(
    {
        ENGINE_ID_COLUMN,
        SUBSET_COLUMN,
        "cycle",
        SCHEMA_VERSION_COLUMN,
        TARGET_COLUMN,
    }
)

KNOWN_OUTPUT_COLUMNS: frozenset[str] = frozenset(
    {
        TARGET_COLUMN,
        "priority",
        "maintenance_urgency",
        "requires_human_review",
        "supervisor_score",
    }
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(slots=True)
class FeatureEngineeringCounters:
    """Counters recorded during feature engineering."""

    rows_before: int = 0
    columns_before: int = 0
    rows_after: int = 0
    columns_after: int = 0

    total_features_created: int = 0
    numeric_features_created: list[str] = field(default_factory=list)
    temporal_features_created: list[str] = field(default_factory=list)
    categorical_features_created: list[str] = field(default_factory=list)

    missing_source_columns: list[str] = field(default_factory=list)
    invalid_numeric_values_replaced: dict[str, int] = field(
        default_factory=dict
    )
    remaining_missing_values_imputed: dict[str, int] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class FeatureEngineeringResult:
    """Returned result from the feature-engineering pipeline."""

    status: str
    input_path: str
    output_path: str
    report_path: str
    rows: int
    columns: int
    features_created: int
    duration_seconds: float
    output_sha256: str
    message: str


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def utc_now_iso() -> str:
    """Return current UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a SHA-256 hash for a file."""

    digest = hashlib.sha256()

    with path.open("rb") as source:
        while True:
            chunk = source.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    """Return DataFrame memory usage in megabytes."""

    size_bytes = int(
        dataframe.memory_usage(
            index=True,
            deep=True,
        ).sum()
    )

    return size_bytes / (1024 ** 2)


def atomic_write_csv(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    """Write a CSV atomically."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            suffix=".tmp",
            prefix=f".{destination.stem}_",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            dataframe.to_csv(
                temporary_file,
                index=False,
                lineterminator="\n",
            )

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )

        raise


def atomic_write_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write JSON atomically."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            suffix=".tmp",
            prefix=f".{destination.stem}_",
            dir=destination.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

            json.dump(
                payload,
                temporary_file,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(
                missing_ok=True,
            )

        raise


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class SupervisorFeatureEngineer:
    """
    Create model-ready supervisor decision features.

    Temporal features are always calculated separately for each engine and
    subset. This prevents values from one engine leaking into another.
    """

    def __init__(
        self,
        input_path: Path = CLEANED_DATASET_PATH,
        output_path: Path = ENGINEERED_DATASET_PATH,
        report_path: Path = FEATURE_ENGINEERING_REPORT_PATH,
        *,
        rolling_windows: Sequence[int] = ROLLING_WINDOWS,
        lag_periods: Sequence[int] = LAG_PERIODS,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path)

        self.rolling_windows = tuple(
            sorted(
                {
                    int(window)
                    for window in rolling_windows
                    if int(window) > 1
                }
            )
        )

        self.lag_periods = tuple(
            sorted(
                {
                    int(period)
                    for period in lag_periods
                    if int(period) > 0
                }
            )
        )

        self.counters = FeatureEngineeringCounters()

        self.started_at: str | None = None
        self.finished_at: str | None = None

        self.original_columns: set[str] = set()

    # =========================================================================
    # LOADING AND VALIDATION
    # =========================================================================

    def _validate_input_path(self) -> None:
        """Validate the cleaned dataset path."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                "Cleaned supervisor dataset was not found: "
                f"{self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                "Input path is not a file: "
                f"{self.input_path}"
            )

        if self.input_path.suffix.lower() != ".csv":
            raise ValueError(
                "feature_engineering expects a CSV input file."
            )

        if self.input_path.stat().st_size <= 0:
            raise ValueError(
                f"Input dataset is empty: {self.input_path}"
            )

    def _load_dataset(self) -> pd.DataFrame:
        """Load the cleaned dataset."""

        self._validate_input_path()

        logger.info(
            "Loading cleaned dataset: %s",
            self.input_path,
        )

        try:
            dataframe = pd.read_csv(
                self.input_path,
                low_memory=False,
            )

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                "The cleaned CSV contains no readable data."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                "The cleaned CSV could not be parsed."
            ) from exc

        if dataframe.empty:
            raise ValueError(
                "The cleaned dataset contains no rows."
            )

        self.counters.rows_before = int(
            len(dataframe)
        )
        self.counters.columns_before = int(
            len(dataframe.columns)
        )

        self.original_columns = set(
            dataframe.columns
        )

        logger.info(
            "Loaded rows=%d, columns=%d, memory=%.2f MB",
            len(dataframe),
            len(dataframe.columns),
            dataframe_memory_mb(dataframe),
        )

        return dataframe

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Ensure key metadata and target columns exist."""

        required = {
            ENGINE_ID_COLUMN,
            SUBSET_COLUMN,
            "cycle",
            TARGET_COLUMN,
        }

        missing = sorted(
            required.difference(
                dataframe.columns
            )
        )

        if missing:
            raise KeyError(
                "Feature engineering cannot continue because required "
                f"columns are missing: {missing}"
            )

    def _sort_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sort records chronologically within each engine."""

        return (
            dataframe.sort_values(
                by=[
                    SUBSET_COLUMN,
                    ENGINE_ID_COLUMN,
                    "cycle",
                ],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    # =========================================================================
    # FEATURE REGISTRATION
    # =========================================================================

    def _register_feature(
        self,
        column: str,
        feature_type: str = "numeric",
    ) -> None:
        """Record a newly created feature."""

        if column in self.original_columns:
            return

        if feature_type == "temporal":
            target_list = self.counters.temporal_features_created
        elif feature_type == "categorical":
            target_list = self.counters.categorical_features_created
        else:
            target_list = self.counters.numeric_features_created

        if column not in target_list:
            target_list.append(column)

    # =========================================================================
    # NUMERIC CONVERSION
    # =========================================================================

    def _convert_available_numeric_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert known numerical source columns safely."""

        dataframe = dataframe.copy()

        candidates = {
            "cycle",
            "predicted_rul",
            "lower_bound",
            "upper_bound",
            "uncertainty_width",
            "degradation_speed",
            "degradation_acceleration",
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
            "anomaly_score",
            "anomaly_threshold",
            "residual_anomaly_score",
            "isolation_forest_score",
            "mahalanobis_score",
            "top_sensor_1_score",
            "top_sensor_2_score",
            "top_sensor_3_score",
            "context_confidence",
            "supervisor_score",
        }

        for column in candidates:
            if column not in dataframe.columns:
                continue

            before_non_missing = int(
                dataframe[column].notna().sum()
            )

            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            after_non_missing = int(
                converted.notna().sum()
            )

            invalid_count = max(
                before_non_missing - after_non_missing,
                0,
            )

            if invalid_count:
                self.counters.invalid_numeric_values_replaced[
                    column
                ] = invalid_count

            dataframe[column] = converted

        return dataframe

    # =========================================================================
    # RUL FEATURES
    # =========================================================================

    def _create_rul_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create RUL state, interval, and degradation features."""

        dataframe = dataframe.copy()

        if "predicted_rul" in dataframe.columns:
            dataframe["rul_inverse"] = (
                1.0
                / (
                    dataframe["predicted_rul"].clip(lower=0.0)
                    + 1.0
                )
            )
            self._register_feature("rul_inverse")

            dataframe["rul_log"] = np.log1p(
                dataframe["predicted_rul"].clip(lower=0.0)
            )
            self._register_feature("rul_log")

            dataframe["rul_normalized_100"] = (
                dataframe["predicted_rul"].clip(
                    lower=0.0,
                    upper=100.0,
                )
                / 100.0
            )
            self._register_feature("rul_normalized_100")

            dataframe["rul_urgency_score"] = (
                1.0 - dataframe["rul_normalized_100"]
            ).clip(
                lower=0.0,
                upper=1.0,
            )
            self._register_feature("rul_urgency_score")

        if {
            "lower_bound",
            "upper_bound",
        }.issubset(dataframe.columns):
            dataframe["rul_interval_width"] = (
                dataframe["upper_bound"]
                - dataframe["lower_bound"]
            ).clip(lower=0.0)
            self._register_feature("rul_interval_width")

            dataframe["rul_interval_midpoint"] = (
                dataframe["upper_bound"]
                + dataframe["lower_bound"]
            ) / 2.0
            self._register_feature("rul_interval_midpoint")

        if {
            "predicted_rul",
            "uncertainty_width",
        }.issubset(dataframe.columns):
            dataframe["rul_relative_uncertainty"] = (
                dataframe["uncertainty_width"]
                / (
                    dataframe["predicted_rul"].abs()
                    + 1.0
                )
            )
            self._register_feature("rul_relative_uncertainty")

            dataframe["rul_confidence_proxy"] = (
                1.0
                / (
                    1.0
                    + dataframe["rul_relative_uncertainty"]
                )
            ).clip(
                lower=0.0,
                upper=1.0,
            )
            self._register_feature("rul_confidence_proxy")

        if {
            "degradation_speed",
            "degradation_acceleration",
        }.issubset(dataframe.columns):
            dataframe["degradation_intensity"] = (
                dataframe["degradation_speed"].abs()
                + dataframe["degradation_acceleration"].abs()
            )
            self._register_feature("degradation_intensity")

            dataframe["accelerating_degradation"] = (
                dataframe["degradation_acceleration"] > 0
            ).astype("int8")
            self._register_feature("accelerating_degradation")

        return dataframe

    # =========================================================================
    # RISK FEATURES
    # =========================================================================

    def _create_risk_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create horizon-risk aggregation and escalation features."""

        dataframe = dataframe.copy()

        available_risks = [
            column
            for column in RISK_COLUMNS
            if column in dataframe.columns
        ]

        if available_risks:
            dataframe["risk_mean"] = dataframe[
                available_risks
            ].mean(axis=1)
            self._register_feature("risk_mean")

            dataframe["risk_max"] = dataframe[
                available_risks
            ].max(axis=1)
            self._register_feature("risk_max")

            dataframe["risk_min"] = dataframe[
                available_risks
            ].min(axis=1)
            self._register_feature("risk_min")

            dataframe["risk_range"] = (
                dataframe["risk_max"]
                - dataframe["risk_min"]
            )
            self._register_feature("risk_range")

            dataframe["risk_std"] = dataframe[
                available_risks
            ].std(
                axis=1,
                ddof=0,
            )
            self._register_feature("risk_std")

        if {
            "risk_10",
            "risk_30",
        }.issubset(dataframe.columns):
            dataframe["risk_growth_10_to_30"] = (
                dataframe["risk_30"]
                - dataframe["risk_10"]
            )
            self._register_feature("risk_growth_10_to_30")

        if {
            "risk_30",
            "risk_50",
        }.issubset(dataframe.columns):
            dataframe["risk_growth_30_to_50"] = (
                dataframe["risk_50"]
                - dataframe["risk_30"]
            )
            self._register_feature("risk_growth_30_to_50")

        if {
            "risk_10",
            "risk_50",
        }.issubset(dataframe.columns):
            dataframe["risk_growth_10_to_50"] = (
                dataframe["risk_50"]
                - dataframe["risk_10"]
            )
            self._register_feature("risk_growth_10_to_50")

        if set(RISK_COLUMNS).issubset(dataframe.columns):
            dataframe["risk_monotonic"] = (
                (
                    dataframe["risk_10"]
                    <= dataframe["risk_30"]
                )
                & (
                    dataframe["risk_30"]
                    <= dataframe["risk_50"]
                )
            ).astype("int8")
            self._register_feature("risk_monotonic")

            dataframe["risk_weighted_horizon"] = (
                0.50 * dataframe["risk_10"]
                + 0.30 * dataframe["risk_30"]
                + 0.20 * dataframe["risk_50"]
            )
            self._register_feature("risk_weighted_horizon")

        available_uncertainties = [
            column
            for column in RISK_UNCERTAINTY_COLUMNS
            if column in dataframe.columns
        ]

        if available_uncertainties:
            dataframe["risk_uncertainty_mean"] = dataframe[
                available_uncertainties
            ].mean(axis=1)
            self._register_feature("risk_uncertainty_mean")

            dataframe["risk_uncertainty_max"] = dataframe[
                available_uncertainties
            ].max(axis=1)
            self._register_feature("risk_uncertainty_max")

        available_thresholds = [
            column
            for column in RISK_THRESHOLD_COLUMNS
            if column in dataframe.columns
        ]

        if available_thresholds:
            dataframe["risk_threshold_mean"] = dataframe[
                available_thresholds
            ].mean(axis=1)
            self._register_feature("risk_threshold_mean")

        for horizon in (10, 30, 50):
            risk_column = f"risk_{horizon}"
            threshold_column = f"threshold_{horizon}"

            if {
                risk_column,
                threshold_column,
            }.issubset(dataframe.columns):
                margin_column = (
                    f"risk_margin_{horizon}"
                )

                dataframe[margin_column] = (
                    dataframe[risk_column]
                    - dataframe[threshold_column]
                )
                self._register_feature(margin_column)

                breach_column = (
                    f"risk_threshold_breach_{horizon}"
                )

                dataframe[breach_column] = (
                    dataframe[risk_column]
                    >= dataframe[threshold_column]
                ).astype("int8")
                self._register_feature(breach_column)

        return dataframe

    # =========================================================================
    # ANOMALY FEATURES
    # =========================================================================

    def _create_anomaly_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create anomaly ensemble and agreement features."""

        dataframe = dataframe.copy()

        available_scores = [
            column
            for column in ANOMALY_SCORE_COLUMNS
            if column in dataframe.columns
        ]

        if available_scores:
            dataframe["anomaly_score_mean"] = dataframe[
                available_scores
            ].mean(axis=1)
            self._register_feature("anomaly_score_mean")

            dataframe["anomaly_score_max"] = dataframe[
                available_scores
            ].max(axis=1)
            self._register_feature("anomaly_score_max")

            dataframe["anomaly_score_min"] = dataframe[
                available_scores
            ].min(axis=1)
            self._register_feature("anomaly_score_min")

            dataframe["anomaly_score_range"] = (
                dataframe["anomaly_score_max"]
                - dataframe["anomaly_score_min"]
            )
            self._register_feature("anomaly_score_range")

            dataframe["anomaly_model_disagreement"] = dataframe[
                available_scores
            ].std(
                axis=1,
                ddof=0,
            )
            self._register_feature("anomaly_model_disagreement")

        available_flags = [
            column
            for column in ANOMALY_BOOLEAN_COLUMNS
            if column in dataframe.columns
        ]

        if available_flags:
            flag_frame = dataframe[
                available_flags
            ].apply(
                lambda column: column.astype(
                    "boolean"
                ).fillna(False).astype("int8")
            )

            dataframe["anomaly_vote_count"] = flag_frame.sum(
                axis=1
            )
            self._register_feature("anomaly_vote_count")

            dataframe["anomaly_vote_ratio"] = (
                dataframe["anomaly_vote_count"]
                / float(len(available_flags))
            )
            self._register_feature("anomaly_vote_ratio")

            dataframe["anomaly_majority_vote"] = (
                dataframe["anomaly_vote_ratio"] >= 0.5
            ).astype("int8")
            self._register_feature("anomaly_majority_vote")

            dataframe["anomaly_unanimous_vote"] = (
                dataframe["anomaly_vote_count"]
                == len(available_flags)
            ).astype("int8")
            self._register_feature("anomaly_unanimous_vote")

        if {
            "anomaly_score",
            "anomaly_threshold",
        }.issubset(dataframe.columns):
            dataframe["anomaly_threshold_margin"] = (
                dataframe["anomaly_score"]
                - dataframe["anomaly_threshold"]
            )
            self._register_feature("anomaly_threshold_margin")

            dataframe["anomaly_threshold_ratio"] = (
                dataframe["anomaly_score"]
                / (
                    dataframe["anomaly_threshold"].abs()
                    + EPSILON
                )
            )
            self._register_feature("anomaly_threshold_ratio")

        top_score_columns = [
            column
            for column in (
                "top_sensor_1_score",
                "top_sensor_2_score",
                "top_sensor_3_score",
            )
            if column in dataframe.columns
        ]

        if top_score_columns:
            dataframe["root_cause_score_sum"] = dataframe[
                top_score_columns
            ].sum(axis=1)
            self._register_feature("root_cause_score_sum")

            dataframe["root_cause_score_max"] = dataframe[
                top_score_columns
            ].max(axis=1)
            self._register_feature("root_cause_score_max")

            dataframe["root_cause_concentration"] = (
                dataframe["root_cause_score_max"]
                / (
                    dataframe["root_cause_score_sum"].abs()
                    + EPSILON
                )
            )
            self._register_feature("root_cause_concentration")

        return dataframe

    # =========================================================================
    # CROSS-COMPONENT FEATURES
    # =========================================================================

    def _create_cross_component_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create features combining RUL, risk, anomaly, and confidence."""

        dataframe = dataframe.copy()

        if {
            "rul_urgency_score",
            "risk_max",
        }.issubset(dataframe.columns):
            dataframe["rul_risk_interaction"] = (
                dataframe["rul_urgency_score"]
                * dataframe["risk_max"]
            )
            self._register_feature("rul_risk_interaction")

        if {
            "risk_max",
            "anomaly_score_max",
        }.issubset(dataframe.columns):
            dataframe["risk_anomaly_interaction"] = (
                dataframe["risk_max"]
                * dataframe["anomaly_score_max"]
            )
            self._register_feature("risk_anomaly_interaction")

            dataframe["risk_anomaly_gap"] = (
                dataframe["risk_max"]
                - dataframe["anomaly_score_max"]
            ).abs()
            self._register_feature("risk_anomaly_gap")

            dataframe["risk_anomaly_agreement"] = (
                1.0
                - dataframe[
                    "risk_anomaly_gap"
                ].clip(
                    lower=0.0,
                    upper=1.0,
                )
            )
            self._register_feature("risk_anomaly_agreement")

        if {
            "rul_urgency_score",
            "risk_max",
            "anomaly_score_max",
        }.issubset(dataframe.columns):
            dataframe["combined_condition_score"] = (
                0.35
                * dataframe["rul_urgency_score"]
                + 0.40
                * dataframe["risk_max"]
                + 0.25
                * dataframe["anomaly_score_max"]
            ).clip(
                lower=0.0,
                upper=1.0,
            )
            self._register_feature("combined_condition_score")

            dataframe["critical_condition_count"] = (
                (
                    dataframe["rul_urgency_score"]
                    >= 0.7
                ).astype("int8")
                + (
                    dataframe["risk_max"]
                    >= 0.7
                ).astype("int8")
                + (
                    dataframe["anomaly_score_max"]
                    >= 0.7
                ).astype("int8")
            )
            self._register_feature("critical_condition_count")

        confidence_sources = [
            column
            for column in (
                "context_confidence",
                "rul_confidence_proxy",
            )
            if column in dataframe.columns
        ]

        if confidence_sources:
            dataframe["combined_confidence"] = dataframe[
                confidence_sources
            ].mean(axis=1).clip(
                lower=0.0,
                upper=1.0,
            )
            self._register_feature("combined_confidence")

            dataframe["decision_uncertainty"] = (
                1.0 - dataframe["combined_confidence"]
            ).clip(
                lower=0.0,
                upper=1.0,
            )
            self._register_feature("decision_uncertainty")

        if {
            "combined_condition_score",
            "combined_confidence",
        }.issubset(dataframe.columns):
            dataframe["confidence_adjusted_condition"] = (
                dataframe["combined_condition_score"]
                * dataframe["combined_confidence"]
            )
            self._register_feature(
                "confidence_adjusted_condition"
            )

        return dataframe

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    def _grouping_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """Return grouping columns used for temporal calculations."""

        columns = [
            SUBSET_COLUMN,
            ENGINE_ID_COLUMN,
        ]

        return [
            column
            for column in columns
            if column in dataframe.columns
        ]

    def _create_temporal_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create lags, differences, rolling means, rolling standard deviations,
        and rolling trend features independently for each engine.
        """

        dataframe = dataframe.copy()

        grouping_columns = self._grouping_columns(
            dataframe
        )

        available_sources = [
            column
            for column in TEMPORAL_SOURCE_COLUMNS
            if column in dataframe.columns
        ]

        for source_column in available_sources:
            grouped = dataframe.groupby(
                grouping_columns,
                sort=False,
                observed=True,
                dropna=False,
            )[source_column]

            difference_column = (
                f"{source_column}_diff_1"
            )

            dataframe[difference_column] = grouped.diff(
                periods=1
            )
            self._register_feature(
                difference_column,
                "temporal",
            )

            percentage_column = (
                f"{source_column}_pct_change_1"
            )

            previous = grouped.shift(1)

            dataframe[percentage_column] = (
                (
                    dataframe[source_column]
                    - previous
                )
                / (
                    previous.abs()
                    + EPSILON
                )
            )
            self._register_feature(
                percentage_column,
                "temporal",
            )

            for lag in self.lag_periods:
                lag_column = (
                    f"{source_column}_lag_{lag}"
                )

                dataframe[lag_column] = grouped.shift(
                    lag
                )

                self._register_feature(
                    lag_column,
                    "temporal",
                )

            for window in self.rolling_windows:
                rolling_mean_column = (
                    f"{source_column}_roll_mean_{window}"
                )

                rolling_std_column = (
                    f"{source_column}_roll_std_{window}"
                )

                rolling_min_column = (
                    f"{source_column}_roll_min_{window}"
                )

                rolling_max_column = (
                    f"{source_column}_roll_max_{window}"
                )

                dataframe[rolling_mean_column] = (
                    grouped.transform(
                        lambda series: (
                            series.rolling(
                                window=window,
                                min_periods=1,
                            ).mean()
                        )
                    )
                )

                dataframe[rolling_std_column] = (
                    grouped.transform(
                        lambda series: (
                            series.rolling(
                                window=window,
                                min_periods=2,
                            ).std(ddof=0)
                        )
                    )
                )

                dataframe[rolling_min_column] = (
                    grouped.transform(
                        lambda series: (
                            series.rolling(
                                window=window,
                                min_periods=1,
                            ).min()
                        )
                    )
                )

                dataframe[rolling_max_column] = (
                    grouped.transform(
                        lambda series: (
                            series.rolling(
                                window=window,
                                min_periods=1,
                            ).max()
                        )
                    )
                )

                for created_column in (
                    rolling_mean_column,
                    rolling_std_column,
                    rolling_min_column,
                    rolling_max_column,
                ):
                    self._register_feature(
                        created_column,
                        "temporal",
                    )

                trend_column = (
                    f"{source_column}_trend_{window}"
                )

                dataframe[trend_column] = (
                    dataframe[source_column]
                    - dataframe[rolling_mean_column]
                )

                self._register_feature(
                    trend_column,
                    "temporal",
                )

        return dataframe

    def _create_anomaly_persistence_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create rolling anomaly persistence features."""

        dataframe = dataframe.copy()
        grouping_columns = self._grouping_columns(
            dataframe
        )

        source_column: str | None = None

        if "is_anomaly" in dataframe.columns:
            source_column = "is_anomaly"
        elif "anomaly_majority_vote" in dataframe.columns:
            source_column = "anomaly_majority_vote"

        if source_column is None:
            return dataframe

        anomaly_binary = (
            dataframe[source_column]
            .astype("boolean")
            .fillna(False)
            .astype("int8")
        )

        temporary_column = "__anomaly_binary__"
        dataframe[temporary_column] = anomaly_binary

        grouped = dataframe.groupby(
            grouping_columns,
            sort=False,
            observed=True,
            dropna=False,
        )[temporary_column]

        for window in self.rolling_windows:
            feature_column = (
                f"anomaly_persistence_{window}"
            )

            dataframe[feature_column] = grouped.transform(
                lambda series: (
                    series.rolling(
                        window=window,
                        min_periods=1,
                    ).mean()
                )
            )

            self._register_feature(
                feature_column,
                "temporal",
            )

        dataframe = dataframe.drop(
            columns=[temporary_column]
        )

        return dataframe

    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================

    def _create_categorical_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create simple categorical interaction features."""

        dataframe = dataframe.copy()

        if {
            "lifecycle_state",
            "risk_state",
        }.issubset(dataframe.columns):
            dataframe["lifecycle_risk_context"] = (
                dataframe["lifecycle_state"]
                .astype("string")
                .fillna("unknown")
                + "__"
                + dataframe["risk_state"]
                .astype("string")
                .fillna("unknown")
            )

            self._register_feature(
                "lifecycle_risk_context",
                "categorical",
            )

        if {
            "risk_state",
            "anomaly_severity",
        }.issubset(dataframe.columns):
            dataframe["risk_anomaly_context"] = (
                dataframe["risk_state"]
                .astype("string")
                .fillna("unknown")
                + "__"
                + dataframe["anomaly_severity"]
                .astype("string")
                .fillna("unknown")
            )

            self._register_feature(
                "risk_anomaly_context",
                "categorical",
            )

        if {
            "lifecycle_state",
            "anomaly_severity",
        }.issubset(dataframe.columns):
            dataframe["health_anomaly_context"] = (
                dataframe["lifecycle_state"]
                .astype("string")
                .fillna("unknown")
                + "__"
                + dataframe["anomaly_severity"]
                .astype("string")
                .fillna("unknown")
            )

            self._register_feature(
                "health_anomaly_context",
                "categorical",
            )

        return dataframe

    # =========================================================================
    # MISSING-VALUE FINALIZATION
    # =========================================================================

    def _impute_engineered_numeric_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fill missing values created by lags and rolling calculations.

        Temporal values are first filled within each engine, then by global
        median, and finally by zero if an entire feature is missing.
        """

        dataframe = dataframe.copy()

        created_numeric_columns = [
            column
            for column in (
                self.counters.numeric_features_created
                + self.counters.temporal_features_created
            )
            if column in dataframe.columns
        ]

        grouping_columns = self._grouping_columns(
            dataframe
        )

        for column in created_numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            ).replace(
                [np.inf, -np.inf],
                np.nan,
            )

            missing_before = int(
                dataframe[column].isna().sum()
            )

            if missing_before == 0:
                continue

            if grouping_columns:
                dataframe[column] = dataframe.groupby(
                    grouping_columns,
                    sort=False,
                    observed=True,
                    dropna=False,
                )[column].transform(
                    lambda series: (
                        series.ffill().bfill()
                    )
                )

            global_median = dataframe[column].median(
                skipna=True
            )

            if pd.isna(global_median):
                global_median = 0.0

            dataframe[column] = dataframe[column].fillna(
                float(global_median)
            )

            missing_after = int(
                dataframe[column].isna().sum()
            )

            self.counters.remaining_missing_values_imputed[
                column
            ] = (
                missing_before - missing_after
            )

        return dataframe

    def _finalize_dtypes(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply memory-efficient final feature dtypes."""

        dataframe = dataframe.copy()

        created_numeric_columns = (
            self.counters.numeric_features_created
            + self.counters.temporal_features_created
        )

        for column in created_numeric_columns:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                    downcast="float",
                )

        for column in (
            self.counters.categorical_features_created
        ):
            if column in dataframe.columns:
                dataframe[column] = dataframe[
                    column
                ].astype("category")

        if "cycle" in dataframe.columns:
            dataframe["cycle"] = pd.to_numeric(
                dataframe["cycle"],
                errors="raise",
                downcast="integer",
            )

        return dataframe

    # =========================================================================
    # LEAKAGE AND FINAL VALIDATION
    # =========================================================================

    def _validate_no_target_derived_features(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Ensure no feature was generated directly from the target."""

        forbidden_names = {
            f"{TARGET_COLUMN}_encoded",
            f"{TARGET_COLUMN}_lag_1",
            f"{TARGET_COLUMN}_roll_mean_3",
            "target_encoded",
            "label_encoded",
        }

        present = sorted(
            forbidden_names.intersection(
                dataframe.columns
            )
        )

        if present:
            raise ValueError(
                "Target-derived leakage features were detected: "
                f"{present}"
            )

    def _validate_final_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Perform strict final validation."""

        if dataframe.empty:
            raise ValueError(
                "Feature engineering produced an empty dataset."
            )

        missing_required = sorted(
            {
                ENGINE_ID_COLUMN,
                SUBSET_COLUMN,
                "cycle",
                TARGET_COLUMN,
            }.difference(
                dataframe.columns
            )
        )

        if missing_required:
            raise KeyError(
                "Engineered dataset is missing required columns: "
                f"{missing_required}"
            )

        duplicated_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        if duplicated_columns:
            raise ValueError(
                "Duplicate columns were created: "
                f"{duplicated_columns}"
            )

        duplicated_keys = dataframe.duplicated(
            subset=list(
                UNIQUE_KEY_COLUMNS
            ),
            keep=False,
        )

        if duplicated_keys.any():
            raise ValueError(
                "Duplicate engine/subset/cycle keys remain after "
                "feature engineering."
            )

        numeric_columns = dataframe.select_dtypes(
            include=[np.number],
        ).columns

        for column in numeric_columns:
            values = dataframe[column].to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            if np.isinf(values).any():
                raise ValueError(
                    f"Engineered column '{column}' contains infinity."
                )

        self._validate_no_target_derived_features(
            dataframe
        )

    # =========================================================================
    # REPORT
    # =========================================================================

    def _build_report(
        self,
        dataframe: pd.DataFrame,
        *,
        duration_seconds: float,
        output_hash: str,
    ) -> dict[str, Any]:
        """Build the feature-engineering report."""

        created_features = (
            self.counters.numeric_features_created
            + self.counters.temporal_features_created
            + self.counters.categorical_features_created
        )

        return {
            "status": "success",
            "component": str(
                config_value(
                    "COMPONENT_NAME",
                    config_value(
                        "PROJECT_NAME",
                        "Autonomous Maintenance Supervisor Agent",
                    ),
                )
            ),
            "stage": "feature_engineering",
            "started_at_utc": self.started_at,
            "finished_at_utc": self.finished_at,
            "duration_seconds": round(
                duration_seconds,
                6,
            ),
            "input": {
                "path": str(
                    self.input_path.resolve()
                ),
                "rows": self.counters.rows_before,
                "columns": self.counters.columns_before,
            },
            "output": {
                "path": str(
                    self.output_path.resolve()
                ),
                "report_path": str(
                    self.report_path.resolve()
                ),
                "rows": int(
                    len(dataframe)
                ),
                "columns": int(
                    len(dataframe.columns)
                ),
                "memory_mb": round(
                    dataframe_memory_mb(dataframe),
                    4,
                ),
                "sha256": output_hash,
            },
            "configuration": {
                "rolling_windows": list(
                    self.rolling_windows
                ),
                "lag_periods": list(
                    self.lag_periods
                ),
                "grouping_columns": [
                    SUBSET_COLUMN,
                    ENGINE_ID_COLUMN,
                ],
                "target_column": TARGET_COLUMN,
            },
            "summary": asdict(
                self.counters
            ),
            "created_features": created_features,
            "feature_count": len(
                created_features
            ),
            "column_profiles": [
                {
                    "name": column,
                    "dtype": str(
                        dataframe[column].dtype
                    ),
                    "missing_count": int(
                        dataframe[column].isna().sum()
                    ),
                    "unique_count": int(
                        dataframe[column].nunique(
                            dropna=True
                        )
                    ),
                }
                for column in dataframe.columns
            ],
        }

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def engineer_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute all feature-engineering transformations."""

        self._validate_required_columns(
            dataframe
        )

        dataframe = self._sort_dataset(
            dataframe
        )

        dataframe = self._convert_available_numeric_columns(
            dataframe
        )

        logger.info(
            "Creating RUL-derived features."
        )
        dataframe = self._create_rul_features(
            dataframe
        )

        logger.info(
            "Creating risk-derived features."
        )
        dataframe = self._create_risk_features(
            dataframe
        )

        logger.info(
            "Creating anomaly-derived features."
        )
        dataframe = self._create_anomaly_features(
            dataframe
        )

        logger.info(
            "Creating cross-component features."
        )
        dataframe = self._create_cross_component_features(
            dataframe
        )

        logger.info(
            "Creating temporal rolling and lag features."
        )
        dataframe = self._create_temporal_features(
            dataframe
        )

        logger.info(
            "Creating anomaly persistence features."
        )
        dataframe = self._create_anomaly_persistence_features(
            dataframe
        )

        logger.info(
            "Creating categorical interaction features."
        )
        dataframe = self._create_categorical_features(
            dataframe
        )

        logger.info(
            "Imputing engineered temporal values."
        )
        dataframe = self._impute_engineered_numeric_features(
            dataframe
        )

        dataframe = self._finalize_dtypes(
            dataframe
        )

        dataframe = self._sort_dataset(
            dataframe
        )

        self.counters.rows_after = int(
            len(dataframe)
        )
        self.counters.columns_after = int(
            len(dataframe.columns)
        )

        self.counters.total_features_created = (
            len(
                self.counters.numeric_features_created
            )
            + len(
                self.counters.temporal_features_created
            )
            + len(
                self.counters.categorical_features_created
            )
        )

        self._validate_final_dataset(
            dataframe
        )

        return dataframe

    def run(self) -> FeatureEngineeringResult:
        """Run feature engineering and write all outputs."""

        timer_start = perf_counter()
        self.started_at = utc_now_iso()

        logger.info("=" * 78)
        logger.info("SUPERVISOR FEATURE ENGINEERING STARTED")
        logger.info("=" * 78)
        logger.info("Input dataset : %s", self.input_path)
        logger.info("Output dataset: %s", self.output_path)
        logger.info("Report path   : %s", self.report_path)

        try:
            dataframe = self._load_dataset()

            engineered_dataframe = self.engineer_dataframe(
                dataframe
            )

            logger.info(
                "Writing engineered dataset atomically."
            )

            atomic_write_csv(
                engineered_dataframe,
                self.output_path,
            )

            output_hash = sha256_file(
                self.output_path
            )

            self.finished_at = utc_now_iso()
            duration_seconds = (
                perf_counter() - timer_start
            )

            report = self._build_report(
                engineered_dataframe,
                duration_seconds=duration_seconds,
                output_hash=output_hash,
            )

            atomic_write_json(
                report,
                self.report_path,
            )

            result = FeatureEngineeringResult(
                status="success",
                input_path=str(
                    self.input_path.resolve()
                ),
                output_path=str(
                    self.output_path.resolve()
                ),
                report_path=str(
                    self.report_path.resolve()
                ),
                rows=int(
                    len(engineered_dataframe)
                ),
                columns=int(
                    len(engineered_dataframe.columns)
                ),
                features_created=(
                    self.counters.total_features_created
                ),
                duration_seconds=round(
                    duration_seconds,
                    6,
                ),
                output_sha256=output_hash,
                message=(
                    "Supervisor feature engineering completed successfully."
                ),
            )

            logger.info("-" * 78)
            logger.info(
                "Rows                       : %d",
                result.rows,
            )
            logger.info(
                "Columns                    : %d",
                result.columns,
            )
            logger.info(
                "Numeric features created   : %d",
                len(
                    self.counters.numeric_features_created
                ),
            )
            logger.info(
                "Temporal features created  : %d",
                len(
                    self.counters.temporal_features_created
                ),
            )
            logger.info(
                "Categorical features       : %d",
                len(
                    self.counters.categorical_features_created
                ),
            )
            logger.info(
                "Total new features         : %d",
                result.features_created,
            )
            logger.info(
                "Duration                   : %.3f seconds",
                result.duration_seconds,
            )
            logger.info(
                "Output SHA-256             : %s",
                result.output_sha256,
            )
            logger.info("=" * 78)
            logger.info(
                "SUPERVISOR FEATURE ENGINEERING COMPLETED"
            )
            logger.info("=" * 78)

            return result

        except Exception:
            self.finished_at = utc_now_iso()

            logger.exception(
                "Supervisor feature engineering failed. "
                "Existing outputs were not intentionally deleted."
            )

            raise


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def parse_integer_list(
    raw_value: str,
) -> tuple[int, ...]:
    """Parse comma-separated positive integers."""

    values: list[int] = []

    for item in raw_value.split(","):
        item = item.strip()

        if not item:
            continue

        value = int(item)

        if value <= 0:
            raise argparse.ArgumentTypeError(
                "All periods must be positive integers."
            )

        values.append(value)

    if not values:
        raise argparse.ArgumentTypeError(
            "At least one integer must be supplied."
        )

    return tuple(
        sorted(set(values))
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Create model-ready features for the Autonomous "
            "Maintenance Supervisor dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=CLEANED_DATASET_PATH,
        help=(
            "Cleaned supervisor CSV path. "
            f"Default: {CLEANED_DATASET_PATH}"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=ENGINEERED_DATASET_PATH,
        help=(
            "Engineered CSV destination. "
            f"Default: {ENGINEERED_DATASET_PATH}"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=FEATURE_ENGINEERING_REPORT_PATH,
        help=(
            "Feature-engineering report destination. "
            f"Default: {FEATURE_ENGINEERING_REPORT_PATH}"
        ),
    )

    parser.add_argument(
        "--rolling-windows",
        type=parse_integer_list,
        default=ROLLING_WINDOWS,
        help=(
            "Comma-separated rolling windows. "
            "Example: 3,5,10"
        ),
    )

    parser.add_argument(
        "--lag-periods",
        type=parse_integer_list,
        default=LAG_PERIODS,
        help=(
            "Comma-separated lag periods. "
            "Example: 1,3,5"
        ),
    )

    return parser


def main(
    arguments: Sequence[str] | None = None,
) -> int:
    """CLI entry point."""

    parser = build_argument_parser()
    parsed = parser.parse_args(
        arguments
    )

    engineer = SupervisorFeatureEngineer(
        input_path=parsed.input,
        output_path=parsed.output,
        report_path=parsed.report,
        rolling_windows=parsed.rolling_windows,
        lag_periods=parsed.lag_periods,
    )

    try:
        result = engineer.run()

        print(
            json.dumps(
                asdict(result),
                indent=2,
            )
        )

        return 0

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        PermissionError,
        OSError,
    ) as exc:
        logger.error(
            "Feature engineering failed: %s",
            exc,
        )

        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "input_path": str(
                        parsed.input
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )

        return 1

    except Exception as exc:
        logger.exception(
            "Unexpected feature-engineering error."
        )

        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "input_path": str(
                        parsed.input
                    ),
                },
                indent=2,
            ),
            file=sys.stderr,
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(main())