"""
data_validator.py

Production-grade dataset validation service for the Autonomous Maintenance
Supervisor Agent.

This module validates the generated supervisor training dataset before the
cleaning, feature-engineering, splitting, training, and evaluation stages.

Main responsibilities
---------------------
1. Confirm that the input CSV exists and is readable.
2. Normalize column names for validation.
3. Validate required metadata, model-output, feature, and target columns.
4. Validate unique engine-cycle records.
5. Validate target decision labels.
6. Validate numerical, Boolean, categorical, probability, and RUL fields.
7. Detect possible data leakage columns.
8. Detect missing values, infinite values, duplicates, and invalid ranges.
9. Save:
   - a detailed JSON validation report;
   - a CSV containing row-level validation issues.
10. Return a non-zero exit code when critical validation errors are found.

Recommended execution
---------------------
Run from the Backend directory:

    "C:\\Python313\\python.exe" -m \
    app.services.Maintenance_Supervisor.Data_preprocessing.data_validator
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# PROJECT PATH BOOTSTRAP
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# Current file:
# Backend/app/services/Maintenance_Supervisor/Data_preprocessing/data_validator.py
#
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
    from app.config import supervisor_config as config
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
# CONFIGURATION ACCESS WITH SAFE FALLBACKS
# =============================================================================

def config_value(name: str, default: Any) -> Any:
    """
    Return a value from supervisor_config with a safe fallback.

    This prevents the validator from crashing merely because an optional
    reporting-path constant has not yet been added to supervisor_config.py.
    """

    return getattr(config, name, default)


SAMPLE_DATASET_PATH = Path(
    config_value(
        "SAMPLE_DATASET_PATH",
        BACKEND_ROOT
        / "data"
        / "Maintenance_Supervisor"
        / "sample"
        / "supervisor_sample_dataset.csv",
    )
)

TARGET_COLUMN = str(
    config_value(
        "TARGET_COLUMN",
        "final_decision",
    )
)

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

DECISION_CLASSES = tuple(
    config_value(
        "DECISION_CLASSES",
        (
            "continue_operation",
            "monitor_closely",
            "schedule_inspection",
            "schedule_maintenance",
            "immediate_maintenance",
        ),
    )
)

METADATA_COLUMNS = tuple(
    config_value(
        "METADATA_COLUMNS",
        (
            ENGINE_ID_COLUMN,
            SUBSET_COLUMN,
            "cycle",
            SCHEMA_VERSION_COLUMN,
        ),
    )
)

BASE_INPUT_FEATURES = tuple(
    config_value(
        "BASE_INPUT_FEATURES",
        (),
    )
)

NUMERICAL_FEATURES = tuple(
    config_value(
        "NUMERICAL_FEATURES",
        (),
    )
)

BOOLEAN_FEATURES = tuple(
    config_value(
        "BOOLEAN_FEATURES",
        (),
    )
)

CATEGORICAL_FEATURES = tuple(
    config_value(
        "CATEGORICAL_FEATURES",
        (),
    )
)

MODEL_TRACE_COLUMNS = tuple(
    config_value(
        "MODEL_TRACE_COLUMNS",
        (),
    )
)

FORBIDDEN_FEATURE_COLUMNS = tuple(
    config_value(
        "FORBIDDEN_FEATURE_COLUMNS",
        (),
    )
)

REPORT_ROOT = Path(
    config_value(
        "DATA_VALIDATION_REPORT_ROOT",
        BACKEND_ROOT
        / "reports"
        / "Maintenance_Supervisor"
        / "data_validation",
    )
)

DATA_VALIDATION_REPORT_PATH = Path(
    config_value(
        "DATA_VALIDATION_REPORT_PATH",
        REPORT_ROOT / "dataset_validation_report.json",
    )
)

DATA_VALIDATION_ISSUES_PATH = Path(
    config_value(
        "DATA_VALIDATION_ISSUES_PATH",
        REPORT_ROOT / "dataset_validation_issues.csv",
    )
)


# =============================================================================
# VALIDATION CONSTANTS
# =============================================================================

EMPTY_TEXT_VALUES: frozenset[str] = frozenset(
    {
        "",
        " ",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "nil",
        "missing",
        "-",
        "--",
    }
)

TRUE_VALUES: frozenset[str] = frozenset(
    {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
        "enabled",
    }
)

FALSE_VALUES: frozenset[str] = frozenset(
    {
        "0",
        "false",
        "f",
        "no",
        "n",
        "off",
        "disabled",
    }
)

PROBABILITY_COLUMNS: tuple[str, ...] = (
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
    "failure_risk_score",
    "failure_probability",
    "risk_probability",
    "predicted_failure_risk",
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
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "predicted_rul",
    "lower_bound",
    "upper_bound",
    "uncertainty_width",
    "degradation_speed",
)

RUL_COLUMNS: tuple[str, ...] = (
    "predicted_rul",
    "lower_bound",
    "upper_bound",
    "uncertainty_width",
)

OPTIONAL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "priority",
    "maintenance_urgency",
    "requires_human_review",
    "supervisor_score",
)

TARGET_ALIASES: tuple[str, ...] = (
    TARGET_COLUMN,
    "supervisor_label",
    "supervisor_decision",
    "maintenance_decision",
    "recommended_action",
    "target",
)

TARGET_NORMALIZATION_MAP: dict[str, str] = {
    "continue": "continue_operation",
    "continue_operating": "continue_operation",
    "continue_operation": "continue_operation",
    "normal_operation": "continue_operation",
    "monitor": "monitor_closely",
    "monitor_close": "monitor_closely",
    "monitor_closely": "monitor_closely",
    "inspection": "schedule_inspection",
    "inspect": "schedule_inspection",
    "schedule_inspection": "schedule_inspection",
    "maintenance": "schedule_maintenance",
    "schedule_maintenance": "schedule_maintenance",
    "immediate": "immediate_maintenance",
    "immediate_maintenance": "immediate_maintenance",
    "emergency_maintenance": "immediate_maintenance",
    "stop_and_maintain": "immediate_maintenance",
}

COLUMN_ALIASES: dict[str, str] = {
    "engine": ENGINE_ID_COLUMN,
    "engineid": ENGINE_ID_COLUMN,
    "engine_identifier": ENGINE_ID_COLUMN,
    "unit_id": ENGINE_ID_COLUMN,
    "subset": SUBSET_COLUMN,
    "fdsubset": SUBSET_COLUMN,
    "cycle_number": "cycle",
    "cycle_id": "cycle",
    "decision": TARGET_COLUMN,
    "maintenance_decision": TARGET_COLUMN,
    "supervisor_decision": TARGET_COLUMN,
}

CRITICAL_MISSING_COLUMNS = {
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    "cycle",
    TARGET_COLUMN,
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(slots=True)
class ValidationIssue:
    """Represents one validation issue or issue category."""

    severity: str
    check_name: str
    column: str | None
    row_index: int | None
    value: Any
    message: str


@dataclass(slots=True)
class ValidationCounters:
    """Summary counters generated during validation."""

    rows: int = 0
    columns: int = 0

    missing_required_columns: list[str] = field(default_factory=list)
    missing_configured_columns: list[str] = field(default_factory=list)
    unexpected_columns: list[str] = field(default_factory=list)

    duplicate_rows: int = 0
    duplicate_key_rows: int = 0
    missing_key_rows: int = 0

    missing_target_rows: int = 0
    invalid_target_rows: int = 0

    missing_values_by_column: dict[str, int] = field(default_factory=dict)
    infinite_values_by_column: dict[str, int] = field(default_factory=dict)
    invalid_numeric_values_by_column: dict[str, int] = field(
        default_factory=dict
    )
    invalid_boolean_values_by_column: dict[str, int] = field(
        default_factory=dict
    )
    invalid_probability_values_by_column: dict[str, int] = field(
        default_factory=dict
    )
    negative_values_by_column: dict[str, int] = field(default_factory=dict)

    invalid_cycle_rows: int = 0
    invalid_rul_bound_rows: int = 0
    inconsistent_uncertainty_rows: int = 0
    risk_monotonicity_violations: int = 0

    possible_leakage_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)

    error_count: int = 0
    warning_count: int = 0
    info_count: int = 0


@dataclass(slots=True)
class ValidationResult:
    """Result returned by the validation pipeline."""

    status: str
    is_valid: bool
    input_path: str
    report_path: str
    issues_path: str
    rows: int
    columns: int
    error_count: int
    warning_count: int
    duration_seconds: float
    message: str


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def utc_now_iso() -> str:
    """Return the current UTC time in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def normalize_column_name(value: Any) -> str:
    """Normalize a column name to snake_case."""

    text = str(value).strip()
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def normalize_label(value: Any) -> str | None:
    """Normalize a categorical value to snake_case."""

    if pd.isna(value):
        return None

    text = str(value).strip()

    if text.lower() in EMPTY_TEXT_VALUES:
        return None

    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_").lower()

    return text or None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 hash of a file."""

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


def atomic_write_json(
    payload: dict[str, Any],
    destination: Path,
) -> None:
    """Write JSON atomically without corrupting an existing report."""

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


# =============================================================================
# DATASET VALIDATOR
# =============================================================================

class SupervisorDatasetValidator:
    """
    Validate a supervisor-agent dataset before data cleaning and training.

    Parameters
    ----------
    input_path:
        Dataset CSV to validate.
    report_path:
        JSON validation report destination.
    issues_path:
        CSV issue report destination.
    strict:
        When True, missing configured feature columns are errors.
        When False, only core missing columns are errors.
    maximum_issue_rows:
        Maximum number of detailed row-level issues stored in the issue CSV.
        Summary counters still include all detected issues.
    """

    def __init__(
        self,
        input_path: Path = SAMPLE_DATASET_PATH,
        report_path: Path = DATA_VALIDATION_REPORT_PATH,
        issues_path: Path = DATA_VALIDATION_ISSUES_PATH,
        *,
        strict: bool = False,
        maximum_issue_rows: int = 10_000,
    ) -> None:
        self.input_path = Path(input_path)
        self.report_path = Path(report_path)
        self.issues_path = Path(issues_path)

        self.strict = bool(strict)
        self.maximum_issue_rows = max(
            int(maximum_issue_rows),
            1,
        )

        self.counters = ValidationCounters()
        self.issues: list[ValidationIssue] = []

        self.started_at: str | None = None
        self.finished_at: str | None = None

        self.original_columns: list[str] = []
        self.normalized_columns: list[str] = []
        self.column_renames: dict[str, str] = {}

    # =========================================================================
    # ISSUE MANAGEMENT
    # =========================================================================

    def _add_issue(
        self,
        severity: str,
        check_name: str,
        message: str,
        *,
        column: str | None = None,
        row_index: int | None = None,
        value: Any = None,
    ) -> None:
        """Add a validation issue and update severity counters."""

        normalized_severity = severity.lower().strip()

        if normalized_severity == "error":
            self.counters.error_count += 1
        elif normalized_severity == "warning":
            self.counters.warning_count += 1
        else:
            normalized_severity = "info"
            self.counters.info_count += 1

        if len(self.issues) >= self.maximum_issue_rows:
            return

        self.issues.append(
            ValidationIssue(
                severity=normalized_severity,
                check_name=check_name,
                column=column,
                row_index=row_index,
                value=value,
                message=message,
            )
        )

    def _add_mask_issues(
        self,
        dataframe: pd.DataFrame,
        mask: pd.Series,
        *,
        severity: str,
        check_name: str,
        message: str,
        column: str | None = None,
        value_column: str | None = None,
        maximum_examples: int = 20,
    ) -> None:
        """Store examples from a Boolean validation mask."""

        invalid_indices = dataframe.index[
            mask.fillna(False)
        ].tolist()[:maximum_examples]

        for row_index in invalid_indices:
            issue_value: Any = None

            if value_column and value_column in dataframe.columns:
                issue_value = dataframe.at[
                    row_index,
                    value_column,
                ]
            elif column and column in dataframe.columns:
                issue_value = dataframe.at[
                    row_index,
                    column,
                ]

            self._add_issue(
                severity,
                check_name,
                message,
                column=column,
                row_index=int(row_index),
                value=issue_value,
            )

    # =========================================================================
    # INPUT LOADING
    # =========================================================================

    def _validate_input_path(self) -> None:
        """Validate the dataset file before reading it."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                "Supervisor dataset was not found: "
                f"{self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                "Supervisor dataset path is not a file: "
                f"{self.input_path}"
            )

        if self.input_path.suffix.lower() != ".csv":
            raise ValueError(
                "The validator currently expects a CSV file. "
                f"Received: {self.input_path.suffix or '<no extension>'}"
            )

        if self.input_path.stat().st_size <= 0:
            raise ValueError(
                f"Supervisor dataset is empty: {self.input_path}"
            )

    def _load_dataset(self) -> pd.DataFrame:
        """Load the input CSV into a DataFrame."""

        self._validate_input_path()

        logger.info(
            "Loading dataset: %s",
            self.input_path,
        )

        try:
            dataframe = pd.read_csv(
                self.input_path,
                low_memory=False,
            )

        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                "The input CSV contains no readable data."
            ) from exc

        except pd.errors.ParserError as exc:
            raise ValueError(
                "The input CSV is malformed and could not be parsed."
            ) from exc

        if dataframe.empty:
            raise ValueError(
                "The input CSV contains a header but no data records."
            )

        self.counters.rows = int(
            len(dataframe)
        )
        self.counters.columns = int(
            len(dataframe.columns)
        )

        self.original_columns = [
            str(column)
            for column in dataframe.columns
        ]

        logger.info(
            "Loaded rows=%d, columns=%d, memory=%.2f MB",
            len(dataframe),
            len(dataframe.columns),
            dataframe_memory_mb(dataframe),
        )

        return dataframe

    # =========================================================================
    # COLUMN NORMALIZATION
    # =========================================================================

    def _normalize_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize column names for reliable validation."""

        dataframe = dataframe.copy()

        normalized_columns: list[str] = []
        renames: dict[str, str] = {}

        for original_column in dataframe.columns:
            normalized = normalize_column_name(
                original_column
            )

            normalized = COLUMN_ALIASES.get(
                normalized,
                normalized,
            )

            normalized_columns.append(
                normalized
            )

            if str(original_column) != normalized:
                renames[str(original_column)] = normalized

        duplicated_normalized_names = [
            name
            for name, count in Counter(
                normalized_columns
            ).items()
            if count > 1
        ]

        if duplicated_normalized_names:
            self._add_issue(
                "error",
                "duplicate_column_names",
                "Column normalization produced duplicate names: "
                f"{duplicated_normalized_names}",
            )

            raise ValueError(
                "Duplicate column names were found after normalization: "
                f"{duplicated_normalized_names}"
            )

        dataframe.columns = normalized_columns

        self.normalized_columns = normalized_columns
        self.column_renames = renames

        return dataframe

    # =========================================================================
    # SCHEMA VALIDATION
    # =========================================================================

    def _resolve_target_column(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rename a known target alias to TARGET_COLUMN when required.
        """

        dataframe = dataframe.copy()

        if TARGET_COLUMN in dataframe.columns:
            return dataframe

        available_aliases = [
            alias
            for alias in TARGET_ALIASES
            if alias in dataframe.columns
            and alias != TARGET_COLUMN
        ]

        if len(available_aliases) == 1:
            alias = available_aliases[0]

            dataframe = dataframe.rename(
                columns={
                    alias: TARGET_COLUMN,
                }
            )

            self.column_renames[alias] = TARGET_COLUMN

            self._add_issue(
                "warning",
                "target_alias",
                f"Target alias '{alias}' was interpreted as "
                f"'{TARGET_COLUMN}'.",
                column=alias,
            )

        elif len(available_aliases) > 1:
            self._add_issue(
                "error",
                "ambiguous_target_columns",
                "Multiple possible target columns were found: "
                f"{available_aliases}",
            )

        return dataframe

    def _validate_required_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate required and configured columns."""

        available_columns = set(
            dataframe.columns
        )

        missing_core = sorted(
            CRITICAL_MISSING_COLUMNS.difference(
                available_columns
            )
        )

        self.counters.missing_required_columns = missing_core

        for column in missing_core:
            self._add_issue(
                "error",
                "missing_required_column",
                f"Required column '{column}' is missing.",
                column=column,
            )

        configured_columns = set(
            [
                *METADATA_COLUMNS,
                *BASE_INPUT_FEATURES,
                *NUMERICAL_FEATURES,
                *BOOLEAN_FEATURES,
                *CATEGORICAL_FEATURES,
                *MODEL_TRACE_COLUMNS,
                TARGET_COLUMN,
            ]
        )

        configured_columns.discard("")

        missing_configured = sorted(
            configured_columns.difference(
                available_columns
            )
        )

        self.counters.missing_configured_columns = (
            missing_configured
        )

        for column in missing_configured:
            if column in missing_core:
                continue

            severity = (
                "error"
                if self.strict
                else "warning"
            )

            self._add_issue(
                severity,
                "missing_configured_column",
                f"Configured column '{column}' is not present.",
                column=column,
            )

        expected_columns = configured_columns.union(
            OPTIONAL_OUTPUT_COLUMNS
        )

        unexpected_columns = sorted(
            available_columns.difference(
                expected_columns
            )
        )

        self.counters.unexpected_columns = unexpected_columns

        for column in unexpected_columns[:100]:
            self._add_issue(
                "info",
                "unexpected_column",
                f"Column '{column}' is not listed in the configured schema.",
                column=column,
            )

    # =========================================================================
    # BASIC DATA QUALITY
    # =========================================================================

    def _validate_missing_values(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Count missing values for every column."""

        missing_counts = dataframe.isna().sum()

        for column, count in missing_counts.items():
            count = int(count)

            if count <= 0:
                continue

            self.counters.missing_values_by_column[
                column
            ] = count

            severity = (
                "error"
                if column in CRITICAL_MISSING_COLUMNS
                else "warning"
            )

            self._add_issue(
                severity,
                "missing_values",
                f"Column '{column}' contains {count} missing values.",
                column=column,
                value=count,
            )

    def _validate_duplicate_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Detect exact duplicate rows."""

        duplicate_mask = dataframe.duplicated(
            keep=False
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        self.counters.duplicate_rows = duplicate_count

        if duplicate_count:
            self._add_issue(
                "warning",
                "duplicate_rows",
                f"The dataset contains {duplicate_count} rows "
                "participating in exact duplicates.",
                value=duplicate_count,
            )

            self._add_mask_issues(
                dataframe,
                duplicate_mask,
                severity="warning",
                check_name="duplicate_row_example",
                message="This row participates in an exact duplicate.",
                maximum_examples=20,
            )

    def _validate_unique_keys(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate engine, subset, and cycle keys."""

        available_keys = [
            column
            for column in UNIQUE_KEY_COLUMNS
            if column in dataframe.columns
        ]

        if len(available_keys) != len(
            UNIQUE_KEY_COLUMNS
        ):
            return

        missing_key_mask = dataframe[
            available_keys
        ].isna().any(axis=1)

        for column in (
            ENGINE_ID_COLUMN,
            SUBSET_COLUMN,
        ):
            if column not in dataframe.columns:
                continue

            normalized = (
                dataframe[column]
                .astype("string")
                .str.strip()
                .str.lower()
            )

            missing_key_mask |= normalized.isin(
                EMPTY_TEXT_VALUES
            )

        missing_key_count = int(
            missing_key_mask.sum()
        )

        self.counters.missing_key_rows = (
            missing_key_count
        )

        if missing_key_count:
            self._add_issue(
                "error",
                "missing_unique_key",
                f"{missing_key_count} rows contain missing unique-key values.",
                value=missing_key_count,
            )

            self._add_mask_issues(
                dataframe,
                missing_key_mask,
                severity="error",
                check_name="missing_unique_key_example",
                message="This row contains a missing unique-key value.",
                maximum_examples=20,
            )

        duplicate_key_mask = dataframe.duplicated(
            subset=available_keys,
            keep=False,
        )

        duplicate_key_count = int(
            duplicate_key_mask.sum()
        )

        self.counters.duplicate_key_rows = (
            duplicate_key_count
        )

        if duplicate_key_count:
            self._add_issue(
                "error",
                "duplicate_unique_key",
                f"{duplicate_key_count} rows participate in duplicate "
                f"{tuple(available_keys)} keys.",
                value=duplicate_key_count,
            )

            self._add_mask_issues(
                dataframe,
                duplicate_key_mask,
                severity="error",
                check_name="duplicate_unique_key_example",
                message=(
                    "This row has a duplicate engine/subset/cycle key."
                ),
                maximum_examples=20,
            )

    # =========================================================================
    # TARGET VALIDATION
    # =========================================================================

    def _validate_target(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate supervisor decision labels."""

        if TARGET_COLUMN not in dataframe.columns:
            return

        original_target = dataframe[
            TARGET_COLUMN
        ]

        normalized_target = original_target.map(
            normalize_label
        )

        normalized_target = normalized_target.map(
            lambda value: (
                TARGET_NORMALIZATION_MAP.get(
                    value,
                    value,
                )
                if value is not None
                else None
            )
        )

        missing_mask = pd.Series(
            normalized_target,
            index=dataframe.index,
        ).isna()

        missing_count = int(
            missing_mask.sum()
        )

        self.counters.missing_target_rows = (
            missing_count
        )

        if missing_count:
            self._add_issue(
                "error",
                "missing_target",
                f"{missing_count} rows have no valid target value.",
                column=TARGET_COLUMN,
                value=missing_count,
            )

        valid_decisions = set(
            DECISION_CLASSES
        )

        invalid_mask = (
            ~missing_mask
            & ~pd.Series(
                normalized_target,
                index=dataframe.index,
            ).isin(valid_decisions)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        self.counters.invalid_target_rows = (
            invalid_count
        )

        if invalid_count:
            invalid_values = sorted(
                {
                    str(value)
                    for value in pd.Series(
                        normalized_target,
                        index=dataframe.index,
                    )[invalid_mask].dropna().unique()
                }
            )

            self._add_issue(
                "error",
                "invalid_target",
                f"{invalid_count} rows contain unsupported target labels: "
                f"{invalid_values}",
                column=TARGET_COLUMN,
                value=invalid_values,
            )

            self._add_mask_issues(
                dataframe,
                invalid_mask,
                severity="error",
                check_name="invalid_target_example",
                message="This row has an unsupported target label.",
                column=TARGET_COLUMN,
                maximum_examples=20,
            )

    # =========================================================================
    # NUMERICAL VALIDATION
    # =========================================================================

    def _numeric_candidate_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> list[str]:
        """Return all configured numerical columns that exist."""

        candidates = [
            *NUMERICAL_FEATURES,
            "cycle",
            "supervisor_score",
        ]

        return [
            column
            for column in dict.fromkeys(candidates)
            if column in dataframe.columns
        ]

    def _validate_numeric_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate numerical conversion and infinite values."""

        for column in self._numeric_candidate_columns(
            dataframe
        ):
            original = dataframe[column]

            converted = pd.to_numeric(
                original,
                errors="coerce",
            )

            original_non_missing = (
                original.notna()
                & ~original.astype("string")
                .str.strip()
                .str.lower()
                .isin(EMPTY_TEXT_VALUES)
            )

            invalid_numeric_mask = (
                original_non_missing
                & converted.isna()
            )

            invalid_numeric_count = int(
                invalid_numeric_mask.sum()
            )

            if invalid_numeric_count:
                self.counters.invalid_numeric_values_by_column[
                    column
                ] = invalid_numeric_count

                self._add_issue(
                    "error",
                    "invalid_numeric_value",
                    f"Column '{column}' contains "
                    f"{invalid_numeric_count} non-numeric values.",
                    column=column,
                    value=invalid_numeric_count,
                )

                self._add_mask_issues(
                    dataframe,
                    invalid_numeric_mask,
                    severity="error",
                    check_name="invalid_numeric_example",
                    message=(
                        f"Column '{column}' contains a value that "
                        "cannot be converted to a number."
                    ),
                    column=column,
                    maximum_examples=10,
                )

            finite_values = converted.to_numpy(
                dtype=float,
                na_value=np.nan,
            )

            infinite_mask_array = np.isinf(
                finite_values
            )

            infinite_count = int(
                infinite_mask_array.sum()
            )

            if infinite_count:
                self.counters.infinite_values_by_column[
                    column
                ] = infinite_count

                self._add_issue(
                    "error",
                    "infinite_numeric_value",
                    f"Column '{column}' contains "
                    f"{infinite_count} infinite values.",
                    column=column,
                    value=infinite_count,
                )

    def _validate_cycle(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate positive integer cycle numbers."""

        if "cycle" not in dataframe.columns:
            return

        cycle = pd.to_numeric(
            dataframe["cycle"],
            errors="coerce",
        )

        invalid_mask = (
            cycle.isna()
            | ~np.isfinite(cycle)
            | (cycle < 1)
            | ~np.isclose(
                cycle.fillna(0.0),
                np.round(
                    cycle.fillna(0.0)
                ),
            )
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        self.counters.invalid_cycle_rows = (
            invalid_count
        )

        if invalid_count:
            self._add_issue(
                "error",
                "invalid_cycle",
                f"{invalid_count} rows contain invalid cycle values. "
                "Cycle must be a positive integer.",
                column="cycle",
                value=invalid_count,
            )

            self._add_mask_issues(
                dataframe,
                invalid_mask,
                severity="error",
                check_name="invalid_cycle_example",
                message="Cycle must be a positive integer.",
                column="cycle",
                maximum_examples=20,
            )

    # =========================================================================
    # BOOLEAN VALIDATION
    # =========================================================================

    @staticmethod
    def _is_valid_boolean(value: Any) -> bool:
        """Return True when a value is Boolean-compatible."""

        if pd.isna(value):
            return True

        if isinstance(value, (bool, np.bool_)):
            return True

        if isinstance(value, (int, np.integer)):
            return int(value) in {0, 1}

        if isinstance(value, (float, np.floating)):
            return (
                np.isfinite(value)
                and float(value) in {0.0, 1.0}
            )

        normalized = str(value).strip().lower()

        return normalized in (
            TRUE_VALUES
            | FALSE_VALUES
            | EMPTY_TEXT_VALUES
        )

    def _validate_boolean_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate configured Boolean columns."""

        candidates = [
            *BOOLEAN_FEATURES,
            "requires_human_review",
        ]

        for column in dict.fromkeys(candidates):
            if column not in dataframe.columns:
                continue

            valid_mask = dataframe[column].map(
                self._is_valid_boolean
            )

            invalid_mask = ~valid_mask
            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count:
                self.counters.invalid_boolean_values_by_column[
                    column
                ] = invalid_count

                self._add_issue(
                    "error",
                    "invalid_boolean_value",
                    f"Column '{column}' contains "
                    f"{invalid_count} invalid Boolean values.",
                    column=column,
                    value=invalid_count,
                )

                self._add_mask_issues(
                    dataframe,
                    invalid_mask,
                    severity="error",
                    check_name="invalid_boolean_example",
                    message=(
                        f"Column '{column}' contains an unsupported "
                        "Boolean representation."
                    ),
                    column=column,
                    maximum_examples=10,
                )

    # =========================================================================
    # DOMAIN VALIDATION
    # =========================================================================

    def _validate_probability_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate probabilities, scores, confidence, and thresholds."""

        for column in PROBABILITY_COLUMNS:
            if column not in dataframe.columns:
                continue

            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            invalid_mask = (
                converted.notna()
                & (
                    ~np.isfinite(converted)
                    | (converted < 0.0)
                    | (converted > 1.0)
                )
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count:
                self.counters.invalid_probability_values_by_column[
                    column
                ] = invalid_count

                self._add_issue(
                    "error",
                    "invalid_probability_range",
                    f"Column '{column}' contains {invalid_count} values "
                    "outside the valid [0, 1] range.",
                    column=column,
                    value=invalid_count,
                )

                self._add_mask_issues(
                    dataframe,
                    invalid_mask,
                    severity="error",
                    check_name="invalid_probability_example",
                    message=(
                        f"Column '{column}' must remain between 0 and 1."
                    ),
                    column=column,
                    maximum_examples=10,
                )

    def _validate_non_negative_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate physical values that cannot be negative."""

        for column in NON_NEGATIVE_COLUMNS:
            if column not in dataframe.columns:
                continue

            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            negative_mask = (
                converted.notna()
                & (converted < 0.0)
            )

            negative_count = int(
                negative_mask.sum()
            )

            if negative_count:
                self.counters.negative_values_by_column[
                    column
                ] = negative_count

                self._add_issue(
                    "error",
                    "negative_value",
                    f"Column '{column}' contains "
                    f"{negative_count} negative values.",
                    column=column,
                    value=negative_count,
                )

    def _validate_rul_intervals(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Validate lower/upper RUL bounds and uncertainty width."""

        required = {
            "lower_bound",
            "upper_bound",
        }

        if not required.issubset(
            dataframe.columns
        ):
            return

        lower = pd.to_numeric(
            dataframe["lower_bound"],
            errors="coerce",
        )

        upper = pd.to_numeric(
            dataframe["upper_bound"],
            errors="coerce",
        )

        reversed_mask = (
            lower.notna()
            & upper.notna()
            & (lower > upper)
        )

        reversed_count = int(
            reversed_mask.sum()
        )

        self.counters.invalid_rul_bound_rows = (
            reversed_count
        )

        if reversed_count:
            self._add_issue(
                "error",
                "invalid_rul_bounds",
                f"{reversed_count} rows have lower_bound > upper_bound.",
                value=reversed_count,
            )

            self._add_mask_issues(
                dataframe,
                reversed_mask,
                severity="error",
                check_name="invalid_rul_bounds_example",
                message="lower_bound cannot be greater than upper_bound.",
                maximum_examples=20,
            )

        if "uncertainty_width" not in dataframe.columns:
            return

        existing_width = pd.to_numeric(
            dataframe["uncertainty_width"],
            errors="coerce",
        )

        calculated_width = upper - lower

        comparable_mask = (
            existing_width.notna()
            & calculated_width.notna()
        )

        inconsistent_mask = (
            comparable_mask
            & ~np.isclose(
                existing_width.fillna(0.0),
                calculated_width.fillna(0.0),
                rtol=1e-5,
                atol=1e-6,
            )
        )

        inconsistent_count = int(
            inconsistent_mask.sum()
        )

        self.counters.inconsistent_uncertainty_rows = (
            inconsistent_count
        )

        if inconsistent_count:
            self._add_issue(
                "warning",
                "inconsistent_uncertainty_width",
                f"{inconsistent_count} rows have uncertainty_width "
                "different from upper_bound - lower_bound.",
                column="uncertainty_width",
                value=inconsistent_count,
            )

    def _validate_risk_monotonicity(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """
        Validate that longer-horizon failure risk is not lower than
        shorter-horizon failure risk.

        Expected relationship:

            risk_10 <= risk_30 <= risk_50
        """

        required = {
            "risk_10",
            "risk_30",
            "risk_50",
        }

        if not required.issubset(
            dataframe.columns
        ):
            return

        risk_10 = pd.to_numeric(
            dataframe["risk_10"],
            errors="coerce",
        )
        risk_30 = pd.to_numeric(
            dataframe["risk_30"],
            errors="coerce",
        )
        risk_50 = pd.to_numeric(
            dataframe["risk_50"],
            errors="coerce",
        )

        complete_mask = (
            risk_10.notna()
            & risk_30.notna()
            & risk_50.notna()
        )

        violation_mask = (
            complete_mask
            & (
                (risk_10 > risk_30)
                | (risk_30 > risk_50)
            )
        )

        violation_count = int(
            violation_mask.sum()
        )

        self.counters.risk_monotonicity_violations = (
            violation_count
        )

        if violation_count:
            self._add_issue(
                "warning",
                "risk_horizon_monotonicity",
                f"{violation_count} rows violate "
                "risk_10 <= risk_30 <= risk_50.",
                value=violation_count,
            )

            self._add_mask_issues(
                dataframe,
                violation_mask,
                severity="warning",
                check_name="risk_monotonicity_example",
                message=(
                    "Longer-horizon failure risk is unexpectedly lower "
                    "than shorter-horizon risk."
                ),
                maximum_examples=20,
            )

    # =========================================================================
    # LEAKAGE AND DISTRIBUTION CHECKS
    # =========================================================================

    def _validate_leakage_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Detect prohibited future-information columns."""

        permitted_outputs = {
            TARGET_COLUMN,
            *OPTIONAL_OUTPUT_COLUMNS,
        }

        possible_leakage = sorted(
            column
            for column in FORBIDDEN_FEATURE_COLUMNS
            if column in dataframe.columns
            and column not in permitted_outputs
        )

        self.counters.possible_leakage_columns = (
            possible_leakage
        )

        for column in possible_leakage:
            self._add_issue(
                "warning",
                "possible_data_leakage",
                f"Potential leakage column '{column}' is present. "
                "It must not be included in model input features.",
                column=column,
            )

    def _validate_constant_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Detect columns with zero or one distinct non-missing value."""

        constant_columns: list[str] = []

        for column in dataframe.columns:
            distinct_count = int(
                dataframe[column].nunique(
                    dropna=True
                )
            )

            if distinct_count <= 1:
                constant_columns.append(
                    column
                )

        self.counters.constant_columns = (
            constant_columns
        )

        for column in constant_columns:
            severity = (
                "info"
                if column == SCHEMA_VERSION_COLUMN
                else "warning"
            )

            self._add_issue(
                severity,
                "constant_column",
                f"Column '{column}' has no meaningful variation.",
                column=column,
            )

    def _validate_target_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Check whether all configured decision classes are represented."""

        if TARGET_COLUMN not in dataframe.columns:
            return

        normalized = dataframe[
            TARGET_COLUMN
        ].map(normalize_label)

        normalized = normalized.map(
            lambda value: (
                TARGET_NORMALIZATION_MAP.get(
                    value,
                    value,
                )
                if value is not None
                else None
            )
        )

        present_classes = set(
            pd.Series(normalized)
            .dropna()
            .unique()
            .tolist()
        )

        missing_classes = sorted(
            set(DECISION_CLASSES).difference(
                present_classes
            )
        )

        if missing_classes:
            self._add_issue(
                "warning",
                "missing_target_classes",
                "The dataset does not contain all configured decision "
                f"classes. Missing: {missing_classes}",
                column=TARGET_COLUMN,
                value=missing_classes,
            )

    # =========================================================================
    # REPORT CREATION
    # =========================================================================

    def _issues_dataframe(self) -> pd.DataFrame:
        """Convert collected issues into a DataFrame."""

        columns = [
            "severity",
            "check_name",
            "column",
            "row_index",
            "value",
            "message",
        ]

        if not self.issues:
            return pd.DataFrame(
                columns=columns
            )

        return pd.DataFrame(
            [
                asdict(issue)
                for issue in self.issues
            ],
            columns=columns,
        )

    def _target_distribution(
        self,
        dataframe: pd.DataFrame,
    ) -> dict[str, int]:
        """Return normalized target counts."""

        if TARGET_COLUMN not in dataframe.columns:
            return {}

        normalized = dataframe[
            TARGET_COLUMN
        ].map(normalize_label)

        normalized = normalized.map(
            lambda value: (
                TARGET_NORMALIZATION_MAP.get(
                    value,
                    value,
                )
                if value is not None
                else None
            )
        )

        counts = (
            pd.Series(normalized)
            .fillna("<missing>")
            .value_counts(dropna=False)
            .sort_index()
        )

        return {
            str(label): int(count)
            for label, count in counts.items()
        }

    def _build_report(
        self,
        dataframe: pd.DataFrame,
        *,
        duration_seconds: float,
        input_hash: str,
    ) -> dict[str, Any]:
        """Build the detailed JSON validation report."""

        status = (
            "valid"
            if self.counters.error_count == 0
            else "invalid"
        )

        numeric_summary: dict[str, dict[str, Any]] = {}

        for column in self._numeric_candidate_columns(
            dataframe
        ):
            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            numeric_summary[column] = {
                "count": int(
                    converted.notna().sum()
                ),
                "missing": int(
                    converted.isna().sum()
                ),
                "minimum": (
                    float(converted.min())
                    if converted.notna().any()
                    else None
                ),
                "maximum": (
                    float(converted.max())
                    if converted.notna().any()
                    else None
                ),
                "mean": (
                    float(converted.mean())
                    if converted.notna().any()
                    else None
                ),
                "median": (
                    float(converted.median())
                    if converted.notna().any()
                    else None
                ),
                "standard_deviation": (
                    float(converted.std())
                    if converted.notna().sum() > 1
                    else None
                ),
            }

        return {
            "status": status,
            "is_valid": (
                self.counters.error_count == 0
            ),
            "component": str(
                config_value(
                    "COMPONENT_NAME",
                    config_value(
                        "PROJECT_NAME",
                        "Autonomous Maintenance Supervisor Agent",
                    ),
                )
            ),
            "stage": "data_validation",
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
                "size_bytes": int(
                    self.input_path.stat().st_size
                ),
                "sha256": input_hash,
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
            },
            "outputs": {
                "report_path": str(
                    self.report_path.resolve()
                ),
                "issues_path": str(
                    self.issues_path.resolve()
                ),
            },
            "configuration": {
                "strict_validation": self.strict,
                "target_column": TARGET_COLUMN,
                "unique_key_columns": list(
                    UNIQUE_KEY_COLUMNS
                ),
                "decision_classes": list(
                    DECISION_CLASSES
                ),
                "maximum_issue_rows": (
                    self.maximum_issue_rows
                ),
            },
            "schema": {
                "original_columns": self.original_columns,
                "normalized_columns": list(
                    dataframe.columns
                ),
                "column_renames": self.column_renames,
            },
            "summary": asdict(
                self.counters
            ),
            "target_distribution": self._target_distribution(
                dataframe
            ),
            "numeric_summary": numeric_summary,
            "column_profiles": [
                {
                    "name": column,
                    "dtype": str(
                        dataframe[column].dtype
                    ),
                    "missing_count": int(
                        dataframe[column].isna().sum()
                    ),
                    "non_missing_count": int(
                        dataframe[column].notna().sum()
                    ),
                    "unique_count": int(
                        dataframe[column].nunique(
                            dropna=True
                        )
                    ),
                }
                for column in dataframe.columns
            ],
            "issue_storage": {
                "total_issue_messages": (
                    self.counters.error_count
                    + self.counters.warning_count
                    + self.counters.info_count
                ),
                "stored_issue_rows": len(
                    self.issues
                ),
                "storage_limit": (
                    self.maximum_issue_rows
                ),
            },
        }

    # =========================================================================
    # FULL VALIDATION PIPELINE
    # =========================================================================

    def validate_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Execute all DataFrame validation checks."""

        logger.info(
            "Normalizing columns."
        )

        dataframe = self._normalize_columns(
            dataframe
        )

        dataframe = self._resolve_target_column(
            dataframe
        )

        logger.info(
            "Validating schema."
        )

        self._validate_required_columns(
            dataframe
        )

        logger.info(
            "Validating missing values and duplicates."
        )

        self._validate_missing_values(
            dataframe
        )
        self._validate_duplicate_rows(
            dataframe
        )
        self._validate_unique_keys(
            dataframe
        )

        logger.info(
            "Validating target labels."
        )

        self._validate_target(
            dataframe
        )
        self._validate_target_distribution(
            dataframe
        )

        logger.info(
            "Validating numerical and Boolean fields."
        )

        self._validate_numeric_columns(
            dataframe
        )
        self._validate_cycle(
            dataframe
        )
        self._validate_boolean_columns(
            dataframe
        )

        logger.info(
            "Validating probability, risk, anomaly, and RUL fields."
        )

        self._validate_probability_columns(
            dataframe
        )
        self._validate_non_negative_columns(
            dataframe
        )
        self._validate_rul_intervals(
            dataframe
        )
        self._validate_risk_monotonicity(
            dataframe
        )

        logger.info(
            "Checking leakage and constant columns."
        )

        self._validate_leakage_columns(
            dataframe
        )
        self._validate_constant_columns(
            dataframe
        )

        return dataframe

    def run(self) -> ValidationResult:
        """Run validation and write JSON/CSV reports atomically."""

        timer_start = perf_counter()
        self.started_at = utc_now_iso()

        logger.info("=" * 78)
        logger.info("SUPERVISOR DATASET VALIDATION STARTED")
        logger.info("=" * 78)
        logger.info("Backend root : %s", BACKEND_ROOT)
        logger.info("Input dataset: %s", self.input_path)
        logger.info("JSON report  : %s", self.report_path)
        logger.info("Issues CSV   : %s", self.issues_path)

        try:
            dataframe = self._load_dataset()
            input_hash = sha256_file(
                self.input_path
            )

            dataframe = self.validate_dataframe(
                dataframe
            )

            self.finished_at = utc_now_iso()
            duration_seconds = (
                perf_counter() - timer_start
            )

            issues_dataframe = self._issues_dataframe()

            report = self._build_report(
                dataframe,
                duration_seconds=duration_seconds,
                input_hash=input_hash,
            )

            logger.info(
                "Writing validation reports atomically."
            )

            atomic_write_csv(
                issues_dataframe,
                self.issues_path,
            )

            atomic_write_json(
                report,
                self.report_path,
            )

            is_valid = (
                self.counters.error_count == 0
            )

            result = ValidationResult(
                status=(
                    "success"
                    if is_valid
                    else "validation_failed"
                ),
                is_valid=is_valid,
                input_path=str(
                    self.input_path.resolve()
                ),
                report_path=str(
                    self.report_path.resolve()
                ),
                issues_path=str(
                    self.issues_path.resolve()
                ),
                rows=int(
                    len(dataframe)
                ),
                columns=int(
                    len(dataframe.columns)
                ),
                error_count=(
                    self.counters.error_count
                ),
                warning_count=(
                    self.counters.warning_count
                ),
                duration_seconds=round(
                    duration_seconds,
                    6,
                ),
                message=(
                    "Dataset validation passed."
                    if is_valid
                    else (
                        "Dataset validation completed, but critical "
                        "validation errors were detected."
                    )
                ),
            )

            logger.info("-" * 78)
            logger.info(
                "Rows                 : %d",
                result.rows,
            )
            logger.info(
                "Columns              : %d",
                result.columns,
            )
            logger.info(
                "Validation errors    : %d",
                result.error_count,
            )
            logger.info(
                "Validation warnings  : %d",
                result.warning_count,
            )
            logger.info(
                "Duplicate rows       : %d",
                self.counters.duplicate_rows,
            )
            logger.info(
                "Duplicate key rows   : %d",
                self.counters.duplicate_key_rows,
            )
            logger.info(
                "Missing target rows  : %d",
                self.counters.missing_target_rows,
            )
            logger.info(
                "Invalid target rows  : %d",
                self.counters.invalid_target_rows,
            )
            logger.info(
                "Duration             : %.3f seconds",
                result.duration_seconds,
            )
            logger.info("=" * 78)

            if result.is_valid:
                logger.info(
                    "SUPERVISOR DATASET VALIDATION PASSED"
                )
            else:
                logger.error(
                    "SUPERVISOR DATASET VALIDATION FAILED"
                )

            logger.info("=" * 78)

            return result

        except Exception:
            self.finished_at = utc_now_iso()

            logger.exception(
                "Supervisor dataset validation crashed. "
                "Existing reports were not intentionally deleted."
            )

            raise


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate the Autonomous Maintenance Supervisor dataset."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=SAMPLE_DATASET_PATH,
        help=(
            "Input supervisor CSV path. "
            f"Default: {SAMPLE_DATASET_PATH}"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=DATA_VALIDATION_REPORT_PATH,
        help=(
            "JSON validation report destination. "
            f"Default: {DATA_VALIDATION_REPORT_PATH}"
        ),
    )

    parser.add_argument(
        "--issues",
        type=Path,
        default=DATA_VALIDATION_ISSUES_PATH,
        help=(
            "CSV validation-issues destination. "
            f"Default: {DATA_VALIDATION_ISSUES_PATH}"
        ),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Treat every missing configured feature column as an error."
        ),
    )

    parser.add_argument(
        "--maximum-issue-rows",
        type=int,
        default=10_000,
        help=(
            "Maximum number of detailed issue rows stored in the CSV. "
            "Default: 10000."
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

    validator = SupervisorDatasetValidator(
        input_path=parsed.input,
        report_path=parsed.report,
        issues_path=parsed.issues,
        strict=parsed.strict,
        maximum_issue_rows=(
            parsed.maximum_issue_rows
        ),
    )

    try:
        result = validator.run()

        print(
            json.dumps(
                asdict(result),
                indent=2,
            )
        )

        return (
            0
            if result.is_valid
            else 2
        )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        PermissionError,
        OSError,
    ) as exc:
        logger.error(
            "Dataset validator failed: %s",
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
            "Unexpected dataset-validation error."
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