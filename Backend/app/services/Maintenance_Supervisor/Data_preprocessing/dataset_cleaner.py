"""
dataset_cleaner.py

Production-grade dataset-cleaning service for the Autonomous Maintenance
Supervisor Agent.

Responsibilities
----------------
1. Load the validated supervisor dataset.
2. Normalize column names and remove accidental CSV index columns.
3. Standardize missing-value representations.
4. Convert numeric, Boolean and categorical columns safely.
5. Correct RUL interval inconsistencies.
6. Clip probability and confidence values to valid ranges.
7. Normalize target and categorical labels.
8. Remove duplicate rows and duplicate engine-cycle records.
9. Validate final output schema.
10. Save the cleaned dataset and a detailed JSON cleaning report atomically.

Important
---------
- This cleaner does not split the dataset.
- This cleaner does not perform feature engineering.
- This cleaner does not fit a machine-learning model.
- The target column is preserved.
- Future-information and leakage columns are removed, except the target.
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
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


# =============================================================================
# STANDALONE EXECUTION SUPPORT
# =============================================================================

CURRENT_FILE = Path(__file__).resolve()

# Current file:
# Backend/app/services/Maintenance_Supervisor/Data_Preprocessing/dataset_cleaner.py
#
# parents[0] -> Data_Preprocessing
# parents[1] -> Maintenance_Supervisor
# parents[2] -> services
# parents[3] -> app
# parents[4] -> Backend
BACKEND_ROOT = CURRENT_FILE.parents[4]

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    ANOMALY_BOOLEAN_FEATURES,
    ANOMALY_CATEGORICAL_FEATURES,
    ANOMALY_NUMERICAL_FEATURES,
    BASE_INPUT_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
    CLEANED_DATASET_PATH,
    CLEANING_REPORT_PATH,
    CONFIG,
    DECISION_CLASSES,
    ENGINE_ID_COLUMN,
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_COLUMNS,
    MODEL_TRACE_COLUMNS,
    NUMERICAL_FEATURES,
    RISK_BOOLEAN_FEATURES,
    RISK_CATEGORICAL_FEATURES,
    RISK_NUMERICAL_FEATURES,
    RUL_BOOLEAN_FEATURES,
    RUL_CATEGORICAL_FEATURES,
    RUL_NUMERICAL_FEATURES,
    SAMPLE_DATASET_PATH,
    SCHEMA_VERSION_COLUMN,
    SUBSET_COLUMN,
    TARGET_COLUMN,
    UNIQUE_KEY_COLUMNS,
    initialize_supervisor_environment,
)


# =============================================================================
# OPTIONAL PROJECT LOGGER
# =============================================================================

try:
    from app.utils.Maintenance_Supervisor.logger import get_logger

    try:
        logger = get_logger(__name__)
    except TypeError:
        logger = get_logger()

except (ImportError, ModuleNotFoundError):
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        ),
    )
    logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
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
        "unknown_value",
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

ACCIDENTAL_INDEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^unnamed:\s*\d+$", re.IGNORECASE),
    re.compile(r"^level_0$", re.IGNORECASE),
    re.compile(r"^index$", re.IGNORECASE),
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
    "anomaly_score",
    "anomaly_threshold",
    "residual_anomaly_score",
    "isolation_forest_score",
    "mahalanobis_score",
    "top_sensor_1_score",
    "top_sensor_2_score",
    "top_sensor_3_score",
    "context_confidence",
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "predicted_rul",
    "lower_bound",
    "upper_bound",
    "uncertainty_width",
    "degradation_speed",
)

PRESERVED_OUTPUT_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    "priority",
    "maintenance_urgency",
    "requires_human_review",
    "supervisor_score",
)

KNOWN_CATEGORICAL_DEFAULTS: dict[str, str] = {
    "lifecycle_state": "unknown",
    "risk_state": "unknown",
    "action_hint_for_supervisor": "unknown",
    "anomaly_severity": "unknown",
    "top_sensor_1": "unknown",
    "top_sensor_2": "unknown",
    "top_sensor_3": "unknown",
    "rul_model_used": "unknown",
    "risk_model_used": "unknown",
    "anomaly_model_used": "unknown",
    TARGET_COLUMN: "unknown",
    "priority": "unknown",
    "maintenance_urgency": "unknown",
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

TARGET_NORMALIZATION_MAP: dict[str, str] = {
    "continue": "continue_operation",
    "continue_operating": "continue_operation",
    "continue_operation": "continue_operation",
    "normal_operation": "continue_operation",
    "monitor": "monitor_closely",
    "monitor_close": "monitor_closely",
    "monitor_closely": "monitor_closely",
    "schedule_inspection": "schedule_inspection",
    "inspection": "schedule_inspection",
    "inspect": "schedule_inspection",
    "schedule_maintenance": "schedule_maintenance",
    "maintenance": "schedule_maintenance",
    "immediate": "immediate_maintenance",
    "immediate_maintenance": "immediate_maintenance",
    "stop_and_maintain": "immediate_maintenance",
    "emergency_maintenance": "immediate_maintenance",
}

PRIORITY_NORMALIZATION_MAP: dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "moderate": "Medium",
    "high": "High",
    "critical": "Critical",
}

ANOMALY_SEVERITY_NORMALIZATION_MAP: dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "moderate": "Medium",
    "high": "High",
    "critical": "Critical",
}

SCHEMA_VERSION_DEFAULT = "1.0"


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass(slots=True)
class CleaningCounters:
    """Mutable counters collected during dataset cleaning."""

    original_rows: int = 0
    original_columns: int = 0
    final_rows: int = 0
    final_columns: int = 0

    duplicate_rows_removed: int = 0
    duplicate_key_rows_removed: int = 0
    rows_removed_missing_keys: int = 0
    rows_removed_invalid_target: int = 0

    accidental_index_columns_removed: list[str] = field(
        default_factory=list
    )
    renamed_columns: dict[str, str] = field(default_factory=dict)
    leakage_columns_removed: list[str] = field(default_factory=list)

    numeric_values_coerced_to_missing: dict[str, int] = field(
        default_factory=dict
    )
    numeric_values_imputed: dict[str, int] = field(default_factory=dict)
    boolean_values_imputed: dict[str, int] = field(default_factory=dict)
    categorical_values_imputed: dict[str, int] = field(default_factory=dict)
    invalid_boolean_values: dict[str, int] = field(default_factory=dict)

    probability_values_clipped: dict[str, int] = field(
        default_factory=dict
    )
    non_negative_values_corrected: dict[str, int] = field(
        default_factory=dict
    )

    rul_bounds_swapped: int = 0
    uncertainty_width_recalculated: int = 0
    invalid_cycles_corrected: int = 0

    categorical_values_normalized: dict[str, int] = field(
        default_factory=dict
    )


@dataclass(slots=True)
class CleaningResult:
    """Result returned by the dataset-cleaning service."""

    status: str
    input_path: str
    output_path: str
    report_path: str
    rows_before: int
    rows_after: int
    columns_before: int
    columns_after: int
    duration_seconds: float
    output_sha256: str
    message: str


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


def normalize_column_name(value: Any) -> str:
    """Convert a column name into normalized snake_case."""

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

    return text.strip("_").lower() or None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def dataframe_memory_mb(dataframe: pd.DataFrame) -> float:
    """Return DataFrame memory usage in megabytes."""

    bytes_used = int(
        dataframe.memory_usage(
            index=True,
            deep=True,
        ).sum()
    )

    return bytes_used / (1024 ** 2)


def atomic_write_csv(
    dataframe: pd.DataFrame,
    destination: Path,
) -> None:
    """
    Write a DataFrame atomically.

    The existing destination remains untouched if writing fails.
    """

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
    """Write a JSON object atomically."""

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
# DATASET CLEANER
# =============================================================================

class SupervisorDatasetCleaner:
    """
    Clean and normalize Autonomous Maintenance Supervisor datasets.

    Parameters
    ----------
    input_path:
        Source CSV path.
    output_path:
        Destination for the cleaned CSV.
    report_path:
        Destination for the cleaning report.
    remove_invalid_targets:
        Whether rows with missing or unsupported target labels should be
        removed. For supervised training datasets, this should remain True.
    preserve_supervisor_outputs:
        Preserve mapped outputs such as priority and supervisor_score.
        These columns remain in the cleaned dataset but must not later be used
        as model features.
    """

    def __init__(
        self,
        input_path: Path = SAMPLE_DATASET_PATH,
        output_path: Path = CLEANED_DATASET_PATH,
        report_path: Path = CLEANING_REPORT_PATH,
        *,
        remove_invalid_targets: bool = True,
        preserve_supervisor_outputs: bool = True,
    ) -> None:
        initialize_supervisor_environment()

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.report_path = Path(report_path)

        self.remove_invalid_targets = remove_invalid_targets
        self.preserve_supervisor_outputs = preserve_supervisor_outputs

        self.counters = CleaningCounters()
        self.started_at: str | None = None
        self.finished_at: str | None = None

    # =========================================================================
    # LOADING
    # =========================================================================

    def _validate_input_file(self) -> None:
        """Validate the source dataset path."""

        if not self.input_path.exists():
            raise FileNotFoundError(
                "Supervisor input dataset was not found: "
                f"{self.input_path}"
            )

        if not self.input_path.is_file():
            raise ValueError(
                "Supervisor input path is not a file: "
                f"{self.input_path}"
            )

        if self.input_path.suffix.lower() != ".csv":
            raise ValueError(
                "dataset_cleaner currently expects a CSV input. "
                f"Received: {self.input_path.suffix or '<no extension>'}"
            )

        if self.input_path.stat().st_size == 0:
            raise ValueError(
                f"Supervisor input dataset is empty: {self.input_path}"
            )

    def _load_dataset(self) -> pd.DataFrame:
        """Load the complete input CSV."""

        self._validate_input_file()

        logger.info(
            "Loading supervisor dataset from %s",
            self.input_path,
        )

        try:
            dataframe = pd.read_csv(
                self.input_path,
                low_memory=False,
            )
        except pd.errors.EmptyDataError as exc:
            raise ValueError(
                f"Supervisor dataset contains no readable rows: "
                f"{self.input_path}"
            ) from exc
        except pd.errors.ParserError as exc:
            raise ValueError(
                f"Supervisor dataset CSV is malformed: "
                f"{self.input_path}"
            ) from exc

        if dataframe.empty:
            raise ValueError(
                "Supervisor dataset contains a header but no data rows."
            )

        self.counters.original_rows = int(len(dataframe))
        self.counters.original_columns = int(len(dataframe.columns))

        logger.info(
            "Loaded dataset: rows=%d, columns=%d, memory=%.2f MB",
            len(dataframe),
            len(dataframe.columns),
            dataframe_memory_mb(dataframe),
        )

        return dataframe

    # =========================================================================
    # COLUMN CLEANING
    # =========================================================================

    def _normalize_column_names(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize headers and apply known aliases."""

        dataframe = dataframe.copy()

        normalized_columns: list[str] = []
        rename_report: dict[str, str] = {}

        for original_column in dataframe.columns:
            normalized = normalize_column_name(original_column)
            normalized = COLUMN_ALIASES.get(
                normalized,
                normalized,
            )

            normalized_columns.append(normalized)

            if str(original_column) != normalized:
                rename_report[str(original_column)] = normalized

        duplicate_names = {
            column
            for column in normalized_columns
            if normalized_columns.count(column) > 1
        }

        if duplicate_names:
            raise ValueError(
                "Column normalization created duplicate names: "
                f"{sorted(duplicate_names)}"
            )

        dataframe.columns = normalized_columns
        self.counters.renamed_columns = rename_report

        return dataframe

    def _remove_accidental_index_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove CSV index columns such as `Unnamed: 0` or a sequential `index`.
        """

        dataframe = dataframe.copy()
        removable: list[str] = []

        for column in dataframe.columns:
            matches_pattern = any(
                pattern.fullmatch(column)
                for pattern in ACCIDENTAL_INDEX_PATTERNS
            )

            if not matches_pattern:
                continue

            if column in {
                ENGINE_ID_COLUMN,
                SUBSET_COLUMN,
                "cycle",
            }:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            expected_zero_based = pd.Series(
                np.arange(len(dataframe)),
                index=dataframe.index,
                dtype=float,
            )
            expected_one_based = expected_zero_based + 1.0

            is_sequential = (
                series.notna().all()
                and (
                    np.array_equal(
                        series.to_numpy(dtype=float),
                        expected_zero_based.to_numpy(dtype=float),
                    )
                    or np.array_equal(
                        series.to_numpy(dtype=float),
                        expected_one_based.to_numpy(dtype=float),
                    )
                )
            )

            if column.startswith("unnamed") or is_sequential:
                removable.append(column)

        if removable:
            dataframe = dataframe.drop(
                columns=removable,
            )

        self.counters.accidental_index_columns_removed = removable

        return dataframe

    def _remove_leakage_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Remove prohibited future-information columns.

        The target and mapped outputs may be preserved because this stage
        prepares a complete research dataset. Feature selection later must
        still exclude them from X.
        """

        protected = {TARGET_COLUMN}

        if self.preserve_supervisor_outputs:
            protected.update(PRESERVED_OUTPUT_COLUMNS)

        removable = sorted(
            column
            for column in FORBIDDEN_FEATURE_COLUMNS
            if column in dataframe.columns
            and column not in protected
        )

        if removable:
            dataframe = dataframe.drop(
                columns=removable,
            )

        self.counters.leakage_columns_removed = removable

        return dataframe

    # =========================================================================
    # MISSING VALUES
    # =========================================================================

    def _standardize_missing_text(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert common textual missing markers to pandas missing values."""

        dataframe = dataframe.copy()

        object_columns = dataframe.select_dtypes(
            include=["object", "string"],
        ).columns

        for column in object_columns:
            series = dataframe[column].astype("string")
            stripped = series.str.strip()
            missing_mask = stripped.str.lower().isin(
                EMPTY_TEXT_VALUES
            )

            dataframe[column] = stripped.mask(
                missing_mask,
                pd.NA,
            )

        return dataframe

    # =========================================================================
    # TYPE CONVERSION
    # =========================================================================

    def _convert_numeric_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert configured numerical and metadata columns to numbers."""

        dataframe = dataframe.copy()

        numeric_candidates = list(
            dict.fromkeys(
                [
                    *NUMERICAL_FEATURES,
                    "cycle",
                    "supervisor_score",
                ]
            )
        )

        for column in numeric_candidates:
            if column not in dataframe.columns:
                continue

            original_non_missing = int(
                dataframe[column].notna().sum()
            )

            converted = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            converted_non_missing = int(
                converted.notna().sum()
            )

            coerced_count = max(
                original_non_missing - converted_non_missing,
                0,
            )

            if coerced_count:
                self.counters.numeric_values_coerced_to_missing[
                    column
                ] = coerced_count

            dataframe[column] = converted.replace(
                [np.inf, -np.inf],
                np.nan,
            )

        return dataframe

    @staticmethod
    def _parse_boolean_value(value: Any) -> bool | pd._libs.missing.NAType:
        """Parse a single value into Boolean or pandas missing."""

        if pd.isna(value):
            return pd.NA

        if isinstance(value, (bool, np.bool_)):
            return bool(value)

        if isinstance(value, (int, np.integer)):
            if int(value) == 1:
                return True

            if int(value) == 0:
                return False

            return pd.NA

        if isinstance(value, (float, np.floating)):
            if not np.isfinite(value):
                return pd.NA

            if float(value) == 1.0:
                return True

            if float(value) == 0.0:
                return False

            return pd.NA

        normalized = str(value).strip().lower()

        if normalized in TRUE_VALUES:
            return True

        if normalized in FALSE_VALUES:
            return False

        if normalized in EMPTY_TEXT_VALUES:
            return pd.NA

        return pd.NA

    def _convert_boolean_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Convert configured Boolean columns to pandas Boolean dtype."""

        dataframe = dataframe.copy()

        boolean_candidates = list(
            dict.fromkeys(
                [
                    *BOOLEAN_FEATURES,
                    "requires_human_review",
                ]
            )
        )

        for column in boolean_candidates:
            if column not in dataframe.columns:
                continue

            original = dataframe[column]
            parsed = original.map(
                self._parse_boolean_value
            ).astype("boolean")

            invalid_mask = (
                original.notna()
                & parsed.isna()
                & ~original.astype("string")
                .str.strip()
                .str.lower()
                .isin(EMPTY_TEXT_VALUES)
            )

            invalid_count = int(
                invalid_mask.sum()
            )

            if invalid_count:
                self.counters.invalid_boolean_values[
                    column
                ] = invalid_count

            dataframe[column] = parsed

        return dataframe

    # =========================================================================
    # CATEGORICAL NORMALIZATION
    # =========================================================================

    def _normalize_categorical_column(
        self,
        dataframe: pd.DataFrame,
        column: str,
        mapping: dict[str, str] | None = None,
    ) -> None:
        """Normalize a categorical column in place."""

        if column not in dataframe.columns:
            return

        before = dataframe[column].astype("string")
        normalized = before.map(normalize_label)

        if mapping is not None:
            normalized = normalized.map(
                lambda value: (
                    mapping.get(value, value)
                    if value is not None
                    else None
                )
            )

        changed_mask = (
            before.notna()
            & normalized.notna()
            & (
                before.str.strip()
                != normalized.astype("string")
            )
        )

        changed_count = int(
            changed_mask.sum()
        )

        if changed_count:
            self.counters.categorical_values_normalized[
                column
            ] = changed_count

        dataframe[column] = pd.Series(
            normalized,
            index=dataframe.index,
            dtype="string",
        )

    def _normalize_categorical_values(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Normalize configured categorical and trace columns."""

        dataframe = dataframe.copy()

        snake_case_columns = list(
            dict.fromkeys(
                [
                    *CATEGORICAL_FEATURES,
                    *MODEL_TRACE_COLUMNS,
                    "maintenance_urgency",
                ]
            )
        )

        for column in snake_case_columns:
            self._normalize_categorical_column(
                dataframe,
                column,
            )

        self._normalize_categorical_column(
            dataframe,
            TARGET_COLUMN,
            TARGET_NORMALIZATION_MAP,
        )

        self._normalize_categorical_column(
            dataframe,
            "priority",
            PRIORITY_NORMALIZATION_MAP,
        )

        self._normalize_categorical_column(
            dataframe,
            "anomaly_severity",
            ANOMALY_SEVERITY_NORMALIZATION_MAP,
        )

        if SUBSET_COLUMN in dataframe.columns:
            subset = (
                dataframe[SUBSET_COLUMN]
                .astype("string")
                .str.strip()
                .str.upper()
            )

            subset = subset.str.replace(
                r"[^A-Z0-9]+",
                "",
                regex=True,
            )

            dataframe[SUBSET_COLUMN] = subset

        if ENGINE_ID_COLUMN in dataframe.columns:
            dataframe[ENGINE_ID_COLUMN] = (
                dataframe[ENGINE_ID_COLUMN]
                .astype("string")
                .str.strip()
                .str.upper()
            )

        if SCHEMA_VERSION_COLUMN in dataframe.columns:
            dataframe[SCHEMA_VERSION_COLUMN] = (
                dataframe[SCHEMA_VERSION_COLUMN]
                .astype("string")
                .str.strip()
            )

        return dataframe

    # =========================================================================
    # IMPUTATION
    # =========================================================================

    def _impute_numeric_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Impute missing numeric values.

        Strategy:
        1. Within-engine median where possible.
        2. Dataset-wide median.
        3. Zero only when the entire column is missing.
        """

        dataframe = dataframe.copy()

        numeric_candidates = list(
            dict.fromkeys(
                [
                    *NUMERICAL_FEATURES,
                    "supervisor_score",
                ]
            )
        )

        group_columns = [
            column
            for column in (
                ENGINE_ID_COLUMN,
                SUBSET_COLUMN,
            )
            if column in dataframe.columns
        ]

        for column in numeric_candidates:
            if column not in dataframe.columns:
                continue

            missing_before = int(
                dataframe[column].isna().sum()
            )

            if missing_before == 0:
                continue

            series = dataframe[column]

            if group_columns:
                group_medians = dataframe.groupby(
                    group_columns,
                    dropna=False,
                    observed=True,
                )[column].transform("median")

                series = series.fillna(
                    group_medians
                )

            global_median = series.median(
                skipna=True
            )

            if pd.isna(global_median):
                global_median = 0.0

            dataframe[column] = series.fillna(
                float(global_median)
            )

            missing_after = int(
                dataframe[column].isna().sum()
            )

            self.counters.numeric_values_imputed[
                column
            ] = missing_before - missing_after

        return dataframe

    def _impute_boolean_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Impute missing Boolean values conservatively as False."""

        dataframe = dataframe.copy()

        boolean_candidates = list(
            dict.fromkeys(
                [
                    *BOOLEAN_FEATURES,
                    "requires_human_review",
                ]
            )
        )

        for column in boolean_candidates:
            if column not in dataframe.columns:
                continue

            missing_count = int(
                dataframe[column].isna().sum()
            )

            if missing_count:
                dataframe[column] = (
                    dataframe[column]
                    .fillna(False)
                    .astype(bool)
                )

                self.counters.boolean_values_imputed[
                    column
                ] = missing_count
            else:
                dataframe[column] = dataframe[column].astype(
                    bool
                )

        return dataframe

    def _impute_categorical_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Fill missing categorical values with explicit defaults."""

        dataframe = dataframe.copy()

        categorical_candidates = list(
            dict.fromkeys(
                [
                    *CATEGORICAL_FEATURES,
                    *MODEL_TRACE_COLUMNS,
                    TARGET_COLUMN,
                    "priority",
                    "maintenance_urgency",
                ]
            )
        )

        for column in categorical_candidates:
            if column not in dataframe.columns:
                continue

            missing_count = int(
                dataframe[column].isna().sum()
            )

            if missing_count:
                default = KNOWN_CATEGORICAL_DEFAULTS.get(
                    column,
                    "unknown",
                )

                dataframe[column] = (
                    dataframe[column]
                    .fillna(default)
                    .astype("string")
                )

                self.counters.categorical_values_imputed[
                    column
                ] = missing_count
            else:
                dataframe[column] = dataframe[column].astype(
                    "string"
                )

        return dataframe

    # =========================================================================
    # DOMAIN CORRECTIONS
    # =========================================================================

    def _clean_cycle_column(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Ensure cycle values are positive integers."""

        dataframe = dataframe.copy()

        if "cycle" not in dataframe.columns:
            return dataframe

        cycle = pd.to_numeric(
            dataframe["cycle"],
            errors="coerce",
        )

        invalid_mask = (
            cycle.isna()
            | ~np.isfinite(cycle)
            | (cycle < 1)
        )

        self.counters.invalid_cycles_corrected = int(
            invalid_mask.sum()
        )

        cycle = cycle.round()

        if invalid_mask.any():
            if ENGINE_ID_COLUMN in dataframe.columns:
                fallback = (
                    dataframe.groupby(
                        ENGINE_ID_COLUMN,
                        dropna=False,
                        observed=True,
                    )
                    .cumcount()
                    .add(1)
                )

                cycle = cycle.mask(
                    invalid_mask,
                    fallback,
                )
            else:
                fallback = pd.Series(
                    np.arange(1, len(dataframe) + 1),
                    index=dataframe.index,
                )

                cycle = cycle.mask(
                    invalid_mask,
                    fallback,
                )

        dataframe["cycle"] = cycle.astype(
            "int64"
        )

        return dataframe

    def _clean_probability_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clip probability, score and confidence values into [0, 1]."""

        dataframe = dataframe.copy()

        for column in PROBABILITY_COLUMNS:
            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            invalid_range = (
                series.notna()
                & (
                    (series < 0.0)
                    | (series > 1.0)
                )
            )

            clipped_count = int(
                invalid_range.sum()
            )

            if clipped_count:
                self.counters.probability_values_clipped[
                    column
                ] = clipped_count

            dataframe[column] = series.clip(
                lower=0.0,
                upper=1.0,
            )

        return dataframe

    def _clean_non_negative_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Clip physically non-negative values to zero."""

        dataframe = dataframe.copy()

        for column in NON_NEGATIVE_COLUMNS:
            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            negative_mask = (
                series.notna()
                & (series < 0.0)
            )

            corrected_count = int(
                negative_mask.sum()
            )

            if corrected_count:
                self.counters.non_negative_values_corrected[
                    column
                ] = corrected_count

            dataframe[column] = series.clip(
                lower=0.0,
            )

        return dataframe

    def _correct_rul_intervals(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Correct RUL interval ordering and derive uncertainty width.

        Rules:
        - predicted_rul >= 0
        - lower_bound >= 0
        - upper_bound >= 0
        - lower_bound <= upper_bound
        - uncertainty_width = upper_bound - lower_bound
        """

        dataframe = dataframe.copy()

        required = {
            "lower_bound",
            "upper_bound",
        }

        if not required.issubset(dataframe.columns):
            return dataframe

        lower = pd.to_numeric(
            dataframe["lower_bound"],
            errors="coerce",
        ).clip(lower=0.0)

        upper = pd.to_numeric(
            dataframe["upper_bound"],
            errors="coerce",
        ).clip(lower=0.0)

        reversed_mask = (
            lower.notna()
            & upper.notna()
            & (lower > upper)
        )

        self.counters.rul_bounds_swapped = int(
            reversed_mask.sum()
        )

        corrected_lower = lower.where(
            ~reversed_mask,
            upper,
        )

        corrected_upper = upper.where(
            ~reversed_mask,
            lower,
        )

        dataframe["lower_bound"] = corrected_lower
        dataframe["upper_bound"] = corrected_upper

        calculated_width = (
            corrected_upper - corrected_lower
        ).clip(lower=0.0)

        if "uncertainty_width" in dataframe.columns:
            existing_width = pd.to_numeric(
                dataframe["uncertainty_width"],
                errors="coerce",
            )

            inconsistent_mask = (
                existing_width.isna()
                | ~np.isclose(
                    existing_width.fillna(0.0),
                    calculated_width.fillna(0.0),
                    rtol=1e-5,
                    atol=1e-6,
                )
            )

            self.counters.uncertainty_width_recalculated = int(
                inconsistent_mask.sum()
            )

        else:
            self.counters.uncertainty_width_recalculated = int(
                calculated_width.notna().sum()
            )

        dataframe["uncertainty_width"] = calculated_width

        return dataframe

    # =========================================================================
    # ROW VALIDATION AND DUPLICATES
    # =========================================================================

    def _remove_rows_missing_keys(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove rows missing engine, subset or cycle keys."""

        dataframe = dataframe.copy()

        available_keys = [
            column
            for column in UNIQUE_KEY_COLUMNS
            if column in dataframe.columns
        ]

        if not available_keys:
            return dataframe

        missing_key_mask = dataframe[
            available_keys
        ].isna().any(axis=1)

        for column in (
            ENGINE_ID_COLUMN,
            SUBSET_COLUMN,
        ):
            if column in available_keys:
                missing_key_mask |= (
                    dataframe[column]
                    .astype("string")
                    .str.strip()
                    .isin(["", "<NA>", "UNKNOWN"])
                )

        removed = int(
            missing_key_mask.sum()
        )

        self.counters.rows_removed_missing_keys = removed

        if removed:
            dataframe = dataframe.loc[
                ~missing_key_mask
            ].copy()

        return dataframe

    def _remove_invalid_target_rows(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove missing or unsupported training targets."""

        dataframe = dataframe.copy()

        if TARGET_COLUMN not in dataframe.columns:
            if self.remove_invalid_targets:
                raise KeyError(
                    f"Required target column is missing: {TARGET_COLUMN}"
                )

            return dataframe

        valid_targets = set(
            DECISION_CLASSES
        )

        invalid_mask = ~dataframe[
            TARGET_COLUMN
        ].isin(valid_targets)

        invalid_count = int(
            invalid_mask.sum()
        )

        self.counters.rows_removed_invalid_target = invalid_count

        if invalid_count and self.remove_invalid_targets:
            dataframe = dataframe.loc[
                ~invalid_mask
            ].copy()

        return dataframe

    def _remove_duplicates(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Remove exact and unique-key duplicate rows."""

        dataframe = dataframe.copy()

        rows_before = len(dataframe)

        dataframe = dataframe.drop_duplicates(
            keep="first",
        )

        self.counters.duplicate_rows_removed = (
            rows_before - len(dataframe)
        )

        available_keys = [
            column
            for column in UNIQUE_KEY_COLUMNS
            if column in dataframe.columns
        ]

        if len(available_keys) == len(UNIQUE_KEY_COLUMNS):
            rows_before_keys = len(dataframe)

            dataframe = dataframe.drop_duplicates(
                subset=available_keys,
                keep="first",
            )

            self.counters.duplicate_key_rows_removed = (
                rows_before_keys - len(dataframe)
            )

        return dataframe

    # =========================================================================
    # FINALIZATION
    # =========================================================================

    def _sort_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Sort deterministically by subset, engine and cycle."""

        sorting_columns = [
            column
            for column in (
                SUBSET_COLUMN,
                ENGINE_ID_COLUMN,
                "cycle",
            )
            if column in dataframe.columns
        ]

        if sorting_columns:
            dataframe = dataframe.sort_values(
                by=sorting_columns,
                kind="mergesort",
            )

        return dataframe.reset_index(
            drop=True,
        )

    def _optimize_dtypes(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Use memory-efficient final dtypes without losing information."""

        dataframe = dataframe.copy()

        float_columns = [
            column
            for column in (
                *NUMERICAL_FEATURES,
                "supervisor_score",
            )
            if column in dataframe.columns
        ]

        for column in float_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce",
                downcast="float",
            )

        if "cycle" in dataframe.columns:
            dataframe["cycle"] = pd.to_numeric(
                dataframe["cycle"],
                errors="raise",
                downcast="integer",
            )

        categorical_candidates = [
            *CATEGORICAL_FEATURES,
            *MODEL_TRACE_COLUMNS,
            TARGET_COLUMN,
            "priority",
            "maintenance_urgency",
            SUBSET_COLUMN,
        ]

        for column in dict.fromkeys(categorical_candidates):
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(
                    "category"
                )

        for column in (
            *BOOLEAN_FEATURES,
            "requires_human_review",
        ):
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].astype(
                    bool
                )

        return dataframe

    def _validate_final_dataset(
        self,
        dataframe: pd.DataFrame,
    ) -> None:
        """Perform strict final checks before writing output."""

        if dataframe.empty:
            raise ValueError(
                "Cleaning removed all dataset rows. "
                "The cleaned dataset cannot be written."
            )

        required_columns = {
            *UNIQUE_KEY_COLUMNS,
            TARGET_COLUMN,
        }

        missing_required = sorted(
            required_columns.difference(
                dataframe.columns
            )
        )

        if missing_required:
            raise KeyError(
                "Cleaned dataset is missing required columns: "
                f"{missing_required}"
            )

        duplicate_columns = dataframe.columns[
            dataframe.columns.duplicated()
        ].tolist()

        if duplicate_columns:
            raise ValueError(
                "Cleaned dataset contains duplicate columns: "
                f"{duplicate_columns}"
            )

        duplicated_keys = dataframe.duplicated(
            subset=UNIQUE_KEY_COLUMNS,
            keep=False,
        )

        if duplicated_keys.any():
            raise ValueError(
                "Cleaned dataset still contains duplicate "
                "(engine_id, fd_subset, cycle) keys."
            )

        invalid_targets = sorted(
            set(
                dataframe[TARGET_COLUMN]
                .astype("string")
                .dropna()
                .tolist()
            ).difference(
                DECISION_CLASSES
            )
        )

        if invalid_targets:
            raise ValueError(
                "Cleaned dataset contains unsupported target labels: "
                f"{invalid_targets}"
            )

        for column in PROBABILITY_COLUMNS:
            if column not in dataframe.columns:
                continue

            series = pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )

            invalid = (
                series.notna()
                & ~series.between(
                    0.0,
                    1.0,
                    inclusive="both",
                )
            )

            if invalid.any():
                raise ValueError(
                    f"Column '{column}' contains values outside [0, 1]."
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
                    f"Column '{column}' contains infinite values."
                )

    def _build_report(
        self,
        dataframe: pd.DataFrame,
        duration_seconds: float,
        output_hash: str,
    ) -> dict[str, Any]:
        """Build the complete cleaning report."""

        target_distribution: dict[str, int] = {}

        if TARGET_COLUMN in dataframe.columns:
            target_distribution = {
                str(label): int(count)
                for label, count in (
                    dataframe[TARGET_COLUMN]
                    .astype("string")
                    .value_counts(dropna=False)
                    .sort_index()
                    .items()
                )
            }

        missing_values = {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
            if int(count) > 0
        }

        report = {
            "status": "success",
            "component": CONFIG.component_name,
            "stage": "dataset_cleaning",
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
            },
            "output": {
                "path": str(
                    self.output_path.resolve()
                ),
                "report_path": str(
                    self.report_path.resolve()
                ),
                "sha256": output_hash,
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
            "configuration": {
                "remove_invalid_targets": (
                    self.remove_invalid_targets
                ),
                "preserve_supervisor_outputs": (
                    self.preserve_supervisor_outputs
                ),
                "target_column": TARGET_COLUMN,
                "unique_key_columns": list(
                    UNIQUE_KEY_COLUMNS
                ),
                "decision_classes": list(
                    DECISION_CLASSES
                ),
            },
            "cleaning_counters": asdict(
                self.counters
            ),
            "remaining_missing_values": missing_values,
            "target_distribution": target_distribution,
            "columns": [
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

        return report

    # =========================================================================
    # PIPELINE
    # =========================================================================

    def clean_dataframe(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """Run all cleaning transformations on a DataFrame."""

        logger.info(
            "Normalizing dataset columns."
        )
        dataframe = self._normalize_column_names(
            dataframe
        )
        dataframe = self._remove_accidental_index_columns(
            dataframe
        )
        dataframe = self._remove_leakage_columns(
            dataframe
        )

        logger.info(
            "Standardizing missing values and data types."
        )
        dataframe = self._standardize_missing_text(
            dataframe
        )
        dataframe = self._convert_numeric_columns(
            dataframe
        )
        dataframe = self._convert_boolean_columns(
            dataframe
        )
        dataframe = self._normalize_categorical_values(
            dataframe
        )

        logger.info(
            "Applying domain corrections."
        )
        dataframe = self._clean_probability_columns(
            dataframe
        )
        dataframe = self._clean_non_negative_columns(
            dataframe
        )
        dataframe = self._correct_rul_intervals(
            dataframe
        )
        dataframe = self._clean_cycle_column(
            dataframe
        )

        logger.info(
            "Imputing missing values."
        )
        dataframe = self._impute_numeric_columns(
            dataframe
        )
        dataframe = self._impute_boolean_columns(
            dataframe
        )
        dataframe = self._impute_categorical_columns(
            dataframe
        )

        logger.info(
            "Removing invalid and duplicate records."
        )
        dataframe = self._remove_rows_missing_keys(
            dataframe
        )
        dataframe = self._remove_invalid_target_rows(
            dataframe
        )
        dataframe = self._remove_duplicates(
            dataframe
        )

        dataframe = self._sort_dataset(
            dataframe
        )
        dataframe = self._optimize_dtypes(
            dataframe
        )

        self.counters.final_rows = int(
            len(dataframe)
        )
        self.counters.final_columns = int(
            len(dataframe.columns)
        )

        self._validate_final_dataset(
            dataframe
        )

        return dataframe

    def run(self) -> CleaningResult:
        """Execute the full cleaning pipeline and save outputs atomically."""

        started_timer = perf_counter()
        self.started_at = utc_now_iso()

        logger.info("=" * 78)
        logger.info("SUPERVISOR DATASET CLEANING STARTED")
        logger.info("=" * 78)
        logger.info("Input dataset : %s", self.input_path)
        logger.info("Cleaned output: %s", self.output_path)
        logger.info("Report output : %s", self.report_path)

        try:
            dataframe = self._load_dataset()
            cleaned_dataframe = self.clean_dataframe(
                dataframe
            )

            logger.info(
                "Writing cleaned dataset atomically."
            )

            atomic_write_csv(
                cleaned_dataframe,
                self.output_path,
            )

            output_hash = sha256_file(
                self.output_path
            )

            self.finished_at = utc_now_iso()
            duration_seconds = (
                perf_counter() - started_timer
            )

            report = self._build_report(
                cleaned_dataframe,
                duration_seconds,
                output_hash,
            )

            atomic_write_json(
                report,
                self.report_path,
            )

            result = CleaningResult(
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
                rows_before=self.counters.original_rows,
                rows_after=self.counters.final_rows,
                columns_before=self.counters.original_columns,
                columns_after=self.counters.final_columns,
                duration_seconds=round(
                    duration_seconds,
                    6,
                ),
                output_sha256=output_hash,
                message=(
                    "Supervisor dataset cleaning completed successfully."
                ),
            )

            logger.info("-" * 78)
            logger.info(
                "Rows before                    : %d",
                result.rows_before,
            )
            logger.info(
                "Rows after                     : %d",
                result.rows_after,
            )
            logger.info(
                "Exact duplicates removed       : %d",
                self.counters.duplicate_rows_removed,
            )
            logger.info(
                "Duplicate engine-cycle removed : %d",
                self.counters.duplicate_key_rows_removed,
            )
            logger.info(
                "Invalid targets removed        : %d",
                self.counters.rows_removed_invalid_target,
            )
            logger.info(
                "Missing-key rows removed       : %d",
                self.counters.rows_removed_missing_keys,
            )
            logger.info(
                "Duration                       : %.3f seconds",
                result.duration_seconds,
            )
            logger.info(
                "Output SHA-256                 : %s",
                result.output_sha256,
            )
            logger.info("=" * 78)
            logger.info(
                "SUPERVISOR DATASET CLEANING COMPLETED"
            )
            logger.info("=" * 78)

            return result

        except Exception:
            self.finished_at = utc_now_iso()

            logger.exception(
                "Supervisor dataset cleaning failed. "
                "Existing outputs were not intentionally deleted."
            )

            raise


# =============================================================================
# COMMAND-LINE INTERFACE
# =============================================================================

def build_argument_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Clean and normalize the Autonomous Maintenance "
            "Supervisor dataset."
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
        "--output",
        type=Path,
        default=CLEANED_DATASET_PATH,
        help=(
            "Cleaned CSV destination. "
            f"Default: {CLEANED_DATASET_PATH}"
        ),
    )

    parser.add_argument(
        "--report",
        type=Path,
        default=CLEANING_REPORT_PATH,
        help=(
            "Cleaning report JSON destination. "
            f"Default: {CLEANING_REPORT_PATH}"
        ),
    )

    parser.add_argument(
        "--keep-invalid-targets",
        action="store_true",
        help=(
            "Keep rows whose final_decision is missing or unsupported. "
            "Not recommended for model-training datasets."
        ),
    )

    parser.add_argument(
        "--drop-derived-outputs",
        action="store_true",
        help=(
            "Drop priority, maintenance_urgency, "
            "requires_human_review and supervisor_score."
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

    cleaner = SupervisorDatasetCleaner(
        input_path=parsed.input,
        output_path=parsed.output,
        report_path=parsed.report,
        remove_invalid_targets=(
            not parsed.keep_invalid_targets
        ),
        preserve_supervisor_outputs=(
            not parsed.drop_derived_outputs
        ),
    )

    try:
        result = cleaner.run()

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
            "Dataset cleaner failed: %s",
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
            "Unexpected dataset-cleaning error."
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