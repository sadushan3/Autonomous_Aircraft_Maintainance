"""
categorical_encoder.py

Categorical Feature Encoder for the Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module applies deterministic, reproducible categorical encoding to the
engineered supervisor dataset. It handles two fundamentally different types
of categorical features using the most appropriate strategy for each:

1. ORDINAL (ordered) encoding
   Applied to columns whose categories have a meaningful severity order.
   Each category is mapped to an integer that preserves the ordering, so
   that tree-based and linear models can exploit the ordinal relationship.

   Columns encoded ordinally:
   - anomaly_severity   : none < low < medium < high < critical
   - lifecycle_state    : early < nominal < degrading < critical
   - risk_state         : nominal < elevated < high < critical

2. ONE-HOT encoding
   Applied to nominal columns whose categories have no inherent ordering.
   Each unique category becomes a binary 0/1 column.

   Columns encoded with one-hot:
   - action_hint_for_supervisor  (supervisor action recommendation)
   - top_sensor_1, top_sensor_2, top_sensor_3 (sensor names)

   Unknown categories seen during inference (not in training vocabulary) are
   mapped to an all-zero row rather than raising an exception.

Categorical columns that appear in the engineered dataset but are not
listed in either group (e.g., new columns added by future feature
engineering steps) are auto-detected and encoded with one-hot encoding as
the conservative default, with a WARNING in the report.

Design decisions
----------------
- Fit on the FULL engineered dataset (pre-split).
  Ordinal and one-hot encoders do not use the target label, so fitting on
  the full dataset before the split does not introduce statistical leakage.
  Target encoding (which uses target statistics) is explicitly NOT performed
  here; that is reserved for the training pipeline where it can be safely
  fitted on the training fold only.

- All encoder objects are saved as .joblib artifacts so that the training
  pipeline and inference pipeline can reload and re-use them without
  re-fitting.

- The encoded CSV replaces the original categorical columns with their
  numeric representations in-place. The original string columns are dropped.
  One-hot output columns are appended at the end.

- Columns already absent from the dataset (e.g., dropped by leakage_guard)
  are silently skipped with an INFO log so the pipeline remains robust.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.categorical_encoder

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_engineered_dataset.csv
  (output of feature_engineering.py)

Expected outputs
----------------
- processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv
- artifacts/Maintenance_Supervisor/encoders/ordinal_encoder.joblib
- artifacts/Maintenance_Supervisor/encoders/onehot_encoder.joblib
- artifacts/Maintenance_Supervisor/encoders/encoding_manifest.json
- reports/Maintenance_Supervisor/categorical_encoding_report.json

Exit codes
----------
0 — encoding completed successfully
1 — internal failure (IO, config, import, unexpected error)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path regardless of launch method
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()

# Data_preprocessing -> Maintenance_Supervisor -> services -> app -> Backend
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    ENGINEERED_DATASET_PATH,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    PROCESSED_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    METADATA_COLUMNS,
    DERIVED_OUTPUT_COLUMNS,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_csv,
)

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_ENCODED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_encoded_dataset.csv"
)
_ENCODER_ROOT: Final[Path] = ARTIFACT_ROOT / "encoders"
_ORDINAL_ENCODER_PATH: Final[Path] = _ENCODER_ROOT / "ordinal_encoder.joblib"
_ONEHOT_ENCODER_PATH: Final[Path] = _ENCODER_ROOT / "onehot_encoder.joblib"
_ENCODING_MANIFEST_PATH: Final[Path] = _ENCODER_ROOT / "encoding_manifest.json"
_ENCODING_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT / "categorical_encoding_report.json"
)

# ---------------------------------------------------------------------------
# Ordinal category definitions
# Severity order is ASCENDING — higher integer = more severe.
# ---------------------------------------------------------------------------

ANOMALY_SEVERITY_ORDER: Final[list[str]] = [
    "none",
    "low",
    "medium",
    "high",
    "critical",
]

LIFECYCLE_STATE_ORDER: Final[list[str]] = [
    "early",
    "nominal",
    "degrading",
    "critical",
]

RISK_STATE_ORDER: Final[list[str]] = [
    "nominal",
    "elevated",
    "high",
    "critical",
]

# Mapping: column name -> ordered category list (lowest to highest)
ORDINAL_COLUMN_ORDERS: Final[dict[str, list[str]]] = {
    "anomaly_severity": ANOMALY_SEVERITY_ORDER,
    "lifecycle_state": LIFECYCLE_STATE_ORDER,
    "risk_state": RISK_STATE_ORDER,
}

# Nominal columns: no ordering — one-hot encoded
NOMINAL_COLUMNS: Final[list[str]] = [
    "action_hint_for_supervisor",
    "top_sensor_1",
    "top_sensor_2",
    "top_sensor_3",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class EncodingStats:
    """Statistics for a single encoded column."""

    original_column: str
    encoding_strategy: str           # "ordinal" or "onehot"
    unique_values_found: list[str]
    output_columns: list[str]
    null_count_before: int
    null_count_after: int
    unknown_categories_found: int


@dataclass
class CategoricalEncoderResult:
    """Aggregated result of the full encoding run."""

    input_path: str
    output_path: str
    rows: int
    columns_before: int
    columns_after: int
    ordinal_columns_encoded: list[str] = field(default_factory=list)
    onehot_columns_encoded: list[str] = field(default_factory=list)
    columns_skipped: list[str] = field(default_factory=list)
    auto_detected_nominal: list[str] = field(default_factory=list)
    encoding_stats: list[EncodingStats] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "rows": self.rows,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "ordinal_columns_encoded": self.ordinal_columns_encoded,
            "onehot_columns_encoded": self.onehot_columns_encoded,
            "columns_skipped": self.columns_skipped,
            "auto_detected_nominal": self.auto_detected_nominal,
            "encoding_stats": [
                {
                    "original_column": s.original_column,
                    "encoding_strategy": s.encoding_strategy,
                    "unique_values_found": s.unique_values_found,
                    "output_columns": s.output_columns,
                    "null_count_before": s.null_count_before,
                    "null_count_after": s.null_count_after,
                    "unknown_categories_found": s.unknown_categories_found,
                }
                for s in self.encoding_stats
            ],
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_string_column(series: pd.Series) -> pd.Series:
    """
    Lowercase, strip whitespace, and fill NaN with 'unknown' so that
    encoders receive clean, consistent input with no raw NaN values.

    Only for NOMINAL (one-hot) columns -- see _normalise_ordinal_column for
    why ordinal columns must not go through this.
    """

    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"nan": "unknown", "none": "unknown", "": "unknown"})
    )


def _normalise_ordinal_column(series: pd.Series) -> pd.Series:
    """
    Lowercase and strip whitespace for ordinal columns, WITHOUT collapsing
    the literal string "none" into "unknown".

    Bug this fixes: anomaly_severity's defined category order is
    ["none", "low", "medium", "high", "critical"] (see ANOMALY_SEVERITY_ORDER)
    -- "none" is a real, ordered category (encoded 0), not a stand-in for a
    missing value. _normalise_string_column's blanket
    ``.replace({"none": "unknown", ...})`` ran before the fitted
    OrdinalEncoder ever saw the data, so every genuine "none" reading was
    rewritten to the string "unknown" first. "unknown" is not in the fitted
    category list, so handle_unknown="use_encoded_value" (unknown_value=-1)
    encoded it as -1 -- confirmed directly: 33,099 of 40,463 rows
    (81.8%, the overwhelming majority) in
    processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv have
    anomaly_severity == -1, and the value 0 (the correct code for "none")
    never appears at all. Every model trained on that column effectively
    never saw "no anomaly" as 0; the live serving path
    (decision_fusion_engine.py's _encode_live_ordinal_columns) does NOT
    apply this "none"->"unknown" collapse, so live "none" readings correctly
    encode to 0 -- a value the model was trained to treat as -1's neighbour,
    not as its own thing. This was a material contributor to the champion
    PPO policy collapsing to a single action on live batches (see
    ppo_supervisor.py's module docstring): most live rows are healthy
    ("none" anomaly severity), and 0 sat far outside what the network was
    ever trained to see for this feature.

    True missing values (real NaN, never stringified here) are left as NaN
    so the encoder's own encoded_missing_value=-2 handles them -- genuinely
    absent data should not be conflated with the unknown-category code -1
    either.
    """

    def _clean(value: object) -> object:
        if pd.isna(value):
            return np.nan
        return str(value).strip().lower()

    return series.apply(_clean)


def _detect_unknown_categories(
    series: pd.Series,
    known_categories: list[str],
) -> int:
    """
    Count how many values in *series* are not in *known_categories*.
    Used for diagnostics only; does not raise an exception.
    """

    known_set: set[str] = set(known_categories)
    return int((~series.isin(known_set)).sum())


# ---------------------------------------------------------------------------
# Ordinal encoding helpers
# ---------------------------------------------------------------------------


def _build_ordinal_encoder(
    dataframe: pd.DataFrame,
    columns: list[str],
    category_orders: dict[str, list[str]],
) -> tuple[OrdinalEncoder, list[str]]:
    """
    Fit a sklearn OrdinalEncoder on the given columns.

    Returns
    -------
    (fitted OrdinalEncoder, list of columns actually encoded)
    """

    present_columns = [c for c in columns if c in dataframe.columns]

    if not present_columns:
        return OrdinalEncoder(), []

    categories = [category_orders[col] for col in present_columns]

    encoder = OrdinalEncoder(
        categories=categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1,          # -1 = unseen category during inference
        encoded_missing_value=-2,  # -2 = missing value during inference
        dtype=np.float32,
    )

    normalised = dataframe[present_columns].apply(_normalise_ordinal_column)
    encoder.fit(normalised)

    return encoder, present_columns


def _apply_ordinal_encoder(
    dataframe: pd.DataFrame,
    encoder: OrdinalEncoder,
    columns: list[str],
    category_orders: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[EncodingStats]]:
    """
    Transform ordinal columns in-place (column names are preserved).
    Original string values are replaced by float codes.
    """

    if not columns:
        return dataframe, []

    stats: list[EncodingStats] = []
    normalised = dataframe[columns].apply(_normalise_ordinal_column)
    encoded_array = encoder.transform(normalised)

    # Build new columns dict for efficient assignment (avoids fragmentation)
    new_cols: dict[str, np.ndarray] = {}

    for idx, col in enumerate(columns):
        null_before = int(dataframe[col].isna().sum())
        unique_vals = sorted(normalised[col].dropna().unique().tolist())
        unknown_count = _detect_unknown_categories(
            normalised[col], category_orders[col]
        )

        new_cols[col] = encoded_array[:, idx].astype(np.float32)

        stats.append(
            EncodingStats(
                original_column=col,
                encoding_strategy="ordinal",
                unique_values_found=unique_vals,
                output_columns=[col],
                null_count_before=null_before,
                null_count_after=0,
                unknown_categories_found=unknown_count,
            )
        )

        if unknown_count > 0:
            logger.warning(
                "Ordinal column '%s': %d unknown category value(s) "
                "(encoded as -1).",
                col, unknown_count,
            )

    result_df = dataframe.assign(**new_cols)
    return result_df, stats


# ---------------------------------------------------------------------------
# One-hot encoding helpers
# ---------------------------------------------------------------------------


def _build_onehot_encoder(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> tuple[OneHotEncoder, list[str]]:
    """
    Fit a sklearn OneHotEncoder on the given nominal columns.

    - handle_unknown='ignore'  → unseen categories become all-zero rows
    - drop='if_binary'         → avoids perfect multicollinearity
    - max_categories=50        → caps high-cardinality columns
    - min_frequency=2          → merges rare categories into 'infrequent'

    Returns
    -------
    (fitted OneHotEncoder, list of columns actually encoded)
    """

    present_columns = [c for c in columns if c in dataframe.columns]

    if not present_columns:
        return OneHotEncoder(), []

    encoder = OneHotEncoder(
        drop="if_binary",
        handle_unknown="ignore",
        sparse_output=False,
        dtype=np.float32,
        min_frequency=2,
        max_categories=50,
    )

    normalised = dataframe[present_columns].apply(_normalise_string_column)
    encoder.fit(normalised)

    return encoder, present_columns


def _apply_onehot_encoder(
    dataframe: pd.DataFrame,
    encoder: OneHotEncoder,
    columns: list[str],
) -> tuple[pd.DataFrame, list[EncodingStats]]:
    """
    Transform nominal columns using one-hot encoding.

    Original string columns are dropped. Binary output columns named
    <original_col>_<category_value> are appended to the DataFrame.
    Uses pd.concat once for efficiency (avoids repeated fragmentation).
    """

    if not columns:
        return dataframe, []

    stats: list[EncodingStats] = []
    normalised = dataframe[columns].apply(_normalise_string_column)

    encoded_array = encoder.transform(normalised)
    feature_names: list[str] = encoder.get_feature_names_out(columns).tolist()

    encoded_df = pd.DataFrame(
        encoded_array,
        columns=feature_names,
        index=dataframe.index,
        dtype=np.float32,
    )

    # Drop original string columns and concat encoded columns at once
    result_df = pd.concat(
        [dataframe.drop(columns=columns), encoded_df],
        axis=1,
    )

    for col_idx, col in enumerate(columns):
        col_output_features = [
            name for name in feature_names
            if name.startswith(f"{col}_")
        ]
        null_before = int(dataframe[col].isna().sum())
        unique_vals = sorted(normalised[col].dropna().unique().tolist())
        vocab: list[str] = list(encoder.categories_[col_idx])
        unknown_count = _detect_unknown_categories(normalised[col], vocab)

        stats.append(
            EncodingStats(
                original_column=col,
                encoding_strategy="onehot",
                unique_values_found=unique_vals,
                output_columns=col_output_features,
                null_count_before=null_before,
                null_count_after=0,  # OHE fills with zeros — no nulls
                unknown_categories_found=unknown_count,
            )
        )

        if unknown_count > 0:
            logger.warning(
                "One-hot column '%s': %d unknown category value(s) "
                "(encoded as all-zero row).",
                col, unknown_count,
            )

    return result_df, stats


def _auto_detect_remaining_categoricals(
    dataframe: pd.DataFrame,
    already_handled: set[str],
    protected_columns: set[str],
) -> list[str]:
    """
    Find any remaining object/string columns not yet scheduled for encoding.
    These may be categorical interaction columns created by
    feature_engineering.py that are not listed in the config.
    Returns a list of column names to be one-hot encoded as a safe default.
    """

    return [
        col for col in dataframe.columns
        if col not in already_handled
        and col not in protected_columns
        and dataframe[col].dtype == object
    ]


# ---------------------------------------------------------------------------
# Main encoder orchestrator
# ---------------------------------------------------------------------------


def run_categorical_encoder(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> CategoricalEncoderResult:
    """
    Run the full categorical encoding pipeline on the engineered dataset.

    Parameters
    ----------
    input_path:
        Path to the engineered CSV. Defaults to ENGINEERED_DATASET_PATH.
    output_path:
        Path to write the encoded CSV. Defaults to _ENCODED_DATASET_PATH.

    Returns
    -------
    CategoricalEncoderResult

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    RuntimeError
        If the dataset is empty or unreadable.
    """

    if input_path is None:
        input_path = ENGINEERED_DATASET_PATH
    if output_path is None:
        output_path = _ENCODED_DATASET_PATH

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    section("SUPERVISOR CATEGORICAL ENCODING STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Output dataset: %s", output_path)
    logger.info("Encoder root  : %s", _ENCODER_ROOT)
    logger.info("Report path   : %s", _ENCODING_REPORT_PATH)

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Engineered dataset not found: {input_path}. "
            "Run feature_engineering.py first."
        )

    logger.info("Loading engineered dataset: %s", input_path)
    dataframe = pd.read_csv(input_path, low_memory=False)

    if dataframe.empty or dataframe.shape[1] == 0:
        raise RuntimeError(
            f"Loaded dataset from '{input_path}' is empty or has no columns."
        )

    rows, cols_before = dataframe.shape
    logger.info(
        "Loaded rows=%d, columns=%d, memory=%.2f MB",
        rows, cols_before,
        dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
    )

    # priority, maintenance_urgency, requires_human_review and
    # supervisor_score are outputs the live engine derives FROM
    # final_decision (see urgency_mapper.py / action_mapper.py), not
    # inputs. Left in place, priority/maintenance_urgency get one-hot
    # encoded by the auto-detect pass below and the other two pass straight
    # through as already-numeric columns -- in both cases the label leaks
    # into the feature space, which is why champion validation metrics were
    # reported near 1.0. Drop them before any encoding happens.
    leaked_columns_present = [
        col for col in DERIVED_OUTPUT_COLUMNS if col in dataframe.columns
    ]
    if leaked_columns_present:
        logger.warning(
            "Dropping %d target-derived output column(s) before encoding "
            "(leakage prevention): %s",
            len(leaked_columns_present), leaked_columns_present,
        )
        dataframe = dataframe.drop(columns=leaked_columns_present)
        rows, cols_before = dataframe.shape

    result = CategoricalEncoderResult(
        input_path=str(input_path),
        output_path=str(output_path),
        rows=rows,
        columns_before=cols_before,
        columns_after=cols_before,
    )

    # ------------------------------------------------------------------
    # Columns that must never be touched by the encoder
    # ------------------------------------------------------------------

    protected_columns: set[str] = set(METADATA_COLUMNS) | {
        TARGET_COLUMN,
        ENGINE_ID_COLUMN,
        SUBSET_COLUMN,
        CYCLE_COLUMN,
    }

    all_handled: set[str] = set(protected_columns)

    # ------------------------------------------------------------------
    # Pass 1 — Ordinal encoding (ordered categoricals)
    # ------------------------------------------------------------------

    logger.info("Pass 1 — Fitting ordinal encoder on ordered categoricals.")

    ordinal_columns_to_encode = [
        col for col in ORDINAL_COLUMN_ORDERS.keys()
        if col in dataframe.columns and col not in all_handled
    ]

    skipped_ordinal = [
        col for col in ORDINAL_COLUMN_ORDERS.keys()
        if col not in dataframe.columns
    ]

    if skipped_ordinal:
        logger.info(
            "Ordinal columns absent from dataset (skipped): %s", skipped_ordinal
        )
        result.columns_skipped.extend(skipped_ordinal)

    if ordinal_columns_to_encode:
        ordinal_encoder, fitted_ordinal_cols = _build_ordinal_encoder(
            dataframe, ordinal_columns_to_encode, ORDINAL_COLUMN_ORDERS,
        )
        dataframe, ordinal_stats = _apply_ordinal_encoder(
            dataframe, ordinal_encoder, fitted_ordinal_cols, ORDINAL_COLUMN_ORDERS,
        )
        result.ordinal_columns_encoded = fitted_ordinal_cols
        result.encoding_stats.extend(ordinal_stats)
        all_handled.update(fitted_ordinal_cols)
        logger.info(
            "Ordinal encoding applied to %d column(s): %s",
            len(fitted_ordinal_cols), fitted_ordinal_cols,
        )
    else:
        logger.info("No ordinal columns found in dataset to encode.")
        ordinal_encoder = OrdinalEncoder()
        fitted_ordinal_cols = []

    # ------------------------------------------------------------------
    # Pass 2 — One-hot encoding (nominal categoricals + auto-detected)
    # ------------------------------------------------------------------

    logger.info(
        "Pass 2 — Fitting one-hot encoder on nominal categoricals."
    )

    nominal_configured = [
        col for col in NOMINAL_COLUMNS
        if col in dataframe.columns and col not in all_handled
    ]

    skipped_nominal = [
        col for col in NOMINAL_COLUMNS
        if col not in dataframe.columns
    ]

    if skipped_nominal:
        logger.info(
            "Nominal columns absent from dataset (skipped): %s", skipped_nominal
        )
        result.columns_skipped.extend(skipped_nominal)

    # Auto-detect any remaining object columns not yet accounted for
    auto_nominal = _auto_detect_remaining_categoricals(
        dataframe,
        all_handled | set(nominal_configured),
        protected_columns,
    )

    if auto_nominal:
        logger.warning(
            "Auto-detected %d additional object column(s) not in config — "
            "one-hot encoding as safe default: %s",
            len(auto_nominal), auto_nominal,
        )
        result.auto_detected_nominal = auto_nominal

    all_nominal_to_encode = nominal_configured + auto_nominal

    if all_nominal_to_encode:
        onehot_encoder, fitted_onehot_cols = _build_onehot_encoder(
            dataframe, all_nominal_to_encode,
        )
        dataframe, onehot_stats = _apply_onehot_encoder(
            dataframe, onehot_encoder, fitted_onehot_cols,
        )
        result.onehot_columns_encoded = fitted_onehot_cols
        result.encoding_stats.extend(onehot_stats)
        all_handled.update(fitted_onehot_cols)

        new_col_count = sum(len(s.output_columns) for s in onehot_stats)
        logger.info(
            "One-hot encoding: %d column(s) → %d binary output column(s).",
            len(fitted_onehot_cols), new_col_count,
        )
    else:
        logger.info("No nominal columns found in dataset to encode.")
        onehot_encoder = OneHotEncoder()
        fitted_onehot_cols = []

    # ------------------------------------------------------------------
    # Defragment the DataFrame (consolidates memory after concat ops)
    # ------------------------------------------------------------------

    dataframe = dataframe.copy()
    cols_after = dataframe.shape[1]
    result.columns_after = cols_after

    # ------------------------------------------------------------------
    # Save encoder artifacts
    # ------------------------------------------------------------------

    logger.info("Saving encoder artifacts to: %s", _ENCODER_ROOT)
    _ENCODER_ROOT.mkdir(parents=True, exist_ok=True)

    joblib.dump(ordinal_encoder, _ORDINAL_ENCODER_PATH)
    logger.info("Ordinal encoder saved : %s", _ORDINAL_ENCODER_PATH)

    joblib.dump(onehot_encoder, _ONEHOT_ENCODER_PATH)
    logger.info("One-hot encoder saved : %s", _ONEHOT_ENCODER_PATH)

    # Encoding manifest — consumed by training and inference pipelines
    onehot_feature_names: list[str] = []
    if fitted_onehot_cols:
        try:
            onehot_feature_names = (
                onehot_encoder.get_feature_names_out(fitted_onehot_cols).tolist()
            )
        except Exception:  # pylint: disable=broad-except
            onehot_feature_names = []

    encoding_manifest = {
        "ordinal_encoder_path": str(_ORDINAL_ENCODER_PATH),
        "onehot_encoder_path": str(_ONEHOT_ENCODER_PATH),
        "ordinal_columns": fitted_ordinal_cols,
        "ordinal_category_orders": ORDINAL_COLUMN_ORDERS,
        "onehot_columns": fitted_onehot_cols,
        "onehot_output_features": onehot_feature_names,
        "auto_detected_nominal": result.auto_detected_nominal,
        "skipped_columns": result.columns_skipped,
        "total_input_categorical_columns": len(
            ordinal_columns_to_encode + all_nominal_to_encode
        ),
        "total_output_columns": cols_after,
    }

    atomic_write_json(encoding_manifest, _ENCODING_MANIFEST_PATH)
    logger.info("Encoding manifest saved: %s", _ENCODING_MANIFEST_PATH)

    # ------------------------------------------------------------------
    # Save encoded dataset atomically
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing encoded dataset atomically.")
    atomic_write_csv(dataframe, output_path)
    logger.info("Encoded dataset written: %s", output_path)

    # ------------------------------------------------------------------
    # Save encoding report
    # ------------------------------------------------------------------

    duration = time.perf_counter() - start_time
    result.duration_seconds = round(duration, 4)

    report = result.to_dict()
    report["status"] = "success"

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _ENCODING_REPORT_PATH)
    logger.info("Encoding report written: %s", _ENCODING_REPORT_PATH)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------

    logger.info("-" * 78)
    logger.info("Rows                        : %d", rows)
    logger.info("Columns before encoding     : %d", cols_before)
    logger.info("Columns after encoding      : %d", cols_after)
    logger.info(
        "Ordinal columns encoded     : %d", len(result.ordinal_columns_encoded)
    )
    logger.info(
        "Nominal columns encoded     : %d", len(result.onehot_columns_encoded)
    )
    logger.info(
        "Auto-detected nominal       : %d", len(result.auto_detected_nominal)
    )
    logger.info(
        "Columns skipped (absent)    : %d", len(result.columns_skipped)
    )
    logger.info("Duration                    : %.3f seconds", duration)

    section("SUPERVISOR CATEGORICAL ENCODING COMPLETED")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: CategoricalEncoderResult) -> None:
    """Print a compact JSON summary to stdout."""

    summary = {
        "status": "success",
        "input_path": result.input_path,
        "output_path": result.output_path,
        "report_path": str(_ENCODING_REPORT_PATH),
        "rows": result.rows,
        "columns_before": result.columns_before,
        "columns_after": result.columns_after,
        "ordinal_columns_encoded": result.ordinal_columns_encoded,
        "onehot_columns_encoded": result.onehot_columns_encoded,
        "auto_detected_nominal": result.auto_detected_nominal,
        "columns_skipped": result.columns_skipped,
        "duration_seconds": result.duration_seconds,
        "message": (
            "Supervisor categorical encoding completed successfully."
        ),
    }

    print(json.dumps(summary, indent=2))


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 — encoding completed successfully
        1 — internal failure
    """

    try:
        result = run_categorical_encoder()
        _print_summary(result)
        return 0

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        return 1

    except RuntimeError as exc:
        logger.error("Runtime error during encoding: %s", exc)
        return 1

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error during categorical encoding: %s",
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
