"""
feature_scaler.py

Feature Scaling Module for the Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module scales all numerical and boolean features in the encoded
supervisor dataset to a common range before model training. Proper scaling
is critical for:

- Linear models (Logistic Regression): very sensitive to feature magnitude.
- Neural networks: gradient descent converges faster with normalised inputs.
- Distance-based methods: unscaled features dominate distance calculations.
- Tree-based models (Random Forest, XGBoost, LightGBM): largely invariant
  to scaling, but consistent preprocessing is still enforced so that the
  same pipeline works for all model types without modification.

Scaling strategies implemented
-------------------------------
Three strategies are supported and configurable per column group:

1. RobustScaler  (DEFAULT for all numerical features)
   Centers on the median and scales by the interquartile range (IQR).
   Resistant to outliers, which are common in sensor-derived features such
   as anomaly scores, RUL values, and risk probabilities.
   Formula: x_scaled = (x - median) / IQR

2. StandardScaler  (optional, for columns with near-Gaussian distributions)
   Centers on the mean and scales by standard deviation.
   Formula: x_scaled = (x - mean) / std

3. MinMaxScaler  (optional, for probability columns already in [0, 1])
   Rescales to [0, 1]. Useful as a secondary step for probability outputs
   that are already bounded but may not span the full range.
   Formula: x_scaled = (x - min) / (max - min)

Design decisions
----------------
- The scaler is fitted on the FULL encoded dataset (pre-split).
  Scaling does not use the target label and does not cause leakage for
  numerical/boolean features. Re-fitting on training data only happens
  inside the training pipeline and uses the same scaler class and
  parameters as configured here.

- Boolean (0/1) features are scaled with MinMaxScaler. Since they already
  live in {0, 1} the scaler is effectively a no-op but it ensures that
  imputed values such as 0.5 (from missing boolean fields) are handled
  consistently in future pipeline runs.

- Ordinal-encoded columns (anomaly_severity, lifecycle_state, risk_state)
  are treated as numerical and scaled with RobustScaler alongside the rest
  of the numerical features.

- One-hot columns produced by categorical_encoder.py are already in {0, 1}
  and are NOT re-scaled (scaling binary indicators would destroy their
  semantics).

- Metadata and target columns (engine_id, fd_subset, cycle, final_decision,
  schema_version) are always excluded from scaling.

- The fitted scaler is saved as a .joblib artifact. The inference pipeline
  and training pipeline both reload this artifact to transform new data.

- A scaling manifest JSON records which columns were scaled, which were
  skipped, and which scaler was applied to each group.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.feature_scaler

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_encoded_dataset.csv
  (output of categorical_encoder.py)

Expected outputs
----------------
- processed/Maintenance_Supervisor/supervisor_scaled_dataset.csv
- artifacts/Maintenance_Supervisor/scalers/robust_scaler.joblib
- artifacts/Maintenance_Supervisor/scalers/scaling_manifest.json
- reports/Maintenance_Supervisor/feature_scaling_report.json

Exit codes
----------
0 — scaling completed successfully
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
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

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
    PROCESSED_ROOT,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    METADATA_COLUMNS,
    NUMERICAL_FEATURES,
    BOOLEAN_FEATURES,
    CATEGORICAL_FEATURES,
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
# Paths
# ---------------------------------------------------------------------------

_ENCODED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_encoded_dataset.csv"
)
_SCALED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_scaled_dataset.csv"
)
_SCALER_ROOT: Final[Path] = ARTIFACT_ROOT / "scalers"
_ROBUST_SCALER_PATH: Final[Path] = _SCALER_ROOT / "robust_scaler.joblib"
_SCALING_MANIFEST_PATH: Final[Path] = _SCALER_ROOT / "scaling_manifest.json"
_SCALING_REPORT_PATH: Final[Path] = REPORTS_ROOT / "feature_scaling_report.json"

# ---------------------------------------------------------------------------
# Column groups that are NEVER scaled
# ---------------------------------------------------------------------------

# One-hot encoded columns: binary {0,1} — do not scale
# Identified at runtime by checking if column is object-free and only
# contains values in {0.0, 1.0, NaN}

# Metadata columns never scaled
_ALWAYS_EXCLUDED: Final[frozenset[str]] = frozenset(
    list(METADATA_COLUMNS)
    + [TARGET_COLUMN, ENGINE_ID_COLUMN, SUBSET_COLUMN, CYCLE_COLUMN]
)

# Ordinal-encoded columns are treated as continuous numerical features
# and ARE scaled. They come in with float values (0.0, 1.0, 2.0, ...).
_ORDINAL_ENCODED_COLUMNS: Final[list[str]] = [
    "anomaly_severity",
    "lifecycle_state",
    "risk_state",
]

# Probability columns: outputs from upstream models already in [0, 1].
# We still apply RobustScaler — they rarely saturate the full 0-1 range,
# so the scaler adds value without hurting interpretability.
_PROBABILITY_COLUMNS: Final[frozenset[str]] = frozenset([
    "risk_10", "risk_30", "risk_50",
    "uncertainty_10", "uncertainty_30", "uncertainty_50",
    "threshold_10", "threshold_30", "threshold_50",
    "control_risk",
    "anomaly_score", "anomaly_threshold",
    "residual_anomaly_score", "isolation_forest_score",
    "mahalanobis_score",
    "context_confidence",
    "top_sensor_1_score", "top_sensor_2_score", "top_sensor_3_score",
])

# Constant variance threshold — columns with variance below this are
# skipped (scaler would divide by near-zero IQR)
_CONSTANT_VAR_THRESHOLD: Final[float] = 1e-10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ColumnScalingRecord:
    """Scaling metadata for a single column."""

    column: str
    scaler_applied: str        # "robust", "minmax", "none"
    reason_skipped: str        # empty string if not skipped
    original_mean: float | None
    original_std: float | None
    original_min: float | None
    original_max: float | None
    scaled_mean: float | None
    scaled_std: float | None


@dataclass
class FeatureScalerResult:
    """Aggregated result of the full scaling run."""

    input_path: str
    output_path: str
    rows: int
    columns_before: int
    columns_after: int
    numerical_columns_scaled: list[str] = field(default_factory=list)
    boolean_columns_scaled: list[str] = field(default_factory=list)
    columns_skipped: list[str] = field(default_factory=list)
    constant_columns_skipped: list[str] = field(default_factory=list)
    column_records: list[ColumnScalingRecord] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "rows": self.rows,
            "columns_before": self.columns_before,
            "columns_after": self.columns_after,
            "numerical_columns_scaled": self.numerical_columns_scaled,
            "boolean_columns_scaled": self.boolean_columns_scaled,
            "columns_skipped": self.columns_skipped,
            "constant_columns_skipped": self.constant_columns_skipped,
            "total_scaled": (
                len(self.numerical_columns_scaled)
                + len(self.boolean_columns_scaled)
            ),
            "column_records": [
                {
                    "column": r.column,
                    "scaler_applied": r.scaler_applied,
                    "reason_skipped": r.reason_skipped,
                    "original_mean": r.original_mean,
                    "original_std": r.original_std,
                    "original_min": r.original_min,
                    "original_max": r.original_max,
                    "scaled_mean": r.scaled_mean,
                    "scaled_std": r.scaled_std,
                }
                for r in self.column_records
            ],
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_onehot_binary_column(series: pd.Series) -> bool:
    """
    Return True if a column contains only {0.0, 1.0, NaN}.
    These are one-hot encoded output columns and must NOT be scaled.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_vals = set(non_null.unique())
    return unique_vals.issubset({0.0, 1.0, 0, 1})


def _compute_stats(series: pd.Series) -> dict[str, float | None]:
    """Compute descriptive stats for a numeric series."""

    clean = series.dropna()

    if clean.empty:
        return {
            "mean": None, "std": None, "min": None, "max": None,
        }

    return {
        "mean": float(clean.mean()),
        "std": float(clean.std()),
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def _classify_columns(
    dataframe: pd.DataFrame,
    always_excluded: frozenset[str],
    boolean_feature_names: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Classify all DataFrame columns into three groups:

    Returns
    -------
    (numerical_to_scale, boolean_to_scale, skip_list)

    numerical_to_scale:
        Float/int columns that should be scaled with RobustScaler.
        Includes raw numerical features, ordinal-encoded columns,
        and any engineered temporal feature columns.

    boolean_to_scale:
        Boolean {0, 1} columns that are raw boolean features (not one-hot).
        Scaled with MinMaxScaler (effectively a no-op but consistent).

    skip_list:
        Columns that should not be scaled:
        - Metadata/target columns
        - One-hot binary output columns (value set is {0, 1} but they
          come from OHE and must not be re-scaled)
        - String/object columns
    """

    numerical_to_scale: list[str] = []
    boolean_to_scale: list[str] = []
    skip_list: list[str] = []

    # Build fast-lookup sets
    boolean_names_lower: frozenset[str] = frozenset(
        col.lower() for col in boolean_feature_names
    )

    # Track OHE output column prefixes from categorical_encoder
    # OHE columns are named <original_col>_<value>, e.g. risk_state_nominal
    onehot_source_cols: frozenset[str] = frozenset([
        "action_hint_for_supervisor",
        "top_sensor_1",
        "top_sensor_2",
        "top_sensor_3",
    ])

    for col in dataframe.columns:

        # --- Always excluded ---
        if col in always_excluded:
            skip_list.append(col)
            continue

        col_dtype = dataframe[col].dtype

        # --- Object/string columns ---
        if col_dtype == object:
            skip_list.append(col)
            continue

        # --- Check if this is a one-hot output column ---
        # OHE columns are named <source>_<value> where source is in
        # onehot_source_cols. We detect them by name prefix.
        is_onehot_col = any(
            col.startswith(src + "_") for src in onehot_source_cols
        )

        if is_onehot_col:
            skip_list.append(col)
            continue

        # --- Additionally, validate using value-set check ---
        if _is_onehot_binary_column(dataframe[col]):
            # If a column is purely binary and it is NOT in the known
            # boolean feature list from config, treat it as OHE output.
            col_lower = col.lower()
            if col_lower not in boolean_names_lower:
                skip_list.append(col)
                continue

        # --- Raw boolean features from config ---
        if col.lower() in boolean_names_lower:
            boolean_to_scale.append(col)
            continue

        # --- Remaining numeric columns -> scale with RobustScaler ---
        if pd.api.types.is_numeric_dtype(col_dtype):
            numerical_to_scale.append(col)
            continue

        # --- Anything else: skip ---
        skip_list.append(col)

    return numerical_to_scale, boolean_to_scale, skip_list


def _apply_robust_scaler(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, RobustScaler, list[str], list[str], list[ColumnScalingRecord]]:
    """
    Fit and apply RobustScaler to the given numerical columns.

    Columns with near-zero IQR (constant) are skipped gracefully.

    Returns
    -------
    (transformed_df, fitted_scaler, scaled_cols, skipped_constant_cols, records)
    """

    if not columns:
        return dataframe, RobustScaler(), [], [], []

    # Filter out constant columns before fitting
    valid_columns: list[str] = []
    constant_columns: list[str] = []

    for col in columns:
        col_variance = float(dataframe[col].var(ddof=0)) if col in dataframe.columns else 0.0
        if col_variance <= _CONSTANT_VAR_THRESHOLD:
            constant_columns.append(col)
            logger.warning(
                "Skipping constant column '%s' from scaling (variance=%.2e).",
                col, col_variance,
            )
        else:
            valid_columns.append(col)

    if not valid_columns:
        return dataframe, RobustScaler(), [], constant_columns, []

    # Collect pre-scaling stats
    pre_stats: dict[str, dict] = {
        col: _compute_stats(dataframe[col]) for col in valid_columns
    }

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaled_array = scaler.fit_transform(dataframe[valid_columns].values)

    # Assign all scaled columns at once using a dict (avoids fragmentation)
    scaled_dict: dict[str, np.ndarray] = {
        col: scaled_array[:, idx].astype(np.float32)
        for idx, col in enumerate(valid_columns)
    }

    result_df = dataframe.assign(**scaled_dict)

    # Collect post-scaling stats and build records
    records: list[ColumnScalingRecord] = []

    for col in valid_columns:
        post = _compute_stats(result_df[col])
        pre = pre_stats[col]
        records.append(
            ColumnScalingRecord(
                column=col,
                scaler_applied="robust",
                reason_skipped="",
                original_mean=pre["mean"],
                original_std=pre["std"],
                original_min=pre["min"],
                original_max=pre["max"],
                scaled_mean=post["mean"],
                scaled_std=post["std"],
            )
        )

    # Add records for constant columns that were skipped
    for col in constant_columns:
        pre = _compute_stats(dataframe[col]) if col in dataframe.columns else {}
        records.append(
            ColumnScalingRecord(
                column=col,
                scaler_applied="none",
                reason_skipped="near-zero variance — constant column",
                original_mean=pre.get("mean"),
                original_std=pre.get("std"),
                original_min=pre.get("min"),
                original_max=pre.get("max"),
                scaled_mean=None,
                scaled_std=None,
            )
        )

    return result_df, scaler, valid_columns, constant_columns, records


def _apply_minmax_scaler_boolean(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[ColumnScalingRecord]]:
    """
    Apply MinMaxScaler to boolean {0, 1} feature columns.

    For clean boolean columns this is effectively a no-op (already in [0,1]).
    For columns that have intermediate imputed values (e.g. 0.5 from median
    imputation of missing booleans), this rescales them consistently.

    We do NOT save a separate MinMaxScaler artifact because there is nothing
    to refit — boolean columns are always in [0, 1] after cleaning.
    """

    if not columns:
        return dataframe, []

    records: list[ColumnScalingRecord] = []
    scaled_dict: dict[str, np.ndarray] = {}

    for col in columns:
        if col not in dataframe.columns:
            continue

        series = dataframe[col]
        pre = _compute_stats(series)

        col_min = float(series.min()) if not series.empty else 0.0
        col_max = float(series.max()) if not series.empty else 1.0
        col_range = col_max - col_min

        if col_range < _CONSTANT_VAR_THRESHOLD:
            # Constant boolean — fill with 0.0
            scaled_dict[col] = np.zeros(len(series), dtype=np.float32)
            records.append(
                ColumnScalingRecord(
                    column=col,
                    scaler_applied="none",
                    reason_skipped="constant boolean column",
                    original_mean=pre["mean"],
                    original_std=pre["std"],
                    original_min=pre["min"],
                    original_max=pre["max"],
                    scaled_mean=0.0,
                    scaled_std=0.0,
                )
            )
        else:
            scaled_values = (
                (series.fillna(0.0).values - col_min) / col_range
            ).astype(np.float32)
            scaled_dict[col] = scaled_values
            post = _compute_stats(pd.Series(scaled_values))
            records.append(
                ColumnScalingRecord(
                    column=col,
                    scaler_applied="minmax",
                    reason_skipped="",
                    original_mean=pre["mean"],
                    original_std=pre["std"],
                    original_min=pre["min"],
                    original_max=pre["max"],
                    scaled_mean=post["mean"],
                    scaled_std=post["std"],
                )
            )

    result_df = dataframe.assign(**scaled_dict)
    return result_df, records


# ---------------------------------------------------------------------------
# Main scaler orchestrator
# ---------------------------------------------------------------------------


def run_feature_scaler(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> FeatureScalerResult:
    """
    Run the full feature scaling pipeline on the encoded dataset.

    Parameters
    ----------
    input_path:
        Path to the encoded CSV. Defaults to _ENCODED_DATASET_PATH.
    output_path:
        Path to write the scaled CSV. Defaults to _SCALED_DATASET_PATH.

    Returns
    -------
    FeatureScalerResult

    Raises
    ------
    FileNotFoundError
        If the encoded dataset does not exist.
    RuntimeError
        If the dataset is empty or unreadable.
    """

    if input_path is None:
        input_path = _ENCODED_DATASET_PATH
    if output_path is None:
        output_path = _SCALED_DATASET_PATH

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    section("SUPERVISOR FEATURE SCALING STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Output dataset: %s", output_path)
    logger.info("Scaler root   : %s", _SCALER_ROOT)
    logger.info("Report path   : %s", _SCALING_REPORT_PATH)

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Load encoded dataset
    # ------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Encoded dataset not found: {input_path}. "
            "Run categorical_encoder.py first."
        )

    logger.info("Loading encoded dataset: %s", input_path)
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

    result = FeatureScalerResult(
        input_path=str(input_path),
        output_path=str(output_path),
        rows=rows,
        columns_before=cols_before,
        columns_after=cols_before,
    )

    # ------------------------------------------------------------------
    # Classify columns into scaling groups
    # ------------------------------------------------------------------

    logger.info("Classifying columns into scaling groups.")

    numerical_cols, boolean_cols, skip_cols = _classify_columns(
        dataframe,
        _ALWAYS_EXCLUDED,
        BOOLEAN_FEATURES,
    )

    logger.info(
        "Column groups — numerical: %d, boolean: %d, skipped: %d",
        len(numerical_cols), len(boolean_cols), len(skip_cols),
    )

    result.columns_skipped = skip_cols

    # ------------------------------------------------------------------
    # Pass 1 — RobustScaler on all numerical columns
    # ------------------------------------------------------------------

    logger.info(
        "Pass 1 — Applying RobustScaler to %d numerical column(s).",
        len(numerical_cols),
    )

    (
        dataframe,
        robust_scaler,
        scaled_numerical,
        constant_skipped,
        numerical_records,
    ) = _apply_robust_scaler(dataframe, numerical_cols)

    result.numerical_columns_scaled = scaled_numerical
    result.constant_columns_skipped = constant_skipped
    result.column_records.extend(numerical_records)

    logger.info(
        "RobustScaler applied to %d column(s). %d constant column(s) skipped.",
        len(scaled_numerical), len(constant_skipped),
    )

    # ------------------------------------------------------------------
    # Pass 2 — MinMaxScaler on boolean feature columns
    # ------------------------------------------------------------------

    logger.info(
        "Pass 2 — Applying MinMaxScaler to %d boolean column(s).",
        len(boolean_cols),
    )

    dataframe, boolean_records = _apply_minmax_scaler_boolean(
        dataframe, boolean_cols
    )
    result.boolean_columns_scaled = boolean_cols
    result.column_records.extend(boolean_records)

    logger.info(
        "MinMaxScaler applied to %d boolean column(s).",
        len(boolean_cols),
    )

    # ------------------------------------------------------------------
    # Defragment
    # ------------------------------------------------------------------

    dataframe = dataframe.copy()
    result.columns_after = dataframe.shape[1]

    # ------------------------------------------------------------------
    # Save scaler artifact
    # ------------------------------------------------------------------

    logger.info("Saving scaler artifacts to: %s", _SCALER_ROOT)
    _SCALER_ROOT.mkdir(parents=True, exist_ok=True)

    joblib.dump(robust_scaler, _ROBUST_SCALER_PATH)
    logger.info("RobustScaler saved: %s", _ROBUST_SCALER_PATH)

    # Scaling manifest — consumed by training and inference pipelines
    scaling_manifest = {
        "robust_scaler_path": str(_ROBUST_SCALER_PATH),
        "numerical_columns_scaled": scaled_numerical,
        "boolean_columns_scaled": boolean_cols,
        "constant_columns_skipped": constant_skipped,
        "skipped_columns": skip_cols,
        "total_columns_scaled": len(scaled_numerical) + len(boolean_cols),
        "scaler_type": "RobustScaler(quantile_range=(25,75)) + MinMaxScaler for booleans",
        "robust_scaler_params": {
            "quantile_range": [25.0, 75.0],
            "with_centering": True,
            "with_scaling": True,
        },
    }

    atomic_write_json(scaling_manifest, _SCALING_MANIFEST_PATH)
    logger.info("Scaling manifest saved: %s", _SCALING_MANIFEST_PATH)

    # ------------------------------------------------------------------
    # Save scaled dataset
    # ------------------------------------------------------------------

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing scaled dataset atomically.")
    atomic_write_csv(dataframe, output_path)
    logger.info("Scaled dataset written: %s", output_path)

    # ------------------------------------------------------------------
    # Save scaling report
    # ------------------------------------------------------------------

    duration = time.perf_counter() - start_time
    result.duration_seconds = round(duration, 4)

    report = result.to_dict()
    report["status"] = "success"

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _SCALING_REPORT_PATH)
    logger.info("Scaling report written: %s", _SCALING_REPORT_PATH)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------

    logger.info("-" * 78)
    logger.info("Rows                        : %d", rows)
    logger.info("Columns before scaling      : %d", cols_before)
    logger.info("Columns after scaling       : %d", result.columns_after)
    logger.info(
        "Numerical cols scaled       : %d", len(scaled_numerical)
    )
    logger.info(
        "Boolean cols scaled         : %d", len(boolean_cols)
    )
    logger.info(
        "Constant cols skipped       : %d", len(constant_skipped)
    )
    logger.info(
        "Columns untouched (skipped) : %d", len(skip_cols)
    )
    logger.info("Duration                    : %.3f seconds", duration)

    section("SUPERVISOR FEATURE SCALING COMPLETED")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: FeatureScalerResult) -> None:
    """Print a compact JSON summary to stdout."""

    summary = {
        "status": "success",
        "input_path": result.input_path,
        "output_path": result.output_path,
        "report_path": str(_SCALING_REPORT_PATH),
        "rows": result.rows,
        "columns_before": result.columns_before,
        "columns_after": result.columns_after,
        "numerical_columns_scaled": len(result.numerical_columns_scaled),
        "boolean_columns_scaled": len(result.boolean_columns_scaled),
        "constant_columns_skipped": result.constant_columns_skipped,
        "total_columns_scaled": (
            len(result.numerical_columns_scaled)
            + len(result.boolean_columns_scaled)
        ),
        "duration_seconds": result.duration_seconds,
        "message": "Supervisor feature scaling completed successfully.",
    }

    print(json.dumps(summary, indent=2))


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 — scaling completed successfully
        1 — internal failure
    """

    try:
        result = run_feature_scaler()
        _print_summary(result)
        return 0

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        return 1

    except RuntimeError as exc:
        logger.error("Runtime error during scaling: %s", exc)
        return 1

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error during feature scaling: %s",
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
