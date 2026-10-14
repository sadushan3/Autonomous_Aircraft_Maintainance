"""
supervisor_cleaner.py

Data Cleaning Utilities & Pre-Preprocessing Sanitizer for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module provides lower-level domain-specific cleaning, validation, and missing
value imputation helper functions tailored specifically for raw integrated upstream
predictive maintenance datasets (RUL, Risk, Anomaly outputs).

It serves both as:
1. An operational helper module for dataset_cleaner.py and real_output_merger.py.
2. A standalone pre-cleaning sanitizer that can be run on raw/synthetic upstream CSVs
   before full dataset cleaning and feature engineering.

Cleaning Strategy
-----------------
- Engine & Cycle Validation:
  Enforces non-negative integers for engine_id and cycle. Removes orphaned or zero-cycle rows.

- Outlier & Range Clipping:
  Clips probabilities (risk_10, risk_30, risk_50, etc.) strictly to [0.0, 1.0].
  Ensures non-negative RUL bounds (predicted_rul, lower_bound, upper_bound >= 0).
  Ensures uncertainty_width >= 0 and lower_bound <= upper_bound.

- Categorical Sanitization:
  Normalizes text fields (lowercase, stripped whitespace).
  Fills missing categorical labels with domain-safe defaults ("nominal", "none", etc.).

- Missing Numerical Imputation:
  Imputes missing upstream numerical values with logical defaults (e.g. median or 0.0)
  rather than dropping rows, preserving sequence continuity for engine lifecycles.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.supervisor_cleaner

Expected inputs
---------------
- data/Maintenance_Supervisor/sample/sample_supervisor_input.csv (or specified path)

Expected outputs
---------------
- processed/Maintenance_Supervisor/supervisor_sanitized_input.csv
- reports/Maintenance_Supervisor/supervisor_cleaner_report.json

Exit codes
----------
0 — cleaning completed successfully
1 — internal failure
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    SAMPLE_DATASET_PATH,
    PROCESSED_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    RUL_NUMERICAL_FEATURES,
    RISK_NUMERICAL_FEATURES,
    ANOMALY_NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    BOOLEAN_FEATURES,
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
# Output Paths
# ---------------------------------------------------------------------------

_SANITIZED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_sanitized_input.csv"
)
_CLEANER_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT / "supervisor_cleaner_report.json"
)

# ---------------------------------------------------------------------------
# Domain Defaults & Boundaries
# ---------------------------------------------------------------------------

PROBABILITY_COLUMNS: Final[tuple[str, ...]] = (
    "risk_10", "risk_30", "risk_50",
    "uncertainty_10", "uncertainty_30", "uncertainty_50",
    "threshold_10", "threshold_30", "threshold_50",
    "control_risk", "anomaly_score", "anomaly_threshold",
    "residual_anomaly_score", "isolation_forest_score",
    "mahalanobis_score", "context_confidence",
    "top_sensor_1_score", "top_sensor_2_score", "top_sensor_3_score",
)

CATEGORICAL_DEFAULTS: Final[dict[str, str]] = {
    "lifecycle_state": "nominal",
    "risk_state": "nominal",
    "action_hint_for_supervisor": "continue_operation",
    "anomaly_severity": "none",
    "top_sensor_1": "none",
    "top_sensor_2": "none",
    "top_sensor_3": "none",
    "rul_model_used": "unknown",
    "risk_model_used": "unknown",
    "anomaly_model_used": "unknown",
}


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class CleaningSummary:
    """Statistics for supervisor cleaning operations."""

    input_path: str
    output_path: str
    initial_rows: int
    final_rows: int
    rows_dropped: int
    nulls_imputed: int
    outliers_clipped: int
    invalid_cycles_removed: int
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "output_path": self.output_path,
            "initial_rows": self.initial_rows,
            "final_rows": self.final_rows,
            "rows_dropped": self.rows_dropped,
            "nulls_imputed": self.nulls_imputed,
            "outliers_clipped": self.outliers_clipped,
            "invalid_cycles_removed": self.invalid_cycles_removed,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Domain Cleaning Functions
# ---------------------------------------------------------------------------

def clean_engine_and_cycle(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clean and validate engine_id and cycle columns."""
    initial_len = len(df)
    clean_df = df.copy()

    if ENGINE_ID_COLUMN in clean_df.columns:
        clean_df = clean_df[clean_df[ENGINE_ID_COLUMN].notna()]
        clean_df[ENGINE_ID_COLUMN] = clean_df[ENGINE_ID_COLUMN].astype(str).str.strip()
        clean_df = clean_df[clean_df[ENGINE_ID_COLUMN] != ""]

    if CYCLE_COLUMN in clean_df.columns:
        clean_df[CYCLE_COLUMN] = pd.to_numeric(clean_df[CYCLE_COLUMN], errors="coerce")
        clean_df = clean_df[clean_df[CYCLE_COLUMN].notna() & (clean_df[CYCLE_COLUMN] > 0)]
        clean_df[CYCLE_COLUMN] = clean_df[CYCLE_COLUMN].astype(int)

    dropped = initial_len - len(clean_df)
    return clean_df, dropped


def sanitize_probabilities_and_bounds(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Clip probabilities to [0, 1] and enforce non-negative RUL bounds."""
    clipped_count = 0
    clean_df = df.copy()

    # Probability columns [0, 1]
    for col in PROBABILITY_COLUMNS:
        if col in clean_df.columns:
            series = pd.to_numeric(clean_df[col], errors="coerce")
            out_of_bounds = (series < 0.0) | (series > 1.0)
            clipped_count += int(out_of_bounds.sum())
            clean_df[col] = series.clip(lower=0.0, upper=1.0)

    # RUL non-negative clipping
    for col in ["predicted_rul", "lower_bound", "upper_bound"]:
        if col in clean_df.columns:
            series = pd.to_numeric(clean_df[col], errors="coerce")
            clipped_count += int((series < 0.0).sum())
            clean_df[col] = series.clip(lower=0.0)

    # Enforce lower_bound <= upper_bound
    if "lower_bound" in clean_df.columns and "upper_bound" in clean_df.columns:
        invalid_bounds = clean_df["lower_bound"] > clean_df["upper_bound"]
        if invalid_bounds.any():
            clipped_count += int(invalid_bounds.sum())
            # Swap or set upper = lower
            clean_df.loc[invalid_bounds, "upper_bound"] = clean_df.loc[invalid_bounds, "lower_bound"]

    # Enforce uncertainty_width calculation consistency
    if all(c in clean_df.columns for c in ["lower_bound", "upper_bound", "uncertainty_width"]):
        clean_df["uncertainty_width"] = (clean_df["upper_bound"] - clean_df["lower_bound"]).clip(lower=0.0)

    return clean_df, clipped_count


def impute_missing_values(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Impute missing numerical and categorical values with domain defaults."""
    clean_df = df.copy()
    imputed_count = 0

    # Categorical defaults
    for col, default_val in CATEGORICAL_DEFAULTS.items():
        if col in clean_df.columns:
            null_mask = clean_df[col].isna() | (clean_df[col].astype(str).str.strip() == "")
            imputed_count += int(null_mask.sum())
            clean_df[col] = clean_df[col].astype(str).str.strip().str.lower()
            clean_df.loc[null_mask, col] = default_val

    # Boolean defaults
    for col in BOOLEAN_FEATURES:
        if col in clean_df.columns:
            null_mask = clean_df[col].isna()
            imputed_count += int(null_mask.sum())
            clean_df[col] = clean_df[col].fillna(False).astype(bool)

    # Numerical defaults: forward-fill per engine first, then a global
    # median/0.0 fallback only for whatever no per-engine fill could reach.
    #
    # Bug this fixes: the previous implementation filled every gap with the
    # GLOBAL column median regardless of which engine or cycle the gap
    # belonged to -- directly contradicting this module's own stated design
    # ("preserving sequence continuity for engine lifecycles", see module
    # docstring). In practice this module's caller, real_output_merger.py,
    # outer-joins a RUL/Risk component stream that (when the caller of
    # rul_failure_batch_service.py passes stride > 1 -- see that module's
    # docstring) only scores every Nth cycle against a dense per-cycle
    # Anomaly stream. Every un-scored cycle is a "missing value" here, and
    # the global-median fill made ALL of them collapse to one
    # dataset-wide constant regardless of the individual engine's actual,
    # continuously-degrading health -- confirmed directly on
    # data/Maintenance_Supervisor/Maintenance_Supervisor/integrated/
    # supervisor_integrated_input.csv: 33,110 of 33,991 rows (97.4%) shared
    # the exact same imputed predicted_rul/risk_10/risk_30 triple, erasing
    # per-engine signal for the vast majority of that batch
    # and starving any downstream model (including PPO) of real information
    # to react to. An engine's most recent real reading is a far better
    # estimate for the next few un-scored cycles than a dataset-wide
    # constant, since health degrades gradually, not in dataset-wide lockstep.
    #
    # Falls back to bfill (for gaps before an engine's first real reading)
    # and finally the global median/0.0 only for an engine with NO real
    # reading at all for that column.
    num_cols = RUL_NUMERICAL_FEATURES + RISK_NUMERICAL_FEATURES + ANOMALY_NUMERICAL_FEATURES
    has_trajectory_cols = ENGINE_ID_COLUMN in clean_df.columns and CYCLE_COLUMN in clean_df.columns
    if has_trajectory_cols:
        sort_cols = [ENGINE_ID_COLUMN, CYCLE_COLUMN]
        order = clean_df[sort_cols].apply(pd.to_numeric, errors="coerce").sort_values(sort_cols).index
    for col in num_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")
            null_mask = clean_df[col].isna()
            if not null_mask.any():
                continue
            imputed_count += int(null_mask.sum())

            if has_trajectory_cols:
                clean_df.loc[order, col] = (
                    clean_df.loc[order]
                    .groupby(clean_df.loc[order, ENGINE_ID_COLUMN])[col]
                    .transform(lambda s: s.ffill().bfill())
                )

            still_missing = clean_df[col].isna()
            if still_missing.any():
                median_val = clean_df[col].median()
                fill_val = median_val if not pd.isna(median_val) else 0.0
                clean_df.loc[still_missing, col] = fill_val

    return clean_df, imputed_count


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def run_supervisor_cleaner(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> CleaningSummary:
    """Run standalone supervisor data cleaning and sanitization pipeline."""
    if input_path is None:
        input_path = SAMPLE_DATASET_PATH
    if output_path is None:
        output_path = _SANITIZED_DATASET_PATH

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    section("SUPERVISOR CLEANER STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Output dataset: %s", output_path)

    start_time = time.perf_counter()

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    initial_rows = len(df)
    logger.info("Loaded %d rows.", initial_rows)

    # 1. Clean engine and cycle IDs
    df, dropped_cycles = clean_engine_and_cycle(df)

    # 2. Sanitize probabilities and RUL bounds
    df, clipped_outliers = sanitize_probabilities_and_bounds(df)

    # 3. Impute missing values
    df, imputed_nulls = impute_missing_values(df)

    final_rows = len(df)
    rows_dropped = initial_rows - final_rows
    duration = time.perf_counter() - start_time

    summary = CleaningSummary(
        input_path=str(input_path),
        output_path=str(output_path),
        initial_rows=initial_rows,
        final_rows=final_rows,
        rows_dropped=rows_dropped,
        nulls_imputed=imputed_nulls,
        outliers_clipped=clipped_outliers,
        invalid_cycles_removed=dropped_cycles,
        duration_seconds=round(duration, 4),
    )

    # Save outputs
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(df, output_path)
    logger.info("Sanitized dataset written to: %s", output_path)

    _CLEANER_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report_data = summary.to_dict()
    report_data["status"] = "success"
    atomic_write_json(report_data, _CLEANER_REPORT_PATH)
    logger.info("Report written to: %s", _CLEANER_REPORT_PATH)

    logger.info("-" * 78)
    logger.info("Initial rows          : %d", initial_rows)
    logger.info("Final rows            : %d", final_rows)
    logger.info("Rows dropped          : %d", rows_dropped)
    logger.info("Nulls imputed         : %d", imputed_nulls)
    logger.info("Outliers clipped      : %d", clipped_outliers)
    logger.info("Duration              : %.3f seconds", duration)
    section("SUPERVISOR CLEANER COMPLETED")

    return summary


def main() -> int:
    """CLI Entrypoint."""
    try:
        summary = run_supervisor_cleaner()
        print(json.dumps(summary.to_dict(), indent=2))
        return 0
    except Exception as exc:
        logger.error("Error running supervisor_cleaner: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
