"""
feature_selector.py

Feature Selection Module for the Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module performs automated feature selection on the scaled supervisor
dataset to identify the most informative features for model training.
Reducing the feature set to only relevant variables has three benefits:

1. Improved model generalisation  — removes noise features that cause
   overfitting.
2. Reduced training time           — fewer columns to process.
3. Improved interpretability       — selected features are the ones that
   actually drive the supervisor's maintenance decisions.

Feature selection strategy
--------------------------
The module applies THREE independent selection methods and combines their
results via intersection voting, where a feature must be selected by at
least K out of N methods to be retained.

Method 1 — Mutual Information (filter method)
    Computes the mutual information between each feature and the
    ordinal-encoded target. MI measures non-linear dependency and does not
    assume any particular relationship shape. Features below a configurable
    percentile threshold are dropped.

Method 2 — Random Forest feature importance (embedded method)
    Trains a lightweight Random Forest classifier on the full dataset and
    extracts mean impurity-decrease importances. Features with zero or
    near-zero importance are dropped.

Method 3 — Variance threshold (filter method)
    Removes features whose variance is below a configurable threshold.
    Catches any near-constant features that survived earlier preprocessing
    steps.

Consensus voting
    A feature is selected if it passes at least 2 of the 3 methods.
    This consensus approach is more robust than any single method and
    reduces the risk of dropping a useful feature due to a quirk of one
    particular criterion.

Design decisions
----------------
- Feature selection is performed on the FULL dataset (pre-split) because:
  • MI and RF importance are computed against the target, but they use the
    full distribution — not held-out performance — so this does not
    constitute leakage. (Leakage only occurs when test-set labels influence
    training-set feature values.)
  • The selected feature list is locked before any train/val/test split.

- Metadata columns (engine_id, fd_subset, cycle), the target column, and
  any remaining string columns are always excluded from feature selection.

- The selected feature list is saved as a JSON artifact so that the
  training pipeline, inference pipeline, and evaluation pipeline all use
  an identical feature set.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.feature_selector

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_scaled_dataset.csv
  (output of feature_scaler.py)

Expected outputs
----------------
- artifacts/Maintenance_Supervisor/selected_features.json
- reports/Maintenance_Supervisor/feature_selection_report.json

Exit codes
----------
0 — feature selection completed successfully
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

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
    PROCESSED_ROOT,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    METADATA_COLUMNS,
    DECISION_TO_SEVERITY,
    FORBIDDEN_FEATURE_COLUMNS,
    DERIVED_OUTPUT_COLUMNS,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
)

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCALED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_scaled_dataset.csv"
)
_SELECTED_FEATURES_PATH: Final[Path] = (
    ARTIFACT_ROOT / "selected_features.json"
)
_SELECTION_REPORT_PATH: Final[Path] = (
    REPORTS_ROOT / "feature_selection_report.json"
)

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

# Minimum number of selection methods (out of 3) that must approve a feature
# for it to be retained. 2 = majority vote.
MIN_VOTES_REQUIRED: Final[int] = 2

# Mutual information: features below this percentile of MI scores are dropped
MI_PERCENTILE_THRESHOLD: Final[float] = 10.0

# Random Forest: features with importance below this fraction of the maximum
# importance are dropped
RF_IMPORTANCE_FRACTION_THRESHOLD: Final[float] = 0.005

# Variance: features with variance below this value are dropped
VARIANCE_THRESHOLD: Final[float] = 1e-8

# Random Forest configuration for importance computation (lightweight)
RF_N_ESTIMATORS: Final[int] = 100
RF_MAX_DEPTH: Final[int] = 12
RF_RANDOM_STATE: Final[int] = 42
RF_N_JOBS: Final[int] = -1
RF_MIN_SAMPLES_LEAF: Final[int] = 5

# Columns that must never be used as features.
#
# FORBIDDEN_FEATURE_COLUMNS and DERIVED_OUTPUT_COLUMNS are leakage guards
# duplicated here (not just relied on via leakage_guard.py's separate
# report) because this module can run against a scaled dataset that never
# passed through the leakage guard. categorical_encoder.py already drops
# DERIVED_OUTPUT_COLUMNS before encoding, but this exclusion stays as a
# second layer in case the scaled dataset was produced some other way.
_ALWAYS_EXCLUDED: Final[frozenset[str]] = frozenset(
    list(METADATA_COLUMNS)
    + list(FORBIDDEN_FEATURE_COLUMNS)
    + list(DERIVED_OUTPUT_COLUMNS)
    + [TARGET_COLUMN, ENGINE_ID_COLUMN, SUBSET_COLUMN, CYCLE_COLUMN,
       "schema_version"]
)

# One-hot encoding expands a forbidden column like "priority" into
# "priority_critical", "priority_high", etc. Those expanded columns carry
# the same leakage but don't match _ALWAYS_EXCLUDED by exact name, so any
# candidate column whose name starts with one of these prefixes is excluded
# too.
_FORBIDDEN_PREFIXES: Final[tuple[str, ...]] = tuple(
    f"{name}_" for name in set(FORBIDDEN_FEATURE_COLUMNS) | set(DERIVED_OUTPUT_COLUMNS)
)


def _is_forbidden(column: str) -> bool:
    """True if `column` is an excluded name or a one-hot expansion of one."""

    return column in _ALWAYS_EXCLUDED or column.startswith(_FORBIDDEN_PREFIXES)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FeatureScore:
    """Score record for a single feature across all selection methods."""

    column: str
    mutual_information: float
    mi_selected: bool
    rf_importance: float
    rf_selected: bool
    variance: float
    variance_selected: bool
    vote_count: int
    final_selected: bool


@dataclass
class FeatureSelectorResult:
    """Aggregated result of the full feature selection run."""

    input_path: str
    rows: int
    total_features_evaluated: int
    features_selected: int
    features_dropped: int
    method1_mi_selected: int
    method2_rf_selected: int
    method3_var_selected: int
    min_votes_required: int
    selected_feature_names: list[str] = field(default_factory=list)
    dropped_feature_names: list[str] = field(default_factory=list)
    feature_scores: list[FeatureScore] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "input_path": self.input_path,
            "rows": self.rows,
            "total_features_evaluated": self.total_features_evaluated,
            "features_selected": self.features_selected,
            "features_dropped": self.features_dropped,
            "method1_mi_selected": self.method1_mi_selected,
            "method2_rf_selected": self.method2_rf_selected,
            "method3_var_selected": self.method3_var_selected,
            "min_votes_required": self.min_votes_required,
            "selected_feature_names": self.selected_feature_names,
            "dropped_feature_names": self.dropped_feature_names,
            "feature_scores": [
                {
                    "column": fs.column,
                    "mutual_information": fs.mutual_information,
                    "mi_selected": fs.mi_selected,
                    "rf_importance": fs.rf_importance,
                    "rf_selected": fs.rf_selected,
                    "variance": fs.variance,
                    "variance_selected": fs.variance_selected,
                    "vote_count": fs.vote_count,
                    "final_selected": fs.final_selected,
                }
                for fs in self.feature_scores
            ],
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Target encoding helper
# ---------------------------------------------------------------------------


def _encode_target(series: pd.Series) -> pd.Series:
    """
    Ordinal-encode the target column using DECISION_TO_SEVERITY.

    continue_operation    -> 0
    monitor_closely       -> 1
    schedule_inspection   -> 2
    schedule_maintenance  -> 3
    immediate_maintenance -> 4
    """

    encoded = (
        series.astype(str).str.strip().str.lower().map(DECISION_TO_SEVERITY)
    )
    return pd.to_numeric(encoded, errors="coerce")


# ---------------------------------------------------------------------------
# Method 1 — Mutual Information
# ---------------------------------------------------------------------------


def _compute_mutual_information(
    features: pd.DataFrame,
    target: pd.Series,
    percentile_threshold: float,
) -> tuple[dict[str, float], set[str]]:
    """
    Compute mutual information between each feature column and the target.

    Parameters
    ----------
    features:
        DataFrame of numeric feature columns (NaN filled with 0).
    target:
        Integer-encoded target series.
    percentile_threshold:
        Features below this percentile of MI scores are dropped.

    Returns
    -------
    (mi_scores_dict, selected_set)
    """

    logger.info("Computing mutual information for %d features.", features.shape[1])

    mi_scores = mutual_info_classif(
        features.values,
        target.values,
        discrete_features=False,
        random_state=RF_RANDOM_STATE,
        n_neighbors=5,
    )

    mi_dict: dict[str, float] = {
        col: float(score) for col, score in zip(features.columns, mi_scores)
    }

    threshold = float(np.percentile(mi_scores, percentile_threshold))
    selected: set[str] = {col for col, score in mi_dict.items() if score > threshold}

    logger.info(
        "MI threshold (%.0f%% percentile): %.6f — %d features selected.",
        percentile_threshold, threshold, len(selected),
    )

    return mi_dict, selected


# ---------------------------------------------------------------------------
# Method 2 — Random Forest importance
# ---------------------------------------------------------------------------


def _compute_rf_importance(
    features: pd.DataFrame,
    target: pd.Series,
    importance_fraction_threshold: float,
) -> tuple[dict[str, float], set[str]]:
    """
    Train a lightweight Random Forest and extract feature importances.

    Parameters
    ----------
    features:
        DataFrame of numeric feature columns (NaN filled with 0).
    target:
        Integer-encoded target series.
    importance_fraction_threshold:
        Features with importance below this fraction of the maximum
        importance value are dropped.

    Returns
    -------
    (importance_dict, selected_set)
    """

    logger.info(
        "Training Random Forest (n=%d, depth=%d) for importance estimation.",
        RF_N_ESTIMATORS, RF_MAX_DEPTH,
    )

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=RF_RANDOM_STATE,
        n_jobs=RF_N_JOBS,
        class_weight="balanced",
    )

    rf.fit(features.values, target.values)

    importance_dict: dict[str, float] = {
        col: float(imp)
        for col, imp in zip(features.columns, rf.feature_importances_)
    }

    max_importance = max(importance_dict.values()) if importance_dict else 1.0
    abs_threshold = max_importance * importance_fraction_threshold

    selected: set[str] = {
        col for col, imp in importance_dict.items()
        if imp >= abs_threshold
    }

    logger.info(
        "RF importance threshold: %.6f (%.1f%% of max=%.6f) — %d features selected.",
        abs_threshold,
        importance_fraction_threshold * 100,
        max_importance,
        len(selected),
    )

    return importance_dict, selected


# ---------------------------------------------------------------------------
# Method 3 — Variance threshold
# ---------------------------------------------------------------------------


def _compute_variance_selection(
    features: pd.DataFrame,
    variance_threshold: float,
) -> tuple[dict[str, float], set[str]]:
    """
    Filter features by variance.

    Parameters
    ----------
    features:
        DataFrame of numeric feature columns.
    variance_threshold:
        Features with variance at or below this value are dropped.

    Returns
    -------
    (variance_dict, selected_set)
    """

    variance_dict: dict[str, float] = {
        col: float(features[col].var(ddof=0))
        for col in features.columns
    }

    selected: set[str] = {
        col for col, var in variance_dict.items()
        if var > variance_threshold
    }

    dropped_count = len(variance_dict) - len(selected)

    logger.info(
        "Variance threshold: %.2e — %d features selected, %d dropped.",
        variance_threshold, len(selected), dropped_count,
    )

    return variance_dict, selected


# ---------------------------------------------------------------------------
# Main feature selection orchestrator
# ---------------------------------------------------------------------------


def run_feature_selector(
    input_path: Path | None = None,
) -> FeatureSelectorResult:
    """
    Run the full feature selection pipeline on the scaled dataset.

    Parameters
    ----------
    input_path:
        Path to the scaled CSV. Defaults to _SCALED_DATASET_PATH.

    Returns
    -------
    FeatureSelectorResult

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    RuntimeError
        If the dataset is empty or target encoding fails.
    """

    if input_path is None:
        input_path = _SCALED_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("SUPERVISOR FEATURE SELECTION STARTED")
    logger.info("Input dataset     : %s", input_path)
    logger.info("Selected features : %s", _SELECTED_FEATURES_PATH)
    logger.info("Report path       : %s", _SELECTION_REPORT_PATH)
    logger.info("Min votes required: %d / 3", MIN_VOTES_REQUIRED)

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"Scaled dataset not found: {input_path}. "
            "Run feature_scaler.py first."
        )

    logger.info("Loading scaled dataset: %s", input_path)
    dataframe = pd.read_csv(input_path, low_memory=False)

    if dataframe.empty or dataframe.shape[1] == 0:
        raise RuntimeError(
            f"Loaded dataset from '{input_path}' is empty or has no columns."
        )

    rows = len(dataframe)
    logger.info(
        "Loaded rows=%d, columns=%d, memory=%.2f MB",
        rows, dataframe.shape[1],
        dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
    )

    # ------------------------------------------------------------------
    # Encode target
    # ------------------------------------------------------------------

    if TARGET_COLUMN not in dataframe.columns:
        raise RuntimeError(
            f"Target column '{TARGET_COLUMN}' not found in dataset."
        )

    target_encoded = _encode_target(dataframe[TARGET_COLUMN])
    valid_mask = target_encoded.notna()
    valid_count = int(valid_mask.sum())

    if valid_count < 100:
        raise RuntimeError(
            f"Only {valid_count} valid target values after encoding. "
            "Need at least 100 for reliable feature selection."
        )

    logger.info(
        "Target encoded: %d valid rows (%.1f%% of total).",
        valid_count, 100.0 * valid_count / rows,
    )

    # ------------------------------------------------------------------
    # Identify candidate feature columns
    # ------------------------------------------------------------------

    candidate_columns: list[str] = [
        col for col in dataframe.columns
        if not _is_forbidden(col)
        and pd.api.types.is_numeric_dtype(dataframe[col])
    ]

    logger.info(
        "Candidate feature columns: %d (excluded %d metadata/target/string cols).",
        len(candidate_columns),
        dataframe.shape[1] - len(candidate_columns),
    )

    if not candidate_columns:
        raise RuntimeError("No numeric candidate features found in dataset.")

    # ------------------------------------------------------------------
    # Prepare feature matrix (use only valid-target rows, fill NaN)
    # ------------------------------------------------------------------

    features_df = dataframe.loc[valid_mask, candidate_columns].fillna(0.0)
    target_series = target_encoded.loc[valid_mask].astype(int)

    logger.info(
        "Feature matrix: %d rows x %d columns.",
        features_df.shape[0], features_df.shape[1],
    )

    # ------------------------------------------------------------------
    # Method 1 — Mutual Information
    # ------------------------------------------------------------------

    logger.info("Method 1/3 — Mutual Information analysis.")
    mi_scores, mi_selected = _compute_mutual_information(
        features_df, target_series, MI_PERCENTILE_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Method 2 — Random Forest importance
    # ------------------------------------------------------------------

    logger.info("Method 2/3 — Random Forest importance estimation.")
    rf_scores, rf_selected = _compute_rf_importance(
        features_df, target_series, RF_IMPORTANCE_FRACTION_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Method 3 — Variance threshold
    # ------------------------------------------------------------------

    logger.info("Method 3/3 — Variance threshold filter.")
    var_scores, var_selected = _compute_variance_selection(
        features_df, VARIANCE_THRESHOLD,
    )

    # ------------------------------------------------------------------
    # Consensus voting
    # ------------------------------------------------------------------

    logger.info("Computing consensus votes (min %d / 3).", MIN_VOTES_REQUIRED)

    feature_score_list: list[FeatureScore] = []
    selected_features: list[str] = []
    dropped_features: list[str] = []

    for col in candidate_columns:
        mi_val = mi_scores.get(col, 0.0)
        rf_val = rf_scores.get(col, 0.0)
        var_val = var_scores.get(col, 0.0)

        mi_ok = col in mi_selected
        rf_ok = col in rf_selected
        var_ok = col in var_selected

        votes = sum([mi_ok, rf_ok, var_ok])
        is_selected = votes >= MIN_VOTES_REQUIRED

        feature_score_list.append(
            FeatureScore(
                column=col,
                mutual_information=mi_val,
                mi_selected=mi_ok,
                rf_importance=rf_val,
                rf_selected=rf_ok,
                variance=var_val,
                variance_selected=var_ok,
                vote_count=votes,
                final_selected=is_selected,
            )
        )

        if is_selected:
            selected_features.append(col)
        else:
            dropped_features.append(col)

    # Sort selected features by RF importance (highest first) for readability
    selected_features.sort(
        key=lambda c: rf_scores.get(c, 0.0), reverse=True
    )

    # Sort dropped features by vote count then name
    dropped_features.sort(
        key=lambda c: (
            -sum([
                c in mi_selected,
                c in rf_selected,
                c in var_selected,
            ]),
            c,
        )
    )

    logger.info(
        "Consensus: %d features selected, %d features dropped.",
        len(selected_features), len(dropped_features),
    )

    # ------------------------------------------------------------------
    # Build result
    # ------------------------------------------------------------------

    duration = time.perf_counter() - start_time

    result = FeatureSelectorResult(
        input_path=str(input_path),
        rows=rows,
        total_features_evaluated=len(candidate_columns),
        features_selected=len(selected_features),
        features_dropped=len(dropped_features),
        method1_mi_selected=len(mi_selected),
        method2_rf_selected=len(rf_selected),
        method3_var_selected=len(var_selected),
        min_votes_required=MIN_VOTES_REQUIRED,
        selected_feature_names=selected_features,
        dropped_feature_names=dropped_features,
        feature_scores=feature_score_list,
        duration_seconds=round(duration, 4),
    )

    # ------------------------------------------------------------------
    # Save selected features artifact
    # ------------------------------------------------------------------

    logger.info("Saving feature selection artifacts.")
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    selected_manifest = {
        "generated_by": "feature_selector.py",
        "input_dataset": str(input_path),
        "total_features_evaluated": len(candidate_columns),
        "features_selected": len(selected_features),
        "features_dropped": len(dropped_features),
        "min_votes_required": MIN_VOTES_REQUIRED,
        "selection_methods": [
            "mutual_information",
            "random_forest_importance",
            "variance_threshold",
        ],
        "selected_features": selected_features,
        "dropped_features": dropped_features,
    }

    atomic_write_json(selected_manifest, _SELECTED_FEATURES_PATH)
    logger.info("Selected features saved: %s", _SELECTED_FEATURES_PATH)

    # ------------------------------------------------------------------
    # Save detailed report
    # ------------------------------------------------------------------

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    report = result.to_dict()
    report["status"] = "success"

    # Add top-10 and bottom-10 features by RF importance for quick review
    sorted_by_rf = sorted(
        feature_score_list, key=lambda fs: fs.rf_importance, reverse=True
    )
    report["top_10_features_by_rf_importance"] = [
        {"column": fs.column, "rf_importance": fs.rf_importance, "mi": fs.mutual_information}
        for fs in sorted_by_rf[:10]
    ]
    report["bottom_10_features_by_rf_importance"] = [
        {"column": fs.column, "rf_importance": fs.rf_importance, "mi": fs.mutual_information}
        for fs in sorted_by_rf[-10:]
    ]

    atomic_write_json(report, _SELECTION_REPORT_PATH)
    logger.info("Selection report saved: %s", _SELECTION_REPORT_PATH)

    # ------------------------------------------------------------------
    # Summary log
    # ------------------------------------------------------------------

    logger.info("-" * 78)
    logger.info("Rows                        : %d", rows)
    logger.info("Features evaluated          : %d", len(candidate_columns))
    logger.info("Method 1 (MI) selected      : %d", len(mi_selected))
    logger.info("Method 2 (RF) selected      : %d", len(rf_selected))
    logger.info("Method 3 (Var) selected     : %d", len(var_selected))
    logger.info("Consensus (>=%d votes)       : %d selected", MIN_VOTES_REQUIRED, len(selected_features))
    logger.info("Features dropped            : %d", len(dropped_features))
    logger.info("Duration                    : %.3f seconds", duration)

    if dropped_features:
        logger.info("Dropped features: %s", dropped_features)

    section("SUPERVISOR FEATURE SELECTION COMPLETED")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: FeatureSelectorResult) -> None:
    """Print a compact JSON summary to stdout."""

    summary = {
        "status": "success",
        "input_path": result.input_path,
        "report_path": str(_SELECTION_REPORT_PATH),
        "selected_features_path": str(_SELECTED_FEATURES_PATH),
        "rows": result.rows,
        "features_evaluated": result.total_features_evaluated,
        "features_selected": result.features_selected,
        "features_dropped": result.features_dropped,
        "method1_mi_selected": result.method1_mi_selected,
        "method2_rf_selected": result.method2_rf_selected,
        "method3_var_selected": result.method3_var_selected,
        "selected_features": result.selected_feature_names,
        "dropped_features": result.dropped_feature_names,
        "duration_seconds": result.duration_seconds,
        "message": (
            f"Feature selection completed: {result.features_selected} "
            f"features retained from {result.total_features_evaluated} "
            f"candidates."
        ),
    }

    print(json.dumps(summary, indent=2))


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 — feature selection completed successfully
        1 — internal failure
    """

    try:
        result = run_feature_selector()
        _print_summary(result)
        return 0

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        return 1

    except RuntimeError as exc:
        logger.error("Runtime error during feature selection: %s", exc)
        return 1

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error during feature selection: %s",
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
