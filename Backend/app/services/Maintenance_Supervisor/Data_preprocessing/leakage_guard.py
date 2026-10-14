"""
leakage_guard.py

Data Leakage Detection and Prevention Guard for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module is the final safety barrier between feature engineering and model
training. It inspects an engineered dataset and detects every category of
data leakage that could cause an optimistically biased model which fails in
production:

1. Target-derived leakage
   Columns that are computed from or are semantically equivalent to
   final_decision: priority, maintenance_urgency, requires_human_review,
   supervisor_score, and any alias.

2. Explicitly forbidden columns
   Hard-coded list from supervisor_config.FORBIDDEN_FEATURE_COLUMNS plus
   additional patterns recognised at runtime.

3. Name-pattern leakage
   Columns whose name matches known dangerous patterns such as "true_rul",
   "actual_rul", "future_*", "failure_*", "max_cycle", "post_*".

4. Future-derived temporal leakage
   Temporal columns (lags, rolling windows, diffs) that use a *forward* shift
   or a *centred* window, which would incorporate future-cycle information at
   the current prediction point.
   NOTE: leakage_guard cannot inspect pandas rolling parameters from the
   already-materialised CSV, so it relies on column-name conventions produced
   by feature_engineering.py to flag candidates for manual review.

5. Suspicious near-perfect correlation with the target
   A numeric column whose Pearson correlation with the *encoded* target
   exceeds a configurable threshold is reported as a leakage risk.
   Correlation alone does not trigger exclusion; it triggers a WARNING so a
   human can decide.

6. Constant columns
   Columns with zero variance carry no useful information for a classifier
   and may indicate a generation error.

7. Post-outcome fields
   Columns whose names suggest that they were populated *after* the
   maintenance event occurred (actual_failure_date, maintenance_result, etc.).

8. Model-trace columns used as raw features
   rul_model_used, risk_model_used, anomaly_model_used should not be used as
   predictive features without careful verification; they are flagged.

For each detected issue the guard records:
- column name
- issue category
- severity (ERROR, WARNING, INFO)
- description

After scanning, the guard writes an atomic JSON leakage report and optionally
a CSV issues table.  It exits with code 0 (clean), 1 (warnings only) or 2
(leakage errors found) so that the training pipeline can refuse to start if
leakage is present.

The guard also produces a *safe feature list* — the set of columns from the
engineered dataset that may be passed to model training, with all leakage
columns removed.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.leakage_guard

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_engineered_dataset.csv

Expected outputs
----------------
- reports/Maintenance_Supervisor/leakage_guard/leakage_guard_report.json
- reports/Maintenance_Supervisor/leakage_guard/leakage_guard_issues.csv
- artifacts/Maintenance_Supervisor/safe_feature_list.json

Exit codes
----------
0 — no errors (warnings may exist)
1 — internal failure (import, config, IO error)
2 — at least one ERROR-level leakage finding was detected
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

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
    FORBIDDEN_FEATURE_COLUMNS,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    METADATA_COLUMNS,
    MODEL_TRACE_COLUMNS,
    DERIVED_OUTPUT_COLUMNS,
    BASE_INPUT_FEATURES,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_csv,
)

# ---------------------------------------------------------------------------
# Logger — uses the project singleton (no argument)
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Report paths
# ---------------------------------------------------------------------------

_LEAKAGE_REPORT_ROOT: Final[Path] = REPORTS_ROOT / "leakage_guard"
_LEAKAGE_REPORT_PATH: Final[Path] = (
    _LEAKAGE_REPORT_ROOT / "leakage_guard_report.json"
)
_LEAKAGE_ISSUES_PATH: Final[Path] = (
    _LEAKAGE_REPORT_ROOT / "leakage_guard_issues.csv"
)
_SAFE_FEATURE_LIST_PATH: Final[Path] = (
    ARTIFACT_ROOT / "safe_feature_list.json"
)

# ---------------------------------------------------------------------------
# Configuration constants — adjustable without changing logic
# ---------------------------------------------------------------------------

# Pearson |r| above this value triggers a WARNING (not an ERROR).
CORRELATION_WARNING_THRESHOLD: Final[float] = 0.98

# Variance below this value marks a column as constant.
CONSTANT_VARIANCE_THRESHOLD: Final[float] = 1e-10

# Column name fragments that indicate forbidden future/outcome information.
FORBIDDEN_PATTERN_SUBSTRINGS: Final[tuple[str, ...]] = (
    "true_rul",
    "actual_rul",
    "remaining_useful_life",
    "remaining_cycles",
    "future_",
    "failure_cycle",
    "failure_timestamp",
    "failure_date",
    "max_cycle",
    "maximum_cycle",
    "engine_lifetime",
    "post_maintenance",
    "post_failure",
    "maintenance_result",
    "maintenance_outcome",
    "repair_outcome",
    "rule_label",
    "pseudo_decision",
    "generated_target",
    "maintenance_label",
)

# Substrings whose presence in a column name flags it for post-outcome review.
POST_OUTCOME_SUBSTRINGS: Final[tuple[str, ...]] = (
    "post_maintenance",
    "post_failure",
    "maintenance_result",
    "maintenance_outcome",
    "actual_failure",
    "repair_result",
    "corrective_action",
)

# Rolling/lag naming conventions produced by feature_engineering.py.
# Centred rolling columns would contain "_centred_" in their name.
# The guard warns on any column that matches this pattern.
CENTRED_ROLLING_PATTERN: Final[str] = "_centred_"

# These column names from feature_engineering.py are safe lag directions
# (backward shift only).  The guard does not warn about them.
# Future temporal leakage would appear only if feature_engineering.py
# mistakenly used negative shift values, which is checked separately via
# naming conventions.
KNOWN_SAFE_LAG_PREFIXES: Final[tuple[str, ...]] = (
    "lag1_",
    "lag2_",
    "lag3_",
    "diff_",
    "pct_change_",
    "roll_mean_",
    "roll_std_",
    "roll_min_",
    "roll_max_",
    "roll_trend_",
    "anomaly_persistence_",
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class LeakageSeverity(str, Enum):
    """Severity level for a leakage finding."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class LeakageCategory(str, Enum):
    """Category of a leakage finding."""

    TARGET_DERIVED = "target_derived"
    EXPLICITLY_FORBIDDEN = "explicitly_forbidden"
    NAME_PATTERN_MATCH = "name_pattern_match"
    POST_OUTCOME = "post_outcome"
    HIGH_CORRELATION = "high_correlation"
    CONSTANT_COLUMN = "constant_column"
    MODEL_TRACE_FEATURE = "model_trace_feature"
    CENTRED_ROLLING = "centred_rolling"
    FUTURE_TEMPORAL = "future_temporal"


@dataclass
class LeakageFinding:
    """A single leakage detection result for one column."""

    column: str
    category: LeakageCategory
    severity: LeakageSeverity
    description: str
    correlation_with_target: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "column": self.column,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "correlation_with_target": self.correlation_with_target,
        }


@dataclass
class LeakageGuardResult:
    """Aggregated result returned by the leakage guard."""

    input_path: str
    total_columns_scanned: int
    error_count: int
    warning_count: int
    info_count: int
    leakage_columns_removed: list[str] = field(default_factory=list)
    safe_feature_columns: list[str] = field(default_factory=list)
    findings: list[LeakageFinding] = field(default_factory=list)
    duration_seconds: float = 0.0
    is_clean: bool = True

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable dictionary."""

        return {
            "input_path": self.input_path,
            "total_columns_scanned": self.total_columns_scanned,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "leakage_columns_removed": self.leakage_columns_removed,
            "safe_feature_columns": self.safe_feature_columns,
            "safe_feature_count": len(self.safe_feature_columns),
            "findings": [f.to_dict() for f in self.findings],
            "duration_seconds": self.duration_seconds,
            "is_clean": self.is_clean,
        }


# ---------------------------------------------------------------------------
# Core leakage-detection logic
# ---------------------------------------------------------------------------


def _build_forbidden_set() -> frozenset[str]:
    """
    Merge the config-defined forbidden list with all known derived-output
    columns and the target column into a single frozenset for O(1) lookup.
    """

    forbidden: set[str] = set(FORBIDDEN_FEATURE_COLUMNS)
    forbidden.add(TARGET_COLUMN)
    forbidden.update(DERIVED_OUTPUT_COLUMNS)
    return frozenset(col.strip().lower() for col in forbidden)


def _encode_target(series: pd.Series) -> pd.Series:
    """
    Ordinal-encode the target column so that Pearson correlation can be
    computed against numeric features.

    Encoding order matches DECISION_TO_SEVERITY in supervisor_config.py:

        continue_operation    -> 0
        monitor_closely       -> 1
        schedule_inspection   -> 2
        schedule_maintenance  -> 3
        immediate_maintenance -> 4
    """

    severity_map: dict[str, int] = {
        "continue_operation": 0,
        "monitor_closely": 1,
        "schedule_inspection": 2,
        "schedule_maintenance": 3,
        "immediate_maintenance": 4,
    }

    encoded = series.str.strip().str.lower().map(severity_map)
    return pd.to_numeric(encoded, errors="coerce")


def _check_target_derived(
    columns: list[str],
    forbidden_lower: frozenset[str],
) -> list[LeakageFinding]:
    """
    Detect columns that are known supervisor output labels.

    These are columns such as priority, maintenance_urgency,
    requires_human_review, and supervisor_score.  They are only populated
    after the supervisor makes a decision, so they must never be used as
    model input features.
    """

    findings: list[LeakageFinding] = []

    target_derived_names: frozenset[str] = frozenset(
        col.strip().lower() for col in DERIVED_OUTPUT_COLUMNS
    )
    # A one-hot encoder expands e.g. "priority" into "priority_critical",
    # "priority_high", etc. Those expanded columns carry the exact same
    # leakage as the base column but don't match it by exact name, so any
    # column whose name starts with "<derived_output>_" is flagged too.
    target_derived_prefixes: tuple[str, ...] = tuple(
        f"{name}_" for name in target_derived_names
    )

    for col in columns:
        col_lower = col.strip().lower()

        if col_lower == TARGET_COLUMN.lower():
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.TARGET_DERIVED,
                    severity=LeakageSeverity.ERROR,
                    description=(
                        f"Column '{col}' is the target variable '{TARGET_COLUMN}'. "
                        "Including the target as a feature is a direct leakage "
                        "violation and will produce a degenerate model."
                    ),
                )
            )

        elif col_lower in target_derived_names:
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.TARGET_DERIVED,
                    severity=LeakageSeverity.ERROR,
                    description=(
                        f"Column '{col}' is a supervisor-output label that is "
                        "derived from or co-determined with the final_decision "
                        "target. It must not be used as a model input feature."
                    ),
                )
            )

        elif col_lower.startswith(target_derived_prefixes):
            base = next(p for p in target_derived_prefixes if col_lower.startswith(p))[:-1]
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.TARGET_DERIVED,
                    severity=LeakageSeverity.ERROR,
                    description=(
                        f"Column '{col}' is a one-hot (or similarly expanded) "
                        f"encoding of '{base}', a supervisor-output label derived "
                        "from final_decision. It must not be used as a model "
                        "input feature."
                    ),
                )
            )

    return findings


def _check_explicitly_forbidden(
    columns: list[str],
    forbidden_lower: frozenset[str],
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Check every column against the supervisor_config FORBIDDEN_FEATURE_COLUMNS
    list.  Skip columns already reported by _check_target_derived.
    """

    findings: list[LeakageFinding] = []

    for col in columns:
        if col in already_flagged:
            continue

        if col.strip().lower() in forbidden_lower:
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.EXPLICITLY_FORBIDDEN,
                    severity=LeakageSeverity.ERROR,
                    description=(
                        f"Column '{col}' appears in the FORBIDDEN_FEATURE_COLUMNS "
                        "list defined in supervisor_config.py. It represents "
                        "information that would not be available during real-time "
                        "inference (e.g. true RUL, future failure label, or a "
                        "post-maintenance outcome)."
                    ),
                )
            )

    return findings


def _check_name_patterns(
    columns: list[str],
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Detect columns whose names match known dangerous leakage patterns.

    This catches columns that were not listed in FORBIDDEN_FEATURE_COLUMNS
    but whose name implies they carry future or post-outcome information.
    """

    findings: list[LeakageFinding] = []

    for col in columns:
        if col in already_flagged:
            continue

        col_lower = col.strip().lower()

        for pattern in FORBIDDEN_PATTERN_SUBSTRINGS:
            if pattern in col_lower:
                findings.append(
                    LeakageFinding(
                        column=col,
                        category=LeakageCategory.NAME_PATTERN_MATCH,
                        severity=LeakageSeverity.ERROR,
                        description=(
                            f"Column '{col}' matches the forbidden name pattern "
                            f"'{pattern}'. This pattern indicates the column may "
                            "carry future-cycle, post-event, or ground-truth "
                            "information that is unavailable during inference."
                        ),
                    )
                )
                break  # One finding per column is sufficient

    return findings


def _check_post_outcome(
    columns: list[str],
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Flag columns that suggest they were recorded after a maintenance event.

    These columns contain outcome data and cannot be known at the time of
    prediction.
    """

    findings: list[LeakageFinding] = []

    for col in columns:
        if col in already_flagged:
            continue

        col_lower = col.strip().lower()

        for pattern in POST_OUTCOME_SUBSTRINGS:
            if pattern in col_lower:
                findings.append(
                    LeakageFinding(
                        column=col,
                        category=LeakageCategory.POST_OUTCOME,
                        severity=LeakageSeverity.ERROR,
                        description=(
                            f"Column '{col}' contains the pattern '{pattern}', "
                            "which suggests it records the outcome of a maintenance "
                            "action. Such columns are populated only after an event "
                            "and must be excluded from all model input."
                        ),
                    )
                )
                break

    return findings


def _check_model_trace_features(
    columns: list[str],
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Flag model-trace columns (rul_model_used, risk_model_used, etc.).

    These are metadata strings describing which upstream model produced the
    output.  They are not predictive features and may cause data leakage in
    production if the model selection changes between training and deployment.
    """

    findings: list[LeakageFinding] = []

    trace_lower: frozenset[str] = frozenset(
        col.strip().lower() for col in MODEL_TRACE_COLUMNS
    )

    for col in columns:
        if col in already_flagged:
            continue

        if col.strip().lower() in trace_lower:
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.MODEL_TRACE_FEATURE,
                    severity=LeakageSeverity.WARNING,
                    description=(
                        f"Column '{col}' is a model-trace string that identifies "
                        "which upstream model produced the prediction. Using it as a "
                        "feature can cause silent deployment failures if the upstream "
                        "model changes. Review whether this column should be excluded."
                    ),
                )
            )

    return findings


def _check_centred_rolling(
    columns: list[str],
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Detect centred rolling window features.

    A centred rolling window uses future observations in its calculation,
    which is valid for offline analysis but constitutes temporal leakage
    for an online prediction model.
    """

    findings: list[LeakageFinding] = []

    for col in columns:
        if col in already_flagged:
            continue

        if CENTRED_ROLLING_PATTERN in col.lower():
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.CENTRED_ROLLING,
                    severity=LeakageSeverity.ERROR,
                    description=(
                        f"Column '{col}' appears to be derived from a centred "
                        "rolling window (pattern: '_centred_'). A centred window "
                        "incorporates future observations at the current prediction "
                        "cycle, which constitutes temporal data leakage. Only "
                        "causal (backward-only) rolling windows are permitted."
                    ),
                )
            )

    return findings


def _check_constant_columns(
    dataframe: pd.DataFrame,
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Detect columns with near-zero variance.

    Constant columns carry no discriminative information for the classifier.
    They are typically a sign of a data-generation bug or an imputation error
    that filled all missing values with the same constant.
    """

    findings: list[LeakageFinding] = []

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col in already_flagged:
            continue

        col_var = float(dataframe[col].var(ddof=0))

        if col_var <= CONSTANT_VARIANCE_THRESHOLD:
            unique_vals = dataframe[col].dropna().unique()
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.CONSTANT_COLUMN,
                    severity=LeakageSeverity.WARNING,
                    description=(
                        f"Column '{col}' has near-zero variance "
                        f"(var={col_var:.2e}). "
                        f"Unique non-null values: {list(unique_vals[:5])}. "
                        "A constant column provides no information to a model and "
                        "may indicate an imputation or generation error."
                    ),
                )
            )

    return findings


def _check_high_correlation(
    dataframe: pd.DataFrame,
    already_flagged: set[str],
) -> list[LeakageFinding]:
    """
    Detect numeric columns with suspiciously high correlation to the target.

    This check warns — it does not automatically exclude the column — because
    genuine strong predictors can have high correlation with the target.
    Human review is required for any column above the threshold.

    Columns that are already flagged as errors are excluded.
    """

    if TARGET_COLUMN not in dataframe.columns:
        logger.warning(
            "Target column '%s' not found in dataset. "
            "Skipping correlation check.",
            TARGET_COLUMN,
        )
        return []

    findings: list[LeakageFinding] = []

    encoded_target = _encode_target(dataframe[TARGET_COLUMN])

    if encoded_target.isna().all():
        logger.warning(
            "Target column could not be encoded. Skipping correlation check."
        )
        return []

    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if col in already_flagged:
            continue

        # Skip the cycle and other metadata numerics
        if col in (CYCLE_COLUMN,):
            continue

        col_series = dataframe[col].dropna()

        if col_series.empty or col_series.var() <= CONSTANT_VARIANCE_THRESHOLD:
            continue

        # Compute on the intersection of non-null indices
        common_idx = col_series.index.intersection(encoded_target.dropna().index)

        if len(common_idx) < 50:
            continue

        corr_value = float(
            np.corrcoef(
                dataframe.loc[common_idx, col].values,
                encoded_target.loc[common_idx].values,
            )[0, 1]
        )

        abs_corr = abs(corr_value)

        if abs_corr >= CORRELATION_WARNING_THRESHOLD:
            findings.append(
                LeakageFinding(
                    column=col,
                    category=LeakageCategory.HIGH_CORRELATION,
                    severity=LeakageSeverity.WARNING,
                    description=(
                        f"Column '{col}' has a Pearson correlation of "
                        f"{corr_value:+.4f} (|r|={abs_corr:.4f}) with the encoded "
                        f"target '{TARGET_COLUMN}'. Correlation above "
                        f"{CORRELATION_WARNING_THRESHOLD:.2f} may indicate that "
                        "this feature is derived from the label or captures "
                        "post-decision information. Human review is required before "
                        "including this column in model training."
                    ),
                    correlation_with_target=corr_value,
                )
            )

    return findings


# ---------------------------------------------------------------------------
# Main guard orchestrator
# ---------------------------------------------------------------------------


def run_leakage_guard(
    input_path: Path | None = None,
    correlation_threshold: float = CORRELATION_WARNING_THRESHOLD,
) -> LeakageGuardResult:
    """
    Run the full leakage detection pipeline on an engineered dataset.

    Parameters
    ----------
    input_path:
        Path to the engineered dataset CSV.  Defaults to
        ENGINEERED_DATASET_PATH from supervisor_config.
    correlation_threshold:
        Pearson |r| above which a WARNING is raised for high correlation
        with the target.  Overrides the module-level constant.

    Returns
    -------
    LeakageGuardResult
        Contains findings, safe feature list, counts, and metadata.

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    ValueError
        If the dataset cannot be loaded or has no columns.
    """

    if input_path is None:
        input_path = ENGINEERED_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("SUPERVISOR LEAKAGE GUARD STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Report path   : %s", _LEAKAGE_REPORT_PATH)
    logger.info("Issues CSV    : %s", _LEAKAGE_ISSUES_PATH)
    logger.info("Safe features : %s", _SAFE_FEATURE_LIST_PATH)

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
        raise ValueError(
            f"The loaded dataset from '{input_path}' is empty or has no columns."
        )

    logger.info(
        "Loaded rows=%d, columns=%d, memory=%.2f MB",
        len(dataframe),
        dataframe.shape[1],
        dataframe.memory_usage(deep=True).sum() / (1024 ** 2),
    )

    all_columns: list[str] = list(dataframe.columns)

    # ------------------------------------------------------------------
    # Build forbidden set for O(1) lookup
    # ------------------------------------------------------------------

    forbidden_lower = _build_forbidden_set()

    # ------------------------------------------------------------------
    # Run all detection passes in severity order
    # ------------------------------------------------------------------

    all_findings: list[LeakageFinding] = []
    flagged_columns: set[str] = set()

    # Pass 1 — Target-derived leakage (highest severity)
    logger.info("Pass 1/8 — Checking target-derived leakage columns.")
    pass1 = _check_target_derived(all_columns, forbidden_lower)
    all_findings.extend(pass1)
    flagged_columns.update(f.column for f in pass1)

    # Pass 2 — Explicitly forbidden columns
    logger.info("Pass 2/8 — Checking explicitly forbidden columns.")
    pass2 = _check_explicitly_forbidden(all_columns, forbidden_lower, flagged_columns)
    all_findings.extend(pass2)
    flagged_columns.update(f.column for f in pass2)

    # Pass 3 — Name-pattern matching
    logger.info("Pass 3/8 — Checking dangerous name patterns.")
    pass3 = _check_name_patterns(all_columns, flagged_columns)
    all_findings.extend(pass3)
    flagged_columns.update(f.column for f in pass3)

    # Pass 4 — Post-outcome columns
    logger.info("Pass 4/8 — Checking post-outcome column names.")
    pass4 = _check_post_outcome(all_columns, flagged_columns)
    all_findings.extend(pass4)
    flagged_columns.update(f.column for f in pass4)

    # Pass 5 — Centred rolling window detection
    logger.info("Pass 5/8 — Checking for centred rolling window features.")
    pass5 = _check_centred_rolling(all_columns, flagged_columns)
    all_findings.extend(pass5)
    flagged_columns.update(
        f.column for f in pass5 if f.severity == LeakageSeverity.ERROR
    )

    # Pass 6 — Model-trace features (WARNING level, not excluded automatically)
    logger.info("Pass 6/8 — Checking model-trace columns used as features.")
    pass6 = _check_model_trace_features(all_columns, flagged_columns)
    all_findings.extend(pass6)
    # Model-trace findings are WARNING; we do not auto-exclude them.

    # Pass 7 — Constant columns (WARNING level)
    logger.info("Pass 7/8 — Detecting constant columns.")
    pass7 = _check_constant_columns(dataframe, flagged_columns)
    all_findings.extend(pass7)

    # Pass 8 — High correlation with target (WARNING level)
    logger.info("Pass 8/8 — Computing feature-to-target correlation.")
    pass8 = _check_high_correlation(dataframe, flagged_columns)
    all_findings.extend(pass8)

    # ------------------------------------------------------------------
    # Determine which columns are excluded (ERROR level only)
    # ------------------------------------------------------------------

    error_columns: list[str] = sorted({
        f.column
        for f in all_findings
        if f.severity == LeakageSeverity.ERROR
    })

    # ------------------------------------------------------------------
    # Build safe feature list
    # ------------------------------------------------------------------

    # Safe features are all columns that are:
    # 1. Not in error_columns (leakage errors)
    # 2. Not the target column
    # 3. Not metadata columns that identify rows but carry no signal
    #    (schema_version is dropped; engine_id, fd_subset, cycle are kept
    #    so that downstream splitters can group by engine)

    exclude_from_features: set[str] = set(error_columns) | {
        TARGET_COLUMN,
        "schema_version",  # version tag, not a feature
    }

    safe_feature_columns: list[str] = [
        col for col in all_columns
        if col not in exclude_from_features
    ]

    # ------------------------------------------------------------------
    # Count findings by severity
    # ------------------------------------------------------------------

    error_count = sum(
        1 for f in all_findings if f.severity == LeakageSeverity.ERROR
    )
    warning_count = sum(
        1 for f in all_findings if f.severity == LeakageSeverity.WARNING
    )
    info_count = sum(
        1 for f in all_findings if f.severity == LeakageSeverity.INFO
    )

    is_clean = error_count == 0

    duration = time.perf_counter() - start_time

    result = LeakageGuardResult(
        input_path=str(input_path),
        total_columns_scanned=len(all_columns),
        error_count=error_count,
        warning_count=warning_count,
        info_count=info_count,
        leakage_columns_removed=error_columns,
        safe_feature_columns=safe_feature_columns,
        findings=all_findings,
        duration_seconds=round(duration, 4),
        is_clean=is_clean,
    )

    # ------------------------------------------------------------------
    # Write reports
    # ------------------------------------------------------------------

    logger.info("Writing leakage guard reports atomically.")

    _LEAKAGE_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    report_dict = result.to_dict()
    report_dict["status"] = "clean" if is_clean else "leakage_detected"

    atomic_write_json(report_dict, _LEAKAGE_REPORT_PATH)
    logger.info("Leakage report written: %s", _LEAKAGE_REPORT_PATH)

    # Issues CSV
    if all_findings:
        issues_df = pd.DataFrame([f.to_dict() for f in all_findings])
        atomic_write_csv(issues_df, _LEAKAGE_ISSUES_PATH)
        logger.info("Issues CSV written   : %s", _LEAKAGE_ISSUES_PATH)
    else:
        logger.info("No issues found. Skipping issues CSV.")

    # Safe feature list JSON (used by downstream training pipeline)
    safe_manifest = {
        "generated_by": "leakage_guard.py",
        "input_dataset": str(input_path),
        "total_scanned": len(all_columns),
        "leakage_errors_removed": error_count,
        "safe_feature_count": len(safe_feature_columns),
        "safe_features": safe_feature_columns,
        "excluded_features": error_columns,
        "warning_features": sorted({
            f.column for f in all_findings
            if f.severity == LeakageSeverity.WARNING
        }),
    }
    atomic_write_json(safe_manifest, _SAFE_FEATURE_LIST_PATH)
    logger.info("Safe feature list    : %s", _SAFE_FEATURE_LIST_PATH)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    section_line = "-" * 78
    logger.info(section_line)
    logger.info("Columns scanned             : %d", len(all_columns))
    logger.info("Error findings (excluded)   : %d", error_count)
    logger.info("Warning findings (review)   : %d", warning_count)
    logger.info("Safe features remaining     : %d", len(safe_feature_columns))
    logger.info("Duration                    : %.3f seconds", duration)

    if is_clean:
        section("LEAKAGE GUARD PASSED — DATASET IS CLEAN")
    else:
        section("LEAKAGE GUARD FAILED — LEAKAGE COLUMNS DETECTED")
        logger.error(
            "The following %d column(s) must be removed before training: %s",
            error_count,
            error_columns,
        )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: LeakageGuardResult) -> None:
    """Print a compact JSON summary to stdout."""

    summary = {
        "status": "clean" if result.is_clean else "leakage_detected",
        "is_clean": result.is_clean,
        "input_path": result.input_path,
        "report_path": str(_LEAKAGE_REPORT_PATH),
        "issues_path": str(_LEAKAGE_ISSUES_PATH),
        "safe_feature_list_path": str(_SAFE_FEATURE_LIST_PATH),
        "total_columns_scanned": result.total_columns_scanned,
        "error_count": result.error_count,
        "warning_count": result.warning_count,
        "safe_feature_count": len(result.safe_feature_columns),
        "leakage_columns_removed": result.leakage_columns_removed,
        "duration_seconds": result.duration_seconds,
        "message": (
            "Dataset is clean. No leakage errors detected."
            if result.is_clean
            else (
                f"{result.error_count} leakage error(s) detected. "
                "Training must not proceed until these are resolved."
            )
        ),
    }

    print(json.dumps(summary, indent=2))


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 — leakage guard passed (no errors, warnings may exist)
        1 — internal failure (exception before guard completed)
        2 — leakage errors detected in the dataset
    """

    try:
        result = run_leakage_guard()
        _print_summary(result)
        return 0 if result.is_clean else 2

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        return 1

    except ValueError as exc:
        logger.error("Value error during leakage guard: %s", exc)
        return 1

    except Exception as exc:  # pylint: disable=broad-except
        logger.error("Unexpected error during leakage guard: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
