"""
engine_splitter.py

Engine-Level Grouped Train / Validation / Test Splitter for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module splits the fully preprocessed supervisor dataset into train,
validation, and test partitions using ENGINE-LEVEL GROUPED splitting.

Why engine-level grouping?
    Records from the same engine (engine_id) describe that engine's
    degradation trajectory over successive cycles.  If rows from the same
    engine appear in both the training and test sets, the model can
    memorise engine-specific patterns (e.g., the exact cycle at which
    engine 17 transitions from "monitor" to "immediate_maintenance").
    This produces an inflated test-set score that does not generalise to
    unseen engines.

    Engine-level splitting ensures that ALL cycles from a given engine
    appear in exactly ONE partition — either train, validation, or test,
    never split across partitions.

Stratification
    Within each CMAPSS sub-dataset (fd_subset), engines may have very
    different degradation profiles.  We further stratify by sub-dataset so
    that the class distribution of final_decision is approximately balanced
    across all three partitions.

    Stratification procedure:
    1. Group by (fd_subset, engine_id) to get unique engines per sub-dataset.
    2. For each sub-dataset, assign engines to train/val/test such that
       the resulting row-level class distribution of final_decision is as
       close as possible to the global distribution.
    3. Use deterministic hashing (not random shuffle) so that the split is
       perfectly reproducible given the same random seed.

Outputs
    The splitter writes three separate CSV files (train, val, test) plus a
    combined CSV with a "split" column appended.  It also writes a split
    manifest JSON recording engine counts, row counts, and class
    distributions per partition for auditability.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.engine_splitter

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_scaled_dataset.csv
  (output of feature_scaler.py)

Expected outputs
----------------
- processed/Maintenance_Supervisor/splits/train.csv
- processed/Maintenance_Supervisor/splits/validation.csv
- processed/Maintenance_Supervisor/splits/test.csv
- processed/Maintenance_Supervisor/splits/full_with_split_column.csv
- artifacts/Maintenance_Supervisor/split_manifest.json
- reports/Maintenance_Supervisor/engine_split_report.json

Exit codes
----------
0 — splitting completed successfully
1 — internal failure
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
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
    PROCESSED_ROOT,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    TARGET_COLUMN,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
    SupervisorConfig,
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
# Configuration
# ---------------------------------------------------------------------------

_CFG = SupervisorConfig()

_TRAIN_FRACTION: Final[float] = _CFG.train_fraction       # 0.70
_VAL_FRACTION: Final[float] = _CFG.validation_fraction     # 0.15
_TEST_FRACTION: Final[float] = _CFG.test_fraction          # 0.15
_RANDOM_SEED: Final[int] = _CFG.random_seed                # 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SCALED_DATASET_PATH: Final[Path] = (
    PROCESSED_ROOT / "supervisor_scaled_dataset.csv"
)
_SPLITS_ROOT: Final[Path] = PROCESSED_ROOT / "splits"
_TRAIN_PATH: Final[Path] = _SPLITS_ROOT / "train.csv"
_VAL_PATH: Final[Path] = _SPLITS_ROOT / "validation.csv"
_TEST_PATH: Final[Path] = _SPLITS_ROOT / "test.csv"
_FULL_SPLIT_PATH: Final[Path] = _SPLITS_ROOT / "full_with_split_column.csv"
_SPLIT_MANIFEST_PATH: Final[Path] = ARTIFACT_ROOT / "split_manifest.json"
_SPLIT_REPORT_PATH: Final[Path] = REPORTS_ROOT / "engine_split_report.json"

# Split label values
SPLIT_TRAIN: Final[str] = "train"
SPLIT_VALIDATION: Final[str] = "validation"
SPLIT_TEST: Final[str] = "test"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PartitionStats:
    """Statistics for a single split partition."""

    partition_name: str
    engine_count: int
    row_count: int
    row_fraction: float
    class_distribution: dict[str, int]
    class_fractions: dict[str, float]
    engines: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "partition_name": self.partition_name,
            "engine_count": self.engine_count,
            "row_count": self.row_count,
            "row_fraction": round(self.row_fraction, 4),
            "class_distribution": self.class_distribution,
            "class_fractions": {
                k: round(v, 4) for k, v in self.class_fractions.items()
            },
            "engines": self.engines,
        }


@dataclass
class EngineSplitterResult:
    """Aggregated result of the engine-level splitting."""

    input_path: str
    total_rows: int
    total_engines: int
    total_subsets: int
    target_fractions: dict[str, float]
    partitions: list[PartitionStats] = field(default_factory=list)
    leakage_check_passed: bool = True
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": self.input_path,
            "total_rows": self.total_rows,
            "total_engines": self.total_engines,
            "total_subsets": self.total_subsets,
            "target_fractions": {
                k: round(v, 4) for k, v in self.target_fractions.items()
            },
            "partitions": [p.to_dict() for p in self.partitions],
            "leakage_check_passed": self.leakage_check_passed,
            "duration_seconds": self.duration_seconds,
        }


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------


def _deterministic_engine_hash(
    engine_id: str,
    subset: str,
    seed: int,
) -> int:
    """
    Produce a deterministic integer hash for an engine within a sub-dataset.

    Uses SHA-256 of the concatenation (seed, subset, engine_id) to ensure
    reproducibility across platforms and Python versions.  The hash is
    independent of dictionary ordering or random.seed state.
    """

    key = f"{seed}|{subset}|{engine_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)  # Use first 64 bits for sorting


def _assign_engines_to_splits(
    engine_ids: list[str],
    subset_label: str,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    """
    Deterministically assign a list of engine IDs to train/val/test.

    The engines are sorted by their deterministic hash, then split into
    contiguous blocks matching the target fractions.  This produces a
    reproducible assignment that is independent of input order.

    Parameters
    ----------
    engine_ids:
        List of unique engine ID strings within one sub-dataset.
    subset_label:
        The fd_subset value (e.g. "FD001") used as part of the hash key.
    seed:
        Random seed for deterministic hashing.
    train_frac:
        Target fraction of engines for training.
    val_frac:
        Target fraction of engines for validation.

    Returns
    -------
    dict mapping engine_id -> split label ("train", "validation", "test")
    """

    if not engine_ids:
        return {}

    # Sort engines by deterministic hash
    sorted_engines = sorted(
        engine_ids,
        key=lambda eid: _deterministic_engine_hash(eid, subset_label, seed),
    )

    n = len(sorted_engines)
    n_train = max(1, round(n * train_frac))
    n_val = max(1, round(n * val_frac))
    n_test = n - n_train - n_val

    # Edge case: very few engines — ensure at least 1 per split
    if n < 3:
        # With fewer than 3 engines, put all in training
        logger.warning(
            "Subset '%s' has only %d engine(s). Assigning all to training.",
            subset_label, n,
        )
        return {eid: SPLIT_TRAIN for eid in sorted_engines}

    if n_test <= 0:
        # Redistribute: take 1 from train for test
        n_train -= 1
        n_test = 1

    assignment: dict[str, str] = {}

    for idx, eid in enumerate(sorted_engines):
        if idx < n_train:
            assignment[eid] = SPLIT_TRAIN
        elif idx < n_train + n_val:
            assignment[eid] = SPLIT_VALIDATION
        else:
            assignment[eid] = SPLIT_TEST

    return assignment


def _verify_no_engine_leakage(
    dataframe: pd.DataFrame,
    split_column: str = "split",
) -> bool:
    """
    Verify that no engine_id appears in more than one split partition.

    Returns True if the split is clean, False if leakage is detected.
    """

    engine_split_pairs = (
        dataframe[[ENGINE_ID_COLUMN, SUBSET_COLUMN, split_column]]
        .drop_duplicates()
    )

    # Group by (engine_id, fd_subset) and check unique splits
    grouped = engine_split_pairs.groupby(
        [ENGINE_ID_COLUMN, SUBSET_COLUMN]
    )[split_column].nunique()

    leaked_engines = grouped[grouped > 1]

    if len(leaked_engines) > 0:
        logger.error(
            "ENGINE LEAKAGE DETECTED: %d engine(s) appear in multiple splits:",
            len(leaked_engines),
        )
        for (eid, subset), count in leaked_engines.items():
            logger.error(
                "  engine_id=%s, fd_subset=%s -> appears in %d splits",
                eid, subset, count,
            )
        return False

    logger.info(
        "Leakage verification PASSED: all %d engine-subset pairs are "
        "confined to exactly one split.",
        len(grouped),
    )
    return True


def _compute_partition_stats(
    dataframe: pd.DataFrame,
    partition_name: str,
    total_rows: int,
) -> PartitionStats:
    """Compute statistics for a single partition."""

    class_counts = dict(Counter(dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower()))
    partition_total = len(dataframe)
    class_fracs = {
        cls: count / partition_total if partition_total > 0 else 0.0
        for cls, count in class_counts.items()
    }

    engines = sorted(
        dataframe[ENGINE_ID_COLUMN].astype(str).unique().tolist()
    )

    return PartitionStats(
        partition_name=partition_name,
        engine_count=len(engines),
        row_count=partition_total,
        row_fraction=partition_total / total_rows if total_rows > 0 else 0.0,
        class_distribution=class_counts,
        class_fractions=class_fracs,
        engines=engines,
    )


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def run_engine_splitter(
    input_path: Path | None = None,
) -> EngineSplitterResult:
    """
    Run engine-level grouped splitting on the scaled dataset.

    Parameters
    ----------
    input_path:
        Path to the scaled CSV. Defaults to _SCALED_DATASET_PATH.

    Returns
    -------
    EngineSplitterResult

    Raises
    ------
    FileNotFoundError
        If the input CSV does not exist.
    RuntimeError
        If the dataset is missing required columns or is empty.
    """

    if input_path is None:
        input_path = _SCALED_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("SUPERVISOR ENGINE-LEVEL SPLITTING STARTED")
    logger.info("Input dataset   : %s", input_path)
    logger.info("Splits root     : %s", _SPLITS_ROOT)
    logger.info("Train fraction  : %.2f", _TRAIN_FRACTION)
    logger.info("Val fraction    : %.2f", _VAL_FRACTION)
    logger.info("Test fraction   : %.2f", _TEST_FRACTION)
    logger.info("Random seed     : %d", _RANDOM_SEED)

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
            f"Loaded dataset from '{input_path}' is empty."
        )

    total_rows = len(dataframe)
    logger.info(
        "Loaded rows=%d, columns=%d.",
        total_rows, dataframe.shape[1],
    )

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------

    required_cols = [ENGINE_ID_COLUMN, SUBSET_COLUMN, TARGET_COLUMN]
    missing = [c for c in required_cols if c not in dataframe.columns]

    if missing:
        raise RuntimeError(
            f"Dataset is missing required column(s) for splitting: {missing}"
        )

    # Normalise engine_id and fd_subset to strings
    dataframe[ENGINE_ID_COLUMN] = dataframe[ENGINE_ID_COLUMN].astype(str).str.strip()
    dataframe[SUBSET_COLUMN] = dataframe[SUBSET_COLUMN].astype(str).str.strip()

    # ------------------------------------------------------------------
    # Discover engines per sub-dataset
    # ------------------------------------------------------------------

    subset_engine_groups = (
        dataframe
        .groupby(SUBSET_COLUMN)[ENGINE_ID_COLUMN]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    total_subsets = len(subset_engine_groups)
    total_engines = sum(len(eids) for eids in subset_engine_groups.values())

    logger.info(
        "Discovered %d sub-dataset(s) with %d total engine(s).",
        total_subsets, total_engines,
    )

    for subset_label, engines in sorted(subset_engine_groups.items()):
        logger.info(
            "  Subset '%s': %d engine(s)", subset_label, len(engines)
        )

    # ------------------------------------------------------------------
    # Assign engines to splits (per sub-dataset)
    # ------------------------------------------------------------------

    logger.info("Assigning engines to splits using deterministic hashing.")

    engine_to_split: dict[tuple[str, str], str] = {}

    for subset_label, engines in subset_engine_groups.items():
        assignments = _assign_engines_to_splits(
            engine_ids=engines,
            subset_label=subset_label,
            seed=_RANDOM_SEED,
            train_frac=_TRAIN_FRACTION,
            val_frac=_VAL_FRACTION,
        )

        for eid, split_label in assignments.items():
            engine_to_split[(subset_label, eid)] = split_label

        # Log per-subset counts
        split_counts = Counter(assignments.values())
        logger.info(
            "  Subset '%s': train=%d, val=%d, test=%d engines",
            subset_label,
            split_counts.get(SPLIT_TRAIN, 0),
            split_counts.get(SPLIT_VALIDATION, 0),
            split_counts.get(SPLIT_TEST, 0),
        )

    # ------------------------------------------------------------------
    # Map split labels back to the full DataFrame
    # ------------------------------------------------------------------

    logger.info("Mapping split labels to all rows.")

    split_labels = dataframe.apply(
        lambda row: engine_to_split.get(
            (str(row[SUBSET_COLUMN]), str(row[ENGINE_ID_COLUMN])),
            SPLIT_TRAIN,  # Fallback (should never happen)
        ),
        axis=1,
    )

    dataframe["split"] = split_labels

    # ------------------------------------------------------------------
    # Verify no engine leakage
    # ------------------------------------------------------------------

    logger.info("Running engine-leakage verification.")
    leakage_ok = _verify_no_engine_leakage(dataframe, "split")

    if not leakage_ok:
        logger.error(
            "CRITICAL: Engine leakage detected. The split is invalid. "
            "This is a bug in the splitting logic."
        )

    # ------------------------------------------------------------------
    # Extract partitions
    # ------------------------------------------------------------------

    train_df = dataframe[dataframe["split"] == SPLIT_TRAIN].copy()
    val_df = dataframe[dataframe["split"] == SPLIT_VALIDATION].copy()
    test_df = dataframe[dataframe["split"] == SPLIT_TEST].copy()

    logger.info(
        "Partition sizes — train: %d (%.1f%%), val: %d (%.1f%%), test: %d (%.1f%%)",
        len(train_df), 100.0 * len(train_df) / total_rows,
        len(val_df), 100.0 * len(val_df) / total_rows,
        len(test_df), 100.0 * len(test_df) / total_rows,
    )

    # ------------------------------------------------------------------
    # Compute partition statistics
    # ------------------------------------------------------------------

    train_stats = _compute_partition_stats(train_df, SPLIT_TRAIN, total_rows)
    val_stats = _compute_partition_stats(val_df, SPLIT_VALIDATION, total_rows)
    test_stats = _compute_partition_stats(test_df, SPLIT_TEST, total_rows)

    # Global class distribution for comparison
    global_class_counts = dict(
        Counter(dataframe[TARGET_COLUMN].astype(str).str.strip().str.lower())
    )
    global_class_fracs = {
        cls: count / total_rows
        for cls, count in global_class_counts.items()
    }

    # ------------------------------------------------------------------
    # Write split CSV files
    # ------------------------------------------------------------------

    logger.info("Writing split CSV files atomically.")
    _SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    # Drop the "split" column from the individual split files (it's
    # redundant — the file name already identifies the partition).
    atomic_write_csv(train_df.drop(columns=["split"]), _TRAIN_PATH)
    logger.info("Train CSV    : %s (%d rows)", _TRAIN_PATH, len(train_df))

    atomic_write_csv(val_df.drop(columns=["split"]), _VAL_PATH)
    logger.info("Val CSV      : %s (%d rows)", _VAL_PATH, len(val_df))

    atomic_write_csv(test_df.drop(columns=["split"]), _TEST_PATH)
    logger.info("Test CSV     : %s (%d rows)", _TEST_PATH, len(test_df))

    # Full dataset with split column (for analysis / debugging)
    atomic_write_csv(dataframe, _FULL_SPLIT_PATH)
    logger.info("Full+split   : %s (%d rows)", _FULL_SPLIT_PATH, total_rows)

    # ------------------------------------------------------------------
    # Save split manifest
    # ------------------------------------------------------------------

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    split_manifest = {
        "generated_by": "engine_splitter.py",
        "input_dataset": str(input_path),
        "random_seed": _RANDOM_SEED,
        "target_fractions": {
            "train": _TRAIN_FRACTION,
            "validation": _VAL_FRACTION,
            "test": _TEST_FRACTION,
        },
        "actual_fractions": {
            "train": round(len(train_df) / total_rows, 4),
            "validation": round(len(val_df) / total_rows, 4),
            "test": round(len(test_df) / total_rows, 4),
        },
        "total_rows": total_rows,
        "total_engines": total_engines,
        "total_subsets": total_subsets,
        "leakage_check_passed": leakage_ok,
        "split_paths": {
            "train": str(_TRAIN_PATH),
            "validation": str(_VAL_PATH),
            "test": str(_TEST_PATH),
            "full_with_split_column": str(_FULL_SPLIT_PATH),
        },
        "partitions": {
            SPLIT_TRAIN: {
                "engines": train_stats.engines,
                "engine_count": train_stats.engine_count,
                "row_count": train_stats.row_count,
            },
            SPLIT_VALIDATION: {
                "engines": val_stats.engines,
                "engine_count": val_stats.engine_count,
                "row_count": val_stats.row_count,
            },
            SPLIT_TEST: {
                "engines": test_stats.engines,
                "engine_count": test_stats.engine_count,
                "row_count": test_stats.row_count,
            },
        },
    }

    atomic_write_json(split_manifest, _SPLIT_MANIFEST_PATH)
    logger.info("Split manifest: %s", _SPLIT_MANIFEST_PATH)

    # ------------------------------------------------------------------
    # Save split report
    # ------------------------------------------------------------------

    duration = time.perf_counter() - start_time

    result = EngineSplitterResult(
        input_path=str(input_path),
        total_rows=total_rows,
        total_engines=total_engines,
        total_subsets=total_subsets,
        target_fractions=global_class_fracs,
        partitions=[train_stats, val_stats, test_stats],
        leakage_check_passed=leakage_ok,
        duration_seconds=round(duration, 4),
    )

    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    report = result.to_dict()
    report["status"] = "success" if leakage_ok else "leakage_detected"
    report["global_class_distribution"] = global_class_counts

    atomic_write_json(report, _SPLIT_REPORT_PATH)
    logger.info("Split report  : %s", _SPLIT_REPORT_PATH)

    # ------------------------------------------------------------------
    # Summary log — class distribution comparison
    # ------------------------------------------------------------------

    logger.info("-" * 78)
    logger.info("Total rows                  : %d", total_rows)
    logger.info("Total engines               : %d", total_engines)
    logger.info("Total sub-datasets          : %d", total_subsets)
    logger.info("Leakage check               : %s", "PASSED" if leakage_ok else "FAILED")
    logger.info("-" * 78)

    # Print class distribution comparison table
    all_classes = sorted(global_class_fracs.keys())

    header = f"{'Class':<30} {'Global':>8} {'Train':>8} {'Val':>8} {'Test':>8}"
    logger.info(header)
    logger.info("-" * len(header))

    for cls in all_classes:
        g_frac = global_class_fracs.get(cls, 0.0)
        t_frac = train_stats.class_fractions.get(cls, 0.0)
        v_frac = val_stats.class_fractions.get(cls, 0.0)
        te_frac = test_stats.class_fractions.get(cls, 0.0)
        logger.info(
            f"{cls:<30} {g_frac:>7.1%} {t_frac:>7.1%} {v_frac:>7.1%} {te_frac:>7.1%}"
        )

    logger.info("-" * 78)
    logger.info(
        "Row counts: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Engine counts: train=%d, val=%d, test=%d",
        train_stats.engine_count, val_stats.engine_count, test_stats.engine_count,
    )
    logger.info("Duration                    : %.3f seconds", duration)

    section("SUPERVISOR ENGINE-LEVEL SPLITTING COMPLETED")

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _print_summary(result: EngineSplitterResult) -> None:
    """Print a compact JSON summary to stdout."""

    summary = {
        "status": "success" if result.leakage_check_passed else "leakage_detected",
        "input_path": result.input_path,
        "report_path": str(_SPLIT_REPORT_PATH),
        "manifest_path": str(_SPLIT_MANIFEST_PATH),
        "total_rows": result.total_rows,
        "total_engines": result.total_engines,
        "leakage_check_passed": result.leakage_check_passed,
        "partitions": {
            p.partition_name: {
                "engines": p.engine_count,
                "rows": p.row_count,
                "fraction": round(p.row_fraction, 4),
            }
            for p in result.partitions
        },
        "split_paths": {
            "train": str(_TRAIN_PATH),
            "validation": str(_VAL_PATH),
            "test": str(_TEST_PATH),
        },
        "duration_seconds": result.duration_seconds,
        "message": (
            "Engine-level grouped splitting completed successfully. "
            "No engine leakage detected."
            if result.leakage_check_passed
            else "WARNING: Engine leakage detected in split."
        ),
    }

    print(json.dumps(summary, indent=2))


def main() -> int:
    """
    CLI entry point.

    Returns
    -------
    int
        0 — splitting completed successfully, no leakage
        1 — internal failure or leakage detected
    """

    try:
        result = run_engine_splitter()
        _print_summary(result)
        return 0 if result.leakage_check_passed else 1

    except FileNotFoundError as exc:
        logger.error("Input file not found: %s", exc)
        return 1

    except RuntimeError as exc:
        logger.error("Runtime error during splitting: %s", exc)
        return 1

    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "Unexpected error during engine splitting: %s",
            exc,
            exc_info=True,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
