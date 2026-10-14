"""
Data loading and schema enforcement -- the component's front door.

Responsibilities:

* Ingest either raw C-MAPSS ``.txt`` files (via :mod:`cmapss_reader`) or an
  already-parsed CSV.
* Validate that all 26 required columns are present, naming exactly which are
  missing when they are not.
* Drop foreign columns **after** validation.  A CSV carrying ``rul_prediction``
  or ``failure_risk_score`` from a sibling component is perfectly acceptable
  input; those columns are simply ignored, never read, and never a cause of
  failure.
* Perform the engine-wise train/validation split.  Splitting by row would let
  cycles from one engine appear on both sides and leak the very degradation
  trajectory the twin is supposed to learn, so the split is always by ``unit``.
* Write ``raw_data.csv`` atomically.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import (
    CMAPSSReader,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    SPLIT_COLUMN,
    SUBSET_COLUMN,
    get_identity_columns,
    get_required_columns,
    select_required_columns,
    verify_schema,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    engine_wise_split,
    get_logger,
    read_csv,
    summarize_frame,
)

#: Columns this component keeps alongside the required 26 when they exist.
CARRY_COLUMNS: List[str] = [RUL_COLUMN, SUBSET_COLUMN, SPLIT_COLUMN]


class DataLoader:
    """Loads C-MAPSS data into the canonical schema and splits it by engine."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the loader.

        Args:
            config: Configuration supplying subset selection, split ratio and
                output paths.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.reader = CMAPSSReader(self.config)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Check an incoming frame against the required C-MAPSS schema.

        Args:
            frame: Data to inspect.

        Returns:
            A report containing ``schema_valid``, ``missing_columns`` and
            ``ignored_columns``.  Extra columns are reported but never make the
            data invalid.
        """
        required = get_required_columns()
        missing = [name for name in required if name not in frame.columns]
        ignored = [
            name
            for name in frame.columns
            if name not in required and name not in CARRY_COLUMNS
        ]
        report: Dict[str, Any] = {
            "schema_valid": not missing,
            "missing_columns": missing,
            "ignored_columns": ignored,
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
        }
        if missing:
            self.logger.error(
                "Schema validation failed -- %d required column(s) missing: %s",
                len(missing),
                missing,
            )
        if ignored:
            self.logger.info(
                "%d foreign column(s) will be ignored (not an error): %s",
                len(ignored),
                ignored[:20],
            )
        return report

    def narrow_to_required(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate, then reduce a frame to the columns this component consumes.

        Args:
            frame: Incoming data, possibly with foreign columns.

        Returns:
            A copy narrowed to the 26 required columns plus RUL/subset/split
            when present.

        Raises:
            SchemaValidationError: If a required column is missing.
        """
        return select_required_columns(frame, extra_keep=CARRY_COLUMNS)

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def load(
        self,
        subsets: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        source_csv: Optional[str | Path] = None,
        max_engines: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load data from raw text files or an already-parsed CSV.

        Args:
            subsets: Subsets to read; defaults to config.
            split: ``'train'``, ``'test'`` or ``'both'``; defaults to config.
            source_csv: When given, this CSV is used and raw ingestion skipped.
            max_engines: Sub-sample this many engines for a fast run. 0/None
                means use everything.

        Returns:
            The narrowed, validated, sorted DataFrame.

        Raises:
            AnomalyHealthError: If nothing could be loaded.
        """
        if source_csv:
            frame = self.reader.read_csv_source(source_csv)
        else:
            frame = self.reader.read_subsets(subsets, split)

        frame = self.narrow_to_required(frame)
        frame = self._ensure_rul(frame)
        frame = self._coerce_types(frame)

        limit = max_engines if max_engines is not None else self.config.data.max_engines
        if limit and limit > 0:
            frame = self._subsample_engines(frame, limit)

        frame = frame.sort_values(get_identity_columns()).reset_index(drop=True)
        verify_schema(frame, context="loaded data")
        self.logger.info("Loaded data: %s", summarize_frame(frame, "raw_data"))
        return frame

    def _ensure_rul(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Derive RUL when the source CSV did not already carry it.

        A CSV supplied by a caller may or may not include the label.  When it
        is absent the run-to-failure convention is applied, which is correct
        for training trajectories.

        Args:
            frame: Data possibly lacking a RUL column.

        Returns:
            The frame with a RUL column guaranteed present.
        """
        if RUL_COLUMN in frame.columns:
            return frame
        self.logger.info(
            "No %s column supplied; deriving it as max(cycle) - cycle per engine "
            "(valid for run-to-failure training trajectories).",
            RUL_COLUMN,
        )
        result = frame.copy()
        max_cycle = result.groupby("unit")["cycle"].transform("max")
        result[RUL_COLUMN] = (max_cycle - result["cycle"]).astype(int)
        return result

    def _coerce_types(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Force identity columns to integers and feature columns to floats.

        Args:
            frame: Data to coerce.

        Returns:
            The type-normalised frame.

        Raises:
            AnomalyHealthError: If the identity keys cannot be made integral.
        """
        result = frame.copy()
        try:
            result["unit"] = result["unit"].astype(np.int64)
            result["cycle"] = result["cycle"].astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise AnomalyHealthError(
                f"'unit' and 'cycle' must be integral: {exc}"
            ) from exc

        for name in get_required_columns():
            if name in ("unit", "cycle"):
                continue
            result[name] = pd.to_numeric(result[name], errors="coerce").astype(float)

        if RUL_COLUMN in result.columns:
            result[RUL_COLUMN] = (
                pd.to_numeric(result[RUL_COLUMN], errors="coerce")
                .fillna(0)
                .astype(np.int64)
            )
        return result

    def _subsample_engines(self, frame: pd.DataFrame, max_engines: int) -> pd.DataFrame:
        """Keep only the first ``max_engines`` engines, whole trajectories intact.

        Args:
            frame: Full data.
            max_engines: Engine budget.

        Returns:
            The reduced frame.
        """
        units = np.sort(frame["unit"].unique())
        if units.size <= max_engines:
            return frame
        keep = set(units[:max_engines].tolist())
        reduced = frame[frame["unit"].isin(keep)].copy()
        self.logger.info(
            "Sub-sampled %d of %d engines (%d rows)",
            len(keep),
            units.size,
            len(reduced),
        )
        return reduced

    # ------------------------------------------------------------------ #
    # Engine-wise splitting
    # ------------------------------------------------------------------ #
    def split_by_engine(
        self,
        frame: pd.DataFrame,
        validation_fraction: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split so every engine lands wholly in train or wholly in validation.

        This is the only splitting strategy permitted in this component.  A
        random row split would place early cycles of an engine in training and
        late cycles of the same engine in validation, letting the model see the
        trajectory it is being asked to predict.

        Args:
            frame: Data to split.
            validation_fraction: Fraction of engines held out; defaults to
                config.

        Returns:
            A ``(train, validation)`` tuple.
        """
        fraction = (
            validation_fraction
            if validation_fraction is not None
            else self.config.data.validation_engine_fraction
        )
        train, validation = engine_wise_split(
            frame,
            unit_column="unit",
            validation_fraction=fraction,
            random_state=self.config.data.random_state,
        )
        overlap = set(train["unit"].unique()) & set(validation["unit"].unique())
        if overlap:
            raise AnomalyHealthError(
                f"Engine-wise split leaked {len(overlap)} engine(s) across both "
                f"sides: {sorted(overlap)[:10]}"
            )
        self.logger.info(
            "Engine-wise split: train %d rows / %d engines, val %d rows / %d engines",
            len(train),
            train["unit"].nunique(),
            len(validation),
            validation["unit"].nunique() if len(validation) else 0,
        )
        return train, validation

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:
        """Write ``raw_data.csv`` atomically, keys intact.

        Args:
            frame: Data to persist.
            path: Optional explicit destination.

        Returns:
            The path written.
        """
        target = path or self.config.paths.processed_path(self.config.paths.raw_data_csv)
        verify_schema(frame, get_identity_columns(), context="raw_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:
        """Read back a previously written ``raw_data.csv``.

        Args:
            path: Optional explicit source.

        Returns:
            The loaded frame.
        """
        target = path or self.config.paths.processed_path(self.config.paths.raw_data_csv)
        return read_csv(target, required_columns=get_identity_columns(), stage="data_loader")

    # ------------------------------------------------------------------ #
    # Stage entry point
    # ------------------------------------------------------------------ #
    def run(
        self,
        subsets: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        source_csv: Optional[str | Path] = None,
        max_engines: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the loading stage and persist ``raw_data.csv``.

        Args:
            subsets: Subsets to read.
            split: Split to read.
            source_csv: Optional already-parsed CSV.
            max_engines: Engine budget for a fast run.

        Returns:
            A ``(frame, report)`` tuple.
        """
        self.config.ensure_directories()
        frame = self.load(subsets, split, source_csv, max_engines)
        output = self.save(frame)

        train, validation = self.split_by_engine(frame)
        report: Dict[str, Any] = {
            "rows": int(len(frame)),
            "engines": int(frame["unit"].nunique()),
            "columns": int(frame.shape[1]),
            "subsets": sorted(frame[SUBSET_COLUMN].unique().tolist())
            if SUBSET_COLUMN in frame.columns
            else [],
            "train_engines": int(train["unit"].nunique()),
            "validation_engines": int(validation["unit"].nunique())
            if len(validation)
            else 0,
            "rul_min": int(frame[RUL_COLUMN].min()),
            "rul_max": int(frame[RUL_COLUMN].max()),
            "output_path": str(output),
        }
        self.logger.info("Data loading report: %s", report)
        return frame, report


def main() -> None:
    """Run the loader standalone and report what was ingested."""
    config = get_config()
    logger = get_logger(__name__, config)
    loader = DataLoader(config)
    try:
        frame, report = loader.run()
    except AnomalyHealthError as exc:
        logger.error("Data loading failed: %s", exc)
        raise SystemExit(1) from exc
    logger.info("raw_data.csv written with %d rows", report["rows"])
    print(frame.head(10).to_string())


if __name__ == "__main__":
    main()
