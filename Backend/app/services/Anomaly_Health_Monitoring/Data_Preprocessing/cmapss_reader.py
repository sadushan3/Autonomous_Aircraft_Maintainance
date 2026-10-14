"""
Classic NASA C-MAPSS ``.txt`` reader.

This module replaces the former HDF5 / N-CMAPSS reader entirely.  There is no
``.h5`` file, no ``h5py`` dependency, and no ``X_s`` / ``X_v`` / ``W`` / ``A``
variable groups anywhere in this component.

The raw files are space-delimited plain text with one row per engine per cycle
and **no header row**.  Column names are assigned on read, in this fixed order:

    unit, cycle, op_setting_1..3, sensor_1..21     (26 columns)

Three file families are handled:

* ``train_FD00X.txt`` -- run-to-failure trajectories.  RUL is derived as
  ``max(cycle) - cycle`` per engine, because every training engine runs until
  it fails.
* ``test_FD00X.txt`` -- truncated trajectories.  The engine has *not* failed at
  the last observed cycle, so the derived RUL must be offset.
* ``RUL_FD00X.txt`` -- one integer per test engine giving the true remaining
  cycles *at the final observed cycle*.  Combined with the test file this
  yields ``RUL(t) = (max_cycle - cycle) + rul_at_end``.

Getting that offset right matters: treating test trajectories like training
trajectories would tell the model every test engine fails at its last cycle,
which is false for all of them.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    RUL_COLUMN,
    SPLIT_COLUMN,
    SUBSET_CHARACTERISTICS,
    SUBSET_COLUMN,
    VALID_SUBSETS,
    get_identity_columns,
    get_raw_columns,
    verify_schema,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    ArtifactNotFoundError,
    get_logger,
)


class CMAPSSReader:
    """Reads classic C-MAPSS text files into the canonical 26-column schema.

    The reader is deliberately dependency-injected: it takes a
    :class:`Config` so tests and API callers can point it at a different data
    directory without touching module state.
    """

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the reader.

        Args:
            config: Configuration supplying the raw-data directory and the
                subset / split selection.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.raw_dir: Path = self.config.paths.data_raw

    # ------------------------------------------------------------------ #
    # Low-level file reading
    # ------------------------------------------------------------------ #
    def read_txt(self, path: str | Path) -> pd.DataFrame:
        """Read one C-MAPSS ``.txt`` file and assign the 26 column names.

        The files are whitespace-delimited and often carry two trailing spaces
        per line, which naive parsing turns into phantom all-NaN columns.  Those
        are discarded before naming so the column count always lands on 26.

        Args:
            path: Path to a ``train_*``/``test_*`` C-MAPSS text file.

        Returns:
            A DataFrame with exactly the canonical 26 columns.

        Raises:
            ArtifactNotFoundError: If the file does not exist.
            AnomalyHealthError: If the parsed column count is not 26.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise ArtifactNotFoundError(
                f"C-MAPSS file '{file_path}' not found. Place the NASA "
                f"FD001..FD004 .txt files in {self.raw_dir}."
            )

        frame = pd.read_csv(
            file_path,
            sep=r"\s+",
            header=None,
            engine="python",
            skip_blank_lines=True,
        )

        # Drop the phantom columns produced by trailing whitespace.
        all_nan = [c for c in frame.columns if frame[c].isna().all()]
        if all_nan:
            frame = frame.drop(columns=all_nan)

        expected = get_raw_columns()
        if frame.shape[1] != len(expected):
            raise AnomalyHealthError(
                f"'{file_path.name}' parsed into {frame.shape[1]} columns but the "
                f"C-MAPSS layout requires exactly {len(expected)}. This file does "
                f"not look like a classic C-MAPSS text file."
            )

        frame.columns = expected
        frame["unit"] = frame["unit"].astype(int)
        frame["cycle"] = frame["cycle"].astype(int)
        self.logger.info(
            "Read %s: %d rows, %d engines",
            file_path.name,
            len(frame),
            frame["unit"].nunique(),
        )
        return frame

    def read_rul_file(self, path: str | Path) -> np.ndarray:
        """Read a ``RUL_FD00X.txt`` file of true end-of-trajectory RUL values.

        Args:
            path: Path to the RUL file.

        Returns:
            A 1-D integer array, one entry per test engine in engine-id order.

        Raises:
            ArtifactNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise ArtifactNotFoundError(f"RUL file '{file_path}' not found.")
        values = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        series = values.iloc[:, 0].astype(int).to_numpy()
        self.logger.info("Read %s: %d engine RUL values", file_path.name, series.size)
        return series

    # ------------------------------------------------------------------ #
    # RUL derivation
    # ------------------------------------------------------------------ #
    @staticmethod
    def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
        """Derive RUL for run-to-failure training trajectories.

        Every training engine runs until failure, so the last observed cycle is
        the failure point and ``RUL = max(cycle) - cycle``.

        Args:
            frame: Training data with ``unit`` and ``cycle``.

        Returns:
            The frame with an integer ``RUL`` column added.
        """
        result = frame.copy()
        max_cycle = result.groupby("unit")["cycle"].transform("max")
        result[RUL_COLUMN] = (max_cycle - result["cycle"]).astype(int)
        return result

    @staticmethod
    def add_test_rul(frame: pd.DataFrame, rul_at_end: np.ndarray) -> pd.DataFrame:
        """Derive RUL for truncated test trajectories.

        Test engines are still running at their last observed cycle, so the
        remaining life there is given by the companion RUL file rather than
        being zero:

            RUL(t) = (max_cycle - cycle) + rul_at_end[engine]

        Args:
            frame: Test data with ``unit`` and ``cycle``.
            rul_at_end: True RUL at the final cycle, ordered by ascending
                engine id.

        Returns:
            The frame with an integer ``RUL`` column added.

        Raises:
            AnomalyHealthError: If the RUL file length does not match the
                number of test engines.
        """
        result = frame.copy()
        units = np.sort(result["unit"].unique())
        if units.size != rul_at_end.size:
            raise AnomalyHealthError(
                f"RUL file has {rul_at_end.size} entries but the test set "
                f"contains {units.size} engines; they must correspond 1:1."
            )
        end_rul = pd.Series(rul_at_end, index=units, name="_end_rul")
        max_cycle = result.groupby("unit")["cycle"].transform("max")
        offset = result["unit"].map(end_rul).astype(int)
        result[RUL_COLUMN] = ((max_cycle - result["cycle"]) + offset).astype(int)
        return result

    # ------------------------------------------------------------------ #
    # Subset-level reading
    # ------------------------------------------------------------------ #
    def read_subset(
        self,
        subset_id: str,
        split: str = "train",
        unit_offset: int = 0,
    ) -> pd.DataFrame:
        """Read one subset/split pair and attach RUL plus provenance columns.

        Args:
            subset_id: One of FD001..FD004.
            split: ``'train'`` or ``'test'``.
            unit_offset: Added to every engine id.  When pooling subsets this
                keeps ids from colliding across files.

        Returns:
            A DataFrame with the 26 canonical columns plus ``RUL``,
            ``subset_id`` and ``split``.

        Raises:
            AnomalyHealthError: If the subset or split is unrecognised.
        """
        if subset_id not in VALID_SUBSETS:
            raise AnomalyHealthError(
                f"Unknown subset '{subset_id}'. Valid subsets: {VALID_SUBSETS}"
            )
        split_lower = split.lower().strip()
        if split_lower not in {"train", "test"}:
            raise AnomalyHealthError(f"split must be 'train' or 'test', got '{split}'")

        data_path = self.raw_dir / f"{split_lower}_{subset_id}.txt"
        frame = self.read_txt(data_path)

        if split_lower == "train":
            frame = self.add_train_rul(frame)
        else:
            rul_values = self.read_rul_file(self.raw_dir / f"RUL_{subset_id}.txt")
            frame = self.add_test_rul(frame, rul_values)

        frame[SUBSET_COLUMN] = subset_id
        frame[SPLIT_COLUMN] = split_lower
        if unit_offset:
            frame["unit"] = frame["unit"] + unit_offset

        self._log_subset_check(frame, subset_id, split_lower)
        return frame

    def _log_subset_check(
        self, frame: pd.DataFrame, subset_id: str, split: str
    ) -> None:
        """Compare what was read against the subset's documented characteristics.

        A mismatch is logged rather than raised, since a user may legitimately
        supply a truncated file for a smoke test.

        Args:
            frame: The subset that was read.
            subset_id: Subset label.
            split: Split label.
        """
        expected = SUBSET_CHARACTERISTICS.get(subset_id, {})
        n_engines = int(frame["unit"].nunique())
        if split == "train" and "train_engines" in expected:
            documented = expected["train_engines"]
            if n_engines != documented:
                self.logger.warning(
                    "%s train has %d engines; NASA documents %d. "
                    "Proceeding with what was supplied.",
                    subset_id,
                    n_engines,
                    documented,
                )
        self.logger.info(
            "%s/%s -> %d rows, %d engines, cycles %d-%d, %d expected regime(s)",
            subset_id,
            split,
            len(frame),
            n_engines,
            int(frame["cycle"].min()),
            int(frame["cycle"].max()),
            expected.get("operating_conditions", 1),
        )

    def read_subsets(
        self,
        subsets: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read and pool one or more subsets.

        When pooling, engine ids are offset per subset so that unit 1 of FD001
        and unit 1 of FD002 remain distinguishable downstream.

        Args:
            subsets: Subsets to read.  Defaults to the configured selection.
            split: ``'train'``, ``'test'`` or ``'both'``.  Defaults to config.

        Returns:
            The pooled DataFrame, sorted by ``(unit, cycle)``.

        Raises:
            AnomalyHealthError: If nothing could be read.
        """
        chosen = list(subsets) if subsets else list(self.config.data.subsets)
        split_choice = (split or self.config.data.split).lower().strip()
        splits = ["train", "test"] if split_choice == "both" else [split_choice]

        frames: List[pd.DataFrame] = []
        for index, subset_id in enumerate(chosen):
            offset = index * self.config.data.unit_id_offset_step
            for split_name in splits:
                try:
                    frames.append(self.read_subset(subset_id, split_name, offset))
                except ArtifactNotFoundError as exc:
                    self.logger.warning("Skipping %s/%s: %s", subset_id, split_name, exc)

        if not frames:
            raise AnomalyHealthError(
                f"No C-MAPSS files could be read for subsets={chosen}, "
                f"split='{split_choice}'. Expected files such as "
                f"'train_FD001.txt' in {self.raw_dir}."
            )

        pooled = pd.concat(frames, ignore_index=True)
        pooled = pooled.sort_values(get_identity_columns()).reset_index(drop=True)
        verify_schema(pooled, context="pooled C-MAPSS data")
        self.logger.info(
            "Pooled %d subset/split frame(s): %d rows, %d engines",
            len(frames),
            len(pooled),
            pooled["unit"].nunique(),
        )
        return pooled

    # ------------------------------------------------------------------ #
    # CSV path (already-parsed input)
    # ------------------------------------------------------------------ #
    def read_csv_source(self, path: str | Path) -> pd.DataFrame:
        """Read a caller-supplied source, whichever legitimate form it takes.

        This is the single entry point every ``source_csv=`` upload goes
        through (``/preprocess``, ``/full-pipeline``, ``/integration/run-full``
        via ``stored_path``), so it has to handle both forms a C-MAPSS file
        legitimately arrives in: an already-parsed, comma-delimited CSV with
        canonical headers, or the raw NASA ``.txt`` layout -- 26
        whitespace-delimited, headerless columns -- exactly as uploaded by
        ``/integration/upload``. Routing a raw ``.txt`` through the CSV branch
        collapses every row into a single garbage column (pandas reads the
        first data row as a comma-delimited header) and fails schema
        validation with a confusing "missing 26 columns" error instead of
        actually parsing the file.

        Format is decided the same way :meth:`detect_format` already decides
        it for ``/integration/upload``, so a raw file is recognised
        identically regardless of which endpoint receives it.

        Args:
            path: Path to the caller-supplied file.

        Returns:
            The loaded DataFrame.

        Raises:
            ArtifactNotFoundError: If the file is absent.
            AnomalyHealthError: If it cannot be parsed.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise ArtifactNotFoundError(f"CSV source '{file_path}' not found.")

        if self.detect_format(file_path) == "cmapss_txt":
            frame = self.read_txt(file_path)
            self.logger.info(
                "Read %s as raw C-MAPSS text: %d rows, %d columns "
                "(no RUL column -- derived downstream via the run-to-failure "
                "convention unless the caller supplies one)",
                file_path.name,
                len(frame),
                frame.shape[1],
            )
        else:
            try:
                frame = pd.read_csv(file_path)
            except Exception as exc:  # noqa: BLE001
                raise AnomalyHealthError(
                    f"Could not parse CSV '{file_path}': {exc}"
                ) from exc
            self.logger.info(
                "Read CSV %s: %d rows, %d columns",
                file_path.name,
                len(frame),
                frame.shape[1],
            )

        frame = self._attach_inferred_subset(frame, file_path)
        return self._attach_true_test_rul(frame, file_path)

    def _attach_true_test_rul(
        self, frame: pd.DataFrame, file_path: Path
    ) -> pd.DataFrame:
        """Join the companion ``RUL_FD00X.txt`` when a *test* file is uploaded.

        :meth:`read_subset` already gets this right, because it is told which
        split it is reading and calls :meth:`add_test_rul`. A file arriving
        through :meth:`read_csv_source` carries no split label, so it reached
        ``DataLoader._ensure_rul`` with no RUL column and got the
        *run-to-failure* convention applied: ``RUL = max(cycle) - cycle``,
        which lands every engine at RUL 0 on its last observed cycle.

        That is true of ``train_FD00X.txt`` and false of ``test_FD00X.txt`` --
        test trajectories are truncated while the engine is still running. On
        FD002 the real remaining life at the final cycle averages ~81 cycles
        and is never once zero, so the derived labels asserted 259 imminent
        failures that do not exist. Nothing in scoring reads RUL, so the twin
        and the detectors were unaffected, but ``health_index_target`` (the
        dashboard's true-health reference curve) and every acceptance metric
        that correlates against RUL were measured against invented ground
        truth.

        The companion file is looked for next to the upload first -- operators
        upload ``RUL_FD002.txt`` alongside ``test_FD002.txt`` -- then in the
        configured raw directory. When neither is present, or the engine count
        does not line up 1:1, RUL is left absent so the caller's
        run-to-failure fallback still applies; that case is logged loudly
        rather than raised, since a deliberately truncated file is a valid
        smoke test.

        Args:
            frame: The freshly loaded frame.
            file_path: Source path, whose name may identify the split.

        Returns:
            The frame, with a true ``RUL`` column when one could be derived.
        """
        if RUL_COLUMN in frame.columns:
            return frame
        if "test" not in file_path.stem.lower():
            return frame

        subset_id = self.infer_subset_id(file_path)
        if subset_id is None:
            return frame

        rul_name = f"RUL_{subset_id}.txt"
        candidates = [file_path.parent / rul_name, self.raw_dir / rul_name]
        rul_path = next((p for p in candidates if p.is_file()), None)
        if rul_path is None:
            self.logger.warning(
                "'%s' looks like a truncated C-MAPSS *test* file but no "
                "companion '%s' was found in %s or %s. RUL will be derived as "
                "max(cycle) - cycle, which asserts every engine fails at its "
                "last observed cycle -- false for test trajectories. Scoring "
                "is unaffected, but health_index_target and every "
                "RUL-correlated evaluation metric will be wrong. Upload '%s' "
                "alongside it to fix this.",
                file_path.name, rul_name, file_path.parent, self.raw_dir, rul_name,
            )
            return frame

        rul_values = self.read_rul_file(rul_path)
        n_engines = int(frame["unit"].nunique())
        if n_engines != rul_values.size:
            self.logger.warning(
                "'%s' has %d engine(s) but '%s' carries %d RUL value(s); they "
                "must correspond 1:1, so the true end-of-trajectory RUL cannot "
                "be applied. Falling back to max(cycle) - cycle, which is "
                "wrong for test trajectories -- treat health_index_target and "
                "any RUL-correlated metric from this run as unusable.",
                file_path.name, n_engines, rul_path.name, rul_values.size,
            )
            return frame

        frame = self.add_test_rul(frame, rul_values)
        self.logger.info(
            "Applied true end-of-trajectory RUL from '%s' to %d engine(s) in "
            "'%s' (mean RUL at final cycle: %.1f cycles). Without this the "
            "run-to-failure convention would have labelled every engine as "
            "failing at its last observed cycle.",
            rul_path.name, n_engines, file_path.name, float(rul_values.mean()),
        )
        return frame

    def _attach_inferred_subset(
        self, frame: pd.DataFrame, file_path: Path
    ) -> pd.DataFrame:
        """Label a caller-supplied frame with its C-MAPSS subset when possible.

        :meth:`read_subset` stamps ``subset_id`` because it is told which
        subset it is reading. :meth:`read_txt` cannot -- it only sees 26
        unlabelled numeric columns -- so a frame arriving through
        :meth:`read_csv_source` carries no subset at all unless it is
        recovered from the filename here.

        That label is not cosmetic. The Predictive Maintenance batch selects
        *which trained model to load* from it, and silently defaults to
        ``FD001`` when it is absent: an uploaded ``test_FD002.txt`` was
        scored end to end against FD001's RUL and Failure-Risk models, with
        FD001's 50-cycle window instead of FD002's 70, even though FD002 has
        six operating regimes against FD001's one. The Maintenance Supervisor
        bridge is affected too -- it builds its engine-to-subset map from this
        column and passes an empty one when it is missing, which breaks the
        three-way exchange join.

        Anything already carrying ``subset_id`` (a prepared CSV, say) is left
        untouched; the filename is only a fallback.

        Args:
            frame: The freshly loaded frame.
            file_path: Source path, whose name may name the subset.

        Returns:
            The frame, with ``subset_id`` attached when it could be inferred.
        """
        if SUBSET_COLUMN in frame.columns and frame[SUBSET_COLUMN].notna().any():
            return frame

        subset_id = self.infer_subset_id(file_path)
        if subset_id is None:
            self.logger.warning(
                "Could not infer a C-MAPSS subset from '%s' and the data "
                "carries no '%s' column. Downstream components that select a "
                "trained model per subset will fall back to their default, "
                "which may not match this data. Rename the file to include "
                "the subset (e.g. 'test_FD002.txt') or supply a '%s' column.",
                file_path.name,
                SUBSET_COLUMN,
                SUBSET_COLUMN,
            )
            return frame

        frame = frame.copy()
        frame[SUBSET_COLUMN] = subset_id
        self.logger.info(
            "Inferred subset '%s' from filename '%s' and attached it as '%s' "
            "so per-subset models are selected correctly downstream.",
            subset_id,
            file_path.name,
            SUBSET_COLUMN,
        )
        return frame

    @staticmethod
    def detect_format(path: str | Path) -> str:
        """Classify an input file as a C-MAPSS text file or a CSV.

        Args:
            path: Candidate file.

        Returns:
            ``'cmapss_txt'`` or ``'csv'``.
        """
        suffix = Path(path).suffix.lower()
        return "csv" if suffix == ".csv" else "cmapss_txt"

    @staticmethod
    def infer_subset_id(path: str | Path) -> Optional[str]:
        """Infer the subset label from a filename such as ``train_FD002.txt``.

        Args:
            path: The file whose name should be inspected.

        Returns:
            The subset label, or None when it cannot be determined.
        """
        stem = Path(path).stem.upper()
        for subset_id in VALID_SUBSETS:
            if subset_id in stem:
                return subset_id
        return None

    def available_files(self) -> Dict[str, List[str]]:
        """Report which C-MAPSS files are actually present on disk.

        Returns:
            A mapping of ``'train'``/``'test'``/``'rul'`` to the subset labels
            found in the raw-data directory.
        """
        found: Dict[str, List[str]] = {"train": [], "test": [], "rul": []}
        if not self.raw_dir.is_dir():
            return found
        for subset_id in VALID_SUBSETS:
            if (self.raw_dir / f"train_{subset_id}.txt").is_file():
                found["train"].append(subset_id)
            if (self.raw_dir / f"test_{subset_id}.txt").is_file():
                found["test"].append(subset_id)
            if (self.raw_dir / f"RUL_{subset_id}.txt").is_file():
                found["rul"].append(subset_id)
        return found


def read_cmapss(
    subsets: Optional[Sequence[str]] = None,
    split: Optional[str] = None,
    config: Optional[Config] = None,
) -> pd.DataFrame:
    """Convenience wrapper reading pooled C-MAPSS data.

    Args:
        subsets: Subsets to read; defaults to config.
        split: Split to read; defaults to config.
        config: Optional configuration override.

    Returns:
        The pooled DataFrame.
    """
    return CMAPSSReader(config).read_subsets(subsets, split)


def main() -> None:
    """Read the configured subsets and print a structural summary."""
    config = get_config()
    logger = get_logger(__name__, config)
    reader = CMAPSSReader(config)

    available = reader.available_files()
    logger.info("Available C-MAPSS files: %s", available)
    if not available["train"] and not available["test"]:
        logger.error(
            "No C-MAPSS .txt files found in %s. Place train_FD00X.txt, "
            "test_FD00X.txt and RUL_FD00X.txt there, or run "
            "scripts/generate_synthetic_cmapss.py to produce schema-identical "
            "sample data.",
            reader.raw_dir,
        )
        return

    started = time.perf_counter()
    frame = reader.read_subsets()
    logger.info(
        "Loaded %d rows / %d engines in %.2fs",
        len(frame),
        frame["unit"].nunique(),
        time.perf_counter() - started,
    )
    logger.info("Columns (%d): %s", frame.shape[1], list(frame.columns))
    logger.info(
        "RUL range: %d - %d", int(frame[RUL_COLUMN].min()), int(frame[RUL_COLUMN].max())
    )
    print(frame.head(10).to_string())


if __name__ == "__main__":
    main()
