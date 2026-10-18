from __future__ import annotations

import sys
import time
import re
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


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.raw_dir: Path = self.config.paths.data_raw


    def read_txt(self, path: str | Path) -> pd.DataFrame:


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


        file_path = Path(path)
        if not file_path.is_file():
            raise ArtifactNotFoundError(f"RUL file '{file_path}' not found.")
        values = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
        series = values.iloc[:, 0].astype(int).to_numpy()
        self.logger.info("Read %s: %d engine RUL values", file_path.name, series.size)
        return series


    @staticmethod
    def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:


        result = frame.copy()
        max_cycle = result.groupby("unit")["cycle"].transform("max")
        result[RUL_COLUMN] = (max_cycle - result["cycle"]).astype(int)
        return result

    @staticmethod
    def add_test_rul(frame: pd.DataFrame, rul_at_end: np.ndarray) -> pd.DataFrame:


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


    def read_subset(
        self,
        subset_id: str,
        split: str = "train",
        unit_offset: int = 0,
    ) -> pd.DataFrame:


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


        chosen = list(subsets) if subsets else list(self.config.data.subsets)
        split_choice = (split or self.config.data.split).lower().strip()
        splits = ["train", "test"] if split_choice == "both" else [split_choice]

        frames: List[pd.DataFrame] = []
        for index, subset_id in enumerate(chosen):
            offset = index * self.config.data.unit_id_offset_step
            for split_name in splits:
                try:
                    population_offset = 1_000_000 if split_choice == "both" and split_name == "test" else 0
                    frames.append(self.read_subset(subset_id, split_name, offset + population_offset))
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


    def read_csv_source(self, path: str | Path) -> pd.DataFrame:


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
            except Exception as exc:
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
        if SPLIT_COLUMN not in frame:
            match = re.search(r"(?:^|[_-])(train|test)(?:[_-]|$)", file_path.stem.lower())
            frame[SPLIT_COLUMN] = match.group(1) if match else "inference"
        return self._attach_true_test_rul(frame, file_path)

    def _attach_true_test_rul(
        self, frame: pd.DataFrame, file_path: Path
    ) -> pd.DataFrame:


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
                "companion '%s' was found in %s or %s. Scoring remains "
                "available; verified RUL targets and supervised metrics are "
                "unavailable until '%s' is supplied.",
                file_path.name, rul_name, file_path.parent, self.raw_dir, rul_name,
            )
            return frame

        rul_values = self.read_rul_file(rul_path)
        n_engines = int(frame["unit"].nunique())
        if n_engines != rul_values.size:
            self.logger.warning(
                "'%s' has %d engine(s) but '%s' carries %d RUL value(s); they "
                "must correspond 1:1. True RUL is unavailable; omit supervised "
                "targets and metrics for this inference batch.",
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


        suffix = Path(path).suffix.lower()
        return "csv" if suffix == ".csv" else "cmapss_txt"

    @staticmethod
    def infer_subset_id(path: str | Path) -> Optional[str]:


        stem = Path(path).stem.upper()
        for subset_id in VALID_SUBSETS:
            if subset_id in stem:
                return subset_id
        return None

    def available_files(self) -> Dict[str, List[str]]:


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


    return CMAPSSReader(config).read_subsets(subsets, split)


def main() -> None:

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
