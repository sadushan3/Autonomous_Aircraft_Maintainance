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


CARRY_COLUMNS: List[str] = [RUL_COLUMN, SUBSET_COLUMN, SPLIT_COLUMN]


class DataLoader:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.reader = CMAPSSReader(self.config)


    def validate(self, frame: pd.DataFrame) -> Dict[str, Any]:


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


        return select_required_columns(frame, extra_keep=CARRY_COLUMNS)


    def load(
        self,
        subsets: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        source_csv: Optional[str | Path] = None,
        max_engines: Optional[int] = None,
    ) -> pd.DataFrame:


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


        if RUL_COLUMN in frame.columns:
            return frame
        if "split" not in frame or not frame["split"].eq("train").all():
            self.logger.info("No verified RUL labels supplied for inference; leaving evaluation targets unavailable.")
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
            result[RUL_COLUMN] = pd.to_numeric(result[RUL_COLUMN], errors="coerce")
            if "split" in result and result.split.eq("train").any():
                training_labels = result.loc[result.split.eq("train"), RUL_COLUMN].to_numpy(float)
                if not np.isfinite(training_labels).all():
                    raise AnomalyHealthError("Training RUL labels must be finite; missing labels cannot be replaced by zero.")
        return result

    def _subsample_engines(self, frame: pd.DataFrame, max_engines: int) -> pd.DataFrame:


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


    def split_by_engine(
        self,
        frame: pd.DataFrame,
        validation_fraction: Optional[float] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:


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


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(self.config.paths.raw_data_csv)
        verify_schema(frame, get_identity_columns(), context="raw_data output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(self.config.paths.raw_data_csv)
        return read_csv(target, required_columns=get_identity_columns(), stage="data_loader")


    def run(
        self,
        subsets: Optional[Sequence[str]] = None,
        split: Optional[str] = None,
        source_csv: Optional[str | Path] = None,
        max_engines: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


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
            "rul_min": float(frame[RUL_COLUMN].min()) if RUL_COLUMN in frame and frame[RUL_COLUMN].notna().any() else None,
            "rul_max": float(frame[RUL_COLUMN].max()) if RUL_COLUMN in frame and frame[RUL_COLUMN].notna().any() else None,
            "output_path": str(output),
        }
        self.logger.info("Data loading report: %s", report)
        return frame, report


def main() -> None:

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
