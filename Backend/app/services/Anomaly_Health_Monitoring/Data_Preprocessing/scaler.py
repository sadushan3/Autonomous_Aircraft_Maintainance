from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import (
    get_active_sensors,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    get_identity_columns,
    get_op_setting_columns,
    get_operating_mode_column,
    verify_identity_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    atomic_write_csv,
    atomic_write_json,
    get_logger,
    load_model,
    read_csv,
    read_json,
    save_model,
)


class PerModeScaler:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.mode_column = get_operating_mode_column()
        self.scalers: Dict[int, Any] = {}
        self.columns: List[str] = []
        self.mode_centroids: Dict[int, List[float]] = {}


    def resolve_columns(self, frame: pd.DataFrame) -> List[str]:


        candidates = get_active_sensors(self.config) + get_op_setting_columns()
        return [name for name in candidates if name in frame.columns]

    def _make_scaler(self) -> Any:


        if str(self.config.context.scaler_kind).lower() == "robust":
            return RobustScaler()
        return StandardScaler()


    def fit(self, frame: pd.DataFrame) -> "PerModeScaler":


        from app.utils.Anomaly_Health_Monitoring.training_split import training_scope
        frame = training_scope(frame, self.config)
        if self.mode_column not in frame.columns:
            raise AnomalyHealthError(
                f"'{self.mode_column}' is missing. Per-mode normalisation "
                f"requires the context-modelling stage to run first; scaling "
                f"globally would let regime changes masquerade as faults."
            )

        self.columns = self.resolve_columns(frame)
        if not self.columns:
            raise AnomalyHealthError("No scalable columns found in the input frame.")

        self.scalers = {}
        self.mode_centroids = {}
        settings = [c for c in get_op_setting_columns() if c in frame.columns]

        for mode, group in frame.groupby(self.mode_column, sort=True):
            mode_key = int(mode)
            values = group[self.columns].astype(float)
            scaler = self._make_scaler()

            if len(group) < 2:


                self.logger.warning(
                    "Operating mode %d has only %d row(s); fitting its scaler on "
                    "the full dataset instead.",
                    mode_key,
                    len(group),
                )
                scaler.fit(frame[self.columns].astype(float))
            else:
                scaler.fit(values)

            self.scalers[mode_key] = scaler
            if settings:
                self.mode_centroids[mode_key] = (
                    group[settings].astype(float).mean().tolist()
                )

            self.logger.info(
                "Fitted %s scaler for operating mode %d on %d rows",
                self.config.context.scaler_kind,
                mode_key,
                len(group),
            )

        self.logger.info(
            "Per-mode scaling fitted: %d mode(s), %d column(s)",
            len(self.scalers),
            len(self.columns),
        )
        return self


    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:


        if not self.scalers:
            raise AnomalyHealthError("PerModeScaler.transform called before fit/load.")
        if self.mode_column not in frame.columns:
            raise AnomalyHealthError(
                f"'{self.mode_column}' is missing; cannot apply per-mode scaling."
            )

        result = frame.copy()
        missing = [c for c in self.columns if c not in result.columns]
        if missing:
            raise AnomalyHealthError(
                f"Scaler was fitted on columns absent from this frame: {missing}"
            )

        scaled_block = result[self.columns].astype(float).to_numpy(copy=True)

        for mode, index in result.groupby(self.mode_column, sort=True).groups.items():
            mode_key = int(mode)
            scaler = self.scalers.get(mode_key)
            if scaler is None:


                for row_index in index:
                    row = result.loc[[row_index]]
                    fallback = self._nearest_mode(row)
                    position = result.index.get_loc(row_index)
                    scaled_block[position] = self.scalers[fallback].transform(row[self.columns].astype(float))[0]
                continue
            positions = result.index.get_indexer(index)
            scaled_block[positions, :] = scaler.transform(
                result.loc[index, self.columns].astype(float)
            )

        result[self.columns] = np.nan_to_num(scaled_block, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _nearest_mode(self, rows: pd.DataFrame) -> int:


        settings = [c for c in get_op_setting_columns() if c in rows.columns]
        if not settings or not self.mode_centroids:
            return sorted(self.scalers.keys())[0]

        target = rows[settings].astype(float).mean().to_numpy()
        best_mode = sorted(self.scalers.keys())[0]
        best_distance = float("inf")
        for mode, centroid in self.mode_centroids.items():
            if mode not in self.scalers:
                continue
            distance = float(np.linalg.norm(np.asarray(centroid, dtype=float) - target))
            if distance < best_distance:
                best_distance = distance
                best_mode = mode
        return best_mode

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:


        return self.fit(frame).transform(frame)


    def save_scalers(self) -> Path:


        if not self.scalers:
            raise AnomalyHealthError("No fitted scalers to save.")
        self.config.ensure_directories()

        for mode, scaler in self.scalers.items():
            save_model(scaler, self.config.paths.scalers / f"mode_{mode}.pkl")

        payload = {
            "modes": sorted(int(m) for m in self.scalers.keys()),
            "columns": self.columns,
            "scaler_kind": self.config.context.scaler_kind,
            "mode_centroids": {str(k): v for k, v in self.mode_centroids.items()},
        }
        path = atomic_write_json(payload, self.config.paths.scalers / "scaler_meta.json")
        self.logger.info("Saved %d per-mode scaler(s) to %s", len(self.scalers), self.config.paths.scalers)
        return path

    def load_scalers(self) -> "PerModeScaler":


        meta = read_json(self.config.paths.scalers / "scaler_meta.json", default=None)
        if not meta:
            raise AnomalyHealthError(
                "No scaler metadata found. Run the context-modelling / scaling "
                "stage before inference."
            )
        self.columns = list(meta.get("columns", []))
        self.mode_centroids = {
            int(k): list(v) for k, v in (meta.get("mode_centroids") or {}).items()
        }
        self.scalers = {}
        for mode in meta.get("modes", []):
            self.scalers[int(mode)] = load_model(
                self.config.paths.scalers / f"mode_{int(mode)}.pkl"
            )
        self.logger.info("Loaded %d per-mode scaler(s)", len(self.scalers))
        return self


    def save(self, frame: pd.DataFrame, path: Optional[Path] = None) -> Path:


        target = path or self.config.paths.processed_path(
            self.config.paths.scaled_features_csv
        )
        verify_identity_columns(frame, context="scaled_features output")
        return atomic_write_csv(frame, target, config=self.config)

    def load_saved(self, path: Optional[Path] = None) -> pd.DataFrame:


        target = path or self.config.paths.processed_path(
            self.config.paths.scaled_features_csv
        )
        return read_csv(target, required_columns=get_identity_columns(), stage="scaler")

    def run(
        self,
        frame: Optional[pd.DataFrame] = None,
        refit: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:


        self.config.ensure_directories()
        source = frame
        if source is None:
            source = read_csv(
                self.config.paths.processed_path(self.config.paths.context_clusters_csv),
                required_columns=get_identity_columns(),
                stage="scaler",
            )

        if refit:
            from app.utils.Anomaly_Health_Monitoring.training_split import load_training_split, prepare_training_split, training_scope, record_training_scope
            if load_training_split(self.config) is None:
                prepare_training_split(source, self.config)
            fitting = training_scope(source, self.config, required=True)
            self.fit(fitting)
            record_training_scope("scaling", fitting, self.config)
            scaled = self.transform(source)
            self.save_scalers()
        else:
            self.load_scalers()
            scaled = self.transform(source)

        mode_counts = (
            scaled[self.mode_column].value_counts().sort_index().to_dict()
            if self.mode_column in scaled.columns
            else {}
        )
        report: Dict[str, Any] = {
            "rows": int(len(scaled)),
            "engines": int(scaled["unit"].nunique()),
            "modes": len(self.scalers),
            "scaled_columns": len(self.columns),
            "scaler_kind": self.config.context.scaler_kind,
            "rows_per_mode": {str(k): int(v) for k, v in mode_counts.items()},
            "refit": bool(refit),
        }
        report["output_path"] = str(self.save(scaled))
        self.logger.info("Scaling report: %s", report)
        return scaled, report


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    scaler = PerModeScaler(config)
    scaled, report = scaler.run()
    logger.info("scaled_features.csv written: %d rows", report["rows"])
    logger.info("Rows per operating mode: %s", report["rows_per_mode"])
    print(scaled.iloc[:5, :10].to_string())


if __name__ == "__main__":
    main()
