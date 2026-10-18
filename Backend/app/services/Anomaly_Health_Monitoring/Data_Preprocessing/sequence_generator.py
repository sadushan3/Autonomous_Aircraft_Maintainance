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
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    get_logger,
)


class SequenceGenerator:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.sequence_length: int = int(self.config.features.sequence_length)
        self.stride: int = max(int(self.config.features.sequence_stride), 1)


    def build_engine_windows(
        self,
        values: np.ndarray,
        sequence_length: Optional[int] = None,
    ) -> np.ndarray:


        length = int(sequence_length or self.sequence_length)
        n_cycles, n_features = values.shape

        if n_cycles == 0:
            return np.empty((0, length, n_features), dtype=float)

        pad_count = max(length - 1, 0)
        if pad_count:
            padding = np.repeat(values[:1], pad_count, axis=0)
            padded = np.vstack([padding, values])
        else:
            padded = values


        windows = np.lib.stride_tricks.sliding_window_view(
            padded, window_shape=length, axis=0
        )


        windows = np.transpose(windows, (0, 2, 1))
        return np.ascontiguousarray(windows[:n_cycles], dtype=float)

    def build(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        sequence_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, pd.DataFrame]:


        columns = list(feature_columns)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(
                f"Sequence generation requested absent column(s): {missing}"
            )
        if not columns:
            raise AnomalyHealthError("Sequence generation requires at least one feature.")

        keys = get_identity_columns()
        ordered = frame.sort_values(keys).reset_index(drop=True)
        length = int(sequence_length or self.sequence_length)

        tensors: List[np.ndarray] = []
        key_frames: List[pd.DataFrame] = []

        for unit, group in ordered.groupby("unit", sort=True):
            values = group[columns].astype(float).to_numpy()
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            tensors.append(self.build_engine_windows(values, length))
            key_frames.append(group[keys].reset_index(drop=True))

        if not tensors:
            return (
                np.empty((0, length, len(columns)), dtype=float),
                pd.DataFrame(columns=keys),
            )

        tensor = np.concatenate(tensors, axis=0)
        key_frame = pd.concat(key_frames, ignore_index=True)

        self.logger.info(
            "Built sequences: %s over %d engine(s), %d feature(s), window=%d",
            tensor.shape,
            ordered["unit"].nunique(),
            len(columns),
            length,
        )
        return tensor, key_frame

    def build_training_windows(
        self,
        frame: pd.DataFrame,
        feature_columns: Sequence[str],
        sequence_length: Optional[int] = None,
    ) -> np.ndarray:


        columns = list(feature_columns)
        missing = [c for c in columns if c not in frame.columns]
        if missing:
            raise AnomalyHealthError(
                f"Sequence generation requested absent column(s): {missing}"
            )

        length = int(sequence_length or self.sequence_length)
        ordered = frame.sort_values(get_identity_columns()).reset_index(drop=True)

        windows: List[np.ndarray] = []
        for unit, group in ordered.groupby("unit", sort=True):
            values = group[columns].astype(float).to_numpy()
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            if values.shape[0] < length:
                continue
            view = np.lib.stride_tricks.sliding_window_view(
                values, window_shape=length, axis=0
            )
            view = np.transpose(view, (0, 2, 1))
            windows.append(view[:: self.stride])

        if not windows:
            self.logger.warning(
                "No engine had at least %d cycles; falling back to padded windows "
                "so the auto-encoder still has something to learn from.",
                length,
            )
            tensor, _ = self.build(ordered, columns, length)
            return tensor

        tensor = np.ascontiguousarray(np.concatenate(windows, axis=0), dtype=float)
        self.logger.info(
            "Built %d training window(s) of shape (%d, %d) with stride %d",
            tensor.shape[0],
            length,
            len(columns),
            self.stride,
        )
        return tensor


    def save(self, tensor: np.ndarray, path: str | Path) -> Path:


        import os

        final_path = Path(path)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.with_name(final_path.name + ".tmp.npy")
        try:
            np.save(tmp_path, tensor)
            os.replace(tmp_path, final_path)
        except Exception as exc:
            tmp_path.unlink(missing_ok=True)
            raise AnomalyHealthError(f"Failed to save sequences to {final_path}: {exc}") from exc
        return final_path

    def load(self, path: str | Path) -> np.ndarray:


        file_path = Path(path)
        if not file_path.is_file():
            raise AnomalyHealthError(f"Sequence file '{file_path}' not found.")
        try:
            return np.load(file_path)
        except Exception as exc:
            raise AnomalyHealthError(f"Could not load sequences from {file_path}: {exc}") from exc

    def describe(self, tensor: np.ndarray) -> Dict[str, Any]:


        return {
            "n_windows": int(tensor.shape[0]) if tensor.ndim == 3 else 0,
            "sequence_length": int(tensor.shape[1]) if tensor.ndim == 3 else 0,
            "n_features": int(tensor.shape[2]) if tensor.ndim == 3 else 0,
            "bytes": int(tensor.nbytes),
        }


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    generator = SequenceGenerator(config)

    demo = pd.DataFrame(
        {
            "unit": [1] * 12 + [2] * 8,
            "cycle": list(range(1, 13)) + list(range(1, 9)),
            "a": np.linspace(0.0, 1.0, 20),
            "b": np.linspace(1.0, 0.0, 20),
        }
    )
    tensor, keys = generator.build(demo, ["a", "b"], sequence_length=5)
    logger.info("inference tensor: %s", generator.describe(tensor))
    logger.info("keys aligned row-for-row: %d rows", len(keys))

    training = generator.build_training_windows(demo, ["a", "b"], sequence_length=5)
    logger.info("training tensor: %s", generator.describe(training))


    boundary = tensor[11]
    logger.info("engine-1 final window first/last rows: %s -> %s", boundary[0], boundary[-1])


if __name__ == "__main__":
    main()
