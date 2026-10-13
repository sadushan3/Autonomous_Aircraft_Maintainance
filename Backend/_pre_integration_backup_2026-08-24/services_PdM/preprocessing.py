from __future__ import annotations

import logging
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from app.schemas.Predictive_Maintenance.common import EngineSensorRow
from app.utils.Predictive_Maintenance.errors import InvalidSequenceError, PredictionError

logger = logging.getLogger(__name__)


RAW_FEATURE_COLUMNS = [
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{idx}" for idx in range(1, 22)],
]


class PreprocessingService:
    """Builds model-ready tensors from raw engine time-series rows."""

    def sequence_to_tensor(
        self,
        sequence: Sequence[EngineSensorRow],
        feature_columns: List[str],
        scaler: object,
        window_size: int,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        if len(sequence) < 30:
            raise InvalidSequenceError("Sequence must contain at least 30 rows.")
        if len(sequence) < window_size:
            raise InvalidSequenceError(
                f"Sequence must contain at least {window_size} rows for this model window."
            )

        df = self._sequence_to_dataframe(sequence)
        prepared = self._align_features(df, feature_columns)

        try:
            scaled = scaler.transform(prepared)
        except Exception as exc:
            raise PredictionError(f"Failed to apply scaler: {exc}") from exc

        recent_window = scaled[-window_size:]
        tensor = np.asarray(recent_window, dtype=np.float32).reshape(
            1, window_size, len(feature_columns)
        )
        return tensor, prepared.tail(window_size).copy()

    def tabular_last_row(
        self,
        sequence: Sequence[EngineSensorRow],
        feature_columns: List[str],
        scaler: object,
    ) -> np.ndarray:
        df = self._sequence_to_dataframe(sequence)
        prepared = self._align_features(df, feature_columns)
        try:
            scaled = scaler.transform(prepared)
        except Exception as exc:
            raise PredictionError(f"Failed to apply scaler: {exc}") from exc
        return np.asarray(scaled[-1:], dtype=np.float32)

    def _sequence_to_dataframe(self, sequence: Sequence[EngineSensorRow]) -> pd.DataFrame:
        records = [row.model_dump() for row in sequence]
        df = pd.DataFrame(records)
        missing = [column for column in RAW_FEATURE_COLUMNS if column not in df.columns]
        if missing:
            raise InvalidSequenceError(
                f"Missing required engine row fields: {', '.join(missing)}."
            )

        df = df.sort_values("cycle", ascending=True).reset_index(drop=True)
        if df["cycle"].duplicated().any():
            raise InvalidSequenceError("Sequence contains duplicate cycle values.")
        return df

    def _align_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        working = df.copy()

        op_cluster_columns = [
            column for column in feature_columns if column.startswith("op_cluster_")
        ]
        for column in op_cluster_columns:
            working[column] = 0.0

        if op_cluster_columns and "op_cluster" in working.columns:
            self._apply_op_cluster_one_hot(working, op_cluster_columns)

        # Any feature the caller did not supply is filled with 0.0 and then
        # handed to the scaler, which maps that raw zero to an extreme z-score
        # rather than to "no information". A silent fill therefore turns missing
        # input into a confident wrong prediction. Filling is kept (callers rely
        # on it for the op_cluster one-hots, which are legitimately absent) but
        # it is no longer silent: unexpected fills are recorded on the frame so
        # the calling service can surface them, and logged once per call.
        invented = [
            column
            for column in feature_columns
            if column not in working.columns and column not in op_cluster_columns
        ]
        for column in feature_columns:
            if column not in working.columns:
                working[column] = 0.0

        aligned = working[feature_columns].copy()
        coerced = aligned.apply(pd.to_numeric, errors="coerce")
        non_numeric = [
            column for column in feature_columns if coerced[column].isna().any()
        ]
        aligned = coerced.fillna(0.0)

        if invented or non_numeric:
            logger.warning(
                "Feature alignment zero-filled %d absent and %d non-numeric "
                "column(s) before scaling; predictions from this call are "
                "degraded. Absent: %s. Non-numeric: %s.",
                len(invented),
                len(non_numeric),
                invented or "-",
                non_numeric or "-",
            )
        aligned.attrs["zero_filled_features"] = invented
        aligned.attrs["non_numeric_features"] = non_numeric
        return aligned.astype(float)

    @staticmethod
    def _apply_op_cluster_one_hot(
        df: pd.DataFrame, op_cluster_columns: Iterable[str]
    ) -> None:
        cluster_values = df["op_cluster"]
        for column in op_cluster_columns:
            suffix = column.replace("op_cluster_", "")
            try:
                cluster_id = int(suffix)
            except ValueError:
                continue
            df[column] = (cluster_values == cluster_id).astype(float)
