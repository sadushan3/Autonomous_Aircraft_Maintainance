"""
feature_scale.py

Shared inverse-RobustScaler utility for the Autonomous Maintenance
Supervisor Agent.

Why this exists
----------------
`feature_scaler.py` fits one `RobustScaler` (median/IQR) across ~285
numerical columns and writes it to
``artifacts/Maintenance_Supervisor/scalers/robust_scaler.joblib``, with the
exact column order it was fit on in
``artifacts/Maintenance_Supervisor/scalers/scaling_manifest.json``
(``numerical_columns_scaled``). Several places in this codebase need the
*physical* value of a column (raw cycle count, raw 0-1 probability) even
when the dataframe they were handed is the scaled form
(``processed/Maintenance_Supervisor/splits/*.csv``) -- safety_policy.py's
guardrail thresholds and rl_environment.py's reward thresholds both compare
against physical-unit constants (RUL <= 10 cycles, risk_10 >= 0.82) and are
meaningless if fed a RobustScaler-transformed value instead.

Both of those files previously reimplemented this unscaling with hand-typed
constants. safety_policy.py's RUL constants (``*101.69 + 99.32``) happened
to exactly match the fitted scaler's real center_/scale_ for
``predicted_rul``. Its risk_10 constants (``(x - 0.003) / 1.14``) did not --
the real center_/scale_ for risk_10 are ``0.0061``/``0.006`` -- so that
"unscale" mapped almost every scaled risk_10 value to >= 1.0 after clipping,
making the critical-risk override fire almost unconditionally whenever it
ran against scaled data. rl_environment.py did not unscale at all.

This module reads the actual fitted scaler instead of re-deriving or
hand-typing those constants, so there is exactly one place that can drift
out of calibration, and fixing it fixes every caller at once.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[3]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import ARTIFACT_ROOT

_SCALER_PATH: Final[Path] = ARTIFACT_ROOT / "scalers" / "robust_scaler.joblib"
_MANIFEST_PATH: Final[Path] = ARTIFACT_ROOT / "scalers" / "scaling_manifest.json"


@lru_cache(maxsize=1)
def _load_center_scale() -> dict[str, tuple[float, float]]:
    """Return {column_name: (center_, scale_)} from the fitted RobustScaler.

    Cached for the life of the process: the scaler artifact only changes
    when feature_scaler.py is re-run, which is an offline retraining step,
    not something that happens while this process is serving requests.
    Returns {} if the artifacts are missing (e.g. before the pipeline has
    ever been run) -- callers treat that as "nothing to unscale".
    """
    if not _SCALER_PATH.exists() or not _MANIFEST_PATH.exists():
        return {}
    try:
        import joblib

        scaler = joblib.load(_SCALER_PATH)
        with open(_MANIFEST_PATH, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        columns = manifest.get("numerical_columns_scaled") or []
        if not isinstance(columns, list) or len(columns) != len(scaler.center_):
            return {}
        return {
            col: (float(scaler.center_[i]), float(scaler.scale_[i]))
            for i, col in enumerate(columns)
        }
    except Exception:  # pylint: disable=broad-except
        return {}


def unscale_column(
    values: np.ndarray,
    column: str,
    *,
    raw_min: float,
    raw_max: float,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """Return `values` in physical units, inverse-transforming only if they look scaled.

    Parameters
    ----------
    values:
        The column's values as given -- may already be raw/physical (e.g.
        live exchange data, which is never scaled) or RobustScaler-scaled
        (e.g. ``processed/Maintenance_Supervisor/splits/*.csv``).
    column:
        Column name, used to look up the fitted scaler's center_/scale_.
    raw_min, raw_max:
        The column's known *hard* physical-unit bounds (e.g. 0.0/320.0 for
        predicted_rul, 0.0/1.0 for risk_10 -- a probability or a cycle count
        cannot go outside these even approximately). Data is judged
        "already physical" only if every value sits within this range plus
        a tiny float-rounding tolerance; RobustScaler output is centred
        near 0 and will almost always include values outside these hard
        bounds (RUL going negative, risk exceeding 1.0), which is exactly
        the signal used to tell the two apart. An earlier version of this
        function used a percentage-of-range tolerance instead, which was
        wide enough that small-magnitude scaled values (predicted_rul's
        scaled std is ~0.63, comfortably inside a +/-5%-of-320 band) read as
        "already physical" and were never unscaled at all.
    tolerance:
        Absolute slack applied to the raw bounds, for floating-point
        rounding only -- not a mechanism for tolerating genuinely scaled data.

    Returns
    -------
    np.ndarray of the same shape, in physical units. If no calibration is
    available for `column` (scaler artifacts missing, or this column was
    never scaled), returns `values` unchanged.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    already_physical = bool(
        np.nanmin(finite) >= raw_min - tolerance and np.nanmax(finite) <= raw_max + tolerance
    )
    if already_physical:
        return arr

    center_scale = _load_center_scale().get(column)
    if center_scale is None:
        return arr

    center, scale = center_scale
    return arr * scale + center
