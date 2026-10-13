"""
shap_explainer.py

SHAP & Feature Attribution Explainer for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Computes local feature attributions for model predictions to explain which
specific sensor features drive supervisor decisions.

When a fitted model and a background dataset are supplied, this computes
genuine SHAP values via ``shap.KernelExplainer`` against the model's own
``predict_proba`` -- model-agnostic, so it works for the classifier
baselines and for PPOSupervisorModel's actor network alike, at the cost of
being an approximation (KernelExplainer, not an exact tree/linear
explainer) and too slow to run over more than a handful of rows at once.

Without a model (the caller has only a feature row, no fitted estimator),
this falls back to ranking features by raw magnitude. That fallback is not
SHAP and is labelled as such in its return value -- an earlier version of
this module returned the magnitude ranking unconditionally under the name
"SHAP" with no such label.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()

# KernelExplainer cost scales with background_size * nsamples * n_features;
# both are capped so a handful of explanations stays fast enough to run
# inline (see supervisor_explainer.py, which explains ~5 rows per report).
_MAX_BACKGROUND_ROWS: Final[int] = 30
_KERNEL_NSAMPLES: Final[int] = 100


def _magnitude_fallback(row: pd.Series, top_k: int) -> dict[str, float]:
    """Rank features by raw magnitude. Not SHAP -- a cheap, model-free proxy."""
    num_cols = pd.to_numeric(row, errors="coerce").dropna().drop(
        labels=["engine_id", "cycle", "schema_version"], errors="ignore"
    )
    if num_cols.empty:
        return {}
    sorted_series = num_cols.abs().sort_values(ascending=False).head(top_k)
    return {str(k): float(v) for k, v in sorted_series.items()}


def compute_feature_attributions(
    row: pd.Series,
    top_k: int = 5,
    model: Any | None = None,
    background: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Return the top-K feature attributions for `row`.

    If `model` (an object exposing `predict_proba` and `feature_names`) and
    `background` are supplied, this returns genuine SHAP values for the
    model's predicted class. Otherwise it falls back to a magnitude-based
    proxy ranking -- see `compute_feature_attributions_labelled` for a
    version that reports which one was used.
    """
    result = compute_feature_attributions_labelled(row, top_k, model, background)
    return result["attributions"]


def compute_feature_attributions_labelled(
    row: pd.Series,
    top_k: int = 5,
    model: Any | None = None,
    background: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Same as `compute_feature_attributions`, but also reports the method used."""

    if model is None or background is None or not getattr(model, "is_fitted", True):
        return {
            "attributions": _magnitude_fallback(row, top_k),
            "method": "magnitude_proxy",
        }

    feature_names = getattr(model, "feature_names", None)
    if not feature_names:
        return {
            "attributions": _magnitude_fallback(row, top_k),
            "method": "magnitude_proxy",
        }

    try:
        import shap

        row_df = row.reindex(feature_names).to_frame().T
        row_mat = row_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        bg_df = background.reindex(columns=feature_names, fill_value=0.0)
        bg_mat = bg_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if len(bg_mat) > _MAX_BACKGROUND_ROWS:
            bg_mat = shap.sample(bg_mat, _MAX_BACKGROUND_ROWS, random_state=42)

        predicted_class = int(np.argmax(model.predict_proba(row_mat)[0]))

        explainer = shap.KernelExplainer(model.predict_proba, bg_mat)
        raw_values = explainer.shap_values(row_mat, nsamples=_KERNEL_NSAMPLES, silent=True)

        # shap_values shape varies by version: a list of per-class arrays,
        # or a single (n_rows, n_features, n_classes) array.
        if isinstance(raw_values, list):
            class_values = np.asarray(raw_values[predicted_class])[0]
        else:
            arr = np.asarray(raw_values)
            class_values = arr[0, :, predicted_class] if arr.ndim == 3 else arr[0]

        attribution_pairs = sorted(
            zip(feature_names, class_values), key=lambda kv: abs(float(kv[1])), reverse=True,
        )[:top_k]

        return {
            "attributions": {name: round(float(val), 6) for name, val in attribution_pairs},
            "method": "shap_kernel_explainer",
            "predicted_class_index": predicted_class,
        }
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("SHAP computation failed (%s); falling back to magnitude proxy.", exc)
        return {
            "attributions": _magnitude_fallback(row, top_k),
            "method": "magnitude_proxy_after_shap_failure",
        }
