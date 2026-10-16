"""Training split and engine-disjoint scope management for Anomaly Health Monitoring.

This module provides utilities to manage, load, and enforce engine-disjoint partitions
(e.g., fit, calibration, and selection) to prevent data leakage and ensure reproducible
evaluation across anomaly detection components.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Union

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError

logger = logging.getLogger(__name__)

# In-memory registry of recorded training scopes for auditability and inspection
_RECORDED_TRAINING_SCOPES: Dict[str, Dict[str, Any]] = {}


def load_training_split(config: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """Load the active training split plan if present.

    Searches environment overrides, model paths, and processed data paths.
    Returns None if no plan has been configured or persisted.

    Args:
        config: Optional component configuration.

    Returns:
        The training plan dictionary if found, otherwise None.
    """
    cfg = config or get_config()

    env_path = os.environ.get("AHM_TRAINING_SPLIT_PATH") or os.environ.get("AHM_TRAINING_PLAN_PATH")
    candidate_paths: List[Path] = []
    if env_path:
        candidate_paths.append(Path(env_path))

    candidate_paths.extend([
        cfg.paths.model_path("training_split.json"),
        cfg.paths.model_path("training_plan.json"),
        cfg.paths.processed_path("training_split.json"),
        cfg.paths.processed_path("training_plan.json"),
    ])

    for path in candidate_paths:
        if path.exists() and path.is_file():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    plan = json.load(f)
                if isinstance(plan, dict):
                    return plan
            except Exception as exc:
                logger.warning("Could not read training split plan from %s: %s", path, exc)

    return None


def training_scope(
    frame: pd.DataFrame,
    config: Optional[Config] = None,
    scopes: Union[Sequence[str], str] = ("fit",),
    required: bool = False,
) -> pd.DataFrame:
    """Filter a DataFrame to the engines or rows belonging to the requested training scope(s).

    Args:
        frame: The input DataFrame containing engine data.
        config: Optional component configuration.
        scopes: One or more scope names (e.g. ("fit",), ("calibration",), ("selection",)).
        required: If True, raises AnomalyHealthError if no active plan exists or no units match.

    Returns:
        A filtered copy of the input DataFrame.

    Raises:
        AnomalyHealthError: If required=True and the plan is missing or empty.
    """
    if frame.empty:
        return frame.copy()

    if isinstance(scopes, str):
        requested_scopes: Set[str] = {scopes}
    else:
        requested_scopes = set(scopes)

    # Check for direct scope column in frame
    for col in ("training_scope", "scope"):
        if col in frame.columns:
            filtered = frame.loc[frame[col].isin(requested_scopes)].copy()
            if required and filtered.empty and not frame.empty:
                raise AnomalyHealthError(f"No records matched requested training scope(s): {requested_scopes}")
            return filtered

    plan = load_training_split(config)
    if plan is None:
        if required:
            raise AnomalyHealthError(
                f"Training split plan is required for scopes {sorted(requested_scopes)}, but no active plan was found."
            )
        return frame.copy()

    # Identify engine ID column
    unit_col = "unit" if "unit" in frame.columns else ("engine_id" if "engine_id" in frame.columns else None)
    if not unit_col:
        if required:
            raise AnomalyHealthError("Dataframe missing 'unit' or 'engine_id' column to partition by training scope.")
        return frame.copy()

    # Resolve unit IDs assigned to the requested scopes
    target_units: Set[Any] = set()

    # Format 1: plan["scopes"] = {"fit": [1, 2, ...], "calibration": [...]}
    # Format 2: plan["splits"] = {"fit": [1, 2, ...], ...}
    for dict_key in ("scopes", "splits"):
        if dict_key in plan and isinstance(plan[dict_key], dict):
            for scope_name in requested_scopes:
                target_units.update(plan[dict_key].get(scope_name, []))

    # Format 3: plan["units"] = {"1": "fit", "2": "calibration", ...}
    # Format 4: plan["assignments"] = {1: "fit", ...}
    for dict_key in ("units", "assignments"):
        if dict_key in plan and isinstance(plan[dict_key], dict):
            for unit_id, scope_name in plan[dict_key].items():
                if scope_name in requested_scopes:
                    target_units.add(unit_id)

    # Format 5: Direct keys in plan, e.g. plan["fit"] = [1, 2, ...]
    for scope_name in requested_scopes:
        if scope_name in plan and isinstance(plan[scope_name], (list, tuple, set)):
            target_units.update(plan[scope_name])

    if not target_units:
        if required:
            raise AnomalyHealthError(
                f"Active training plan '{plan.get('plan_id', 'unknown')}' does not contain units for scopes: {sorted(requested_scopes)}"
            )
        return frame.copy()

    # Normalize integer and string forms to prevent type mismatch
    target_int = set()
    target_str = set()
    for u in target_units:
        target_str.add(str(u))
        try:
            target_int.add(int(u))
        except (ValueError, TypeError):
            pass

    mask = frame[unit_col].isin(target_int) | frame[unit_col].astype(str).isin(target_str)
    filtered = frame.loc[mask].copy()

    if required and filtered.empty and not frame.empty:
        raise AnomalyHealthError(f"No engines matched requested training scopes: {sorted(requested_scopes)}")

    return filtered


def record_training_scope(
    name: str,
    frame: pd.DataFrame,
    config: Optional[Config] = None,
    scopes: Union[Sequence[str], str] = ("fit",),
) -> Dict[str, Any]:
    """Record engine provenance and training scope usage for auditable model training.

    Args:
        name: Identifier for the recorded training event/component.
        frame: DataFrame used in the training step.
        config: Optional component configuration.
        scopes: Training scope(s) applied.

    Returns:
        Dictionary of recorded provenance metadata.
    """
    if isinstance(scopes, str):
        scope_list = [scopes]
    else:
        scope_list = list(scopes)

    unit_col = "unit" if "unit" in frame.columns else ("engine_id" if "engine_id" in frame.columns else None)
    unit_ids = sorted(map(int, frame[unit_col].unique())) if unit_col and not frame.empty else []

    record = {
        "name": name,
        "scopes": scope_list,
        "row_count": len(frame),
        "engine_count": len(unit_ids),
        "unit_ids": unit_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    _RECORDED_TRAINING_SCOPES[name] = record
    logger.info(
        "Recorded training scope '%s': %d engines, %d rows (scopes: %s)",
        name,
        len(unit_ids),
        len(frame),
        scope_list,
    )
    return record


def get_recorded_training_scopes() -> Dict[str, Dict[str, Any]]:
    """Retrieve all training scopes recorded in this process."""
    return dict(_RECORDED_TRAINING_SCOPES)


def clear_recorded_training_scopes() -> None:
    """Clear recorded training scope registry (useful for test resets)."""
    _RECORDED_TRAINING_SCOPES.clear()


def save_training_split(
    plan: Dict[str, Any],
    config: Optional[Config] = None,
    filename: str = "training_split.json",
) -> Path:
    """Persist a training split plan to models directory.

    Args:
        plan: Dictionary describing the training split.
        config: Optional component configuration.
        filename: Destination JSON filename.

    Returns:
        Path to the saved file.
    """
    cfg = config or get_config()
    cfg.ensure_directories()
    out_path = cfg.paths.model_path(filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return out_path
