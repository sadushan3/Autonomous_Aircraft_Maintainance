from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError, atomic_write_json, engine_wise_split, read_json,
)

MANIFEST = "training_split.json"
PROVENANCE = "training_provenance.json"
ROLES = ("fit", "calibration", "selection", "confirmation")


def load_training_split(config):
    payload = read_json(config.paths.models / MANIFEST, default=None)
    if payload is not None:
        if payload.get("schema_version") != 1 or set(payload.get("engine_roles", {})) != set(ROLES):
            raise AnomalyHealthError("Invalid shared training partition.")
        sets = [set(payload["engine_roles"][role]) for role in ROLES]
        if any(not values for values in sets) or sum(map(len, sets)) != len(set.union(*sets)):
            raise AnomalyHealthError("Training partitions must contain separate engines.")
        fields = ("schema_version", "engine_roles", "source_identities", "age_reference_cycles", "commissioning_cycles")
        if not all(key in payload for key in fields):
            raise AnomalyHealthError("The shared training partition is missing its provenance.")
        canonical = {key: payload[key] for key in fields}
        expected = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        if payload.get("plan_id") != expected:
            raise AnomalyHealthError("The shared training partition changed without a new bundle identity.")
    return payload


def prepare_training_split(frame: pd.DataFrame, config):

    if "split" not in frame:
        raise AnomalyHealthError("Tag the training population with split='train' before preparing the engine split.")
    source = frame.loc[frame["split"].eq("train")]
    if source.unit.nunique() < 8:
        raise AnomalyHealthError("Leakage-controlled training needs at least eight engines.")
    if "subset_id" in source and source.groupby("unit").subset_id.nunique().gt(1).any():
        raise AnomalyHealthError("Different engine populations share a unit ID; remap identities before fitting.")
    dev, confirmation = engine_wise_split(source,
        validation_fraction=config.data.validation_engine_fraction, random_state=config.data.random_state)
    units = dev.groupby("unit", sort=True).size().index.to_numpy()
    strata = dev.groupby("unit").subset_id.first().reindex(units).to_numpy() if "subset_id" in dev else None

    if strata is not None and (pd.Series(strata).value_counts().min() < 2 or int(np.ceil(len(units) * .3)) < len(set(strata))):
        strata = None
    fitting, held = train_test_split(units, test_size=.3, stratify=strata, random_state=20261015)
    held_strata = dev.groupby("unit").subset_id.first().reindex(held).to_numpy() if strata is not None else None
    if held_strata is not None and (pd.Series(held_strata).value_counts().min() < 2 or len(held) // 2 < len(set(held_strata))):
        held_strata = None
    calibration, selection = train_test_split(held, test_size=.5, stratify=held_strata, random_state=20261016)
    roles = {"fit": sorted(map(int, fitting)), "calibration": sorted(map(int, calibration)),
             "selection": sorted(map(int, selection)), "confirmation": sorted(map(int, confirmation.unit.unique()))}
    identities = {str(int(unit)): str(block.subset_id.iloc[0]) if "subset_id" in block else None
                  for unit, block in source.groupby("unit", sort=True)}
    contents = {"schema_version": 1, "engine_roles": roles, "source_identities": identities,
                "age_reference_cycles": float(config.twin.age_reference_cycles),
                "commissioning_cycles": int(config.twin.commissioning_cycles)}
    contents["plan_id"] = hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()
    contents["evaluation_status"] = "Retrospective engine-disjoint evaluation; this fleet and official benchmark were previously inspected."
    previous = load_training_split(config)
    if previous and previous["plan_id"] != contents["plan_id"]:
        raise AnomalyHealthError("Training fleet/policy differs from the saved split; train a new isolated bundle.")
    atomic_write_json(contents, config.paths.models / MANIFEST)
    return contents


def training_scope(frame: pd.DataFrame, config, roles: Iterable[str] = ("fit",), *, required=False):

    roles = tuple(roles)
    source = frame.loc[frame["split"].eq("train")].copy() if "split" in frame else frame.copy()
    plan = load_training_split(config)
    if plan is None:
        if required:
            raise AnomalyHealthError("Prepare the shared engine split before fitting this stage.")
        if source.empty:
            raise AnomalyHealthError("Fitting requires training engines; official test input is forbidden.")
        return source
    if "split" not in frame:
        raise AnomalyHealthError("Shared-partition fitting requires explicit train/test population metadata.")
    selected = set().union(*(set(plan["engine_roles"][role]) for role in roles))
    result = source.loc[source.unit.isin(selected)].copy()
    if result.empty:
        raise AnomalyHealthError(f"No training engines available for roles {tuple(roles)}.")
    if "subset_id" in result:
        expected = result.unit.astype(int).astype(str).map(plan["source_identities"])
        if (expected.notna() & expected.ne(result.subset_id.astype(str))).any():
            raise AnomalyHealthError("Training population identities differ from the shared split.")
    return result


def record_training_scope(stage: str, frame: pd.DataFrame, config, roles=("fit",)):
    plan = load_training_split(config)
    if plan is None:
        return
    allowed = set().union(*(set(plan["engine_roles"][role]) for role in roles))
    units = set(map(int, frame.unit.unique()))
    if not units or not units <= allowed or ("split" in frame and not frame.split.eq("train").all()):
        raise AnomalyHealthError(f"{stage} attempted fitting outside its permitted training engines.")
    document = read_json(config.paths.models / PROVENANCE, default={})
    document[stage] = {"plan_id": plan["plan_id"], "roles": list(roles), "units": sorted(units), "rows": len(frame)}
    atomic_write_json(document, config.paths.models / PROVENANCE)
