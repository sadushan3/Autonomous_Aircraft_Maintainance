from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)


IDENTITY_COLUMNS: List[str] = ["unit", "cycle"]


OP_SETTING_COLUMNS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]


SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]


RAW_COLUMNS: List[str] = IDENTITY_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS


RUL_COLUMN: str = "RUL"


SUBSET_COLUMN: str = "subset_id"
SPLIT_COLUMN: str = "split"


OPERATING_MODE_COLUMN: str = "operating_mode"


EXPECTED_PREFIX: str = "expected_"
RESIDUAL_PREFIX: str = "residual_"
TWIN_SPREAD_PREFIX: str = "twin_spread_"
RESIDUAL_Z_PREFIX: str = "residual_z_"
FEATURE_PREFIX: str = "feat_"
RESIDUAL_FEATURE_PREFIX: str = "resfeat_"


SUBSET_CHARACTERISTICS: Dict[str, Dict[str, int]] = {
    "FD001": {"train_engines": 100, "operating_conditions": 1, "fault_modes": 1},
    "FD002": {"train_engines": 260, "operating_conditions": 6, "fault_modes": 1},
    "FD003": {"train_engines": 100, "operating_conditions": 1, "fault_modes": 2},
    "FD004": {"train_engines": 249, "operating_conditions": 6, "fault_modes": 2},
}


VALID_SUBSETS: List[str] = list(SUBSET_CHARACTERISTICS.keys())


class SchemaValidationError(ValueError):
    pass


def get_identity_columns() -> List[str]:


    return list(IDENTITY_COLUMNS)


def get_op_setting_columns() -> List[str]:


    return list(OP_SETTING_COLUMNS)


def get_sensor_columns() -> List[str]:


    return list(SENSOR_COLUMNS)


def get_feature_columns() -> List[str]:


    return get_op_setting_columns() + get_sensor_columns()


def get_raw_columns() -> List[str]:


    return list(RAW_COLUMNS)


def get_required_columns() -> List[str]:


    return get_raw_columns()


def get_rul_column() -> str:


    return RUL_COLUMN


def get_operating_mode_column() -> str:


    return OPERATING_MODE_COLUMN


def get_expected_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:


    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{EXPECTED_PREFIX}{name}" for name in active]


def get_residual_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:


    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{RESIDUAL_PREFIX}{name}" for name in active]


def get_residual_z_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:


    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{RESIDUAL_Z_PREFIX}{name}" for name in active]


def get_twin_spread_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:


    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{TWIN_SPREAD_PREFIX}{name}" for name in active]


def present_residual_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:


    return [c for c in get_residual_columns(sensors) if c in frame.columns]


def present_residual_z_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:


    return [c for c in get_residual_z_columns(sensors) if c in frame.columns]


def present_expected_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:


    return [c for c in get_expected_columns(sensors) if c in frame.columns]


def present_twin_spread_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:


    return [c for c in get_twin_spread_columns(sensors) if c in frame.columns]


def sensors_from_residual_frame(frame: pd.DataFrame) -> List[str]:


    columns = set(frame.columns)
    covered = [
        s
        for s in get_sensor_columns()
        if f"{RESIDUAL_PREFIX}{s}" in columns or f"{RESIDUAL_Z_PREFIX}{s}" in columns
    ]
    return covered


def sensor_from_residual(residual_column: str) -> str:


    name = residual_column
    for prefix in (
        RESIDUAL_Z_PREFIX,
        RESIDUAL_PREFIX,
        TWIN_SPREAD_PREFIX,
        EXPECTED_PREFIX,
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def sensor_index(sensor_column: str) -> int:


    if sensor_column in SENSOR_COLUMNS:
        return int(sensor_column.rsplit("_", 1)[1])
    return -1


def resolve_active_sensors(dropped: Optional[Iterable[str]] = None) -> List[str]:


    if not dropped:
        return get_sensor_columns()
    dropped_set = set(dropped)
    unknown = dropped_set - set(SENSOR_COLUMNS)
    if unknown:
        logger.warning("Ignoring unknown sensors in drop list: %s", sorted(unknown))
    active = [name for name in SENSOR_COLUMNS if name not in dropped_set]
    if not active:
        logger.warning("All sensors were dropped; falling back to the full set")
        return get_sensor_columns()
    return active


def detect_constant_sensors(
    frame: pd.DataFrame,
    std_epsilon: float = 1e-6,
    sensors: Optional[Sequence[str]] = None,
) -> List[str]:


    candidates = list(sensors) if sensors is not None else get_sensor_columns()
    present = [name for name in candidates if name in frame.columns]
    if not present:
        return []
    stds = frame[present].std(numeric_only=True)
    constant = [
        name for name in present if float(stds.get(name, 0.0) or 0.0) <= std_epsilon
    ]
    if constant:
        logger.info("Detected %d constant sensor(s): %s", len(constant), constant)
    return constant


def verify_schema(
    frame: pd.DataFrame,
    required: Optional[Sequence[str]] = None,
    context: str = "dataframe",
) -> None:


    needed = list(required) if required is not None else get_required_columns()
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SchemaValidationError(
            f"{context} is missing {len(missing)} required column(s): {missing}. "
            f"Present columns: {list(frame.columns)[:40]}"
        )


def verify_identity_columns(frame: pd.DataFrame, context: str = "dataframe") -> None:


    verify_schema(frame, get_identity_columns(), context=context)


def select_required_columns(
    frame: pd.DataFrame,
    extra_keep: Optional[Sequence[str]] = None,
) -> pd.DataFrame:


    verify_schema(frame, get_required_columns(), context="incoming data")

    keep = get_required_columns()
    for name in extra_keep or []:
        if name in frame.columns and name not in keep:
            keep.append(name)

    ignored = [name for name in frame.columns if name not in keep]
    if ignored:
        logger.info(
            "Ignoring %d column(s) not consumed by this component: %s",
            len(ignored),
            ignored[:20],
        )
    return frame.loc[:, keep].copy()


def describe_schema() -> Dict[str, object]:


    return {
        "identity_columns": get_identity_columns(),
        "op_setting_columns": get_op_setting_columns(),
        "sensor_columns": get_sensor_columns(),
        "raw_columns": get_raw_columns(),
        "n_raw_columns": len(RAW_COLUMNS),
        "derived_label": RUL_COLUMN,
        "valid_subsets": VALID_SUBSETS,
        "subset_characteristics": SUBSET_CHARACTERISTICS,
        "notes": (
            "Raw C-MAPSS .txt files are whitespace-delimited with no header. "
            "Column names are assigned on read in the order listed above. "
            "There is no 'split' column in C-MAPSS."
        ),
    }


def expected_operating_conditions(subsets: Sequence[str]) -> int:


    counts = [
        SUBSET_CHARACTERISTICS[name]["operating_conditions"]
        for name in subsets
        if name in SUBSET_CHARACTERISTICS
    ]
    return max(counts) if counts else 1


def main() -> None:

    import json

    logging.basicConfig(level=logging.INFO)
    print(json.dumps(describe_schema(), indent=2))
    print(f"\nfeature columns ({len(get_feature_columns())}): {get_feature_columns()}")
    print(
        f"residual columns ({len(get_residual_columns())}): "
        f"{get_residual_columns()[:5]} ..."
    )


if __name__ == "__main__":
    main()
