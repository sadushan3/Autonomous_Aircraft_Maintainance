"""
THE single authoritative column resolver for the C-MAPSS schema.

Every module in this component -- feature engineering, scaling, the digital
twin, anomaly detection, reasoning, explainability -- obtains its column names
from this file and nowhere else.

Two rules are enforced here and must never be broken:

1. No module hardcodes its own sensor list. The C-MAPSS layout is fixed and is
   written down exactly once, below.
2. No module discovers columns by pattern sniffing. There is no X_s/X_v/W
   discovery and no get_xs_columns-style regex matching. Columns are declared,
   not guessed.

The raw C-MAPSS .txt files have 26 whitespace-delimited columns and no header
row. Their meaning, in order, is:

    unit, cycle,
    op_setting_1, op_setting_2, op_setting_3,
    sensor_1 ... sensor_21

RUL is a derived label, not a raw column. There is no 'split' column in
C-MAPSS and nothing in this component may require one.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# The canonical C-MAPSS schema -- declared exactly once, here.
# --------------------------------------------------------------------------- #

#: Identity key columns present on every carry-forward CSV.
IDENTITY_COLUMNS: List[str] = ["unit", "cycle"]

#: The three operational setting channels.
OP_SETTING_COLUMNS: List[str] = ["op_setting_1", "op_setting_2", "op_setting_3"]

#: The 21 measured sensor channels.
SENSOR_COLUMNS: List[str] = [f"sensor_{i}" for i in range(1, 22)]

#: The full 26-column raw layout, in file order.
RAW_COLUMNS: List[str] = IDENTITY_COLUMNS + OP_SETTING_COLUMNS + SENSOR_COLUMNS

#: Derived supervised label.
RUL_COLUMN: str = "RUL"

#: Metadata columns this component attaches during ingestion.
SUBSET_COLUMN: str = "subset_id"
SPLIT_COLUMN: str = "split"

#: Operating-mode label produced by the context-modelling layer.
OPERATING_MODE_COLUMN: str = "operating_mode"

#: Prefixes used for derived column families. Declared here so the
#: residual/expected/feature namespaces are also owned by the resolver.
EXPECTED_PREFIX: str = "expected_"
RESIDUAL_PREFIX: str = "residual_"
TWIN_SPREAD_PREFIX: str = "twin_spread_"
RESIDUAL_Z_PREFIX: str = "residual_z_"
FEATURE_PREFIX: str = "feat_"
RESIDUAL_FEATURE_PREFIX: str = "resfeat_"

#: Known characteristics of each subset. The loader uses this to sanity-check
#: what it read and to decide how many operating modes to expect.
SUBSET_CHARACTERISTICS: Dict[str, Dict[str, int]] = {
    "FD001": {"train_engines": 100, "operating_conditions": 1, "fault_modes": 1},
    "FD002": {"train_engines": 260, "operating_conditions": 6, "fault_modes": 1},
    "FD003": {"train_engines": 100, "operating_conditions": 1, "fault_modes": 2},
    "FD004": {"train_engines": 249, "operating_conditions": 6, "fault_modes": 2},
}

#: Valid subset identifiers.
VALID_SUBSETS: List[str] = list(SUBSET_CHARACTERISTICS.keys())


class SchemaValidationError(ValueError):
    """Raised when a DataFrame does not carry the required C-MAPSS columns."""


# --------------------------------------------------------------------------- #
# Primary resolver accessors
# --------------------------------------------------------------------------- #

def get_identity_columns() -> List[str]:
    """Return the identity key columns.

    These must be present on every carry-forward CSV and every output so the
    downstream Maintenance Supervisor can align on (engine_id, cycle).

    Returns:
        A fresh list: ['unit', 'cycle'].
    """
    return list(IDENTITY_COLUMNS)


def get_op_setting_columns() -> List[str]:
    """Return the three operational setting columns.

    Returns:
        ['op_setting_1', 'op_setting_2', 'op_setting_3'].
    """
    return list(OP_SETTING_COLUMNS)


def get_sensor_columns() -> List[str]:
    """Return the 21 measured sensor columns.

    Returns:
        ['sensor_1', ..., 'sensor_21'].
    """
    return list(SENSOR_COLUMNS)


def get_feature_columns() -> List[str]:
    """Return the model-input feature columns (op settings + sensors).

    This is the 24-column block constituting the observable state of an engine
    at a single cycle, excluding identity keys and derived labels.

    Returns:
        Operating settings followed by sensors.
    """
    return get_op_setting_columns() + get_sensor_columns()


def get_raw_columns() -> List[str]:
    """Return the full 26-column raw C-MAPSS layout in file order.

    Returns:
        The exact column names to assign when reading a headerless .txt file.
    """
    return list(RAW_COLUMNS)


def get_required_columns() -> List[str]:
    """Return the columns this component consumes and therefore requires.

    Any column outside this set found on an incoming CSV -- for example
    rul_prediction or failure_risk_score produced by a sibling component -- is
    dropped after validation and never read.

    Returns:
        The 26 required column names.
    """
    return get_raw_columns()


def get_rul_column() -> str:
    """Return the derived RUL label column name.

    Returns:
        The string 'RUL'.
    """
    return RUL_COLUMN


def get_operating_mode_column() -> str:
    """Return the operating-mode label column name.

    Returns:
        The string 'operating_mode'.
    """
    return OPERATING_MODE_COLUMN


# --------------------------------------------------------------------------- #
# Derived column-family resolvers
# --------------------------------------------------------------------------- #

def get_expected_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:
    """Return digital-twin expected-value column names.

    Args:
        sensors: Optional restricted sensor list (e.g. after constant-sensor
            dropping). Defaults to all 21 sensors.

    Returns:
        Names such as 'expected_sensor_1', aligned 1:1 with sensors.
    """
    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{EXPECTED_PREFIX}{name}" for name in active]


def get_residual_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:
    """Return residual column names (actual minus expected, per sensor).

    Args:
        sensors: Optional restricted sensor list. Defaults to all 21 sensors.

    Returns:
        Names such as 'residual_sensor_1', aligned 1:1 with sensors.
    """
    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{RESIDUAL_PREFIX}{name}" for name in active]


def get_residual_z_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:
    """Return robust z-scored residual column names.

    Args:
        sensors: Optional restricted sensor list.

    Returns:
        Names such as 'residual_z_sensor_1'.
    """
    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{RESIDUAL_Z_PREFIX}{name}" for name in active]


def get_twin_spread_columns(sensors: Optional[Sequence[str]] = None) -> List[str]:
    """Return per-sensor twin-disagreement column names.

    The spread is the standard deviation of the individual twin predictions for
    a sensor, and is the raw material of the uncertainty layer.

    Args:
        sensors: Optional restricted sensor list.

    Returns:
        Names such as 'twin_spread_sensor_1'.
    """
    active = list(sensors) if sensors is not None else get_sensor_columns()
    return [f"{TWIN_SPREAD_PREFIX}{name}" for name in active]


def present_residual_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the residual columns this frame actually carries.

    Modules need "the residual columns that are here" -- which varies once
    constant sensors have been dropped. The tempting shortcut is to scan the
    frame for names starting with ``residual_sensor_``, but that makes every
    caller a second, unofficial definition of the schema: rename the prefix
    and a dozen files break silently instead of one. These helpers keep the
    canonical order and the prefix ownership in this module and merely filter
    by presence.

    Args:
        frame: Frame to inspect.
        sensors: Optional restricted sensor list; defaults to all 21.

    Returns:
        Present residual columns, in canonical order.
    """
    return [c for c in get_residual_columns(sensors) if c in frame.columns]


def present_residual_z_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the robust z-scored residual columns this frame actually carries.

    Args:
        frame: Frame to inspect.
        sensors: Optional restricted sensor list; defaults to all 21.

    Returns:
        Present residual z-score columns, in canonical order.
    """
    return [c for c in get_residual_z_columns(sensors) if c in frame.columns]


def present_expected_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the digital-twin expected-value columns this frame carries.

    Args:
        frame: Frame to inspect.
        sensors: Optional restricted sensor list; defaults to all 21.

    Returns:
        Present expected-value columns, in canonical order.
    """
    return [c for c in get_expected_columns(sensors) if c in frame.columns]


def present_twin_spread_columns(
    frame: pd.DataFrame, sensors: Optional[Sequence[str]] = None
) -> List[str]:
    """Return the twin-disagreement columns this frame carries.

    Args:
        frame: Frame to inspect.
        sensors: Optional restricted sensor list; defaults to all 21.

    Returns:
        Present twin-spread columns, in canonical order.
    """
    return [c for c in get_twin_spread_columns(sensors) if c in frame.columns]


def sensors_from_residual_frame(frame: pd.DataFrame) -> List[str]:
    """Return the sensors a residuals frame covers, in canonical order.

    Accepts a frame carrying either the raw residual columns or only their
    z-scored counterparts -- both families track the same sensor set, and
    different consumers hold one or the other.

    Args:
        frame: Residuals frame.

    Returns:
        Sensor column names covered by the frame, in canonical order.
    """
    columns = set(frame.columns)
    covered = [
        s
        for s in get_sensor_columns()
        if f"{RESIDUAL_PREFIX}{s}" in columns or f"{RESIDUAL_Z_PREFIX}{s}" in columns
    ]
    return covered


def sensor_from_residual(residual_column: str) -> str:
    """Map a derived column name back to its originating sensor.

    Args:
        residual_column: A name such as 'residual_sensor_4' or
            'residual_z_sensor_4'.

    Returns:
        The sensor column name, e.g. 'sensor_4'. Unrecognised names are
        returned unchanged.
    """
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
    """Return the 1-based channel number of a sensor column.

    Args:
        sensor_column: A name such as 'sensor_14'.

    Returns:
        The integer channel number, or -1 when the name is not a recognised
        sensor column.
    """
    if sensor_column in SENSOR_COLUMNS:
        return int(sensor_column.rsplit("_", 1)[1])
    return -1


# --------------------------------------------------------------------------- #
# Active-sensor resolution (constant-sensor dropping without misalignment)
# --------------------------------------------------------------------------- #

def resolve_active_sensors(dropped: Optional[Iterable[str]] = None) -> List[str]:
    """Return the sensor columns still in play after constant-sensor dropping.

    Column alignment is preserved by always deriving the active list from the
    canonical order, so a model trained with a given active list can be matched
    against the same list at inference time.

    Args:
        dropped: Sensor names detected as constant or near-constant.

    Returns:
        The canonical sensor list minus dropped, in canonical order. Falls back
        to the full set if the drop list would remove everything.
    """
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
    """Identify sensors that are flat (or effectively flat) in the data.

    Several C-MAPSS sensors are constant in FD001 and carry no information.
    Detecting them lets the pipeline drop them via config without breaking
    column alignment for models that expect a specific active set.

    Args:
        frame: DataFrame containing sensor columns.
        std_epsilon: Standard-deviation threshold below which a sensor counts
            as constant.
        sensors: Optional restricted sensor list to test.

    Returns:
        The names of constant sensors, in canonical order.
    """
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


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #

def verify_schema(
    frame: pd.DataFrame,
    required: Optional[Sequence[str]] = None,
    context: str = "dataframe",
) -> None:
    """Assert that a frame carries every required column.

    Args:
        frame: The DataFrame to validate.
        required: Columns that must be present. Defaults to the 26 required
            C-MAPSS columns.
        context: Human-readable label used in the error message.

    Raises:
        SchemaValidationError: If any required column is missing, naming
            exactly which ones.
    """
    needed = list(required) if required is not None else get_required_columns()
    missing = [name for name in needed if name not in frame.columns]
    if missing:
        raise SchemaValidationError(
            f"{context} is missing {len(missing)} required column(s): {missing}. "
            f"Present columns: {list(frame.columns)[:40]}"
        )


def verify_identity_columns(frame: pd.DataFrame, context: str = "dataframe") -> None:
    """Assert the unit/cycle keys survived a transformation.

    Every stage calls this before writing, guaranteeing the integration
    contract that all carry-forward CSVs remain joinable.

    Args:
        frame: DataFrame about to be written.
        context: Label used in the error message.

    Raises:
        SchemaValidationError: If unit or cycle is absent.
    """
    verify_schema(frame, get_identity_columns(), context=context)


def select_required_columns(
    frame: pd.DataFrame,
    extra_keep: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Validate, then narrow a frame to the columns this component consumes.

    Extra columns produced by sibling components (rul_prediction,
    failure_risk_score, and any other downstream field) are dropped after
    validation succeeds. Their mere presence is never an error.

    Args:
        frame: Incoming DataFrame, possibly carrying foreign columns.
        extra_keep: Additional columns to retain when present, e.g. RUL or
            subset_id.

    Returns:
        A copy narrowed to the required columns plus any requested extras.

    Raises:
        SchemaValidationError: If a required column is missing.
    """
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
    """Return a machine-readable description of the canonical schema.

    Used by the /upload endpoint to tell callers exactly what is expected.

    Returns:
        A mapping of schema facts suitable for JSON serialisation.
    """
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
    """Return how many operating regimes the given subsets should exhibit.

    FD002 and FD004 have six regimes; FD001 and FD003 have one. Pooling a
    six-regime subset with a one-regime subset still yields six.

    Args:
        subsets: Subset identifiers such as ['FD001', 'FD002'].

    Returns:
        The maximum operating-condition count across the requested subsets,
        defaulting to 1 when nothing is recognised.
    """
    counts = [
        SUBSET_CHARACTERISTICS[name]["operating_conditions"]
        for name in subsets
        if name in SUBSET_CHARACTERISTICS
    ]
    return max(counts) if counts else 1


def main() -> None:
    """Print the canonical schema -- a quick way to verify the resolver."""
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
