"""
Build a mixed-condition C-MAPSS file for exercising the whole pipeline.

    python scripts/make_test_dataset.py --subset FD001
    python scripts/make_test_dataset.py --subset FD002

Writes ``data/synthetic/mixed_conditions_<SUBSET>.txt`` (the file to upload)
plus a ground-truth key beside it.

Why it is built from real data
------------------------------
Every row is a real ``train_<SUBSET>`` row. The four agents are fitted on
C-MAPSS sensor physics with persisted per-subset scalers, so a file of invented
numbers scores as noise and tests nothing: the twin's residuals, the health
index and the RUL regression all assume the real sensor manifold. What is
synthetic is the *composition* -- which engine appears, where its trajectory is
cut, and the faults injected into four of them.

That gives the one thing the real files cannot: a known label per engine. The
train split runs every engine to failure, so truncating a trajectory at cycle
``c`` of a total ``L`` leaves an engine whose true remaining life is exactly
``L - c``. Cutting different engines at different depths produces a fleet that
is simultaneously healthy, degrading, near-failure and faulted, and the key
file says which is which.

Naming matters
--------------
The subset is read from the filename, and the loader silently falls back to
FD001 models when it cannot find one -- so the name must carry the subset.
It must NOT contain ``test``: that triggers a join against a companion
``RUL_<SUBSET>.txt``, and the ones in ``data/raw`` describe a different set of
engines. A count mismatch is caught and warned about, but the safe move is to
keep the word out of the name entirely. RUL is then derived by the
run-to-failure convention, which affects only ``health_index_target`` and
RUL-correlated evaluation metrics -- never scoring. Use the key file shipped
alongside as the truth for this dataset.

Injection scale is per operating regime, not fleet-wide
-------------------------------------------------------
This is the one thing that cannot be shared blindly between subsets. FD001 flies
a single operating condition, so a sensor's fleet-wide spread *is* its
degradation spread. FD002 flies six, and the regime dominates everything:
measured on ``train_FD002`` the fleet-wide standard deviation runs from 5x the
within-regime figure (sensor_14) to 1514x it (sensor_6). Sizing an injection
against the fleet-wide number would push sensor_6 roughly 6000 within-regime
sigma off its own operating point -- not a fault, a physically impossible
reading that any detector trips on trivially.

So faults are scaled by the standard deviation of the sensor *inside the regime
that row is flying*, which is the scale the detectors actually judge against.

The injected faults
-------------------
Four engines have 100+ cycles of genuine life left but carry a deliberate
sensor fault in their final stretch, after ~100 clean cycles of baseline. They
are the sharpest test in the file: an agent that only tracks remaining life
will call them healthy, and it will be right to. Only a detector genuinely
modelling sensor behaviour flags them. Each signature differs so root-cause
attribution has something to separate:

* ``hpt_drift``        progressive ramp on sensors 3/4/11 -- turbine efficiency loss
* ``sensor_bias``      step offset on sensors 2/17 -- a miscalibrated probe
* ``transient_spikes`` isolated excursions on sensors 9/14 -- core-speed transients
* ``instability``      variance inflation on sensors 7/12/21, mean unchanged

Sensors that are constant (or quantised to a near-constant) within a regime are
never injected: preprocessing drops them, so a fault there would be silently
discarded. That set is measured per subset rather than hardcoded, because it
differs -- FD001 pins sensors 1/5/10/16/18/19, FD002 only 1/5/18/19.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = BACKEND_ROOT / "data" / "synthetic"

SENSORS = [f"sensor_{i}" for i in range(1, 22)]
OP_SETTINGS = ["op_setting_1", "op_setting_2", "op_setting_3"]
COLUMNS = ["unit", "cycle"] + OP_SETTINGS + SENSORS

#: Quantised to a handful of discrete levels rather than genuinely varying.
#: Their spread is a property of the recording, not of engine condition, so an
#: injection there manufactures an artefact instead of a fault.
QUANTISED_SENSORS = {"sensor_10", "sensor_16"}

#: Decimal places per column, matching the raw C-MAPSS text exactly.
PRECISION: Dict[str, int] = {
    "unit": 0, "cycle": 0,
    "op_setting_1": 4, "op_setting_2": 4, "op_setting_3": 1,
    "sensor_15": 4, "sensor_17": 0, "sensor_18": 0, "sensor_21": 4,
}
DEFAULT_PRECISION = 2

#: (category, engines, min RUL at final cycle, max RUL, min length as a
#: multiple of the subset's RUL window).
#:
#: The floors are relative to the window because FD002 needs 70 cycles to form
#: one where FD001 needs 50. An engine shorter than its window is skipped by the
#: batch entirely and never reaches the decision table.
#:
#: ``anomaly_injected`` carries the largest floor on purpose. Its fault occupies
#: the final 40-50 cycles, and the detectors judge a window against that
#: engine's own earlier behaviour, so a short trajectory would leave the fault
#: covering most of the life it is meant to stand out from.
PLAN: List[Tuple[str, int, int, int, float]] = [
    ("healthy", 8, 110, 155, 1.6),
    ("degrading", 7, 60, 95, 1.8),
    ("warning", 6, 30, 55, 1.8),
    ("critical", 5, 5, 20, 1.8),
    ("anomaly_injected", 4, 100, 125, 2.7),
]

FAULTS = [
    ("hpt_drift", ["sensor_3", "sensor_4", "sensor_11"], 4.0, 45),
    ("sensor_bias", ["sensor_2", "sensor_17"], 3.5, 40),
    ("transient_spikes", ["sensor_9", "sensor_14"], 5.0, 50),
    ("instability", ["sensor_7", "sensor_12", "sensor_21"], 6.0, 40),
]

#: Must match Settings.RUL_WINDOW_SIZES, or the emitted engines can fall under
#: the window and be skipped by the batch.
RUL_WINDOWS = {"FD001": 50, "FD002": 70, "FD003": 50, "FD004": 50}


def regime_labels(frame: pd.DataFrame) -> pd.Series:
    """Label each row with its operating regime.

    C-MAPSS defines its conditions by the operating settings themselves, so
    rounding them recovers the regimes exactly: one for FD001, six for FD002.
    """
    rounded = frame[OP_SETTINGS].round({"op_setting_1": 0, "op_setting_2": 2, "op_setting_3": 0})
    # Adding zero collapses IEEE negative zero onto positive zero. Without it
    # the label is built from the *string* form, where "-0.0" and "0.0" are
    # different keys: FD001's operating settings straddle zero, so its single
    # physical condition was splitting into four spurious regimes and the sigma
    # each injection is scaled by was estimated on a quarter of the rows.
    rounded = rounded + 0.0
    return rounded.astype(str).agg("|".join, axis=1)


def injectable_sensors(frame: pd.DataFrame, regimes: pd.Series) -> Tuple[List[str], pd.DataFrame]:
    """Return the sensors worth injecting into, plus the per-regime sigma table.

    A sensor qualifies when it actually varies inside a regime and is not one of
    the quantised channels. The sigma table is what injections are scaled by --
    see the module docstring on why fleet-wide sigma is wrong here.
    """
    sigma = frame.groupby(regimes)[SENSORS].std().fillna(0.0)
    usable = [
        sensor for sensor in SENSORS
        if sensor not in QUANTISED_SENSORS and float(sigma[sensor].max()) > 1e-6
    ]
    return usable, sigma


def inject(
    frame: pd.DataFrame, kind: str, sensors: List[str], strength: float, span: int,
    sigma: pd.DataFrame, regimes: pd.Series, rng: np.random.Generator,
) -> pd.DataFrame:
    """Apply one fault signature to the final ``span`` cycles of one engine.

    Every row is scaled by the sigma of the regime it is actually flying, so the
    same nominal strength means the same thing in every operating condition.
    """
    frame = frame.copy()
    start = max(0, len(frame) - span)
    idx = frame.index[start:]
    steps = len(idx)
    if steps == 0:
        return frame

    for sensor in sensors:
        scale = regimes.loc[idx].map(sigma[sensor]).to_numpy(dtype=float)
        if not np.isfinite(scale).all() or scale.max() <= 0:
            continue
        values = frame.loc[idx, sensor].to_numpy(dtype=float)

        if kind == "hpt_drift":
            # Ramp in, so onset is gradual -- degradation, not a jump.
            delta = np.linspace(0.0, strength, steps) * scale
        elif kind == "sensor_bias":
            delta = strength * scale
        elif kind == "transient_spikes":
            delta = np.zeros(steps)
            picks = rng.choice(steps, size=max(3, steps // 6), replace=False)
            delta[picks] = rng.choice([-1.0, 1.0], size=picks.size) * strength * scale[picks]
        elif kind == "instability":
            # Mean-preserving: inflate the spread around the local level only.
            noise = rng.normal(0.0, 1.0, steps) * scale * strength / 3.0
            delta = noise - noise.mean()
        else:
            continue

        frame.loc[idx, sensor] = values + delta

    return frame


def render(frame: pd.DataFrame) -> str:
    """Serialise to raw C-MAPSS text: space separated, no header, per-column dp."""
    lines: List[str] = []
    for row in frame[COLUMNS].itertuples(index=False):
        parts = []
        for column, value in zip(COLUMNS, row):
            places = PRECISION.get(column, DEFAULT_PRECISION)
            parts.append(str(int(round(float(value)))) if places == 0 else f"{float(value):.{places}f}")
        lines.append(" ".join(parts))
    return "\n".join(lines) + "\n"


def build(subset: str, seed: int) -> int:
    source_path = BACKEND_ROOT / "data" / "raw" / f"train_{subset}.txt"
    if not source_path.exists():
        print(f"Source not found: {source_path}")
        return 2

    window = RUL_WINDOWS.get(subset, 50)
    rng = np.random.default_rng(seed)

    source = pd.read_csv(source_path, sep=r"\s+", header=None, names=COLUMNS)
    # Several sensors parse as int64; injecting a fractional offset into an int
    # column silently truncates it, so widen up front. The writer restores each
    # column's original decimal places.
    source[SENSORS] = source[SENSORS].astype("float64")
    source["_regime"] = regime_labels(source)

    usable, sigma = injectable_sensors(source, source["_regime"])
    skipped = [s for s in SENSORS if s not in usable]
    print(f"[{subset}] window {window} cycles | {source['_regime'].nunique()} operating regime(s)")
    print(f"[{subset}] not injectable (constant or quantised in-regime): {', '.join(skipped) or 'none'}")

    lengths = source.groupby("unit").size().sort_values(ascending=False)

    # Hardest first: a category needing (RUL + floor) cycles competes for the
    # scarce long trajectories. Allocating in plan order let the early
    # categories consume them and starved the injected engines to the minimum
    # length. Presentation order is restored afterwards.
    order = sorted(range(len(PLAN)), key=lambda i: -(PLAN[i][3] + PLAN[i][4] * window))
    built: List[Tuple[int, pd.DataFrame, Dict[str, object]]] = []
    used: set[int] = set()
    fault_cursor = 0

    for plan_index in order:
        category, count, rul_low, rul_high, floor_multiple = PLAN[plan_index]
        min_length = int(round(floor_multiple * window))
        targets = np.linspace(rul_high, rul_low, count).round().astype(int)

        for target_rul in targets:
            needed = int(target_rul) + min_length
            candidates = [u for u, length in lengths.items() if u not in used and length >= needed]
            if not candidates:
                print(f"  [{subset}] no engine long enough for {category} at RUL {target_rul}; skipped")
                continue
            # Shortest sufficient engine, so the long ones stay available for
            # the high-RUL categories still to be filled.
            source_unit = candidates[-1]
            used.add(source_unit)

            engine = source[source["unit"] == source_unit].sort_values("cycle").reset_index(drop=True)
            total = len(engine)
            engine = engine.iloc[: total - int(target_rul)].reset_index(drop=True)

            fault_name = injected = None
            onset = None
            if category == "anomaly_injected":
                fault_name, fault_sensors, strength, span = FAULTS[fault_cursor % len(FAULTS)]
                fault_cursor += 1
                fault_sensors = [s for s in fault_sensors if s in usable]
                engine = inject(
                    engine, fault_name, fault_sensors, strength, span,
                    sigma, engine["_regime"], rng,
                )
                injected = " ".join(fault_sensors)
                onset = max(1, len(engine) - span + 1)

            engine["cycle"] = np.arange(1, len(engine) + 1)
            built.append((plan_index, engine, {
                "category": category,
                "true_rul_at_final_cycle": int(target_rul),
                "cycles": len(engine),
                "expected_health_state": {
                    "healthy": "Healthy",
                    "degrading": "Degrading",
                    "warning": "Warning",
                    "critical": "Critical",
                    "anomaly_injected": "Healthy (but anomalous)",
                }[category],
                "expect_anomaly_flag": category in ("anomaly_injected", "critical"),
                "injected_fault": fault_name or "",
                "injected_sensors": injected or "",
                "injection_onset_cycle": onset if onset is not None else "",
                "source_engine": int(source_unit),
                "source_total_cycles": int(total),
            }))

    # Restore presentation order and number units only now, so the file reads
    # healthy -> ... -> injected regardless of allocation order.
    built.sort(key=lambda item: (item[0], -int(item[2]["true_rul_at_final_cycle"])))
    records, key_rows = [], []
    for unit_number, (_, engine, meta) in enumerate(built, start=1):
        engine = engine.copy()
        engine["unit"] = unit_number
        records.append(engine)
        key_rows.append({"unit": unit_number, **meta})

    dataset = pd.concat(records, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"mixed_conditions_{subset}"
    data_path = OUT_DIR / f"{stem}.txt"
    data_path.write_text(render(dataset), encoding="utf-8")

    key = pd.DataFrame(key_rows)
    key_path = OUT_DIR / f"{stem}_ground_truth.csv"
    key.to_csv(key_path, index=False)

    per_engine = dataset.groupby("unit").size()
    points = int(sum(max(0, n - window + 1) for n in per_engine))
    print(f"[{subset}] wrote {data_path.name}: {len(dataset):,} rows | "
          f"{dataset['unit'].nunique()} engines | {per_engine.min()}-{per_engine.max()} cycles each")
    print(f"[{subset}] wrote {key_path.name} | ~{points:,} scoring points at stride 1")
    print(key.groupby("category").agg(
        engines=("unit", "count"),
        rul_min=("true_rul_at_final_cycle", "min"),
        rul_max=("true_rul_at_final_cycle", "max"),
        cycles_min=("cycles", "min"),
    ).to_string())
    print()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a mixed-condition C-MAPSS test file.")
    parser.add_argument(
        "--subset", default="FD001", choices=sorted(RUL_WINDOWS),
        help="Which C-MAPSS subset to build from (default FD001).",
    )
    parser.add_argument("--seed", type=int, default=20260826, help="RNG seed for the injected faults.")
    args = parser.parse_args()
    return build(args.subset, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
