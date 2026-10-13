"""Export and validate FD001 test data without running inference or a pipeline."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.Integration.feature_router import FeatureRouter
from app.utils.Anomaly_Health_Monitoring.model_utils import get_raw_columns


def main() -> None:
    raw = ROOT / "data" / "raw"
    destination = ROOT.parent / "upload_datasets"
    destination.mkdir(exist_ok=True)
    columns = get_raw_columns()
    frame = pd.read_csv(raw / "test_FD001.txt", sep=r"\s+", header=None)
    if frame.shape[1] != len(columns):
        raise ValueError("Unexpected source column count")
    frame.columns = columns
    truth = pd.read_csv(raw / "RUL_FD001.txt", sep=r"\s+", header=None)[0]
    units = sorted(frame.unit.unique())
    if units != list(range(1, len(truth) + 1)):
        raise ValueError("Ground-truth engine IDs do not match source")
    final_rul = dict(zip(units, truth.astype(int)))
    frame["RUL"] = (
        frame.groupby("unit").cycle.transform("max") - frame.cycle
        + frame.unit.map(final_rul)
    ).astype(int)
    frame["subset_id"] = "FD001"
    frame["split"] = "test"
    lengths = frame.groupby("unit").size()
    excluded = lengths[lengths < 50].index.tolist()
    frame = frame.loc[~frame.unit.isin(excluded)].sort_values(["unit", "cycle"])
    output = destination / "CMAPSS_FD001_test_pipeline_ready.csv"
    frame.to_csv(output, index=False)

    # Validate the actual saved upload; no models, pipeline stages or API calls.
    saved = pd.read_csv(output)
    if saved.isna().any().any():
        raise ValueError("Missing values in upload")
    if not np.isfinite(saved[columns + ["RUL"]].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite numeric values in upload")
    if saved.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle keys")
    for unit, group in saved.groupby("unit"):
        if len(group) < 50 or not np.array_equal(group.cycle, np.arange(1, len(group) + 1)):
            raise ValueError(f"Invalid sequence for engine {unit}")
        if not (group.RUL.diff().dropna() == -1).all():
            raise ValueError(f"Invalid RUL trajectory for engine {unit}")
        if group.RUL.iloc[-1] != final_rul[unit]:
            raise ValueError(f"Incorrect terminal RUL for engine {unit}")
    routed = FeatureRouter().route(saved)
    report = {
        "file": str(output),
        "source": "Local NASA C-MAPSS test_FD001.txt and RUL_FD001.txt",
        "rows": len(saved),
        "engines": int(saved.unit.nunique()),
        "columns": saved.columns.tolist(),
        "minimum_cycles": int(saved.groupby("unit").size().min()),
        "excluded_short_engine_ids": excluded,
        "selection": "All test engines with at least 50 cycles; no prediction-based selection",
        "rul_formula": "last_observed_cycle - cycle + official_terminal_RUL",
        "component_views": {name: view.summary() for name, view in routed.items()},
        "pipeline_run": False,
        "validation": "Schema, routing, finite values, unique keys, contiguous cycles, true RUL alignment",
        "limitations": "Compatibility checks do not establish accuracy or guarantee runtime success. FD001 test engines have previously been used for calibration/evaluation in this project.",
    }
    (destination / "CMAPSS_FD001_test_pipeline_ready_validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: report[k] for k in ["file", "rows", "engines", "minimum_cycles", "excluded_short_engine_ids", "pipeline_run"]}, indent=2))


if __name__ == "__main__":
    main()
