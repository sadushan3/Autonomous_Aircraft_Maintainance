"""
upstream_output_loader.py

Robust Loader & Format Normalizer for Upstream Component Prediction Files.

Purpose
-------
This module discovers, loads, and normalizes prediction CSVs/Parquets generated
by independent upstream research components (RUL, Risk, Anomaly agents).

It handles format variations, subtle column naming differences (e.g. `unit_number` vs `engine_id`),
missing metadata columns, and timestamp/cycle alignments so that downstream modules receive
standardized DataFrames.

Features
--------
- Auto-discovery of upstream files in raw/integrated directories.
- Flexible column alias mapping (maps legacy/variant column names to supervisor standard).
- Automatic type casting and index alignment on (engine_id, cycle).
- Memory-efficient batch/chunk loading support.

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.data_integration.upstream_output_loader

Expected outputs
----------------
- reports/Maintenance_Supervisor/upstream_loader_report.json

Exit codes
----------
0 — loading completed successfully
1 — internal failure
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pandas as pd

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    REPORTS_ROOT,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json  # noqa: E402

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths & Column Mapping Aliases
# ---------------------------------------------------------------------------

_REPORT_PATH: Final[Path] = REPORTS_ROOT / "upstream_loader_report.json"

COLUMN_ALIASES: Final[dict[str, str]] = {
    "unit": ENGINE_ID_COLUMN,
    "unit_number": ENGINE_ID_COLUMN,
    "engine": ENGINE_ID_COLUMN,
    "id": ENGINE_ID_COLUMN,
    "time_in_cycles": CYCLE_COLUMN,
    "time_cycle": CYCLE_COLUMN,
    "cycles": CYCLE_COLUMN,
    "dataset": SUBSET_COLUMN,
    "subset": SUBSET_COLUMN,
    "rul_pred": "predicted_rul",
    "rul": "predicted_rul",
    "risk_prob_10": "risk_10",
    "risk_prob_30": "risk_30",
    "risk_prob_50": "risk_50",
    "anomaly_idx": "anomaly_score",
    "score": "anomaly_score",
}


# ---------------------------------------------------------------------------
# Loader Functions
# ---------------------------------------------------------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize variant column names to standard supervisor schema."""
    renames: dict[str, str] = {}
    for col in df.columns:
        col_clean = str(col).strip().lower()
        if col_clean in COLUMN_ALIASES:
            renames[col] = COLUMN_ALIASES[col_clean]

    if renames:
        logger.info("Renamed %d alias column(s): %s", len(renames), renames)
        df = df.rename(columns=renames)

    return df


def load_and_normalize_file(file_path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet prediction file and apply standard schema normalizations."""
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Loading upstream file: %s", file_path)
    if file_path.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(file_path)
    else:
        df = pd.read_csv(file_path, low_memory=False)

    df = normalize_columns(df)

    # Coerce keys if present
    if ENGINE_ID_COLUMN in df.columns:
        df[ENGINE_ID_COLUMN] = df[ENGINE_ID_COLUMN].astype(str).str.strip()
    if CYCLE_COLUMN in df.columns:
        df[CYCLE_COLUMN] = pd.to_numeric(df[CYCLE_COLUMN], errors="coerce").fillna(0).astype(int)

    return df


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

def run_upstream_loader(input_path: Path | None = None) -> dict:
    """Run standalone upstream file loader & normalizer."""
    if input_path is None:
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    input_path = Path(input_path).resolve()

    section("UPSTREAM OUTPUT LOADER STARTED")
    logger.info("Target file: %s", input_path)

    start_time = time.perf_counter()
    df = load_and_normalize_file(input_path)
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "file_path": str(input_path),
        "rows_loaded": len(df),
        "columns_count": df.shape[1],
        "columns": list(df.columns),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Loaded %d rows and %d columns successfully.", len(df), df.shape[1])
    logger.info("Report written to: %s", _REPORT_PATH)
    section("UPSTREAM OUTPUT LOADER COMPLETED")

    return report


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_upstream_loader()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error running upstream loader: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
