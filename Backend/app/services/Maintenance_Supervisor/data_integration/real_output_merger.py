"""
real_output_merger.py

Multi-Agent Upstream Prediction Merger & Combiner for the
Autonomous Maintenance Supervisor Agent.

Status: offline/batch path only -- NOT the live serving path
--------------------------------------------------------------
`POST /integration/run-full` (app/routers/Integration/routes.py) is the live
path: it runs the Anomaly & Health pipeline, then RUL + Failure-Risk batch
scoring (`services/Integration/rul_failure_batch_service.py`), then
`services/Integration/supervisor_bridge.py`'s `run_supervisor_bridge`
directly against the CSVs those two stages just produced from the caller's
uploaded file. It never calls this module and never falls back to synthetic
data -- a request with no real upstream data simply fails.

This module's silent fallback to SAMPLE_DATASET_PATH when no upstream paths
are given (see `run_real_output_merger` below) is retained only for
offline/manual use: generating training data, running historical backtests,
or exercising the Supervisor's fusion logic against hand-picked files. Treat
any output from that fallback branch as synthetic-proxy, never as evidence
from a live run -- see `sample_output_generator.py` for the same caveat.

Purpose
-------
This module merges predictions produced independently by the three upstream research components:
1. RUL Prediction Component (predicted_rul, bounds, degradation metrics)
2. Failure Risk Prediction Component (risk_10, risk_30, risk_50, horizon flags)
3. Anomaly & Health Monitoring Component (anomaly_score, top sensors, severity)

It performs an outer/inner relational join on the composite key (`fd_subset`, `engine_id`, `cycle`),
validates completeness across all three components, imputes any missing component predictions,
and generates an integrated dataset ready for full pre-preprocessing or inference.

Merge Strategy
--------------
- Primary Join Key: (fd_subset, engine_id, cycle)
- Handling Unmatched Cycles:
  - If a cycle exists in one component but not others, missing component features are imputed
    using domain-safe fallback rules (e.g. supervisor_cleaner defaults) rather than dropping the cycle.
- Output Dataset:
  - Writes integrated dataset to data/Maintenance_Supervisor/integrated/supervisor_integrated_input.csv

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.data_integration.real_output_merger

Expected inputs
---------------
- Upstream CSV/Parquet files or single combined prediction dataset.

Expected outputs
----------------
- data/Maintenance_Supervisor/integrated/supervisor_integrated_input.csv
- reports/Maintenance_Supervisor/real_output_merger_report.json

Exit codes
----------
0 — merge completed successfully
1 — internal failure
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
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
    DATA_ROOT,
    ENGINE_ID_COLUMN,
    SUBSET_COLUMN,
    CYCLE_COLUMN,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_csv,
)
from app.services.Maintenance_Supervisor.data_integration.upstream_output_loader import (  # noqa: E402
    load_and_normalize_file,
)
from app.services.Maintenance_Supervisor.data_integration.upstream_output_validator import (  # noqa: E402
    validate_upstream_dataframe,
)
from app.services.Maintenance_Supervisor.Data_preprocessing.supervisor_cleaner import (  # noqa: E402
    impute_missing_values,
    sanitize_probabilities_and_bounds,
)

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_INTEGRATED_ROOT: Final[Path] = DATA_ROOT / "Maintenance_Supervisor" / "integrated"
_INTEGRATED_OUTPUT_PATH: Final[Path] = _INTEGRATED_ROOT / "supervisor_integrated_input.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "real_output_merger_report.json"


# ---------------------------------------------------------------------------
# Core Merger Functions
# ---------------------------------------------------------------------------

def merge_component_dataframes(
    rul_df: pd.DataFrame | None = None,
    risk_df: pd.DataFrame | None = None,
    anomaly_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge RUL, Risk, and Anomaly DataFrames on (subset, engine_id, cycle)."""
    dfs = [d for d in [rul_df, risk_df, anomaly_df] if d is not None and not d.empty]

    if not dfs:
        raise ValueError("At least one non-empty DataFrame must be provided for merging.")

    join_keys = [ENGINE_ID_COLUMN, CYCLE_COLUMN]
    if SUBSET_COLUMN in dfs[0].columns:
        join_keys.insert(0, SUBSET_COLUMN)

    logger.info("Merging %d component DataFrame(s) on keys: %s", len(dfs), join_keys)

    merged_df = dfs[0]
    for next_df in dfs[1:]:
        # Avoid duplicate non-key columns
        overlap_cols = [c for c in next_df.columns if c in merged_df.columns and c not in join_keys]
        if overlap_cols:
            logger.info("Dropping overlapping column(s) from right DataFrame: %s", overlap_cols)
            next_df = next_df.drop(columns=overlap_cols)

        merged_df = pd.merge(merged_df, next_df, on=join_keys, how="outer")

    # Sanitize and impute missing fields resulting from outer join
    merged_df, _ = sanitize_probabilities_and_bounds(merged_df)
    merged_df, _ = impute_missing_values(merged_df)

    return merged_df


# ---------------------------------------------------------------------------
# CLI Orchestrator
# ---------------------------------------------------------------------------

def run_real_output_merger(
    rul_path: Path | None = None,
    risk_path: Path | None = None,
    anomaly_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run real upstream output merger pipeline."""
    if output_path is None:
        output_path = _INTEGRATED_OUTPUT_PATH

    output_path = Path(output_path).resolve()

    section("REAL OUTPUT MERGER STARTED")
    start_time = time.perf_counter()

    # Load paths or fallback to sample dataset
    synthetic_fallback_used = rul_path is None and risk_path is None and anomaly_path is None
    if synthetic_fallback_used:
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        logger.warning(
            "No upstream component paths provided -- falling back to the "
            "SYNTHETIC SAMPLE dataset (%s). This is the offline/manual path "
            "only; the live API (/integration/run-full) never reaches this "
            "branch. Do not report this run's output as evidence from real "
            "upstream models.",
            SAMPLE_DATASET_PATH,
        )
        merged_df = load_and_normalize_file(SAMPLE_DATASET_PATH)
    else:
        rul_df = load_and_normalize_file(rul_path) if rul_path and rul_path.exists() else None
        risk_df = load_and_normalize_file(risk_path) if risk_path and risk_path.exists() else None
        anomaly_df = load_and_normalize_file(anomaly_path) if anomaly_path and anomaly_path.exists() else None

        merged_df = merge_component_dataframes(rul_df, risk_df, anomaly_df)

    # Validate merged result
    val_res = validate_upstream_dataframe(merged_df, component="combined")
    logger.info("Validation check on merged output: valid=%s (%d errors, %d warnings)", val_res.is_valid, val_res.error_count, val_res.warning_count)

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(merged_df, output_path)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success" if val_res.is_valid else "validation_failed",
        "output_path": str(output_path),
        "total_rows": len(merged_df),
        "total_columns": merged_df.shape[1],
        "validation_errors": val_res.error_count,
        "validation_warnings": val_res.warning_count,
        "synthetic_fallback_used": synthetic_fallback_used,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Integrated dataset written to: %s (%d rows)", output_path, len(merged_df))
    logger.info("Report written to: %s", _REPORT_PATH)
    section("REAL OUTPUT MERGER COMPLETED")

    return report


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_real_output_merger()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in real_output_merger: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
