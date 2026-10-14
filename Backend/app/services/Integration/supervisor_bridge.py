"""
Bridges the RUL/Failure-Risk/Anomaly exchange CSVs into the existing
(unmodified) Maintenance Supervisor pipeline -- the offline equivalent of
running its own CLI as ``--mode real`` then ``--mode infer``, just as direct
function calls instead of subprocess/CLI invocations.

No decision-fusion logic lives here. This module only:
  1. renames the Anomaly & Health exchange file's columns onto the names the
     Supervisor's ``upstream_output_loader``/``ANOMALY_FEATURES`` expect
     (``severity`` -> ``anomaly_severity``, ``anomaly_flag`` -> ``is_anomalous``;
     everything else -- engine_id, cycle, anomaly_score -- already matches, and
     RUL/Failure-Risk exchange files need no renaming at all, see the
     integration plan), then
  2. calls the ported ``run_real_output_merger`` and ``run_inference_pipeline``
     exactly as they exist today.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from app.services.Maintenance_Supervisor.data_integration.real_output_merger import (
    run_real_output_merger,
)
from app.services.Maintenance_Supervisor.pipeline.run_inference_pipeline import (
    run_inference_pipeline,
)

_ANOMALY_COLUMN_RENAME = {
    "severity": "anomaly_severity",
    "anomaly_flag": "is_anomalous",
}


def prepare_anomaly_agent_csv(
    source_csv: Path,
    dest_csv: Path,
    engine_fd_subset: Optional[Mapping[str, str]] = None,
    default_fd_subset: str = "FD001",
) -> Path:
    """Rename the Anomaly & Health exchange file onto the Supervisor's column names.

    The Anomaly & Health exchange file has no ``fd_subset`` column (that
    component doesn't track it per row), but the Supervisor's merge join key
    is ``(fd_subset, engine_id, cycle)`` whenever the RUL/Failure-Risk frames
    carry ``fd_subset`` -- so one is added here to make the three-way merge
    possible.

    Args:
        source_csv: ``anomaly_health_predictions.csv`` written by the existing,
            unmodified Anomaly & Health dashboard exchange step.
        dest_csv: Destination path for the Supervisor-facing copy.
        engine_fd_subset: Optional per-engine ``{engine_id: fd_subset}`` map
            (engine_id as string), e.g. built from the uploaded file's
            ``subset_id`` column grouped by engine. Every engine's rows are
            stamped with ITS OWN subset -- a single global default silently
            mismatches the RUL/Failure-Risk join key for any engine that
            isn't actually FD001, duplicating rows in the merge instead of
            joining them.
        default_fd_subset: Fallback for any engine missing from the map.

    Returns:
        ``dest_csv``.
    """
    frame = pd.read_csv(source_csv)
    frame = frame.rename(columns=_ANOMALY_COLUMN_RENAME)
    if "fd_subset" not in frame.columns:
        if engine_fd_subset:
            frame["fd_subset"] = (
                frame["engine_id"].astype(str).map(engine_fd_subset).fillna(default_fd_subset)
            )
        else:
            frame["fd_subset"] = default_fd_subset
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest_csv, index=False)
    return dest_csv


def run_supervisor_bridge(
    rul_csv: Path,
    failure_risk_csv: Path,
    anomaly_csv: Path,
    decisions_output_csv: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge the three upstream exchange CSVs and run the Supervisor's fusion engine.

    Args:
        rul_csv: ``rul_predictions.csv`` from ``rul_failure_batch_service``.
        failure_risk_csv: ``failure_risk_predictions.csv`` from the same.
        anomaly_csv: The Anomaly & Health exchange CSV, already renamed via
            :func:`prepare_anomaly_agent_csv`.
        decisions_output_csv: Optional explicit path to also copy the final
            decisions to (defaults to leaving them at the Supervisor's own
            ``artifacts/Maintenance_Supervisor/inference_pipeline_output.csv``).

    Returns:
        A combined report dict with both the merge and inference reports.
    """
    merge_report = run_real_output_merger(
        rul_path=rul_csv,
        risk_path=failure_risk_csv,
        anomaly_path=anomaly_csv,
    )

    inference_report = run_inference_pipeline(input_path=Path(merge_report["output_path"]))

    if decisions_output_csv is not None:
        decisions_output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.read_csv(inference_report["output_path"]).to_csv(decisions_output_csv, index=False)
        inference_report = {**inference_report, "decisions_csv": str(decisions_output_csv)}

    return {
        "status": inference_report.get("status", "unknown"),
        "merge": merge_report,
        "inference": inference_report,
    }
