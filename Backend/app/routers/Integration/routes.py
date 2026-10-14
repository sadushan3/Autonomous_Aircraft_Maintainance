"""
Integration layer tying the three independent branches -- Anomaly & Health,
RUL, Failure Risk -- into the shared "one CSV in, one Supervisor decision out"
flow from the architecture diagram.

This router adds NO new prediction/decision logic: every endpoint here is a
thin orchestrator that calls the existing, unmodified component code
(``app.routers.Anomaly_Health_Monitoring.routes`` for ingest/validate and the
full pipeline, ``app.services.Integration.rul_failure_batch_service`` for the
RUL/Failure-Risk branch, ``app.services.Integration.supervisor_bridge`` for
the Supervisor fusion step) and reports what happened.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field as PydanticField

from app.config.Anomaly_Health_Monitoring.config import get_config as get_ahm_config
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import CMAPSSReader
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.data_loader import DataLoader
from app.services.Integration.feature_router import FeatureRouter
from app.services.Integration.rul_failure_batch_service import run_rul_failure_batch
from app.services.Integration.supervisor_bridge import prepare_anomaly_agent_csv, run_supervisor_bridge
from app.utils.Anomaly_Health_Monitoring.model_utils import get_required_columns
from app.utils.Anomaly_Health_Monitoring.utils import AnomalyHealthError, get_logger

logger = get_logger(__name__)
ahm_config = get_ahm_config()

router = APIRouter(prefix="/integration", tags=["Integration"])

_INTEGRATION_EXCHANGE_DIR = ahm_config.paths.data_exchange

#: Scoring points an automatic (``stride=0``) run will aim to stay under.
#:
#: One point costs a RUL forward pass plus a failure-risk one (with the MC
#: dropout batched into a single call), measured at roughly half a second, so
#: this budget is about a 35-minute ceiling. A whole C-MAPSS subset at stride 1
#: is ~17k points and would take hours; a single uploaded file is usually far
#: smaller and can be scored densely.
_DENSE_POINT_BUDGET = 4000


def _auto_stride(frame: pd.DataFrame, requested: int) -> int:
    """Pick a stride that scores densely when the upload can afford it.

    A fixed stride is wrong in both directions: 20 leaves a small upload almost
    entirely forward-filled, and 1 makes a full subset take hours. This scales
    with the actual file, and returns ``requested`` untouched whenever the
    caller named one.

    Args:
        frame: The validated raw frame about to be scored.
        requested: The caller's stride, or 0 to choose automatically.

    Returns:
        The stride to score at.
    """
    if requested >= 1:
        return requested

    from app.services.Integration.rul_failure_batch_service import (
        _count_eligible_points,
        _settings as _pm_settings,
    )

    dense_points = _count_eligible_points(frame, 1, _pm_settings)
    if dense_points <= _DENSE_POINT_BUDGET:
        return 1
    # Points fall off roughly linearly with stride, so this lands just inside
    # the budget rather than overshooting it.
    return max(2, math.ceil(dense_points / _DENSE_POINT_BUDGET))


def _error_response(exc: Exception, status_code: int = 500) -> JSONResponse:
    logger.error("Integration API error (%d): %s", status_code, exc)
    return JSONResponse(
        status_code=status_code,
        content={"status": "failed", "error_type": type(exc).__name__, "message": str(exc)},
    )


@router.get("/feature-routing")
def feature_routing() -> Any:
    """Describe how one uploaded CSV is split across the component branches.

    Returns:
        The routing contract: required upload columns, and the feature view
        plus identity naming each branch receives.
    """
    try:
        return FeatureRouter().describe()
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.post("/route-preview")
async def route_preview(file: UploadFile = File(...)) -> Any:
    """Validate an upload and report the per-component views without running models.

    Lets a caller confirm a file will route correctly -- and see exactly which
    foreign columns get dropped -- before paying for a full pipeline run.

    Args:
        file: The uploaded C-MAPSS ``.txt`` or CSV.

    Returns:
        Per-branch row/engine/column counts, or a structured error naming any
        missing required column.
    """
    stored: Optional[Path] = None
    try:
        ahm_config.ensure_directories()
        stored = ahm_config.paths.uploads / file.filename
        stored.parent.mkdir(parents=True, exist_ok=True)
        with stored.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        reader = CMAPSSReader(ahm_config)
        if reader.detect_format(stored) == "csv":
            frame = reader.read_csv_source(stored)
        else:
            subset_id = reader.infer_subset_id(stored)
            if subset_id is None:
                raise AnomalyHealthError(
                    f"Could not infer a C-MAPSS subset (FD001..FD004) from "
                    f"'{file.filename}'. Rename it to include the subset, "
                    f"e.g. 'train_FD001.txt'."
                )
            split = "test" if "test" in stored.stem.lower() else "train"
            frame = reader.read_subset(subset_id, split)

        feature_router = FeatureRouter()
        views = feature_router.route(frame)

        kept = set(get_required_columns()) | set(feature_router.carry_columns)
        ignored = [c for c in frame.columns if c not in kept]

        return {
            "status": "success",
            "filename": file.filename,
            "rows_uploaded": int(len(frame)),
            "columns_uploaded": int(frame.shape[1]),
            "ignored_columns": ignored,
            "routes": {name: view.summary() for name, view in views.items()},
        }
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.post("/upload")
async def upload(file: UploadFile = File(...)) -> Any:
    """Ingest and validate one CSV/txt against the shared 26-column C-MAPSS schema.

    Reuses the Anomaly & Health component's own reader/validator -- the "Ingest
    & validate" + "Normalize identity" boxes in the architecture diagram --
    rather than re-implementing schema validation.
    """
    try:
        ahm_config.ensure_directories()
        stored_path = ahm_config.paths.uploads / file.filename
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        with stored_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        reader = CMAPSSReader(ahm_config)
        detected_format = reader.detect_format(stored_path)
        if detected_format != "csv" and reader.infer_subset_id(stored_path) is None:
            raise AnomalyHealthError(
                f"Could not infer a C-MAPSS subset (FD001..FD004) from "
                f"filename '{file.filename}'. Rename it to include the "
                f"subset, e.g. 'train_FD001.txt'."
            )

        # Validate the upload itself rather than `data/raw/{split}_{subset}.txt`
        # -- see the matching note in the Anomaly & Health /upload endpoint.
        frame = reader.read_csv_source(stored_path)

        report = DataLoader(ahm_config).validate(frame)
        return {
            "status": "success" if report["schema_valid"] else "failed",
            "filename": file.filename,
            "stored_path": str(stored_path),
            "detected_format": detected_format,
            "rows": report["rows"],
            "columns": report["columns"],
            "engines": int(frame["unit"].nunique()) if "unit" in frame.columns else 0,
            "schema_valid": report["schema_valid"],
            "missing_columns": report["missing_columns"],
            "ignored_columns": report["ignored_columns"],
        }
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


def _resolve_source(file: Optional[UploadFile], stored_path: Optional[str]) -> Path:
    """Resolve the run's input from either a direct upload or a stored path.

    Accepting the file directly exists because the two-step flow is a usability
    trap on Windows: ``/integration/upload`` hands back a path like
    ``D:\\...\\uploads\\test_FD002.txt``, which then has to survive being pasted
    into a query string. Backslashes get escaped -- and frequently
    double-escaped, producing ``%5C%5C`` and a 404 on a file that is sitting
    right there. Uploading in one call skips the round trip entirely.

    Args:
        file: Optional direct upload; saved into the uploads directory.
        stored_path: Optional path from a prior ``/integration/upload``.

    Returns:
        Path to the file to run the pipeline over.

    Raises:
        HTTPException: If neither or both are supplied, or the path is absent.
    """
    if file is not None and stored_path:
        raise HTTPException(
            status_code=400,
            detail="Supply either 'file' or 'stored_path', not both.",
        )

    if file is not None:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename.")
        ahm_config.ensure_directories()
        target = ahm_config.paths.uploads / Path(file.filename).name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        logger.info("run-full received a direct upload, stored at %s", target)
        return target

    if not stored_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide 'file' (upload a .txt/.csv directly) or 'stored_path' "
                "(from /integration/upload)."
            ),
        )

    # Tolerate the doubled backslashes a copied Windows path picks up on its
    # way through a query string.
    cleaned = stored_path.strip().strip('"').strip("'").replace("\\\\", "\\")
    candidate = Path(cleaned)
    if not candidate.exists():
        # A bare filename is unambiguous enough to resolve against uploads/.
        fallback = ahm_config.paths.uploads / candidate.name
        if fallback.exists():
            logger.info(
                "stored_path '%s' not found; resolved to '%s' by filename.",
                stored_path, fallback,
            )
            return fallback
        raise HTTPException(
            status_code=404,
            detail=(
                f"stored_path not found: {stored_path}. Upload the file "
                f"directly via the 'file' field instead."
            ),
        )
    return candidate


@router.post("/run-full")
def run_full(
    file: Optional[UploadFile] = File(
        default=None,
        description=(
            "Upload a C-MAPSS .txt or .csv directly and run everything in one "
            "call. Alternative to stored_path; supply exactly one."
        ),
    ),
    stored_path: Optional[str] = Query(
        default=None,
        description=(
            "Path returned by /integration/upload. Optional -- prefer uploading "
            "the file directly via 'file' to avoid hand-escaping Windows paths."
        ),
    ),
    stride: int = Query(
        default=0,
        ge=0,
        description=(
            "Score every Nth cycle for RUL/Failure Risk. Each engine's final "
            "cycle is always scored regardless of stride, so every engine still "
            "gets a current figure; stride only thins the intermediate "
            "trajectory. Anomaly detection, health index and reasoning are "
            "unaffected: those always run over every row. "
            "0 (the default) picks the stride from the size of the upload: "
            "every cycle is scored when the file is small enough to afford it, "
            "and larger files are thinned just enough to stay within "
            f"{_DENSE_POINT_BUDGET} scoring points. A fixed default of 20 used "
            "to leave a 30-engine upload with real predictions on 3.4% of its "
            "rows, the rest forward-filled by the Supervisor. Pass an explicit "
            "value to override."
        ),
    ),
    include_uncertainty: bool = Query(
        default=True,
        description=(
            "Run Monte-Carlo dropout to produce real uncertainty_10/30/50 "
            "values. Costs ~20 extra forward passes per scored point; set "
            "false (or raise stride) for a faster uncertainty-free run."
        ),
    ),
) -> Any:
    """Run all three branches over the same uploaded file, then fuse via the Supervisor.

    Inference-only end to end: every branch reloads its persisted models and
    calibrations rather than fitting fresh ones against the upload. Retraining
    any component is a deliberate offline action, never something reachable
    by uploading a file through this API.

    Order: (1) Anomaly & Health full pipeline (unmodified), which also writes
    the anomaly exchange CSV; (2) RUL + Failure Risk batch scoring over the
    same raw frame; (3) Maintenance Supervisor merge + inference over all
    three exchange CSVs.
    """
    source_path = _resolve_source(file, stored_path)

    report: dict = {"status": "success", "stages": {}}

    # --- Stage 1: Anomaly & Health full pipeline (existing, unmodified) ---
    try:
        import importlib.util
        import sys as _sys

        pipeline_dir = (
            Path(__file__).resolve().parents[2]
            / "services"
            / "Anomaly_Health_Monitoring"
            / "pipeline"
        )
        module_path = pipeline_dir / "10_full_pipeline.py"
        spec = importlib.util.spec_from_file_location("ahm_full_pipeline_integration", module_path)
        module = importlib.util.module_from_spec(spec)
        _sys.modules["ahm_full_pipeline_integration"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        ahm_report = module.run_full_pipeline(
            ahm_config, retrain=False, include_shap=False, source_csv=str(source_path)
        )
        report["stages"]["anomaly_health"] = ahm_report
        if ahm_report.get("status") != "success":
            report["status"] = "partial"
    except Exception as exc:  # noqa: BLE001
        report["stages"]["anomaly_health"] = {"status": "failed", "error": str(exc)}
        return _error_response(exc, status_code=500)

    # --- Stage 2: RUL + Failure Risk batch (new glue, existing services) ---
    try:
        frame = DataLoader(ahm_config).load(source_csv=source_path)
        effective_stride = _auto_stride(frame, stride)
        batch_report = run_rul_failure_batch(
            frame,
            output_dir=_INTEGRATION_EXCHANGE_DIR,
            stride=effective_stride,
            include_uncertainty=include_uncertainty,
        )
        # Report the stride actually used, not the one requested: with the
        # automatic default the caller has no other way to know how dense the
        # trajectory columns are, and "how much of this is a real prediction"
        # is exactly what a reader of the decision table needs to know.
        batch_report["stride_used"] = effective_stride
        batch_report["stride_selected_automatically"] = stride < 1
        report["stages"]["predictive_maintenance"] = batch_report
        if batch_report.get("status") != "success":
            report["status"] = "partial"
    except Exception as exc:  # noqa: BLE001
        report["stages"]["predictive_maintenance"] = {"status": "failed", "error": str(exc)}
        return _error_response(exc, status_code=500)

    # --- Stage 3: Maintenance Supervisor merge + inference (existing, unmodified) ---
    try:
        anomaly_source = _INTEGRATION_EXCHANGE_DIR / ahm_config.paths.exchange_predictions_csv
        anomaly_for_supervisor = _INTEGRATION_EXCHANGE_DIR / "anomaly_agent_outputs.csv"
        engine_fd_subset = {}
        if "subset_id" in frame.columns:
            engine_fd_subset = {
                str(int(unit)): str(subset).upper()
                for unit, subset in frame.groupby("unit")["subset_id"].first().items()
                if pd.notna(subset)
            }
        prepare_anomaly_agent_csv(anomaly_source, anomaly_for_supervisor, engine_fd_subset=engine_fd_subset)

        supervisor_report = run_supervisor_bridge(
            rul_csv=Path(batch_report["rul_predictions_csv"]),
            failure_risk_csv=Path(batch_report["failure_risk_predictions_csv"]),
            anomaly_csv=anomaly_for_supervisor,
            decisions_output_csv=_INTEGRATION_EXCHANGE_DIR / "supervisor_decisions.csv",
        )
        report["stages"]["maintenance_supervisor"] = supervisor_report
        if supervisor_report.get("status") != "success":
            report["status"] = "partial"
    except Exception as exc:  # noqa: BLE001
        report["stages"]["maintenance_supervisor"] = {"status": "failed", "error": str(exc)}
        return _error_response(exc, status_code=500)

    return report


@router.get("/dashboard")
def dashboard(
    engine_id: Optional[int] = Query(default=None),
    limit: int = Query(
        default=1000,
        ge=1,
        le=100000,
        description="Maximum decision rows to return. Raise it deliberately.",
    ),
    offset: int = Query(default=0, ge=0, description="Rows to skip, for paging."),
) -> Any:
    """Return the fused Maintenance Supervisor decisions, optionally filtered to one engine.

    Paged. Serialising the full decision set is not viable as a dashboard
    response: a single FD002 run produces ~34k rows x ~60 columns, which came
    to a 54 MB JSON body. ``total`` always reports the full match count so a
    caller can page or filter without having to fetch everything to discover
    how much there is.

    Args:
        engine_id: Restrict to one engine.
        limit: Maximum rows in this response.
        offset: Rows to skip before this page.

    Returns:
        The page of decisions plus the total available and paging echo.
    """
    decisions_path = _INTEGRATION_EXCHANGE_DIR / "supervisor_decisions.csv"
    if not decisions_path.exists():
        raise HTTPException(status_code=404, detail="No supervisor decisions yet -- run /integration/run-full first.")

    frame = pd.read_csv(decisions_path)
    if engine_id is not None and "engine_id" in frame.columns:
        frame = frame[frame["engine_id"].astype(str) == str(engine_id)]

    total = int(len(frame))
    page = frame.iloc[offset : offset + limit]

    return {
        "total": total,
        "returned": int(len(page)),
        "limit": limit,
        "offset": offset,
        "truncated": bool(offset + len(page) < total),
        # Retained for callers written against the previous shape, which
        # returned the row count of the (then unpaged) payload under "rows".
        "rows": int(len(page)),
        "decisions": page.to_dict(orient="records"),
    }


def _load_decisions() -> pd.DataFrame:
    decisions_path = _INTEGRATION_EXCHANGE_DIR / "supervisor_decisions.csv"
    if not decisions_path.exists():
        raise HTTPException(status_code=404, detail="No supervisor decisions yet -- run /integration/run-full first.")
    return pd.read_csv(decisions_path)


def _json_safe_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """``to_dict(orient="records")`` with NaN/inf mapped to null.

    pandas emits float NaN for any missing cell, which serialises as the bare
    token ``NaN`` -- not valid JSON, and ``JSON.parse`` in the browser rejects
    it. Converting to object dtype first keeps ints and strings intact.
    """
    if frame.empty:
        return []
    numeric = frame.select_dtypes(include=[np.number]).columns
    working = frame.copy()
    if len(numeric):
        working[numeric] = working[numeric].replace([np.inf, -np.inf], np.nan)
    cleaned = working.astype(object).where(pd.notnull(working), None)
    return cleaned.to_dict(orient="records")


def _latest_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its highest-cycle record.

    The fused table is ~34k rows and the dashboard endpoint pages it, so any
    caller deriving a fleet figure from a page was summarising whichever rows
    happened to land in that page -- and, worse, taking each engine's last row
    *within the page*, which is mid-life rather than current. Doing the
    reduction here means every caller gets the same, complete answer.
    """
    if "engine_id" not in frame.columns:
        return frame
    if "cycle" not in frame.columns:
        return frame.drop_duplicates(subset="engine_id", keep="last")
    ordered = frame.sort_values(["engine_id", "cycle"])
    return ordered.drop_duplicates(subset="engine_id", keep="last")


@router.get("/fleet-summary")
def fleet_summary() -> Any:
    """Fleet state at each engine's latest cycle, reduced server-side.

    Returns one row per engine plus the aggregate counts a dashboard needs, so
    the client never has to page 34k rows (a 54 MB body) to compute them. This
    exists because deriving these figures from ``/dashboard?limit=N`` silently
    summarised N rows instead of the fleet: at limit=4000 that was 11.8% of the
    rows and 33 of the engines, reporting a mean health index of 75.0 where the
    true latest-cycle value is 18.0.

    Returns:
        ``engines`` (one record each, latest cycle), ``totals`` (row/engine
        counts so coverage is auditable) and ``aggregates`` (means and
        categorical tallies over those latest-cycle records).
    """
    frame = _load_decisions()
    latest = _latest_per_engine(frame)

    def _mean(col: str) -> Optional[float]:
        if col not in latest.columns:
            return None
        series = pd.to_numeric(latest[col], errors="coerce").dropna()
        return round(float(series.mean()), 6) if len(series) else None

    def _tally(col: str) -> Dict[str, int]:
        if col not in latest.columns:
            return {}
        return {
            str(k): int(v)
            for k, v in latest[col].dropna().astype(str).value_counts().items()
        }

    def _count_true(col: str) -> Optional[int]:
        if col not in latest.columns:
            return None
        return int(
            latest[col].astype(str).str.lower().isin({"true", "1", "1.0", "yes"}).sum()
        )

    return {
        "totals": {
            "engines": int(len(latest)),
            "rows_total": int(len(frame)),
            "basis": "latest cycle per engine",
        },
        "aggregates": {
            "mean_health_index": _mean("health_index"),
            "mean_predicted_rul": _mean("predicted_rul"),
            "mean_confidence_score": _mean("confidence_score"),
            "mean_anomaly_score": _mean("anomaly_score"),
            "health_state_counts": _tally("health_state"),
            "final_decision_counts": _tally("final_decision"),
            "anomaly_severity_counts": _tally("anomaly_severity"),
            "lifecycle_state_counts": _tally("lifecycle_state"),
            "risk_state_counts": _tally("risk_state"),
            "anomalous_engines": _count_true("is_anomalous"),
            "review_required_engines": _count_true("requires_human_review"),
            "subsets": sorted(
                latest["fd_subset"].dropna().astype(str).unique().tolist()
            )
            if "fd_subset" in latest.columns
            else [],
        },
        "engines": _json_safe_records(latest),
    }


_LEADERBOARD_METRICS = {
    "predicted_rul": True,  # ascending=True -> worst (lowest) RUL first
    "risk_30": False,  # descending -> highest risk first
    "confidence_score": True,  # ascending -> lowest confidence first
    "health_index": True,  # ascending -> worst (lowest) health first
    "anomaly_score": False,  # descending -> most anomalous first
}


@router.get("/leaderboard")
def leaderboard(
    metric: str = Query(default="predicted_rul", description="predicted_rul | risk_30 | confidence_score"),
    limit: int = Query(default=10, ge=1, le=200),
) -> Any:
    """Rank engines by the given metric, worst-first, for triage.

    ``predicted_rul``/``confidence_score`` are sorted ascending (lowest =
    most urgent); ``risk_30`` is sorted descending (highest = most urgent).
    """
    if metric not in _LEADERBOARD_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported metric '{metric}'. Choose one of: {sorted(_LEADERBOARD_METRICS)}.",
        )
    frame = _load_decisions()
    if metric not in frame.columns:
        raise HTTPException(status_code=404, detail=f"Column '{metric}' not present in current decisions.")

    ascending = _LEADERBOARD_METRICS[metric]
    # Rank engines, not rows. Sorting the raw frame let a single engine occupy
    # many slots (one per cycle), so a "top 10 worst engines" list could show
    # the same three engines ten times.
    latest = _latest_per_engine(frame)
    ranked = latest.sort_values(metric, ascending=ascending).head(limit)
    return {
        "metric": metric,
        "rows": int(len(ranked)),
        "basis": "latest cycle per engine",
        "engines": _json_safe_records(ranked),
    }


_QUEUE_DECISIONS = {"schedule_maintenance", "schedule_inspection", "immediate_maintenance"}
_QUEUE_URGENCY_ORDER = {
    "immediate_maintenance": 0,
    "schedule_maintenance": 1,
    "schedule_inspection": 2,
}


@router.get("/maintenance-queue")
def maintenance_queue() -> Any:
    """Engines currently recommended for maintenance/inspection, most urgent first."""
    frame = _load_decisions()
    if "final_decision" not in frame.columns:
        raise HTTPException(status_code=404, detail="No 'final_decision' column in current decisions.")

    queued = frame[frame["final_decision"].isin(_QUEUE_DECISIONS)].copy()
    queued["_urgency_rank"] = queued["final_decision"].map(_QUEUE_URGENCY_ORDER).fillna(99)
    sort_cols = ["_urgency_rank"] + (["predicted_rul"] if "predicted_rul" in queued.columns else [])
    queued = queued.sort_values(sort_cols).drop(columns="_urgency_rank")
    return {"rows": int(len(queued)), "queue": queued.to_dict(orient="records")}


@router.get("/fleet-modes")
def fleet_modes() -> Any:
    """Fleet-wide operating-mode distribution, reusing the Anomaly & Health context-status endpoint."""
    from app.routers.Anomaly_Health_Monitoring.routes import context_status

    status = context_status()
    engines = status.engines if hasattr(status, "engines") else status.get("engines", [])
    mode_counts: dict = {}
    for engine in engines:
        mode = engine.operating_mode if hasattr(engine, "operating_mode") else engine.get("operating_mode")
        key = str(mode)
        mode_counts[key] = mode_counts.get(key, 0) + 1

    return {
        "mode_distribution": mode_counts,
        "drift": status.drift if hasattr(status, "drift") else status.get("drift"),
    }


@router.get("/compare")
def compare(engine_ids: str = Query(..., description="Comma-separated engine ids, e.g. 1,2,3")) -> Any:
    """Side-by-side decision history for a small set of engines."""
    try:
        ids = [int(x.strip()) for x in engine_ids.split(",") if x.strip()]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="engine_ids must be a comma-separated list of integers.") from exc
    if not ids:
        raise HTTPException(status_code=400, detail="No engine_ids provided.")

    frame = _load_decisions()
    result = {}
    for engine_id in ids:
        subset = frame[frame["engine_id"].astype(str) == str(engine_id)].sort_values(
            "cycle" if "cycle" in frame.columns else frame.columns[0]
        )
        result[str(engine_id)] = subset.to_dict(orient="records")
    return {"engines": ids, "history": result}


@router.get("/engine/{engine_id}/trajectory")
def engine_trajectory(
    engine_id: int,
    max_points: int = Query(default=50, ge=5, le=200),
) -> Any:
    """Forecast an engine's RUL trajectory forward from its own sensor history.

    The trajectory service needs a full window of raw ``EngineSensorRow``
    readings, which a browser client does not hold -- it only ever receives
    scored decision rows. Rather than shipping tens of thousands of raw sensor
    values to the client so it can post them straight back, this assembles the
    window server-side from the last preprocessed upload and calls the existing
    service in-process.

    Args:
        engine_id: Engine to forecast, as it appears in ``raw_data.csv``.
        max_points: Cap on returned trajectory points.

    Returns:
        The trajectory service's response for that engine.

    Raises:
        HTTPException: If no preprocessed data exists, the engine is unknown,
            or its history is shorter than the model's window.
    """
    try:
        from app.schemas.Predictive_Maintenance.trajectory_schema import TrajectoryRequest
        from app.services.Predictive_Maintenance.trajectory_service import (
            TrajectoryService,
        )
        from app.services.Integration.rul_failure_batch_service import (
            DEFAULT_FD_SUBSET,
            _row_to_engine_sensor_row,
            _settings,
            _rul_service,
            _failure_risk_service,
        )

        raw_path = ahm_config.paths.processed_path(ahm_config.paths.raw_data_csv)
        if not Path(raw_path).is_file():
            raise HTTPException(
                status_code=404,
                detail="No preprocessed data yet -- run /integration/run-full first.",
            )
        frame = pd.read_csv(raw_path, low_memory=False)
        engine_frame = frame[frame["unit"].astype(int) == int(engine_id)]
        if engine_frame.empty:
            raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found.")
        engine_frame = engine_frame.sort_values("cycle").reset_index(drop=True)

        fd_subset = DEFAULT_FD_SUBSET
        if "subset_id" in engine_frame.columns and pd.notna(engine_frame["subset_id"].iloc[0]):
            candidate = str(engine_frame["subset_id"].iloc[0]).upper()
            if candidate in _settings.SUPPORTED_FDS:
                fd_subset = candidate

        window = max(_settings.RUL_WINDOW_SIZES.get(fd_subset, 50), 30)
        if len(engine_frame) < window:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Engine {engine_id} has {len(engine_frame)} cycles but "
                    f"{fd_subset} needs at least {window} to forecast."
                ),
            )

        # The service replays the sequence at growing window sizes, so passing
        # exactly `window` rows yields a single point. Send the whole observed
        # history and let it walk forward.
        sequence = [
            _row_to_engine_sensor_row(row, int(engine_id), fd_subset)
            for _, row in engine_frame.iterrows()
        ]
        service = TrajectoryService(_settings, _rul_service, _failure_risk_service)
        response = service.compute(
            TrajectoryRequest(
                fd_subset=fd_subset,
                engine_id=str(engine_id),
                sequence=sequence,
                max_points=max_points,
            )
        )
        return response.model_dump() if hasattr(response, "model_dump") else response
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# ---------------------------------------------------------------------------
# Autonomous Maintenance Supervisor Agent -- operator view
# ---------------------------------------------------------------------------
#
# These endpoints back the Supervisor Agent screen: one engine's fused decision
# with the three upstream agent readings that produced it, the twin-residual
# component heat map, the explainability layer, the forward what-if projection
# and the human-feedback loop. They add no decision logic -- everything is
# assembled by `supervisor_agent_view` from artefacts the pipeline already
# wrote, or delegates to the Supervisor's own existing engines.


class SupervisorFeedbackPayload(BaseModel):
    """An engineer's verdict on one fused decision."""

    engine_id: int
    cycle: int
    engineer_decision: str = PydanticField(
        ..., description="One of the five decision classes."
    )
    engineer_id: str = "CHIEF_ENGINEER"
    comments: str = ""
    adapt_policy: bool = True


_NO_DECISIONS_DETAIL = "No supervisor decisions yet -- run /integration/run-full first."


@router.get("/supervisor/engines")
def supervisor_engines(
    limit: Optional[int] = Query(
        default=None, ge=1, le=1000, description="Cap the returned list; omit for all."
    ),
) -> Any:
    """Engines with fused decisions, most urgent first, for the unit picker."""
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.engine_catalog(limit=limit)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.get("/supervisor/agent-view")
def supervisor_agent_view_endpoint(
    engine_id: int = Query(..., description="Engine to open."),
    cycle: Optional[int] = Query(
        default=None,
        ge=1,
        description="Cycle to inspect. Defaults to the engine's latest observed cycle.",
    ),
) -> Any:
    """Everything the Supervisor Agent screen shows for one engine at one cycle.

    Bundled into a single response deliberately: the screen's cards are all
    facets of the same decision record, and issuing eight requests for one
    engine would let them render mutually inconsistent states while they
    settled.
    """
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.agent_view(engine_id, cycle)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.get("/supervisor/what-if")
def supervisor_what_if(
    engine_id: int = Query(...),
    cycle: Optional[int] = Query(default=None, ge=1),
    cycles_ahead: int = Query(
        default=0,
        ge=0,
        le=200,
        description=(
            "Cycles to simulate forward. Within the engine's observed history this "
            "returns the cycle as actually recorded and scored; past it, the state "
            "is extrapolated from the agents' own reported derivatives and the "
            "response says so in 'basis'."
        ),
    ),
) -> Any:
    """Project one engine's state forward for the what-if scenario simulator."""
    from app.services.Integration import supervisor_agent_view as view

    try:
        rows = view.engine_rows(engine_id)
        base = view.pick_cycle(rows, cycle)
        return view.what_if(engine_id, int(base["cycle"]), cycles_ahead)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.post("/supervisor/run-fusion")
def supervisor_run_fusion(
    engine_id: int = Query(...),
    cycle: Optional[int] = Query(default=None, ge=1),
) -> Any:
    """Execute the Supervisor's decision-fusion engine live for one engine.

    Runs the real ``DecisionFusionEngine`` over that engine's stored upstream
    features and reports the result next to the persisted one. The decision
    table on disk is not modified, and no model is retrained.
    """
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.run_live_fusion(engine_id, cycle)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.get("/supervisor/certificate")
def supervisor_certificate(
    engine_id: int = Query(...),
    cycle: Optional[int] = Query(default=None, ge=1),
) -> Any:
    """A self-contained, auditable record of one decision."""
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.decision_certificate(engine_id, cycle)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.get("/supervisor/fusion-summary")
def supervisor_fusion_summary() -> Any:
    """Fleet-level record of what the fusion step decided on the last run."""
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.fusion_summary()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.post("/supervisor/feedback")
def supervisor_feedback(payload: SupervisorFeedbackPayload) -> Any:
    """Log an engineer's accept/override verdict and re-run the policy pass.

    This closes the loop the Supervisor component always had code for and never
    had a caller for. It records the verdict and re-derives the adaptive policy
    from the full feedback history; it does not retrain a model.
    """
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.submit_feedback(
            engine_id=payload.engine_id,
            cycle=payload.cycle,
            engineer_decision=payload.engineer_decision,
            engineer_id=payload.engineer_id,
            comments=payload.comments,
            adapt_policy=payload.adapt_policy,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=_NO_DECISIONS_DETAIL)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)


@router.get("/supervisor/feedback")
def supervisor_feedback_history(
    engine_id: Optional[int] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
) -> Any:
    """Engineer verdicts logged so far, newest first."""
    from app.services.Integration import supervisor_agent_view as view

    try:
        return view.feedback_history(engine_id, limit)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc)
