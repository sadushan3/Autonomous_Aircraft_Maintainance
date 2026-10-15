"""
FastAPI router for the Explainable Health Monitoring & Anomaly Reasoning Agent.

Twelve endpoints, matching the specification:

    /upload  /preprocess  /train-digital-twin  /generate-health-index
    /generate-health-score  /detect-anomalies  /root-cause-analysis
    /explain  /confidence  /feedback  /dashboard  /full-pipeline

Every endpoint delegates to the same service classes the CLI pipeline stages
use -- there is no logic duplicated between the API and the standalone
scripts.  Errors from the service layer are caught and translated into
:class:`ErrorResponse` bodies with an appropriate HTTP status rather than
leaking raw tracebacks.
"""

from __future__ import annotations

import ast
import json
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from app.config.Anomaly_Health_Monitoring.config import get_config
from app.schemas.Anomaly_Health_Monitoring.anomaly_schema import (
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    HealthIndexResponse,
)
from app.schemas.Anomaly_Health_Monitoring.common_schema import StageStatus
from app.schemas.Anomaly_Health_Monitoring.confidence_schema import (
    ConfidenceResponse,
    FeedbackResponse,
    FeedbackSubmission,
    FeedbackSummaryResponse,
    ThresholdStateResponse,
)
from app.schemas.Anomaly_Health_Monitoring.explainability_schema import (
    DashboardResponse,
    ExplainRequest,
    ExplainResponse,
)
from app.schemas.Anomaly_Health_Monitoring.reasoning_schema import (
    DependencyGraphResponse,
    EngineRootCauseResponse,
    ReasoningResponse,
)
from app.schemas.Anomaly_Health_Monitoring.upload_schema import (
    ContextStatusResponse,
    PreprocessRequest,
    PreprocessResponse,
    SchemaDescription,
    TwinTrainingRequest,
    TwinTrainingResponse,
    UploadResponse,
)
from app.services.Anomaly_Health_Monitoring.anomaly_detection.anomaly_fusion import (
    AnomalyFusion,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.context_drift import (
    ContextDriftDetector,
)
from app.services.Anomaly_Health_Monitoring.context_modeling.operating_mode_detector import (
    OperatingModeDetector,
)
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_api import (
    DashboardApiService,
)
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cleaner import DataCleaner
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.cmapss_reader import (
    CMAPSSReader,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.data_loader import (
    DataLoader,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.feature_engineering import (
    FeatureEngineer,
)
from app.services.Anomaly_Health_Monitoring.Data_Preprocessing.scaler import PerModeScaler
from app.services.Anomaly_Health_Monitoring.digital_twin.ensemble_twin import EnsembleTwin
from app.services.Anomaly_Health_Monitoring.explainability.explanation_generator import (
    ExplanationGenerator,
)
from app.services.Anomaly_Health_Monitoring.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitoring.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitoring.feedback.learning_updater import (
    LearningUpdater,
)
from app.services.Anomaly_Health_Monitoring.feedback.threshold_adapter import (
    ThresholdAdapter,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_alert_engine import (
    HealthAlertEngine,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import (
    HealthIndexCalculator,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import (
    HealthScoreEngine,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.services.Anomaly_Health_Monitoring.reasoning.root_cause_analyzer import (
    RootCauseAnalyzer,
)
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import (
    SensorDependencyGraph,
)
from app.services.Anomaly_Health_Monitoring.uncertainty.confidence_estimator import (
    ConfidenceEstimator,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import (
    describe_schema,
    get_required_columns,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    ArtifactNotFoundError,
    get_logger,
    read_csv,
    utc_now_iso,
)

logger = get_logger(__name__)
config = get_config()

router = APIRouter(prefix=config.api.router_prefix, tags=list(config.api.tags))


def _error_response(exc: Exception, status_code: int = 400) -> JSONResponse:
    """Build a structured error response from a caught exception.

    Args:
        exc: The exception raised by a service call.
        status_code: HTTP status to respond with.

    Returns:
        A JSON response body matching :class:`ErrorResponse`.
    """
    stage = None
    hint = None
    if isinstance(exc, ArtifactNotFoundError):
        hint = "Run the preceding pipeline stage before calling this endpoint."
    payload = {
        "status": StageStatus.FAILED.value,
        "error_type": type(exc).__name__,
        "message": str(exc),
        "stage": stage,
        "hint": hint,
    }
    logger.error("API error (%d): %s", status_code, payload)
    return JSONResponse(status_code=status_code, content=payload)


# --------------------------------------------------------------------------- #
# /upload
# --------------------------------------------------------------------------- #

@router.get("/schema", response_model=SchemaDescription)
def get_schema() -> SchemaDescription:
    """Return the canonical C-MAPSS schema this component expects.

    Returns:
        The 26-column schema description.
    """
    return SchemaDescription(**describe_schema())


@router.post("/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)) -> Any:
    """Accept a raw C-MAPSS ``.txt`` file or an already-parsed CSV.

    Validates the 26-column schema, reporting missing required columns and
    any foreign columns that were ignored (never a failure by themselves).

    Args:
        file: The uploaded file.

    Returns:
        An :class:`UploadResponse`, or a structured error on failure.
    """
    try:
        config.ensure_directories()
        stored_path = config.paths.uploads / file.filename
        stored_path.parent.mkdir(parents=True, exist_ok=True)
        with stored_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)

        reader = CMAPSSReader(config)
        detected_format = reader.detect_format(stored_path)

        if detected_format != "csv" and reader.infer_subset_id(stored_path) is None:
            raise AnomalyHealthError(
                f"Could not infer a C-MAPSS subset (FD001..FD004) from "
                f"filename '{file.filename}'. Rename it to include the "
                f"subset, e.g. 'train_FD001.txt'."
            )

        # Validate the file that was actually uploaded. The .txt branch
        # previously called `read_subset(subset_id, split)`, which reads
        # `data/raw/{split}_{subset}.txt` -- a different file that merely
        # shares a naming convention. Every figure in the response (rows,
        # columns, engines, cycle_range, schema verdict) therefore described
        # the stock raw file rather than the upload, and an upload whose
        # counterpart was absent from data/raw failed on a missing file while
        # sitting right there in uploads/. `read_csv_source` routes on the
        # same `detect_format` result, so both forms still parse correctly.
        frame = reader.read_csv_source(stored_path)

        loader = DataLoader(config)
        report = loader.validate(frame)

        return UploadResponse(
            status=StageStatus.SUCCESS if report["schema_valid"] else StageStatus.FAILED,
            message="Upload validated." if report["schema_valid"] else "Upload missing required columns.",
            timestamp=utc_now_iso(),
            filename=file.filename,
            stored_path=str(stored_path),
            detected_format=detected_format,
            rows=report["rows"],
            columns=report["columns"],
            engines=int(frame["unit"].nunique()) if "unit" in frame.columns else 0,
            subsets=sorted(frame["subset_id"].unique().tolist()) if "subset_id" in frame.columns else [],
            schema_valid=report["schema_valid"],
            missing_columns=report["missing_columns"],
            ignored_columns=report["ignored_columns"],
            cycle_range=[int(frame["cycle"].min()), int(frame["cycle"].max())] if "cycle" in frame.columns and len(frame) else [],
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /preprocess
# --------------------------------------------------------------------------- #

@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess(request: PreprocessRequest) -> Any:
    """Run data loading, cleaning and feature engineering.

    Args:
        request: Preprocessing parameters.

    Returns:
        A :class:`PreprocessResponse`, or a structured error on failure.
    """
    try:
        loader = DataLoader(config)
        raw_frame, load_report = loader.run(
            subsets=request.subsets,
            split=request.split,
            source_csv=request.source_csv,
            max_engines=request.max_engines,
        )
        # fit_policy=False: this endpoint serves ad-hoc / uploaded data, not
        # the authoritative training run. It must apply the persisted
        # sensor/feature policy, never re-derive and overwrite it -- every
        # twin, detector and calibration downstream was fit against a
        # specific active-sensor set, and a single-engine or different-mix
        # upload silently redefining that set would invalidate all of them.
        cleaner = DataCleaner(config)
        cleaned_frame, clean_report = cleaner.run(raw_frame, fit_policy=False)
        engineer = FeatureEngineer(config)
        engineered_frame, feature_report = engineer.run(cleaned_frame, fit_policy=False)

        return PreprocessResponse(
            status=StageStatus.SUCCESS,
            message="Preprocessing complete.",
            timestamp=utc_now_iso(),
            rows=int(len(engineered_frame)),
            engines=int(engineered_frame["unit"].nunique()),
            subsets=load_report.get("subsets", []),
            raw_csv=load_report.get("output_path", ""),
            cleaned_csv=clean_report.get("output_path", ""),
            features_csv=feature_report.get("output_path", ""),
            feature_count=feature_report.get("features_created", 0),
            data_quality=clean_report,
            metrics={"load": load_report, "clean": clean_report, "features": feature_report},
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /train-digital-twin
# --------------------------------------------------------------------------- #

@router.post("/train-digital-twin", response_model=TwinTrainingResponse)
def train_digital_twin(request: TwinTrainingRequest) -> Any:
    """Train (or reload) the Digital Twin ensemble and compute residuals.

    Args:
        request: Twin training parameters.

    Returns:
        A :class:`TwinTrainingResponse`, or a structured error on failure.
    """
    try:
        # This endpoint is inference-only: it always reloads the persisted
        # context model, per-mode scalers and twin ensemble rather than
        # fitting fresh ones. Retraining is a deliberate offline action (the
        # pipeline scripts under app/services/.../pipeline/), never something
        # a request against this API can trigger.
        detector = OperatingModeDetector(config)
        labelled, _ = detector.run(refit=False)
        scaler = PerModeScaler(config)
        scaled, _ = scaler.run(labelled, refit=False)

        ensemble = EnsembleTwin(config)
        predictions, twin_report = ensemble.run(scaled, retrain=False)

        from app.services.Anomaly_Health_Monitoring.digital_twin.residual_calculator import (
            ResidualCalculator,
        )
        from app.services.Anomaly_Health_Monitoring.digital_twin.validate_residuals import (
            ResidualValidator,
        )

        calculator = ResidualCalculator(config)
        residuals, residual_report = calculator.run(scaled, predictions)
        validation = ResidualValidator(config).validate(residuals)

        twin_metrics = [
            {
                "name": name,
                "rmse": metrics.get("rmse", 0.0),
                "mae": metrics.get("mae", 0.0),
                "r2": metrics.get("r2", 0.0),
                "train_rows": metrics.get("train_rows", 0),
                "val_rows": metrics.get("val_rows", 0),
                "fit_seconds": metrics.get("fit_seconds", 0.0),
                "ensemble_weight": twin_report.get("weights", {}).get(name, 0.0),
            }
            for name, metrics in twin_report.get("per_twin_metrics", {}).items()
        ]

        return TwinTrainingResponse(
            status=StageStatus.SUCCESS,
            message="Digital twin stage complete.",
            timestamp=utc_now_iso(),
            twins=twin_metrics,
            ensemble_rmse=twin_report.get("weighted_mean_rmse", 0.0),
            total_rows=int(len(scaled)),
            predictions_csv=twin_report.get("output_path", ""),
            residuals_csv=residual_report.get("output_path", ""),
            residual_validation=validation,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /generate-health-index, /generate-health-score
# --------------------------------------------------------------------------- #

@router.post("/generate-health-index", response_model=HealthIndexResponse)
def generate_health_index() -> Any:
    """Compute the health index from residuals.

    Returns:
        A :class:`HealthIndexResponse`, or a structured error on failure.
    """
    try:
        calculator = HealthIndexCalculator(config)
        # recalibrate=False: this API is inference-only. The default is True
        # for the offline training stage, and leaving it to default here let a
        # single request re-fit the healthy-behaviour reference from the
        # current batch and overwrite the persisted calibration that every
        # later health score, state and alert is judged against.
        health_index, report = calculator.run(recalibrate=False)
        return HealthIndexResponse(
            status=StageStatus.SUCCESS,
            message="Health index computed.",
            timestamp=utc_now_iso(),
            rows=report["rows"],
            engines=report["engines"],
            mean_health_index=report["mean_health_index"],
            min_health_index=report["min_health_index"],
            health_index_csv=report["output_path"],
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


@router.post("/generate-health-score", response_model=HealthIndexResponse)
def generate_health_score() -> Any:
    """Compute health score, trend, state and alert level.

    Returns:
        A :class:`HealthIndexResponse` enriched with state distribution, or a
        structured error on failure.
    """
    try:
        score_engine = HealthScoreEngine(config)
        scores, score_report = score_engine.run()
        classifier = HealthStateClassifier(config)
        states, state_report = classifier.run(scores)
        alert_engine = HealthAlertEngine(config)
        alerted, _ = alert_engine.run(states)

        from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_csv

        atomic_write_csv(alerted, config.paths.processed_path(config.paths.health_states_csv), config=config)

        return HealthIndexResponse(
            status=StageStatus.SUCCESS,
            message="Health score/state computed.",
            timestamp=utc_now_iso(),
            rows=state_report["rows"],
            engines=state_report["engines"],
            mean_health_index=score_report.get("mean_health_score", 0.0),
            state_distribution=state_report.get("state_distribution", {}),
            health_scores_csv=score_report.get("output_path", ""),
            health_states_csv=state_report.get("output_path", ""),
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /detect-anomalies
# --------------------------------------------------------------------------- #

@router.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
def detect_anomalies(request: AnomalyDetectionRequest) -> Any:
    """Train/apply all anomaly detectors, fuse scores and gate operational anomalies.

    Args:
        request: Anomaly detection parameters.

    Returns:
        An :class:`AnomalyDetectionResponse`, or a structured error on failure.
    """
    try:
        # Inference-only: always reload the persisted detectors rather than
        # refitting them against whatever data this request happens to carry.
        fusion = AnomalyFusion(config)
        classified, report = fusion.run(
            retrain=False, fusion_threshold=request.fusion_threshold
        )
        return AnomalyDetectionResponse(
            status=StageStatus.SUCCESS,
            message="Anomaly detection complete.",
            timestamp=utc_now_iso(),
            rows=report["rows"],
            engines=report["engines"],
            total_anomalies=report["total_flagged"],
            fault_anomalies=report["fault_anomalies"],
            operational_anomalies=report["operational_anomalies"],
            severity_distribution=report.get("severity_distribution", {}),
            fusion_threshold=report["fusion_threshold"],
            anomaly_scores_csv=report["scores_output_path"],
            anomaly_fusion_csv=report["fusion_output_path"],
            metrics=report,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /root-cause-analysis
# --------------------------------------------------------------------------- #

@router.post("/root-cause-analysis", response_model=ReasoningResponse)
def root_cause_analysis(only_anomalies: bool = Query(default=True)) -> Any:
    """Run root-cause reasoning over fault-type anomalies.

    Args:
        only_anomalies: Reason only about fault-type rows.

    Returns:
        A :class:`ReasoningResponse`, or a structured error on failure.
    """
    try:
        analyzer = RootCauseAnalyzer(config)
        analyzed, report = analyzer.run(only_anomalies=only_anomalies)
        return ReasoningResponse(
            status=StageStatus.SUCCESS,
            message="Root-cause analysis complete.",
            timestamp=utc_now_iso(),
            rows=report["rows_input"],
            reasoned_rows=report["rows_reasoned"],
            engines=report["engines"],
            cause_distribution=report["cause_distribution"],
            subsystem_distribution=report["subsystem_distribution"],
            evolution_distribution=report["evolution_distribution"],
            memory_matches=report["memory_matches"],
            root_cause_csv=report["output_path"],
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /explain
# --------------------------------------------------------------------------- #

@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> Any:
    """Generate human-readable explanations for reasoned anomalies.

    Args:
        request: Explainability parameters.

    Returns:
        An :class:`ExplainResponse`, or a structured error on failure.
    """
    try:
        generator = ExplanationGenerator(config)
        explanations, report = generator.run(
            include_shap=request.include_shap, engine_id=request.engine_id,
            max_rows=request.max_rows, only_anomalies=request.only_anomalies,
            persist=request.engine_id is None and request.max_rows is None,
        )
        from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_service import DashboardService
        from app.schemas.Anomaly_Health_Monitoring.explainability_schema import ExplanationRecord
        records = []
        for _, row in explanations.iterrows():
            payload = DashboardService._sanitize(row.dropna().to_dict())
            payload["engine_id"] = payload.pop("unit")
            records.append(ExplanationRecord(**payload))
        return ExplainResponse(
            status=StageStatus.SUCCESS,
            message="Explanations generated.",
            timestamp=utc_now_iso(),
            rows=report["rows_input"],
            explained_rows=report["rows_explained"],
            explanation_csv=report["output_path"],
            engines=report["engines"],
            explainer_used=report["explainer_used"],
            shap_computed=report["shap_computed"],
            records=records, metrics=report,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /confidence
# --------------------------------------------------------------------------- #

@router.post("/confidence", response_model=ConfidenceResponse)
def confidence() -> Any:
    """Compute the confidence/reliability/uncertainty triple.

    Returns:
        A :class:`ConfidenceResponse`, or a structured error on failure.
    """
    try:
        estimator = ConfidenceEstimator(config)
        result, report = estimator.run()
        return ConfidenceResponse(
            status=StageStatus.SUCCESS,
            message="Confidence estimation complete.",
            timestamp=utc_now_iso(),
            rows=report["rows"],
            engines=report["engines"],
            mean_confidence=report["mean_confidence"],
            mean_reliability=report["mean_reliability"],
            mean_uncertainty=report["mean_uncertainty"],
            low_confidence_rows=report["low_confidence_rows"],
            confidence_csv=report["output_path"],
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /feedback
# --------------------------------------------------------------------------- #

@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(submission: FeedbackSubmission) -> Any:
    """Record an operator verdict and propagate it through memory and thresholds.

    Args:
        submission: The operator's feedback.

    Returns:
        A :class:`FeedbackResponse`, or a structured error on failure.
    """
    try:
        updater = LearningUpdater(config)
        summary = updater.apply_feedback(
            engine_id=submission.engine_id,
            cycle=submission.cycle,
            verdict=submission.verdict.value,
            operator=submission.operator,
            comment=submission.comment,
            confirmed_root_cause=submission.confirmed_root_cause,
        )
        feedback_summary = updater.feedback_store.summary()
        return FeedbackResponse(
            status=StageStatus.SUCCESS,
            message="Feedback recorded.",
            timestamp=utc_now_iso(),
            feedback_id=summary["feedback_id"],
            total_feedback_records=feedback_summary["total"],
            accepted_count=feedback_summary["by_verdict"].get("accepted", 0),
            false_alarm_count=feedback_summary["by_verdict"].get("false_alarm", 0),
            missed_count=feedback_summary["by_verdict"].get("missed", 0),
            thresholds_path=str(updater.threshold_adapter.path),
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /context-status
# --------------------------------------------------------------------------- #

#: Columns the context view and the drift detector actually read.
_CONTEXT_COLUMNS = (
    "operating_mode",
    "mode_confidence",
    "is_mode_transition",
    "cycles_since_mode_change",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
)


@lru_cache(maxsize=2)
def _context_frame(_stamp: float) -> pd.DataFrame:
    """Load just the context columns of ``context_clusters.csv``.

    That artefact carries every engineered feature and is ~640 MB, so reading
    it whole cost about seven seconds on every request to a panel that needs
    only the regime label and the three operating settings. Cached on its
    mtime, so a pipeline re-run invalidates it.

    Args:
        _stamp: Modification time of ``context_clusters.csv``; cache key only.

    Returns:
        One row per (unit, cycle) with whichever context columns exist.

    Raises:
        ArtifactNotFoundError: If the context stage has not been run.
    """
    path = config.paths.processed_path(config.paths.context_clusters_csv)
    if not path.exists():
        raise ArtifactNotFoundError(
            f"{path.name} not found. Run the context-modeling stage first."
        )
    header = pd.read_csv(path, nrows=0)
    wanted = ["unit", "cycle"] + [c for c in _CONTEXT_COLUMNS if c in header.columns]
    return pd.read_csv(path, usecols=wanted)


@router.get("/context-status", response_model=ContextStatusResponse)
def context_status() -> Any:
    """Return each engine's latest operating mode plus fleet-wide context drift.

    Returns:
        A :class:`ContextStatusResponse`, or a structured error on failure.
    """
    try:
        frame = _context_frame(
            config.paths.processed_path(
                config.paths.context_clusters_csv
            ).stat().st_mtime
        )
        latest = (
            frame.sort_values("cycle")
            .groupby("unit", as_index=False, sort=True)
            .tail(1)
            .sort_values("unit")
            .reset_index(drop=True)
        )

        engines = []
        for _, row in latest.iterrows():
            entry: Dict[str, Any] = {
                "engine_id": int(row["unit"]),
                "cycle": int(row["cycle"]),
            }
            for column in (
                "operating_mode", "mode_confidence",
                "is_mode_transition", "cycles_since_mode_change",
            ):
                if column in row.index and pd.notna(row[column]):
                    entry[column] = row[column]
            engines.append(entry)

        drift = ContextDriftDetector(config).detect(frame)

        return ContextStatusResponse(
            status=StageStatus.SUCCESS,
            message="Context status ready.",
            timestamp=utc_now_iso(),
            engines=engines,
            drift=drift,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /reasoning/sensor-graph, /reasoning/{engine_id}
# --------------------------------------------------------------------------- #

@router.get("/reasoning/sensor-graph", response_model=DependencyGraphResponse)
def sensor_dependency_graph() -> Any:
    """Return the fixed sensor/subsystem dependency graph.

    Returns:
        A :class:`DependencyGraphResponse`, or a structured error on failure.
    """
    try:
        description = SensorDependencyGraph(config).describe()
        return DependencyGraphResponse(
            status=StageStatus.SUCCESS,
            message="Sensor dependency graph.",
            timestamp=utc_now_iso(),
            **description,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


def _parse_stringified_column(value: Any) -> Any:
    """Parse a CSV column that stores a stringified Python list/dict.

    Args:
        value: Either an already-parsed object, a stringified literal, or NaN.

    Returns:
        The parsed value, or ``None`` when parsing fails or the value is NaN.
    """
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return value


@router.get("/reasoning/{engine_id}", response_model=EngineRootCauseResponse)
def reasoning_for_engine(
    engine_id: int, limit: int = Query(default=50, ge=1, le=1000)
) -> Any:
    """Return root-cause analysis history for one engine.

    Args:
        engine_id: Engine identifier.
        limit: Maximum most-recent rows to return.

    Returns:
        An :class:`EngineRootCauseResponse`, or a structured error on failure.
    """
    try:
        frame = read_csv(
            config.paths.processed_path(config.paths.root_cause_analysis_csv),
            required_columns=["unit", "cycle"],
            stage="reasoning_for_engine",
        )
        engine_rows = (
            frame[frame["unit"] == engine_id]
            .sort_values("cycle", ascending=False)
            .head(limit)
        )
        if engine_rows.empty:
            raise AnomalyHealthError(
                f"No root-cause records found for engine {engine_id}."
            )

        records = []
        for _, row in engine_rows.iterrows():
            record = row.to_dict()
            record["engine_id"] = int(record.pop("unit"))
            record["cycle"] = int(record["cycle"])
            for column in ("top_sensors", "sensor_contributions", "candidate_causes"):
                if column in record:
                    parsed = _parse_stringified_column(record[column])
                    record[column] = parsed if parsed is not None else []
            for key, value in list(record.items()):
                if isinstance(value, float) and pd.isna(value):
                    record[key] = None
            records.append(record)

        return EngineRootCauseResponse(
            status=StageStatus.SUCCESS,
            message=f"Root-cause history for engine {engine_id}.",
            timestamp=utc_now_iso(),
            engine_id=engine_id,
            records=records,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /feedback/summary, /feedback/thresholds
# --------------------------------------------------------------------------- #

@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def feedback_summary() -> Any:
    """Return operator-feedback and anomaly-memory rollups.

    Returns:
        A :class:`FeedbackSummaryResponse`, or a structured error on failure.
    """
    try:
        feedback_store = FeedbackStore(config)
        alert_memory = AlertMemory(config)
        return FeedbackSummaryResponse(
            status=StageStatus.SUCCESS,
            message="Feedback summary ready.",
            timestamp=utc_now_iso(),
            feedback=feedback_store.summary(),
            memory=alert_memory.summary(),
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


@router.get("/feedback/thresholds", response_model=ThresholdStateResponse)
def feedback_thresholds() -> Any:
    """Return the current adaptive fusion-threshold state.

    Returns:
        A :class:`ThresholdStateResponse`, or a structured error on failure.
    """
    try:
        state = ThresholdAdapter(config).load_state()
        return ThresholdStateResponse(
            status=StageStatus.SUCCESS,
            message="Threshold adaptation state.",
            timestamp=utc_now_iso(),
            **state,
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /dashboard
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=2)
def _dashboard_payload_cached(_stamp: float) -> Dict[str, Any]:
    """Build the unfiltered dashboard payload, cached on the artefact's mtime.

    Assembling it parses a 216 MB CSV and reduces 160k rows, which costs about
    five seconds -- fine once, wasteful on every poll of a screen whose data
    only changes when the pipeline re-runs. Keying the cache on the CSV's mtime
    means a pipeline run invalidates it automatically. The result is ~6 MB, so
    caching the payload rather than the parsed frame keeps ~360 MB out of
    memory.

    Args:
        _stamp: Modification time of ``dashboard_data.csv``; cache key only.

    Returns:
        A mapping with ``fleet``, ``engines``, ``alerts`` and ``timeline``.
    """
    return DashboardApiService(config).dashboard_payload()


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(engine_id: Optional[int] = Query(default=None)) -> Any:
    """Return the dashboard payload, optionally filtered to one engine.

    Args:
        engine_id: Optional engine filter.

    Returns:
        A :class:`DashboardResponse`, or a structured error on failure.
    """
    try:
        api_service = DashboardApiService(config)

        # Serve the artefact the pipeline already produced rather than
        # rebuilding it. Regenerating on every GET re-joined all 160k rows and
        # rewrote a 216 MB CSV, which cost ~12s per request on top of the ~3.6s
        # the payload itself takes -- and because the page polls, a second
        # request would arrive mid-write and collide with the first on the same
        # fixed `.tmp` path, failing the replace with WinError 32 and 500ing the
        # whole screen. The payload reads the CSV independently, so none of that
        # work was needed to answer the request.
        dashboard_csv = config.paths.processed_path(config.paths.dashboard_data_csv)
        if not dashboard_csv.exists():
            logger.info("dashboard_data.csv absent; building it once for /dashboard.")
            wide, _ = DashboardDataGenerator(config).run()
            api_service.write_exchange_file(wide)

        exchange_path = config.paths.data_exchange / config.paths.exchange_predictions_csv
        payload = _dashboard_payload_cached(dashboard_csv.stat().st_mtime)

        if engine_id is not None:
            engine_payload = api_service.engine_payload(engine_id)
            payload = {
                **payload,
                "engines": [engine_payload["summary"]] if engine_payload["summary"] else [],
                "timeline": {str(engine_id): engine_payload["timeline"]},
            }

        return DashboardResponse(
            status=StageStatus.SUCCESS,
            message="Dashboard data ready.",
            timestamp=utc_now_iso(),
            fleet=payload["fleet"],
            engines=payload["engines"],
            alerts=payload["alerts"],
            timeline=payload["timeline"],
            dashboard_csv=str(dashboard_csv),
            exchange_csv=str(exchange_path),
        )
    except AnomalyHealthError as exc:
        return _error_response(exc, status_code=400)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# /full-pipeline
# --------------------------------------------------------------------------- #

@router.post("/full-pipeline")
def full_pipeline(
    include_shap: bool = Query(default=True),
) -> Any:
    """Run every stage end to end and write the integration exchange file.

    Inference-only: every stage reloads its persisted artifacts (context
    model, per-mode scalers, twin ensemble, detectors, calibrations) rather
    than fitting fresh ones. Retraining is a deliberate offline action run
    via the pipeline scripts directly; this API can never trigger it.

    Args:
        include_shap: Whether the explainability stage computes global SHAP.

    Returns:
        A :class:`PipelineResponse`-shaped payload, or a structured error.
    """
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
        spec = importlib.util.spec_from_file_location("ahm_full_pipeline_api", module_path)
        module = importlib.util.module_from_spec(spec)
        _sys.modules["ahm_full_pipeline_api"] = module
        spec.loader.exec_module(module)  # type: ignore[union-attr]

        report = module.run_full_pipeline(config, retrain=False, include_shap=include_shap)
        status_code = 200 if report["status"] == "success" else 500
        return JSONResponse(status_code=status_code, content=_json_safe(report))
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


def _json_safe(payload: Any) -> Any:
    """Recursively convert a report to JSON-safe primitives.

    Args:
        payload: Arbitrary nested structure possibly containing NumPy scalars.

    Returns:
        A structure safe to pass to ``JSONResponse``.
    """
    import numpy as np

    if isinstance(payload, dict):
        return {str(k): _json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_json_safe(v) for v in payload]
    if isinstance(payload, (np.integer,)):
        return int(payload)
    if isinstance(payload, (np.floating,)):
        return float(payload)
    if isinstance(payload, (np.bool_,)):
        return bool(payload)
    if isinstance(payload, Path):
        return str(payload)
    return payload


# --------------------------------------------------------------------------- #
# Held-out validation metrics (read-only)
#
# Purely additive: serves JSON already produced by the offline evaluation
# pipeline. It trains nothing, fits nothing and writes nothing, so it respects
# the inference-only contract of this API. Added because the component's own
# validation evidence -- held-out AUC, false-alarm rate and detection lead
# time -- existed on disk but was reachable from no endpoint, so the dashboard
# could show that the detector runs but not that it works.
# --------------------------------------------------------------------------- #

_EVAL_CANDIDATES = (
    "holdout_anomaly_evaluation.json",
    "holdout_test_evaluation.json",
    # The anomaly stage's own acceptance report: where flagged faults sit in
    # remaining-life terms, and how much of the raw flag volume the operational
    # gate suppresses. Both were written by the pipeline and read by nothing.
    "evaluate_anomaly.json",
    # The health stage's equivalents: its RUL-correlation/monotonicity gate, and
    # the alert-level mix over every scored row rather than the latest cycle.
    "evaluate_health.json",
    "health_alerts.json",
    # The reasoning and explainability stages: coverage, whether any hard causal
    # claim was made, narrative completeness/consistency, and the full cause and
    # subsystem distributions. `reasoning.json` carries all seven causes, where
    # the dashboard's `top_root_causes` is deliberately a top-5 widget field.
    "reasoning.json",
    "evaluate_reasoning.json",
    "evaluate_explainability.json",
)


@router.get("/evaluation/metrics")
def evaluation_metrics() -> Dict[str, Any]:
    """Return the offline held-out evaluation results for this component.

    Reads ``metrics/holdout_anomaly_evaluation.json`` (detector performance)
    and ``metrics/holdout_test_evaluation.json`` (health-index quality) if the
    offline evaluation has been run. Nothing is computed live -- these need
    ground-truth labels the API never sees.

    Returns:
        ``available`` plus whichever metric documents were found. Missing files
        are reported rather than raising, so a fresh checkout degrades to an
        empty panel instead of a 500.
    """
    metrics_dir = Path(__file__).resolve().parents[3] / "metrics"
    found: Dict[str, Any] = {}
    searched = []
    for name in _EVAL_CANDIDATES:
        path = metrics_dir / name
        searched.append(str(path))
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    found[name.replace(".json", "")] = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                found[name.replace(".json", "")] = {"error": str(exc)}

    return _json_safe(
        {
            "available": bool(found),
            "searched_paths": searched,
            "metrics": found,
        }
    )


# --------------------------------------------------------------------------- #
# Digital twin evidence (read-only)
#
# Purely additive, like the held-out metrics above: the twin's training
# summary, its residual-validation gates, its per-engine fidelity table and its
# actual-vs-expected traces all already existed on disk, but no endpoint served
# them, so the Digital Twin screen could only show operating-context statistics
# that say nothing about whether the twin works.
#
# Nothing here trains, fits or writes -- both endpoints read artefacts the
# offline pipeline produced, so the inference-only contract is preserved.
# --------------------------------------------------------------------------- #

#: The twin models one target per sensor, sensor_1 .. sensor_21.
_TWIN_SENSOR_COUNT = 21


def _load_first_json(candidates) -> Optional[Dict[str, Any]]:
    """Return the first candidate path that parses as JSON, or ``None``.

    The twin stage writes the same evaluation document to both ``metrics/``
    and ``reports/``; whichever exists is equally authoritative.

    Args:
        candidates: Paths to try, in order of preference.

    Returns:
        The parsed document, or ``None`` when no candidate is readable.
    """
    for path in candidates:
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Twin artefact %s unreadable: %s", path, exc)
    return None


@router.get("/twin/metrics")
def twin_metrics() -> Any:
    """Return the digital twin's training, validation and fidelity evidence.

    Merges three offline artefacts into one payload:

    * ``metrics/digital_twin.json`` -- per-model error and ensemble weights.
    * ``metrics/evaluate_digital_twin.json`` -- the residual acceptance gates
      (centring, spread, monotonic growth) that decide whether the residual
      stream is fit for the downstream detectors to consume.
    * ``metrics/twin_comparison.json`` -- deviation-pattern mix and the
      per-engine fidelity table.

    Returns:
        Normalised ``summary`` / ``models`` / ``validation`` / ``engines``
        blocks, or ``available: false`` when the twin stage has not been run.
    """
    try:
        metrics_dir = config.paths.metrics
        reports_dir = config.paths.reports

        training = _load_first_json((metrics_dir / "digital_twin.json",))
        validation = _load_first_json(
            (
                metrics_dir / "evaluate_digital_twin.json",
                reports_dir / "digital_twin_evaluation.json",
            )
        )
        comparison = _load_first_json((metrics_dir / "twin_comparison.json",))

        if training is None and validation is None:
            return _json_safe(
                {
                    "available": False,
                    "message": (
                        "No digital twin artefacts found. Run the offline "
                        "pipeline stage 03_digital_twin.py first."
                    ),
                }
            )

        training = training or {}
        validation = validation or {}
        comparison = comparison or {}

        per_twin = (
            training.get("per_twin_metrics")
            or validation.get("per_twin_metrics")
            or {}
        )
        weights = training.get("weights") or validation.get("ensemble_weights") or {}

        # The ensemble fuses by inverse RMSE, so the "best member" is simply the
        # lowest-RMSE twin. Reported so the UI can show what fusion had to beat.
        best_name = None
        if per_twin:
            best_name = min(
                per_twin, key=lambda k: per_twin[k].get("rmse", float("inf"))
            )

        models = []
        for name, m in per_twin.items():
            models.append(
                {
                    "name": name,
                    "rmse": m.get("rmse"),
                    "mae": m.get("mae"),
                    "weight": weights.get(name),
                    "fit_seconds": m.get("fit_seconds"),
                    "epochs_trained": m.get("epochs_trained"),
                    "train_rows": m.get("train_rows"),
                    "val_rows": m.get("val_rows"),
                    "n_features": m.get("n_features"),
                    "n_targets": m.get("n_targets"),
                    "is_best": name == best_name,
                }
            )
        models.sort(key=lambda r: (r["rmse"] is None, r["rmse"]))

        first = next(iter(per_twin.values()), {})
        rv = validation.get("residual_validation") or {}
        spread = rv.get("spread") or {}
        centring = rv.get("centring") or {}
        growth = rv.get("growth") or {}
        coverage = rv.get("coverage") or {}

        p99 = spread.get("per_sensor_p99_abs_z") or {}
        # A sensor whose p99 |z| is exactly zero is constant across the whole
        # fleet: the twin cannot model it and it carries no residual signal, so
        # it must not be offered in a sensor picker.
        informative = [s for s, v in p99.items() if v not in (None, 0, 0.0)]

        summary = {
            "rows": training.get("rows"),
            "engines": training.get("engines") or comparison.get("engines"),
            "n_twins": len(per_twin),
            "sensors_modelled": training.get("n_sensors_modelled")
            or coverage.get("expected_sensors"),
            "informative_sensors": len(informative) or None,
            "operating_modes": first.get("n_modes"),
            "n_features": first.get("n_features"),
            "train_rows": first.get("train_rows"),
            "val_rows": first.get("val_rows"),
            "weighted_mean_rmse": training.get("weighted_mean_rmse")
            or validation.get("weighted_mean_rmse"),
            "mean_twin_spread": training.get("mean_twin_spread")
            or validation.get("mean_twin_spread"),
            "best_member": best_name,
            "best_member_rmse": (
                per_twin.get(best_name, {}).get("rmse") if best_name else None
            ),
            # Pooled runs offset unit ids per subset so they never collide, which
            # leaves the UI showing opaque ids like 30249. Publishing the scheme
            # lets it render that as "FD004 #249" without hardcoding the
            # convention on the client, where it could drift from the config.
            "subsets": list(config.data.subsets),
            "unit_id_offset_step": config.data.unit_id_offset_step,
        }

        return _json_safe(
            {
                "available": True,
                "summary": summary,
                "models": models,
                "validation": {
                    "passed": rv.get("passed"),
                    "healthy_rows_checked": rv.get("healthy_rows_checked"),
                    "total_rows": rv.get("total_rows"),
                    "centring": {
                        "mean_absolute_bias": centring.get("mean_absolute_bias"),
                        "per_sensor_mean": centring.get("per_sensor_mean") or {},
                    },
                    "spread": {
                        "exceedance_fraction": spread.get("exceedance_fraction"),
                        "ceiling": spread.get("ceiling"),
                        "per_sensor_p99_abs_z": p99,
                    },
                    "growth": {
                        "engines_checked": growth.get("engines_checked"),
                        "engines_with_rising_residual": growth.get(
                            "engines_with_rising_residual"
                        ),
                        "fraction_rising": growth.get("fraction_rising"),
                    },
                    "coverage": coverage,
                    "acceptance": validation.get("acceptance") or {},
                },
                "patterns": comparison.get("pattern_distribution") or {},
                "mean_deviating_sensors": comparison.get("mean_deviating_sensors"),
                "engines": comparison.get("engine_summary") or [],
                "available_sensors": sorted(
                    informative, key=lambda s: int(s.rsplit("_", 1)[-1])
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


@lru_cache(maxsize=6)
def _twin_sensor_frame(sensor: str, _stamp: float) -> pd.DataFrame:
    """Load the fleet-wide twin trace for one sensor.

    Only four columns are pulled from each artefact, which keeps a cold read
    of the 66 MB prediction table and the 99 MB residual table around a
    second; the result is cached per sensor so repeated picks are instant.
    ``_stamp`` is the prediction file's mtime, so re-running the pipeline
    invalidates the cache instead of serving stale traces.

    Args:
        sensor: Sensor column name, e.g. ``"sensor_11"``.
        _stamp: Modification time of ``twin_predictions.csv``; cache key only.

    Returns:
        One row per (unit, cycle) with the twin's expected value, its spread
        across the four members, the baseline-centred residual and, when the
        detector has run, the anomaly flag/score for that cycle.
    """
    processed = config.paths.processed
    expected_col = f"expected_{sensor}"
    spread_col = f"twin_spread_{sensor}"
    residual_col = f"residual_{sensor}"

    # pd.read_csv rather than the shared read_csv helper: only usecols keeps a
    # cold read of these tables sub-second, and the helper does not expose it.
    preds = pd.read_csv(
        processed / "twin_predictions.csv",
        usecols=["unit", "cycle", expected_col, spread_col],
    )
    frame = preds.rename(columns={expected_col: "expected", spread_col: "spread"})

    residual_path = processed / "residuals.csv"
    if residual_path.exists():
        residuals = pd.read_csv(
            residual_path, usecols=["unit", "cycle", residual_col]
        ).rename(columns={residual_col: "residual"})
        frame = frame.merge(residuals, on=["unit", "cycle"], how="left")
    else:
        frame["residual"] = float("nan")

    # The detector is a later stage; a twin-only run must still return a trace.
    fusion_path = processed / "anomaly_fusion.csv"
    if fusion_path.exists():
        try:
            fusion = pd.read_csv(
                fusion_path,
                usecols=["unit", "cycle", "anomaly_flag", "anomaly_score"],
            )
            frame = frame.merge(fusion, on=["unit", "cycle"], how="left")
        except (ValueError, KeyError) as exc:
            logger.warning("Anomaly overlay unavailable for twin trace: %s", exc)

    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


@router.get("/twin/trace")
def twin_trace(
    engine_id: int = Query(..., description="Engine unit number."),
    sensor: str = Query("sensor_11", description="Sensor column, e.g. sensor_11."),
    max_points: int = Query(400, ge=20, le=2000),
) -> Any:
    """Return one engine's actual-vs-twin trace for a single sensor.

    This is the twin made visible: the value the twin says a *healthy* engine
    would read under the same operating conditions, the disagreement band
    across its four members, and the actual reading. Both series are expressed
    relative to the engine's own early-life baseline, so the vertical gap
    between them is exactly the residual the downstream detectors consume --
    a chart where the gap is the signal, rather than two lines whose offset is
    per-unit manufacturing scatter.

    Args:
        engine_id: Unit number to trace.
        sensor: Sensor column name; must be one the twin models.
        max_points: Cap on returned points; longer histories are strided.

    Returns:
        ``points`` plus the healthy-training window, detected onset cycle and
        the engine's fidelity summary.
    """
    try:
        valid = {f"sensor_{i}" for i in range(1, _TWIN_SENSOR_COUNT + 1)}
        if sensor not in valid:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown sensor '{sensor}'. Expected sensor_1..sensor_{_TWIN_SENSOR_COUNT}.",
            )

        preds_path = config.paths.processed / "twin_predictions.csv"
        if not preds_path.exists():
            raise ArtifactNotFoundError(
                "twin_predictions.csv not found. Run the digital twin stage "
                "before requesting a trace."
            )

        frame = _twin_sensor_frame(sensor, preds_path.stat().st_mtime)
        engine = frame[frame["unit"] == engine_id]
        if engine.empty:
            raise HTTPException(
                status_code=404, detail=f"No twin predictions for engine {engine_id}."
            )

        engine = engine.sort_values("cycle")
        last_cycle = int(engine["cycle"].max())

        # The twin trains on the healthy region only -- RUL above the cap and
        # the first `cycle_fraction` of observed life -- so the chart shades
        # that window to show the residual is flat where the model was fitted
        # and grows outside it.
        healthy_until = int(round(last_cycle * config.healthy_region.cycle_fraction))

        onset_cycle = None
        if "anomaly_flag" in engine.columns:
            flagged = engine[engine["anomaly_flag"] == 1]
            if not flagged.empty:
                onset_cycle = int(flagged["cycle"].iloc[0])

        stride = max(1, len(engine) // max_points)
        sampled = engine.iloc[::stride]
        # Keep the final cycle: it is the engine's current state and dropping
        # it to a stride boundary would understate the latest deviation.
        if sampled["cycle"].iloc[-1] != engine["cycle"].iloc[-1]:
            sampled = pd.concat([sampled, engine.iloc[[-1]]])

        points = []
        for row in sampled.itertuples(index=False):
            expected = float(getattr(row, "expected"))
            spread = float(getattr(row, "spread", 0.0) or 0.0)
            residual = getattr(row, "residual", None)
            residual = None if residual is None or pd.isna(residual) else float(residual)
            flag = getattr(row, "anomaly_flag", None)
            score = getattr(row, "anomaly_score", None)
            points.append(
                {
                    "cycle": int(getattr(row, "cycle")),
                    "expected": round(expected, 5),
                    "actual": None if residual is None else round(expected + residual, 5),
                    "band_low": round(expected - spread, 5),
                    "band_high": round(expected + spread, 5),
                    "spread": round(spread, 5),
                    "residual": None if residual is None else round(residual, 5),
                    "anomaly": None if flag is None or pd.isna(flag) else int(flag),
                    "anomaly_score": (
                        None if score is None or pd.isna(score) else round(float(score), 5)
                    ),
                }
            )

        return _json_safe(
            {
                "available": True,
                "engine_id": engine_id,
                "sensor": sensor,
                "last_cycle": last_cycle,
                "healthy_until_cycle": healthy_until,
                "onset_cycle": onset_cycle,
                "points_returned": len(points),
                "cycles_total": int(len(engine)),
                "stride": stride,
                "units": (
                    "per-mode standardised sensor units, offset-corrected to "
                    "this engine's own early-life baseline"
                ),
                "points": points,
            }
        )
    except HTTPException:
        raise
    except ArtifactNotFoundError as exc:
        return _error_response(exc, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


@router.get("/detector-config")
def detector_config() -> Any:
    """Return the fusion contract the detector actually runs with.

    The detector weights and the calibration percentile were duplicated as
    literals in the dashboard, where nothing tied them to the config that the
    fusion stage reads -- a weight retuned in the backend would have left the UI
    quietly describing the old ensemble. Serving them makes the screen a view of
    the configuration rather than a second, drifting copy of it.

    Returns:
        Per-detector weight and principle, plus the fusion calibration rule.
    """
    try:
        weights = dict(config.anomaly.fusion_weights)
        total = sum(weights.values()) or 1.0
        fusion_meta = _load_first_json([config.paths.model_path("anomaly_fusion_meta.json")]) or {}

        #: What each detector keys on. Stated here because it describes the
        #: algorithm, not a tunable, so it cannot drift from a config value.
        principles = {
            "isolation_forest": "tree-partition isolation depth",
            "mahalanobis": "Ledoit-Wolf covariance distance",
            "lstm_autoencoder": "sequence reconstruction error",
            "residual_rule": "direct residual thresholding",
            "condition_consensus": "ExtraTrees consensus over causal residual histories; trained on RUL condition proxies",
        }

        return _json_safe(
            {
                "available": True,
                "detectors": [
                    {
                        "key": name,
                        "weight": weight,
                        "weight_share": weight / total,
                        "principle": principles.get(name),
                    }
                    for name, weight in weights.items()
                ],
                "fusion": {
                    "threshold_percentile": config.anomaly.fusion_threshold_percentile,
                    "fallback_threshold": config.anomaly.fusion_threshold,
                    "normalisation": (
                        "each detector uses its persisted healthy-reference calibration "
                        "before weighting; condition consensus scales critical-condition "
                        "odds by separate-engine healthy maxima"
                    ),
                    "condition_consensus_margin": fusion_meta.get("margin", config.anomaly.condition_consensus["margin"]),
                    "max_subset_false_alarm_increase": config.anomaly.condition_consensus["max_false_alarm_increase"],
                },
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)


# --------------------------------------------------------------------------- #
# Per-engine timeline (read-only)
#
# /dashboard caps its bundled timelines at the first 50 unit ids, which -- once
# subsets are pooled and offset -- means FD001 engines 1..50 and nothing else.
# The health screen defaults to the worst-health engine, which is essentially
# never in that slice, so its main chart rendered empty. This serves any
# engine's timeline on demand and reads only the columns a trajectory needs, so
# it does not pay for the explanation text that dominates the wide table.
# --------------------------------------------------------------------------- #

#: Trajectory columns, in addition to the identity keys.
_TIMELINE_COLUMNS = (
    "health_index",
    "health_index_target",
    "health_state",
    "anomaly_score",
    "anomaly_flag",
    "confidence_score",
    "reliability_score",
    "uncertainty_score",
    "twin_spread_mean",
    "deterioration_rate",
    "operating_mode",
)


@lru_cache(maxsize=2)
def _timeline_frame(_stamp: float) -> pd.DataFrame:
    """Load just the trajectory columns of the dashboard table.

    The wide table is 216 MB, most of it generated explanation text; the dozen
    columns a chart needs are a small fraction of that. Cached on the artefact's
    mtime so every engine after the first is served without touching disk, and
    so a pipeline re-run invalidates it.

    Args:
        _stamp: Modification time of ``dashboard_data.csv``; cache key only.

    Returns:
        One row per (unit, cycle) with whichever trajectory columns exist.
    """
    path = config.paths.processed_path(config.paths.dashboard_data_csv)
    header = pd.read_csv(path, nrows=0)
    wanted = ["unit", "cycle"] + [c for c in _TIMELINE_COLUMNS if c in header.columns]
    frame = pd.read_csv(path, usecols=wanted)
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


@router.get("/engine/{engine_id}/timeline")
def engine_timeline(engine_id: int, max_points: int = Query(600, ge=20, le=5000)) -> Any:
    """Return one engine's full health trajectory.

    Args:
        engine_id: Unit number to trace.
        max_points: Cap on returned points; longer histories are strided.

    Returns:
        ``points`` ordered by cycle, plus the engine's cycle count.
    """
    try:
        path = config.paths.processed_path(config.paths.dashboard_data_csv)
        if not path.exists():
            raise ArtifactNotFoundError(
                "dashboard_data.csv not found. Run the pipeline before "
                "requesting an engine timeline."
            )

        frame = _timeline_frame(path.stat().st_mtime)
        engine = frame[frame["unit"] == engine_id]
        if engine.empty:
            raise HTTPException(
                status_code=404, detail=f"No timeline for engine {engine_id}."
            )

        stride = max(1, len(engine) // max_points)
        sampled = engine.iloc[::stride]
        # Keep the final cycle: it is the engine's current state, and dropping
        # it to a stride boundary would understate the latest deterioration.
        if sampled["cycle"].iloc[-1] != engine["cycle"].iloc[-1]:
            sampled = pd.concat([sampled, engine.iloc[[-1]]])

        points = []
        for row in sampled.to_dict(orient="records"):
            point = {"cycle": int(row["cycle"])}
            for column in _TIMELINE_COLUMNS:
                value = row.get(column)
                if value is not None and not pd.isna(value):
                    point[column] = value
            points.append(point)

        return _json_safe(
            {
                "available": True,
                "engine_id": engine_id,
                "cycles_total": int(len(engine)),
                "points_returned": len(points),
                "stride": stride,
                "points": points,
            }
        )
    except HTTPException:
        raise
    except ArtifactNotFoundError as exc:
        return _error_response(exc, status_code=404)
    except Exception as exc:  # noqa: BLE001
        return _error_response(exc, status_code=500)
