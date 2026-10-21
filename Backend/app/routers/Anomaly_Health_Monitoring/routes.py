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


@router.get("/schema", response_model=SchemaDescription)
def get_schema() -> SchemaDescription:


    return SchemaDescription(**describe_schema())


@router.post("/upload", response_model=UploadResponse)
async def upload_data(file: UploadFile = File(...)) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/preprocess", response_model=PreprocessResponse)
def preprocess(request: PreprocessRequest) -> Any:


    try:
        loader = DataLoader(config)
        raw_frame, load_report = loader.run(
            subsets=request.subsets,
            split=request.split,
            source_csv=request.source_csv,
            max_engines=request.max_engines,
        )


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/train-digital-twin", response_model=TwinTrainingResponse)
def train_digital_twin(request: TwinTrainingRequest) -> Any:


    try:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/generate-health-index", response_model=HealthIndexResponse)
def generate_health_index() -> Any:


    try:
        calculator = HealthIndexCalculator(config)


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/generate-health-score", response_model=HealthIndexResponse)
def generate_health_score() -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/detect-anomalies", response_model=AnomalyDetectionResponse)
def detect_anomalies(request: AnomalyDetectionRequest) -> Any:


    try:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/root-cause-analysis", response_model=ReasoningResponse)
def root_cause_analysis(only_anomalies: bool = Query(default=True)) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/confidence", response_model=ConfidenceResponse)
def confidence() -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(submission: FeedbackSubmission) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.get("/reasoning/sensor-graph", response_model=DependencyGraphResponse)
def sensor_dependency_graph() -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


def _parse_stringified_column(value: Any) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.get("/feedback/summary", response_model=FeedbackSummaryResponse)
def feedback_summary() -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.get("/feedback/thresholds", response_model=ThresholdStateResponse)
def feedback_thresholds() -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@lru_cache(maxsize=2)
def _dashboard_payload_cached(_stamp: float) -> Dict[str, Any]:


    return DashboardApiService(config).dashboard_payload()


@router.get("/dashboard", response_model=DashboardResponse)
def dashboard(engine_id: Optional[int] = Query(default=None)) -> Any:


    try:
        api_service = DashboardApiService(config)


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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.post("/full-pipeline")
def full_pipeline(
    include_shap: bool = Query(default=True),
) -> Any:


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
        spec.loader.exec_module(module)

        report = module.run_full_pipeline(config, retrain=False, include_shap=include_shap)
        status_code = 200 if report["status"] == "success" else 500
        return JSONResponse(status_code=status_code, content=_json_safe(report))
    except Exception as exc:
        return _error_response(exc, status_code=500)


def _json_safe(payload: Any) -> Any:


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


_EVAL_CANDIDATES = (
    "holdout_anomaly_evaluation.json",
    "holdout_test_evaluation.json",


    "evaluate_anomaly.json",


    "evaluate_health.json",
    "health_alerts.json",


    "reasoning.json",
    "evaluate_reasoning.json",
    "evaluate_explainability.json",
)


@router.get("/evaluation/metrics")
def evaluation_metrics() -> Dict[str, Any]:


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


_TWIN_SENSOR_COUNT = 21


def _load_first_json(candidates) -> Optional[Dict[str, Any]]:


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
        recipe = _load_first_json((config.paths.models / "startup_profile_recipe.json",)) or {}

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
        from app.services.Anomaly_Health_Monitoring.dashboard.forecast_metrics import active_forecast_metrics
        forecast = active_forecast_metrics(training, validation, recipe)

        per_twin = (
            training.get("per_twin_metrics")
            or validation.get("per_twin_metrics")
            or {}
        )
        weights = training.get("weights") or validation.get("ensemble_weights") or {}


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


            "subsets": list(config.data.subsets),
            "unit_id_offset_step": config.data.unit_id_offset_step,
        }

        return _json_safe(
            {
                "available": True,
                "summary": summary,
                "forecast": forecast,
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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@lru_cache(maxsize=6)
def _twin_sensor_frame(sensor: str, _stamp: float) -> pd.DataFrame:


    processed = config.paths.processed
    expected_col = f"expected_{sensor}"
    spread_col = f"twin_spread_{sensor}"
    residual_col = f"residual_{sensor}"


    prediction_path = processed / "twin_predictions.csv"
    reference_col = f"reference_expected_{sensor}"
    prediction_columns = pd.read_csv(prediction_path, nrows=0).columns
    has_reference = reference_col in prediction_columns
    columns = ["unit", "cycle", reference_col if has_reference else expected_col, spread_col]
    if "startup_profile_ready" in prediction_columns:
        columns.append("startup_profile_ready")
    preds = pd.read_csv(prediction_path, usecols=columns)
    frame = preds.rename(columns={reference_col if has_reference else expected_col: "expected", spread_col: "spread"})
    frame["forecast"] = float("nan")

    residual_path = processed / "residuals.csv"
    if residual_path.exists():
        prediction_residual_col = f"prediction_residual_{sensor}"
        residual_columns = pd.read_csv(residual_path, nrows=0).columns
        columns = ["unit", "cycle", residual_col]
        if prediction_residual_col in residual_columns:
            columns.append(prediction_residual_col)
        residuals = pd.read_csv(residual_path, usecols=columns).rename(columns={residual_col: "residual"})
        frame = frame.merge(residuals, on=["unit", "cycle"], how="left")
        if has_reference and prediction_residual_col in frame:
            frame["forecast"] = frame["expected"] + frame["residual"] - frame[prediction_residual_col]
            if "startup_profile_ready" in frame:
                frame.loc[~frame["startup_profile_ready"].isin([True, 1, "True", "true", "1"]), "forecast"] = float("nan")
    else:
        frame["residual"] = float("nan")


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


        healthy_until = int(round(last_cycle * config.healthy_region.cycle_fraction))

        onset_cycle = None
        if "anomaly_flag" in engine.columns:
            flagged = engine[engine["anomaly_flag"] == 1]
            if not flagged.empty:
                onset_cycle = int(flagged["cycle"].iloc[0])

        stride = max(1, len(engine) // max_points)
        sampled = engine.iloc[::stride]


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
            forecast = getattr(row, "forecast", None)
            points.append(
                {
                    "cycle": int(getattr(row, "cycle")),
                    "expected": round(expected, 5),
                    "actual": None if residual is None else round(expected + residual, 5),
                    "forecast": None if forecast is None or pd.isna(forecast) else round(float(forecast), 5),
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
    except Exception as exc:
        return _error_response(exc, status_code=500)


@router.get("/detector-config")
def detector_config() -> Any:


    try:
        fusion_meta = _load_first_json([config.paths.model_path("anomaly_fusion_meta.json")]) or {}
        weights = fusion_meta.get("fusion_weights") or dict(config.anomaly.fusion_weights)
        total = sum(weights.values()) or 1.0
        report = _load_first_json([config.paths.metrics / "anomaly_detection.json"]) or {}
        trained = fusion_meta.get("detectors") or report.get("detector_summary") or {}


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
                        "enabled": weight > 0,
                        "trained": name in trained and "error" not in trained[name],
                        "threshold": trained.get(name, {}).get("threshold"),
                    }
                    for name, weight in weights.items()
                ],
                "fusion": {
                    "active_detectors": sum(weight > 0 and name in trained and "error" not in trained[name]
                                            for name, weight in weights.items()),
                    "calibrated_threshold": fusion_meta.get("calibrated_fusion_threshold"),
                    "threshold_percentile": config.anomaly.fusion_threshold_percentile,
                    "fallback_threshold": config.anomaly.fusion_threshold,
                    "normalisation": (
                        "Isolation Forest, LSTM and Residual Rule use healthy-reference percentile ranks; "
                        "Mahalanobis uses its saved rational distance calibration. Enabled scores are "
                        "combined using the persisted weights and a cutoff fitted on separate training engines."
                    ),
                    "condition_consensus_margin": fusion_meta.get("margin", config.anomaly.condition_consensus["margin"]),
                    "max_subset_false_alarm_increase": config.anomaly.condition_consensus["max_false_alarm_increase"],
                },
            }
        )
    except Exception as exc:
        return _error_response(exc, status_code=500)


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


    path = config.paths.processed_path(config.paths.dashboard_data_csv)
    header = pd.read_csv(path, nrows=0)
    wanted = ["unit", "cycle"] + [c for c in _TIMELINE_COLUMNS if c in header.columns]
    frame = pd.read_csv(path, usecols=wanted)
    return frame.sort_values(["unit", "cycle"]).reset_index(drop=True)


@router.get("/engine/{engine_id}/timeline")
def engine_timeline(engine_id: int, max_points: int = Query(600, ge=20, le=5000)) -> Any:


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
    except Exception as exc:
        return _error_response(exc, status_code=500)
