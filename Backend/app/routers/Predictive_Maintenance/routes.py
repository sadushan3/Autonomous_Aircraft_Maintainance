"""
FastAPI router for the Predictive Maintenance component (RUL Lifecycle +
Failure Risk + their combined/supervisor-facing prediction endpoint).

Ported from the standalone predictive-maintenance-backend service. That
service wired its services onto ``app.state`` inside its own
``main.py::create_app()``; here the same objects are built once as
module-level singletons (mirroring that wiring exactly) since this router is
mounted into a shared FastAPI app rather than owning its own app factory.

The full endpoint surface of the source service is ported: RUL, Failure Risk,
Combined prediction, Artifact status/config, Memory, Threshold feedback,
Trajectory, Evaluation metrics and the two demo routes. Paths are prefixed with
``/predictive-maintenance`` rather than the source service's ``/api``, so the
source Postman collection needs its ``baseUrl`` adjusted accordingly (see
``postman/`` alongside this component).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import json

from fastapi import APIRouter, Query

from app.config.Predictive_Maintenance.config import get_settings
from app.schemas.Predictive_Maintenance.combined_schema import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
)
from app.schemas.Predictive_Maintenance.common import (
    ArtifactConfigResponse,
    ArtifactStatusResponse,
    HealthResponse,
)
from app.schemas.Predictive_Maintenance.failure_risk_schema import (
    FailureRiskPredictionRequest,
    FailureRiskPredictionResponse,
)
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionRequest, RULPredictionResponse
from app.schemas.Predictive_Maintenance.threshold_schema import ThresholdFeedbackRequest
from app.schemas.Predictive_Maintenance.trajectory_schema import TrajectoryRequest, TrajectoryResponse
from app.services.Predictive_Maintenance.adaptive_threshold_service import AdaptiveThresholdService
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader
from app.services.Predictive_Maintenance.combined_prediction_service import CombinedPredictionService
from app.services.Predictive_Maintenance.failure_risk_service import FailureRiskService
from app.services.Predictive_Maintenance.memory_service import MemoryService
from app.services.Predictive_Maintenance.preprocessing import PreprocessingService
from app.services.Predictive_Maintenance.rul_lifecycle_service import RULLifecycleService
from app.services.Predictive_Maintenance.trajectory_service import TrajectoryService

# --------------------------------------------------------------------------- #
# Module-level service singletons -- mirrors create_app() in the source
# service's main.py, just built once here instead of on app.state.
# --------------------------------------------------------------------------- #
_settings = get_settings()

_artifact_loader = ArtifactLoader(_settings)
_preprocessing = PreprocessingService()
_adaptive_threshold_service = AdaptiveThresholdService(
    Path(__file__).resolve().parents[3] / "data" / "exchange" / "Predictive_Maintenance" / "adaptive_thresholds.json",
    _settings.FAILURE_RISK_DEFAULT_THRESHOLDS,
)
_rul_service = RULLifecycleService(_settings, _artifact_loader, _preprocessing)
_failure_risk_service = FailureRiskService(
    _settings, _artifact_loader, _preprocessing, _adaptive_threshold_service
)
_memory_service = MemoryService(
    Path(__file__).resolve().parents[3] / "data" / "exchange" / "Predictive_Maintenance" / "memory_store.json"
)
_combined_service = CombinedPredictionService(_rul_service, _failure_risk_service, _memory_service)
_trajectory_service = TrajectoryService(_settings, _rul_service, _failure_risk_service)

router = APIRouter(prefix="/predictive-maintenance", tags=["Predictive Maintenance"])


@router.post("/rul/predict", response_model=RULPredictionResponse)
def predict_rul(payload: RULPredictionRequest) -> RULPredictionResponse:
    return _rul_service.predict(payload)


@router.post("/failure-risk/predict", response_model=FailureRiskPredictionResponse)
def predict_failure_risk(payload: FailureRiskPredictionRequest) -> FailureRiskPredictionResponse:
    return _failure_risk_service.predict(payload)


@router.post("/predict/combined", response_model=CombinedPredictionResponse)
def predict_combined(payload: CombinedPredictionRequest) -> CombinedPredictionResponse:
    return _combined_service.predict(payload)


@router.get("/artifacts/status", response_model=ArtifactStatusResponse)
def artifact_status() -> ArtifactStatusResponse:
    return ArtifactStatusResponse(**_artifact_loader.status())


@router.get("/memory/status")
def memory_status() -> Dict[str, Any]:
    return _memory_service.status()


@router.get("/memory/similar")
def memory_similar(
    fd_subset: str,
    degradation_speed: float = 0.0,
    degradation_acceleration: float = 0.0,
    risk_30: float = 0.0,
    engine_id: Optional[str] = None,
    limit: int = Query(default=5, ge=1, le=20),
) -> Dict[str, Any]:
    fd_subset = _artifact_loader.validate_fd_subset(fd_subset)
    matches = _memory_service.similar(
        fd_subset=fd_subset,
        degradation_speed=degradation_speed,
        degradation_acceleration=degradation_acceleration,
        risk_30=risk_30,
        exclude_engine_id=engine_id,
        limit=limit,
    )
    return {"fd_subset": fd_subset, "matches": matches}


@router.get("/threshold/status")
def threshold_status(fd_subset: str = Query(...)) -> Dict[str, Any]:
    fd_subset = _artifact_loader.validate_fd_subset(fd_subset)
    return {
        "fd_subset": fd_subset,
        "horizons": _adaptive_threshold_service.status(fd_subset),
    }


@router.post("/threshold/feedback")
def threshold_feedback(payload: ThresholdFeedbackRequest) -> Dict[str, Any]:
    # Seed adaptation from the threshold currently being predicted against --
    # the model-calibrated value in the artifact metadata -- so the first
    # verdict nudges that value rather than discarding it for a flat default.
    seed = None
    try:
        artifacts = _artifact_loader.load("failure_risk", payload.fd_subset)
        thresholds = (artifacts.metadata or {}).get("thresholds") or {}
        raw = thresholds.get(payload.horizon, thresholds.get(int(payload.horizon)))
        if raw is not None:
            seed = float(raw)
    except Exception:  # noqa: BLE001
        # No metadata (or an unloadable artifact) just means the service falls
        # back to its configured default, which is the previous behaviour.
        seed = None

    return _adaptive_threshold_service.record_feedback(
        fd_subset=payload.fd_subset,
        horizon=payload.horizon,
        was_false_alarm=payload.was_false_alarm,
        seed=seed,
    )


@router.post("/predict/trajectory", response_model=TrajectoryResponse)
def predict_trajectory(payload: TrajectoryRequest) -> TrajectoryResponse:
    return _trajectory_service.compute(payload)


# --------------------------------------------------------------------------- #
# Endpoints ported from the source service's health_router, artifact_router,
# evaluation_router and combined_router (demo half). Behaviour is unchanged;
# only the dependency wiring is adapted -- the source resolved these from
# ``request.app.state``, which does not exist when this router is mounted into
# the shared app, so they read the module-level singletons above instead.
# --------------------------------------------------------------------------- #

_COMPONENT_ROOT = Path(__file__).resolve().parents[3]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service="predictive-maintenance-backend",
        version=_settings.APP_VERSION,
    )


@router.get("/artifacts/config", response_model=ArtifactConfigResponse)
def artifact_config() -> ArtifactConfigResponse:
    return ArtifactConfigResponse(
        artifact_root=_settings.ARTIFACT_ROOT,
        supported_fds=_settings.SUPPORTED_FDS,
        failure_risk_window_size=_settings.FAILURE_RISK_WINDOW_SIZE,
        rul_window_sizes=_settings.RUL_WINDOW_SIZES,
        failure_risk_mc_passes=_settings.FAILURE_RISK_MC_PASSES,
        default_rul_q_hat=_settings.DEFAULT_RUL_Q_HAT,
        rul_uncertainty_width_threshold=_settings.RUL_UNCERTAINTY_WIDTH_THRESHOLD,
    )


@router.get("/evaluation/metrics")
def evaluation_metrics() -> Dict[str, Any]:
    """Serves offline model-comparison metrics (MAE/RMSE/R2/NASA score/PICP/MPIW)
    exported from the training notebook, if present.

    Nothing in this backend computes these numbers live -- they require
    ground-truth RUL labels the API never sees. Drop a JSON file shaped like
    {"models": [{"name": "...", "mae": ..., "rmse": ..., "r2": ...,
    "nasa_score": ..., "picp": ..., "mpiw": ...}, ...]} at one of the candidate
    paths to make this endpoint (and the dashboard panel) live.

    Neither shipped zip contained this file, so the endpoint reports
    ``available: false`` until the RUL/Failure-Risk team exports it.
    """
    candidates = [
        _COMPONENT_ROOT / "metrics" / "Predictive_Maintenance" / "evaluation_metrics.json",
        _COMPONENT_ROOT / "models" / "Predictive_Maintenance" / "evaluation_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
            return {"available": True, "source": str(path), "data": data}

    return {
        "available": False,
        "source": None,
        "data": None,
        "candidate_paths": [str(path) for path in candidates],
    }


@router.get("/demo/sample-request")
def sample_request() -> Dict[str, Any]:
    sample_path = (
        _COMPONENT_ROOT / "sample_data" / "Predictive_Maintenance" / "sample_combined_request.json"
    )
    with sample_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not payload.get("sequence"):
        payload["sequence"] = build_demo_sequence(length=50)
    return payload


@router.post("/demo/predict", response_model=CombinedPredictionResponse)
def demo_predict() -> CombinedPredictionResponse:
    payload = CombinedPredictionRequest(**sample_request())
    return _combined_service.predict(payload)


def build_demo_sequence(length: int = 50) -> list[dict[str, Any]]:
    """Generate a stable FD001-like sequence for local demos and frontend wiring."""
    rows: list[dict[str, Any]] = []
    for cycle in range(1, length + 1):
        row: dict[str, Any] = {
            "cycle": cycle,
            "unit": 1,
            "op_setting_1": round(-0.0007 + cycle * 0.00001, 6),
            "op_setting_2": round(-0.0004 + cycle * 0.000005, 6),
            "op_setting_3": 100.0,
            "sensor_1": 518.67,
            "sensor_2": round(642.40 + cycle * 0.012, 4),
            "sensor_3": round(1587.90 + cycle * 0.025, 4),
            "sensor_4": round(1400.80 + cycle * 0.035, 4),
            "sensor_5": 14.62,
            "sensor_6": 21.61,
            "sensor_7": round(554.10 - cycle * 0.01, 4),
            "sensor_8": 2388.04,
            "sensor_9": round(9048.20 + cycle * 0.05, 4),
            "sensor_10": 1.30,
            "sensor_11": round(47.10 + cycle * 0.004, 4),
            "sensor_12": round(521.90 - cycle * 0.012, 4),
            "sensor_13": 2388.03,
            "sensor_14": round(8124.50 + cycle * 0.08, 4),
            "sensor_15": round(8.4000 + cycle * 0.0004, 4),
            "sensor_16": 0.03,
            "sensor_17": 392.0,
            "sensor_18": 2388.0,
            "sensor_19": 100.0,
            "sensor_20": round(39.10 - cycle * 0.002, 4),
            "sensor_21": round(23.3200 - cycle * 0.001, 4),
        }
        rows.append(row)
    return rows
