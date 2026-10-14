"""
FastAPI router for the Predictive Maintenance component (RUL Lifecycle +
Failure Risk + their combined/supervisor-facing prediction endpoint).

Ported from the standalone predictive-maintenance-backend service. That
service wired its services onto ``app.state`` inside its own
``main.py::create_app()``; here the same objects are built once as
module-level singletons (mirroring that wiring exactly) since this router is
mounted into a shared FastAPI app rather than owning its own app factory.

Only the RUL, Failure Risk, Combined-prediction and Artifact-status endpoints
are ported. Trajectory, threshold-feedback, memory-inspection and evaluation
endpoints from the source service are intentionally out of scope for this
port -- see the integration report for details.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

from app.config.Predictive_Maintenance.config import get_settings
from app.schemas.Predictive_Maintenance.combined_schema import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
)
from app.schemas.Predictive_Maintenance.common import ArtifactStatusResponse
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
