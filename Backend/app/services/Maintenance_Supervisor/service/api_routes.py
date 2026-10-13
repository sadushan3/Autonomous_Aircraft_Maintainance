"""
api_routes.py

FastAPI API Router Endpoints for the Autonomous Maintenance Supervisor Agent.

Endpoints:
- POST `/api/v1/supervisor/predict`: Single-sample streaming prediction.
- POST `/api/v1/supervisor/predict-batch`: High-throughput batch inference.
- POST `/api/v1/supervisor/explain`: Per-sample decision explanation.
- GET  `/api/v1/supervisor/baselines/compare`: Baseline models vs. champion comparison.
- GET  `/api/v1/supervisor/feedback/history`: Human engineer feedback history.
- GET  `/api/v1/supervisor/cost-avoided`: Estimated cost avoided for a given decision.
- GET  `/api/v1/supervisor/health`: Production health probe.

Federated learning was removed from this project's scope by explicit
instruction (see the master prompt's directive on this) and must not be
regenerated or re-described going forward. `federated/federated_supervisor.py`
still exists as an already-built module -- not deleted, since removing source
history is a bigger call than unwiring an endpoint -- but nothing in this
router calls it, so it is no longer reachable from the live API.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, Optional

try:
    from fastapi import APIRouter, HTTPException, status
    HAS_FASTAPI = True
except ImportError:
    APIRouter = object
    HAS_FASTAPI = False

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.services.Maintenance_Supervisor.service.schemas import (
    TelemetryInputSchema,
    SupervisorDecisionResponseSchema,
    BatchInferenceRequestSchema,
    BatchInferenceResponseSchema,
)
from app.services.Maintenance_Supervisor.service.supervisor_service import MaintenanceSupervisorService

if HAS_FASTAPI:
    router = APIRouter(prefix="/api/v1/supervisor", tags=["Maintenance Supervisor Agent"])

    @router.post("/predict", response_model=SupervisorDecisionResponseSchema)
    def predict_supervisor_decision(telemetry: TelemetryInputSchema):
        try:
            service = MaintenanceSupervisorService.get_instance()
            return service.predict_single(telemetry)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.post("/predict-batch", response_model=BatchInferenceResponseSchema)
    def predict_supervisor_batch(request: BatchInferenceRequestSchema):
        try:
            service = MaintenanceSupervisorService.get_instance()
            decisions = [service.predict_single(rec) for rec in request.records]
            return BatchInferenceResponseSchema(total_processed=len(decisions), decisions=decisions)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.post("/explain")
    def explain_supervisor_decision(telemetry: TelemetryInputSchema):
        try:
            # Previously built a throwaway, unwired SupervisorExplainer() (no
            # model/background -> silent magnitude-proxy fallback, not real
            # SHAP) and never ran decision fusion at all -- it read a
            # "final_decision" field directly off the raw telemetry input,
            # which TelemetryInputSchema doesn't have, so this always
            # explained the hardcoded default "immediate_maintenance"
            # regardless of the actual sensor reading. Routing through the
            # shared singleton fixes both: it fuses a real decision first,
            # then explains that decision with the champion model wired in.
            service = MaintenanceSupervisorService.get_instance()
            return service.explain_single(telemetry)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.get("/baselines/compare")
    def compare_baselines(input_path: Optional[str] = None):
        try:
            from app.services.Maintenance_Supervisor.evaluation.compare_baselines import run_baseline_comparison
            return run_baseline_comparison(Path(input_path) if input_path else None)
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.get("/feedback/history")
    def feedback_history(engine_id: Optional[str] = None):
        try:
            from app.services.Maintenance_Supervisor.feedback.feedback_store import SupervisorFeedbackStore
            store = SupervisorFeedbackStore()
            records = store.load_all_feedback()
            if engine_id:
                # `sample_id` is a free-form string, not a structured field, so this is a
                # simple substring match (e.g. engine_id="ENG_101" matches
                # sample_id="ENG_101_CYCLE_150"). This can produce false positives if one
                # engine_id is a substring of another (e.g. "ENG_1" matching "ENG_101_...").
                records = [r for r in records if engine_id in r.get("sample_id", "")]
            return {"total": len(records), "records": records}
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.get("/cost-avoided")
    def cost_avoided(
        decision: str,
        risk_10: Optional[float] = None,
        risk_30: Optional[float] = None,
        predicted_rul: Optional[float] = None,
    ):
        try:
            from app.services.Maintenance_Supervisor.reporting.cost_avoided import estimate_cost_avoided
            return estimate_cost_avoided(
                decision, risk_10=risk_10, risk_30=risk_30, predicted_rul=predicted_rul,
            )
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    @router.get("/health")
    def health_check():
        return {"status": "healthy", "service": "Autonomous Maintenance Supervisor Agent", "version": "2.1.0"}
else:
    router = None

