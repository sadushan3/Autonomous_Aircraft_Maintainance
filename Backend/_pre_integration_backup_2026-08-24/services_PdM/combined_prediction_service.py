from __future__ import annotations

from typing import Any, Dict, Optional

from app.schemas.Predictive_Maintenance.combined_schema import (
    CombinedPredictionRequest,
    CombinedPredictionResponse,
    SupervisorPreDecision,
)
from app.schemas.Predictive_Maintenance.failure_risk_schema import FailureRiskPredictionRequest
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionRequest
from app.services.Predictive_Maintenance.failure_risk_service import FailureRiskService
from app.services.Predictive_Maintenance.memory_service import MemoryService
from app.services.Predictive_Maintenance.rul_lifecycle_service import RULLifecycleService


class CombinedPredictionService:
    def __init__(
        self,
        rul_service: RULLifecycleService,
        failure_risk_service: FailureRiskService,
        memory_service: Optional[MemoryService] = None,
    ):
        self.rul_service = rul_service
        self.failure_risk_service = failure_risk_service
        self.memory_service = memory_service

    def predict(self, request: CombinedPredictionRequest) -> CombinedPredictionResponse:
        rul_response = self.rul_service.predict(
            RULPredictionRequest(
                fd_subset=request.fd_subset,
                engine_id=request.engine_id,
                sequence=request.sequence,
                previous_rul_predictions=request.previous_rul_predictions,
                include_debug=request.include_debug,
            )
        )
        failure_response = self.failure_risk_service.predict(
            FailureRiskPredictionRequest(
                fd_subset=request.fd_subset,
                engine_id=request.engine_id,
                sequence=request.sequence,
                previous_risk_scores=request.previous_risk_scores,
                include_uncertainty=request.include_uncertainty,
                include_debug=request.include_debug,
            )
        )

        cycle = max(row.cycle for row in request.sequence)
        pre_decision = self._supervisor_pre_decision(
            rul_response.lifecycle_state,
            failure_response.risk_state,
        )

        debug: Optional[Dict[str, Any]] = None
        if request.include_debug:
            debug = {
                "note": "Supervisor pre-decision is rule-based decision support, not the RL supervisor.",
            }

        if self.memory_service is not None:
            self.memory_service.record(
                engine_id=request.engine_id,
                fd_subset=request.fd_subset,
                cycle=cycle,
                predicted_rul=rul_response.predicted_rul,
                degradation_speed=rul_response.degradation_speed,
                degradation_acceleration=rul_response.degradation_acceleration,
                risk_30=failure_response.risk_30,
                lifecycle_state=rul_response.lifecycle_state,
                risk_state=failure_response.risk_state,
            )

        return CombinedPredictionResponse(
            engine_id=request.engine_id,
            fd_subset=request.fd_subset,
            cycle=cycle,
            rul_agent=rul_response,
            failure_risk_agent=failure_response,
            supervisor_pre_decision=pre_decision,
            debug=debug,
        )

    @staticmethod
    def _supervisor_pre_decision(
        rul_state: str, risk_state: str
    ) -> SupervisorPreDecision:
        if rul_state == "Critical" or risk_state == "Critical":
            return SupervisorPreDecision(
                overall_health_state=f"{rul_state}_{risk_state}_Risk",
                priority="High",
                recommendation="schedule_maintenance_review",
                reason="RUL or failure-risk agent indicates a critical engine health condition.",
            )
        if rul_state in {"Accelerating", "Unstable"} or risk_state == "Accelerating":
            return SupervisorPreDecision(
                overall_health_state=f"{rul_state}_{risk_state}_Risk",
                priority="Medium-High",
                recommendation="increase_monitoring_and_prepare_inspection",
                reason="Degradation or failure risk is accelerating and needs closer supervision.",
            )
        if rul_state == "Degrading" or risk_state == "Rising":
            return SupervisorPreDecision(
                overall_health_state=f"{rul_state}_{risk_state}_Risk",
                priority="Medium",
                recommendation="monitor_closely",
                reason="RUL indicates degradation or 30-cycle failure risk is rising.",
            )
        return SupervisorPreDecision(
            overall_health_state=f"{rul_state}_{risk_state}_Risk",
            priority="Low",
            recommendation="continue_operation",
            reason="Both AI components indicate stable health under current decision-support rules.",
        )
