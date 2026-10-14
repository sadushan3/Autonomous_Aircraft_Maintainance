from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.Predictive_Maintenance.common import EngineSensorRow, validate_fd_subset
from app.schemas.Predictive_Maintenance.failure_risk_schema import FailureRiskPredictionResponse
from app.schemas.Predictive_Maintenance.rul_schema import RULPredictionResponse


class CombinedPredictionRequest(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    sequence: List[EngineSensorRow] = Field(..., min_length=30)
    previous_rul_predictions: List[float] = Field(default_factory=list)
    previous_risk_scores: List[float] = Field(default_factory=list)
    include_uncertainty: bool = False
    include_debug: bool = False

    @field_validator("fd_subset")
    @classmethod
    def validate_fd(cls, value: str) -> str:
        return validate_fd_subset(value)


class SupervisorPreDecision(BaseModel):
    overall_health_state: str
    priority: str
    recommendation: str
    reason: str


class CombinedPredictionResponse(BaseModel):
    engine_id: Optional[str] = None
    fd_subset: str
    cycle: int
    rul_agent: RULPredictionResponse
    failure_risk_agent: FailureRiskPredictionResponse
    supervisor_pre_decision: SupervisorPreDecision
    debug: Optional[Dict[str, Any]] = None
