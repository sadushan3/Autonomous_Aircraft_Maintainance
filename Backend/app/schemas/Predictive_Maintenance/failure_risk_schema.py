from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.Predictive_Maintenance.common import EngineSensorRow, validate_fd_subset


class FailureRiskPredictionRequest(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    sequence: List[EngineSensorRow] = Field(..., min_length=30)
    previous_risk_scores: List[float] = Field(default_factory=list)
    include_uncertainty: bool = False
    include_debug: bool = False

    @field_validator("fd_subset")
    @classmethod
    def validate_fd(cls, value: str) -> str:
        return validate_fd_subset(value)


class FailureRiskPredictionResponse(BaseModel):
    fd_subset: str
    risk_10: float
    risk_30: float
    risk_50: float
    uncertainty_10: float
    uncertainty_30: float
    uncertainty_50: float
    threshold_10: float
    threshold_30: float
    threshold_50: float
    predicted_fail_in_10: bool
    predicted_fail_in_30: bool
    predicted_fail_in_50: bool
    control_risk: float
    risk_velocity: float
    risk_acceleration: float
    risk_state: str
    action_hint_for_supervisor: str
    model_used: str
    debug: Optional[Dict[str, Any]] = None
