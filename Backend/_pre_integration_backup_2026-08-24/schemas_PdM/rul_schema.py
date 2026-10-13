from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.Predictive_Maintenance.common import EngineSensorRow, validate_fd_subset


class RULPredictionRequest(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    sequence: List[EngineSensorRow] = Field(..., min_length=30)
    previous_rul_predictions: List[float] = Field(default_factory=list)
    include_debug: bool = False

    @field_validator("fd_subset")
    @classmethod
    def validate_fd(cls, value: str) -> str:
        return validate_fd_subset(value)


class RULPredictionResponse(BaseModel):
    fd_subset: str
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    uncertainty_width: float
    lifecycle_state: str
    degradation_speed: float
    degradation_acceleration: float
    confidence_collapse_flag: bool
    sudden_health_drop_flag: bool
    instability_flag: bool
    model_used: str
    fallback_triggered: bool = False
    fallback_allowed: bool = False
    correction_applied: bool = False
    fallback_model_used: Optional[str] = None
    primary_predicted_rul: Optional[float] = None
    debug: Optional[Dict[str, Any]] = None
