from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.Predictive_Maintenance.common import EngineSensorRow, validate_fd_subset


class RULPredictionRequest(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    sequence: List[EngineSensorRow] = Field(..., min_length=30)
    previous_rul_predictions: List[float] = Field(default_factory=list)
    #: Cycles elapsed between consecutive ``previous_rul_predictions`` entries
    #: (and between the last of them and the window being scored now).
    #:
    #: Degradation speed is a rate, so it only means anything once the spacing
    #: is known. A batch that sub-samples every 20th cycle hands over readings
    #: 20 cycles apart, and a normal ~1 cycle/cycle decline then looks like a
    #: drop of 20 -- tripping the ``sudden_health_drop_flag`` threshold of 5.0
    #: on ordinary wear. Defaults to 1.0, which is the per-cycle case and
    #: reproduces the previous behaviour for callers that scored consecutively.
    prediction_cycle_gap: float = Field(default=1.0, gt=0.0)
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
