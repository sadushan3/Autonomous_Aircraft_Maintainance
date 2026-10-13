from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.schemas.Predictive_Maintenance.common import EngineSensorRow, validate_fd_subset


class TrajectoryRequest(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    sequence: List[EngineSensorRow] = Field(..., min_length=30)
    stride: Optional[int] = Field(default=None, ge=1)
    max_points: int = Field(default=50, ge=5, le=200)

    @field_validator("fd_subset")
    @classmethod
    def validate_fd(cls, value: str) -> str:
        return validate_fd_subset(value)


class TrajectoryPoint(BaseModel):
    cycle: int
    predicted_rul: float
    lower_bound: float
    upper_bound: float
    uncertainty_width: float
    degradation_speed: float
    degradation_acceleration: float
    lifecycle_state: str
    confidence_collapse_flag: bool
    correction_applied: bool
    risk_10: float
    risk_30: float
    risk_50: float
    risk_state: str


class TrajectoryResponse(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    stride: int
    window_start_cycle: int
    points: List[TrajectoryPoint]
