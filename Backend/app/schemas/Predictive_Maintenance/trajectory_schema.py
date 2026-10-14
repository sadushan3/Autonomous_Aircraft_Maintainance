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
    include_risk: bool = Field(
        default=True,
        description=(
            "Also replay the Failure Risk agent. Set false for a RUL-only "
            "trajectory: the RUL Lifecycle module then never invokes the "
            "Failure Risk model, keeping the two agents independent (and "
            "skipping one BiLSTM forward pass per point)."
        ),
    )

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
    # Owned by the Failure Risk agent, not the RUL Lifecycle agent. Null when
    # the caller asked for a RUL-only replay (``include_risk=false``).
    risk_10: Optional[float] = None
    risk_30: Optional[float] = None
    risk_50: Optional[float] = None
    risk_state: Optional[str] = None


class TrajectoryResponse(BaseModel):
    fd_subset: str
    engine_id: Optional[str] = None
    stride: int
    window_start_cycle: int
    points: List[TrajectoryPoint]
