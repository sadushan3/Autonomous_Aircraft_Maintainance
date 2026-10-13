from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

from app.schemas.Predictive_Maintenance.common import validate_fd_subset


class ThresholdFeedbackRequest(BaseModel):
    fd_subset: str
    horizon: Literal["10", "30", "50"]
    was_false_alarm: bool

    @field_validator("fd_subset")
    @classmethod
    def validate_fd(cls, value: str) -> str:
        return validate_fd_subset(value)
