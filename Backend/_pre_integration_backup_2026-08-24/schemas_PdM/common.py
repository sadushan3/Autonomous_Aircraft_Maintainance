from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


SUPPORTED_FDS = {"FD001", "FD002", "FD003", "FD004"}


def validate_fd_subset(value: str) -> str:
    normalized = value.upper()
    if normalized not in SUPPORTED_FDS:
        raise ValueError("fd_subset must be one of FD001, FD002, FD003, FD004")
    return normalized


class EngineSensorRow(BaseModel):
    cycle: int = Field(..., ge=1)
    unit: Optional[int] = None
    fd_subset: Optional[str] = None
    op_setting_1: float
    op_setting_2: float
    op_setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float
    op_cluster: Optional[int] = None

    @field_validator("fd_subset")
    @classmethod
    def validate_optional_fd(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        return validate_fd_subset(value)


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ArtifactItemStatus(BaseModel):
    fd_subset: str
    loaded: bool
    missing_model: bool = False
    missing_scaler: bool = False
    missing_feature_columns: bool = False
    missing_metadata: bool = False
    missing: List[str] = Field(default_factory=list)
    paths: Dict[str, Optional[str]] = Field(default_factory=dict)
    searched_locations: List[str] = Field(default_factory=list)


class ArtifactStatusResponse(BaseModel):
    rul_lifecycle: List[ArtifactItemStatus]
    failure_risk: List[ArtifactItemStatus]


class ArtifactConfigResponse(BaseModel):
    artifact_root: str
    supported_fds: List[str]
    failure_risk_window_size: int
    rul_window_sizes: Dict[str, int]
    failure_risk_mc_passes: int
    default_rul_q_hat: Dict[str, float]
    rul_uncertainty_width_threshold: Dict[str, float]
