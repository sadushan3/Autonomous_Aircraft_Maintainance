from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class StageStatus(str, Enum):


    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class HealthState(str, Enum):


    HEALTHY = "Healthy"
    DEGRADING = "Degrading"
    WARNING = "Warning"
    CRITICAL = "Critical"
    FAILURE_RISK = "Failure Risk"


class Severity(str, Enum):


    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):


    NONE = "none"
    FAULT = "fault"
    OPERATIONAL = "operational"


class AlertLevel(str, Enum):


    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class EvolutionPattern(str, Enum):


    STABLE = "stable"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    INTERMITTENT = "intermittent"


class FeedbackVerdict(str, Enum):


    ACCEPTED = "accepted"
    FALSE_ALARM = "false_alarm"
    MISSED = "missed"
    UNKNOWN = "unknown"


class StrictModel(BaseModel):


    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",
        protected_namespaces=(),
    )


class EngineCycleKey(StrictModel):


    engine_id: int = Field(..., description="Engine identifier (alias of 'unit').")
    cycle: int = Field(..., ge=1, description="Operational cycle number.")


class BaseResponse(StrictModel):


    status: StageStatus = Field(
        default=StageStatus.SUCCESS, description="Outcome of the request."
    )
    message: str = Field(default="", description="Human-readable summary.")
    timestamp: str = Field(default="", description="UTC ISO-8601 completion time.")


class ErrorResponse(StrictModel):


    status: StageStatus = Field(default=StageStatus.FAILED)
    error_type: str = Field(..., description="Exception class name.")
    message: str = Field(..., description="What went wrong.")
    stage: Optional[str] = Field(default=None, description="Stage that failed.")
    hint: Optional[str] = Field(
        default=None, description="Suggested remedy, e.g. which stage to run first."
    )


class StageResponse(BaseResponse):


    stage: str = Field(..., description="Stage identifier, e.g. '03_digital_twin'.")
    rows_in: int = Field(default=0, ge=0, description="Rows read from the input CSV.")
    rows_out: int = Field(default=0, ge=0, description="Rows written to the output CSV.")
    engines: int = Field(default=0, ge=0, description="Distinct engines processed.")
    output_paths: List[str] = Field(
        default_factory=list, description="Absolute paths of artefacts written."
    )
    duration_seconds: float = Field(default=0.0, ge=0.0)
    metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Stage-specific metrics."
    )


class PipelineResponse(BaseResponse):


    stages: List[StageResponse] = Field(
        default_factory=list, description="Per-stage results in execution order."
    )
    total_duration_seconds: float = Field(default=0.0, ge=0.0)
    exchange_file: Optional[str] = Field(
        default=None,
        description="Path of the integration handoff CSV consumed by the "
        "Maintenance Supervisor.",
    )
    summary: Dict[str, Any] = Field(
        default_factory=dict, description="Headline numbers across the run."
    )


class PaginationParams(StrictModel):


    limit: int = Field(default=500, ge=1, le=100_000, description="Maximum rows.")
    offset: int = Field(default=0, ge=0, description="Rows to skip.")
    engine_id: Optional[int] = Field(
        default=None, description="Restrict results to a single engine."
    )


class MetricsPayload(StrictModel):


    name: str = Field(..., description="Metric group name, e.g. 'digital_twin'.")
    values: Dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default="", description="UTC ISO-8601 timestamp.")
