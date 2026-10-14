"""
Shared Pydantic models and enumerations for the component's API surface.

Everything here is transport-level: enums that pin down the vocabulary used
across stages, a common response envelope, and the ``(engine_id, cycle)``
integration key.  Domain logic lives in the service layer, never here.

Note on the key naming convention: internally the component always uses
``unit``.  The name ``engine_id`` appears only at the API / handoff boundary,
which is exactly what these schemas represent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

class StageStatus(str, Enum):
    """Outcome of a pipeline stage."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class HealthState(str, Enum):
    """The five-level health state ladder.

    Ordered from best to worst; the ordering is meaningful and is relied upon
    by the trend tracker and the alert engine.
    """

    HEALTHY = "Healthy"
    DEGRADING = "Degrading"
    WARNING = "Warning"
    CRITICAL = "Critical"
    FAILURE_RISK = "Failure Risk"


class Severity(str, Enum):
    """Anomaly severity bands."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AnomalyType(str, Enum):
    """Whether a detection reflects a fault or merely an operating change.

    Separating these two is the core novelty requirement: a regime change in
    FD002/FD004 must not be reported as a fault.
    """

    NONE = "none"
    FAULT = "fault"
    OPERATIONAL = "operational"


class AlertLevel(str, Enum):
    """Operator-facing alert escalation levels."""

    NONE = "none"
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class EvolutionPattern(str, Enum):
    """How an anomaly developed over time."""

    STABLE = "stable"
    GRADUAL = "gradual"
    ABRUPT = "abrupt"
    INTERMITTENT = "intermittent"


class FeedbackVerdict(str, Enum):
    """Operator judgement on a raised alert."""

    ACCEPTED = "accepted"
    FALSE_ALARM = "false_alarm"
    MISSED = "missed"
    UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Base models
# --------------------------------------------------------------------------- #

class StrictModel(BaseModel):
    """Base model with a consistent configuration for the whole component."""

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",
        protected_namespaces=(),
    )


class EngineCycleKey(StrictModel):
    """The integration key every output row is addressable by."""

    engine_id: int = Field(..., description="Engine identifier (alias of 'unit').")
    cycle: int = Field(..., ge=1, description="Operational cycle number.")


class BaseResponse(StrictModel):
    """Common envelope returned by every endpoint."""

    status: StageStatus = Field(
        default=StageStatus.SUCCESS, description="Outcome of the request."
    )
    message: str = Field(default="", description="Human-readable summary.")
    timestamp: str = Field(default="", description="UTC ISO-8601 completion time.")


class ErrorResponse(StrictModel):
    """Structured error payload returned on failure."""

    status: StageStatus = Field(default=StageStatus.FAILED)
    error_type: str = Field(..., description="Exception class name.")
    message: str = Field(..., description="What went wrong.")
    stage: Optional[str] = Field(default=None, description="Stage that failed.")
    hint: Optional[str] = Field(
        default=None, description="Suggested remedy, e.g. which stage to run first."
    )


class StageResponse(BaseResponse):
    """Result of running a single pipeline stage."""

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
    """Result of a full end-to-end pipeline run."""

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
    """Row-window controls for endpoints returning tabular data."""

    limit: int = Field(default=500, ge=1, le=100_000, description="Maximum rows.")
    offset: int = Field(default=0, ge=0, description="Rows to skip.")
    engine_id: Optional[int] = Field(
        default=None, description="Restrict results to a single engine."
    )


class MetricsPayload(StrictModel):
    """A named block of evaluation metrics."""

    name: str = Field(..., description="Metric group name, e.g. 'digital_twin'.")
    values: Dict[str, Any] = Field(default_factory=dict)
    recorded_at: str = Field(default="", description="UTC ISO-8601 timestamp.")
