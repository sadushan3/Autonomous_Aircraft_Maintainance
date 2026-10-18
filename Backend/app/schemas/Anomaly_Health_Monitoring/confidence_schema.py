from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from app.schemas.Anomaly_Health_Monitoring.common_schema import (
    BaseResponse,
    FeedbackVerdict,
    StrictModel,
)


class ConfidenceRecord(StrictModel):


    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)
    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall trust in this assessment, on a 0-100 scale.",
    )
    reliability_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1 minus normalised disagreement between anomaly detectors.",
    )
    uncertainty_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised spread of the digital-twin predictions. High "
        "spread means the expected value itself is in doubt, so the residual "
        "derived from it cannot be trusted.",
    )
    model_agreement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1 minus normalised twin disagreement.",
    )
    detector_agreement: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Pairwise detector concordance."
    )
    data_quality_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Imputation burden and mode-assignment certainty for the row.",
    )
    twin_spread_mean: float = Field(
        default=0.0, ge=0.0, description="Mean per-sensor twin standard deviation."
    )


class ConfidenceRequest(StrictModel):


    weight_agreement: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override the agreement weight."
    )
    weight_reliability: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override the reliability weight."
    )
    weight_data_quality: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Override the data-quality weight."
    )


class ConfidenceResponse(BaseResponse):


    rows: int = Field(default=0, ge=0)
    engines: int = Field(default=0, ge=0)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    mean_reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    mean_uncertainty: float = Field(default=0.0, ge=0.0, le=1.0)
    low_confidence_rows: int = Field(
        default=0, ge=0, description="Rows below a 50/100 confidence score."
    )
    confidence_csv: str = Field(default="")
    records: List[ConfidenceRecord] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class FeedbackSubmission(StrictModel):


    engine_id: int = Field(..., description="Engine the alert concerned.")
    cycle: int = Field(..., ge=1, description="Cycle the alert concerned.")
    verdict: FeedbackVerdict = Field(
        ..., description="Whether the alert was correct, a false alarm, or a miss."
    )
    operator: str = Field(default="unknown", description="Who submitted the verdict.")
    comment: str = Field(default="", description="Free-text operator note.")
    confirmed_root_cause: Optional[str] = Field(
        default=None,
        description="The cause actually found, when it differed from the "
        "system's hypothesis. Recorded into the anomaly memory.",
    )


class ThresholdChange(StrictModel):


    key: str = Field(..., description="Threshold identifier, e.g. a sensor name.")
    threshold_before: float = Field(default=0.0)
    threshold_after: float = Field(default=0.0)
    scale_applied: float = Field(default=1.0)
    adjustment_reason: str = Field(default="")


class FeedbackResponse(BaseResponse):


    feedback_id: str = Field(default="", description="Identifier of the record stored.")
    total_feedback_records: int = Field(default=0, ge=0)
    accepted_count: int = Field(default=0, ge=0)
    false_alarm_count: int = Field(default=0, ge=0)
    missed_count: int = Field(default=0, ge=0)
    thresholds_updated: int = Field(default=0, ge=0)
    changes: List[ThresholdChange] = Field(default_factory=list)
    feedback_csv: str = Field(default="")
    thresholds_path: str = Field(
        default="",
        description="residual_thresholds.json -- read by the anomaly layer on "
        "the next run. This file is the entire feedback loop.",
    )


class AlertMemoryRecord(StrictModel):


    anomaly_id: str = Field(...)
    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)
    signature: List[str] = Field(
        default_factory=list, description="Top contributing sensors at detection."
    )
    root_cause: str = Field(default="")
    severity: str = Field(default="none")
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    operator_verdict: FeedbackVerdict = Field(default=FeedbackVerdict.UNKNOWN)
    confirmed_root_cause: Optional[str] = Field(default=None)
    recorded_at: str = Field(default="")
    times_matched: int = Field(default=0, ge=0)


class AlertMemoryResponse(BaseResponse):


    total_memories: int = Field(default=0, ge=0)
    false_alarm_patterns: int = Field(
        default=0, ge=0, description="Signatures operators have rejected before."
    )
    confirmed_fault_patterns: int = Field(default=0, ge=0)
    records: List[AlertMemoryRecord] = Field(default_factory=list)


class FeedbackVerdictSummary(StrictModel):


    total: int = Field(default=0, ge=0)
    by_verdict: Dict[str, int] = Field(default_factory=dict)


class MemorySummary(StrictModel):


    total_memories: int = Field(default=0, ge=0)
    false_alarm_patterns: int = Field(default=0, ge=0)
    confirmed_fault_patterns: int = Field(default=0, ge=0)


class FeedbackSummaryResponse(BaseResponse):


    feedback: FeedbackVerdictSummary = Field(default_factory=FeedbackVerdictSummary)
    memory: MemorySummary = Field(default_factory=MemorySummary)


class ThresholdStateResponse(BaseResponse):


    scale: float = Field(
        default=1.0, description="Raw multiplicative scale from the verdict history."
    )
    fusion_threshold: float = Field(
        default=0.0, description="The threshold actually applied, after clamping."
    )
    base_threshold: Optional[float] = Field(default=None)
    effective_scale: Optional[float] = Field(
        default=None,
        description="fusion_threshold / base_threshold. Differs from `scale` "
        "whenever the clamp engaged.",
    )
    clamped: Optional[bool] = Field(
        default=None,
        description="True when the raw scale would have pushed the threshold "
        "past the reachable ceiling and was capped.",
    )
    reachable_ceiling: Optional[float] = Field(
        default=None,
        description="Highest threshold a bounded [0,1] anomaly score can still "
        "exceed; the cut is never allowed above this.",
    )
    feedback_applied: int = Field(default=0, ge=0)
    last_updated: Optional[str] = Field(default=None)
