"""
schemas.py

Pydantic Request & Response Schemas for the Maintenance Supervisor API.

Purpose
-------
Defines strict Pydantic schemas for single-sample and batch real-time API endpoints.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class TelemetryInputSchema(BaseModel):
    engine_id: str = Field(..., example="ENG_001")
    cycle: int = Field(..., example=150)
    fd_subset: Optional[str] = Field("FD001", example="FD001")
    predicted_rul: float = Field(..., example=12.5)
    risk_10: float = Field(0.0, example=0.85)
    risk_30: float = Field(0.0, example=0.92)
    risk_50: float = Field(0.0, example=0.95)
    anomaly_score: float = Field(0.0, example=0.78)
    anomaly_severity: Optional[str] = Field("high", example="high")
    extra_features: Optional[dict[str, Any]] = Field(default_factory=dict)


class SupervisorDecisionResponseSchema(BaseModel):
    engine_id: str
    cycle: int
    final_decision: str
    priority: str
    maintenance_urgency: str
    recommended_action: str
    confidence_score: float
    confidence_level: str
    entropy_score: float
    conflict_score: float
    requires_human_review: bool
    review_reasons: Optional[str] = None
    override_applied: bool
    override_reason: Optional[str] = None


class BatchInferenceRequestSchema(BaseModel):
    records: list[TelemetryInputSchema]


class BatchInferenceResponseSchema(BaseModel):
    total_processed: int
    decisions: list[SupervisorDecisionResponseSchema]
