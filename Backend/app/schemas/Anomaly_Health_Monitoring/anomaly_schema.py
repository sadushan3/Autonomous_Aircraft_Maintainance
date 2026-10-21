from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from app.schemas.Anomaly_Health_Monitoring.common_schema import (
    AlertLevel,
    AnomalyType,
    BaseResponse,
    HealthState,
    Severity,
    StrictModel,
)


class HealthRecord(StrictModel):


    engine_id: int = Field(..., description="Engine identifier.")
    cycle: int = Field(..., ge=1)
    health_index: float = Field(
        ..., ge=0.0, le=100.0, description="Smoothed health indicator, 100 = healthy."
    )
    health_index_raw: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Pre-smoothing health indicator."
    )
    health_score: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Health index after trend blending."
    )
    health_state: HealthState = Field(default=HealthState.HEALTHY)
    health_trend: float = Field(
        default=0.0, description="Slope of the health index; negative = worsening."
    )
    deterioration_rate: float = Field(
        default=0.0, ge=0.0, description="Health points lost per cycle."
    )
    remaining_health_pct: float = Field(
        default=100.0,
        ge=0.0,
        le=100.0,
        description="Fraction of the healthy-baseline health still available.",
    )
    alert_level: AlertLevel = Field(default=AlertLevel.NONE)
    state_changed: bool = Field(
        default=False, description="True on the cycle the state transitioned."
    )
    operating_mode: int = Field(default=0, description="Operating regime label.")


class HealthIndexResponse(BaseResponse):


    rows: int = Field(default=0, ge=0)
    engines: int = Field(default=0, ge=0)
    mean_health_index: float = Field(default=0.0, ge=0.0, le=100.0)
    min_health_index: float = Field(default=0.0, ge=0.0, le=100.0)
    monotonicity_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Fraction of engines whose health index is non-increasing "
        "over the final phase of life. The acceptance gate for the twin.",
    )
    rul_correlation: Optional[float] = Field(
        default=None,
        description="Spearman correlation between health index and true RUL.",
    )
    state_distribution: Dict[str, int] = Field(default_factory=dict)
    health_index_csv: str = Field(default="")
    health_scores_csv: str = Field(default="")
    health_states_csv: str = Field(default="")
    records: List[HealthRecord] = Field(
        default_factory=list, description="Optional sample of per-row records."
    )


class DetectorScores(StrictModel):


    isolation_forest_score: float = Field(default=0.0, ge=0.0, le=1.0)
    mahalanobis_score: float = Field(default=0.0, ge=0.0, le=1.0)
    lstm_ae_score: float = Field(default=0.0, ge=0.0, le=1.0)
    residual_rule_score: float = Field(default=0.0, ge=0.0, le=1.0)
    condition_consensus_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AnomalyRecord(StrictModel):


    engine_id: int = Field(..., description="Engine identifier.")
    cycle: int = Field(..., ge=1)
    anomaly_score: float = Field(
        ..., ge=0.0, le=1.0, description="Weighted fusion of all detectors."
    )
    anomaly_flag: bool = Field(
        default=False, description="True when the score exceeds the adaptive threshold."
    )
    anomaly_type: AnomalyType = Field(
        default=AnomalyType.NONE,
        description="'fault' or 'operational'. Operational detections coincide "
        "with a regime change, revert within the guard window, and are "
        "suppressed rather than escalated.",
    )
    severity: Severity = Field(default=Severity.NONE)
    early_warning_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Likelihood of degradation becoming critical within the "
        "configured horizon.",
    )
    detector_scores: Optional[DetectorScores] = Field(default=None)
    threshold_used: float = Field(
        default=0.5, description="The adaptive threshold in force for this row."
    )
    operating_mode: int = Field(default=0)
    is_mode_transition: bool = Field(
        default=False, description="Row lies within the regime-change guard window."
    )
    top_sensors: List[str] = Field(
        default_factory=list, description="Highest-contributing sensors by residual."
    )


class AnomalyDetectionRequest(StrictModel):


    detectors: Optional[List[str]] = Field(
        default=None,
        description="Subset of ['isolation_forest','mahalanobis','lstm_autoencoder',"
        "'residual_rule']. Defaults to all.",
    )
    fusion_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the fused anomaly threshold for this run.",
    )
    apply_feedback_thresholds: bool = Field(
        default=True,
        description="Apply operator-adapted thresholds from residual_thresholds.json.",
    )


class DetectorSummary(StrictModel):


    name: str = Field(...)
    threshold: float = Field(default=0.0)
    threshold_percentile: float = Field(default=99.0)
    flagged_rows: int = Field(default=0, ge=0)
    flagged_fraction: float = Field(default=0.0, ge=0.0, le=1.0)
    train_rows: int = Field(default=0, ge=0)
    fit_seconds: float = Field(default=0.0, ge=0.0)


class AnomalyDetectionResponse(BaseResponse):


    rows: int = Field(default=0, ge=0)
    engines: int = Field(default=0, ge=0)
    detectors: List[DetectorSummary] = Field(default_factory=list)
    total_anomalies: int = Field(default=0, ge=0)
    fault_anomalies: int = Field(
        default=0, ge=0, description="Escalated to the reasoning layer."
    )
    operational_anomalies: int = Field(
        default=0,
        ge=0,
        description="Suppressed regime-change detections -- the false alarms "
        "this component exists to avoid.",
    )
    severity_distribution: Dict[str, int] = Field(default_factory=dict)
    fusion_threshold: float = Field(default=0.5)
    anomaly_scores_csv: str = Field(default="")
    anomaly_fusion_csv: str = Field(default="")
    records: List[AnomalyRecord] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
