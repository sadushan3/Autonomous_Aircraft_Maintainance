"""
Schemas for the explainability layer and the dashboard feed.

The explainability layer converts numeric residuals into an account a
maintenance engineer can act on, answering four questions per anomaly:

* WHY did the anomaly occur?
* WHICH sensors contributed, and by how much?
* WHAT pattern do those sensors form?
* HOW severe is it?
"""

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


class ShapContribution(StrictModel):
    """A single SHAP feature attribution."""

    feature: str = Field(..., description="Input feature name.")
    shap_value: float = Field(
        default=0.0, description="Signed contribution to the twin's prediction."
    )
    abs_shap_value: float = Field(default=0.0, ge=0.0)
    rank: int = Field(default=0, ge=0)


class SubsystemAttribution(StrictModel):
    """Aggregated deviation attributed to one engine subsystem."""

    subsystem: str = Field(..., description="e.g. 'HPC', 'LPT', 'burner'.")
    contribution: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Share of total deviation."
    )
    sensors: List[str] = Field(
        default_factory=list, description="Sensors belonging to this subsystem."
    )
    mean_residual_z: float = Field(default=0.0)
    status: str = Field(
        default="nominal", description="'nominal', 'elevated' or 'degraded'."
    )


class ExplanationEvidence(StrictModel):
    """Provenance and numeric evidence shared by explanation/dashboard views."""

    shap_status: Optional[str] = None
    shap_sensor: Optional[str] = None
    shap_explained_output: Optional[str] = None
    shap_base_value: Optional[float] = None
    shap_prediction: Optional[float] = None
    shap_commissioning_offset: Optional[float] = None
    shap_additivity_error: Optional[float] = None
    shap_reference_error: Optional[float] = None
    health_explanation_status: Optional[str] = None
    health_explanation: Optional[str] = None
    health_evidence: Dict[str, Any] = Field(default_factory=dict)
    sensor_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    cause_support_type: Optional[str] = None
    cause_likelihood: Optional[float] = None
    candidate_causes: List[Dict[str, Any]] = Field(default_factory=list)
    anomaly_margin: Optional[float] = None
    hard_causal_claim: bool = False


class ExplanationRecord(ExplanationEvidence):
    """A complete human-readable explanation for one engine at one cycle."""

    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)

    why: str = Field(
        default="", description="Why the anomaly was raised, in plain language."
    )
    which_sensors: List[str] = Field(
        default_factory=list, description="Sensors with the largest robust residual deviations."
    )
    what_pattern: str = Field(
        default="", description="The pattern those sensors form together."
    )
    how_severe: str = Field(default="", description="Severity in operational terms.")

    explanation_text: str = Field(
        default="",
        description="The full narrative paragraph shown to the operator.",
    )
    recommended_action: str = Field(default="")

    anomaly_type: AnomalyType = Field(default=AnomalyType.NONE)
    severity: Severity = Field(default=Severity.NONE)
    health_state: HealthState = Field(default=HealthState.HEALTHY)

    shap_top_features: List[ShapContribution] = Field(default_factory=list)
    subsystem_attribution: List[SubsystemAttribution] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    anomaly_score: Optional[float] = None
    threshold_used: Optional[float] = None


class ExplainRequest(StrictModel):
    """Parameters for the explainability stage."""

    max_rows: Optional[int] = Field(
        default=None,
        ge=1,
        description="Cap on rows to explain. SHAP is expensive; the default "
        "comes from config.",
    )
    engine_id: Optional[int] = Field(
        default=None, description="Explain a single engine only."
    )
    include_shap: bool = Field(
        default=True,
        description="Compute SHAP attributions. Disabling leaves residual "
        "ranking and narrative generation intact but is much faster.",
    )
    only_anomalies: bool = Field(default=True)


class ExplainResponse(BaseResponse):
    """Result of the explainability stage."""

    rows: int = Field(default=0, ge=0)
    explained_rows: int = Field(default=0, ge=0)
    engines: int = Field(default=0, ge=0)
    explainer_used: str = Field(
        default="", description="'exact_grouped' for monitoring, or 'none'."
    )
    shap_computed: bool = Field(default=False)
    top_global_features: List[ShapContribution] = Field(
        default_factory=list, description="Mean absolute SHAP across the sample."
    )
    explanation_csv: str = Field(default="")
    records: List[ExplanationRecord] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #

class EngineSummary(ExplanationEvidence):
    """Latest-cycle rollup for one engine, as shown on a dashboard card."""

    engine_id: int = Field(...)
    latest_cycle: int = Field(default=0, ge=0)
    total_cycles: int = Field(default=0, ge=0)
    health_index: float = Field(default=100.0, ge=0.0, le=100.0)
    health_state: HealthState = Field(default=HealthState.HEALTHY)
    health_trend: float = Field(default=0.0)
    remaining_health_pct: float = Field(default=100.0, ge=0.0, le=100.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    anomaly_flag: bool = Field(default=False)
    anomaly_type: AnomalyType = Field(default=AnomalyType.NONE)
    severity: Severity = Field(default=Severity.NONE)
    alert_level: AlertLevel = Field(default=AlertLevel.NONE)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    root_cause: str = Field(default="")
    top_sensors: List[str] = Field(default_factory=list)
    operating_mode: int = Field(default=0)
    explanation_text: str = Field(default="")
    twin_spread_mean: Optional[float] = Field(default=None, ge=0.0)
    deterioration_rate: Optional[float] = Field(default=None)
    reliability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertainty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detector_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evolution_pattern: Optional[str] = Field(default=None)

    # Per-detector contributions. The fused anomaly_score alone says a row was
    # flagged but not which detector drove it -- the first thing an operator
    # asks. These are the four detectors' individually normalised scores plus
    # the calibrated cut they were compared against.
    isolation_forest_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mahalanobis_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lstm_ae_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    residual_rule_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold_used: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    anomaly_persistence_cycles: Optional[int] = Field(default=None, ge=0)
    early_warning_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Structured explainability. `subsystem_attribution` maps the deviation onto
    # physical engine sections with a contribution share each; kept as loose
    # dicts because the reasoning layer owns that shape, not the API.
    subsystem_attribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-subsystem contribution share, member sensors, mean "
        "residual z and elevated/nominal status.",
    )
    which_sensors: List[str] = Field(default_factory=list)
    why: Optional[str] = Field(default=None)
    what_pattern: Optional[str] = Field(default=None)
    how_severe: Optional[str] = Field(default=None)
    shap_top_features: List[Any] = Field(
        default_factory=list,
        description="Empty unless the pipeline was run with include_shap=True.",
    )


class TimelinePoint(StrictModel):
    """One point on an engine's health / anomaly timeline."""

    cycle: int = Field(..., ge=1)
    health_index: float = Field(default=100.0, ge=0.0, le=100.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    anomaly_flag: bool = Field(default=False)
    health_state: HealthState = Field(default=HealthState.HEALTHY)
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    operating_mode: int = Field(default=0)
    twin_spread_mean: Optional[float] = Field(default=None, ge=0.0)
    deterioration_rate: Optional[float] = Field(default=None)
    reliability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertainty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detector_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evolution_pattern: Optional[str] = Field(default=None)


class AlertItem(ExplanationEvidence):
    """An entry in the operator alert feed."""

    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)
    alert_level: AlertLevel = Field(default=AlertLevel.NONE)
    severity: Severity = Field(default=Severity.NONE)
    health_state: HealthState = Field(default=HealthState.HEALTHY)
    root_cause: str = Field(default="")
    recommended_action: str = Field(default="")
    confidence_score: float = Field(default=0.0, ge=0.0, le=100.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
    explanation_text: str = Field(default="")
    twin_spread_mean: Optional[float] = Field(default=None, ge=0.0)
    deterioration_rate: Optional[float] = Field(default=None)
    reliability_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    uncertainty_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    detector_agreement: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    evolution_pattern: Optional[str] = Field(default=None)

    # Per-detector contributions. The fused anomaly_score alone says a row was
    # flagged but not which detector drove it -- the first thing an operator
    # asks. These are the four detectors' individually normalised scores plus
    # the calibrated cut they were compared against.
    isolation_forest_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mahalanobis_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    lstm_ae_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    residual_rule_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    threshold_used: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    anomaly_persistence_cycles: Optional[int] = Field(default=None, ge=0)
    early_warning_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # Structured explainability. `subsystem_attribution` maps the deviation onto
    # physical engine sections with a contribution share each; kept as loose
    # dicts because the reasoning layer owns that shape, not the API.
    subsystem_attribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Per-subsystem contribution share, member sensors, mean "
        "residual z and elevated/nominal status.",
    )
    which_sensors: List[str] = Field(default_factory=list)
    why: Optional[str] = Field(default=None)
    what_pattern: Optional[str] = Field(default=None)
    how_severe: Optional[str] = Field(default=None)
    shap_top_features: List[Any] = Field(
        default_factory=list,
        description="Empty unless the pipeline was run with include_shap=True.",
    )


class FleetStatistics(StrictModel):
    """Fleet-wide rollup shown at the top of the dashboard."""

    total_engines: int = Field(default=0, ge=0)
    total_rows: int = Field(default=0, ge=0)
    mean_health_index: float = Field(default=0.0, ge=0.0, le=100.0)
    engines_healthy: int = Field(default=0, ge=0)
    engines_degrading: int = Field(default=0, ge=0)
    engines_warning: int = Field(default=0, ge=0)
    engines_critical: int = Field(default=0, ge=0)
    engines_failure_risk: int = Field(default=0, ge=0)
    total_anomalies: int = Field(default=0, ge=0)
    fault_anomalies: int = Field(default=0, ge=0)
    operational_anomalies: int = Field(default=0, ge=0)
    mean_confidence: float = Field(default=0.0, ge=0.0, le=100.0)
    top_root_causes: Dict[str, int] = Field(default_factory=dict)
    lifetime_mean_health_index: float = Field(
        default=0.0, ge=0.0, le=100.0,
        description="Mean health_index across every cycle of every engine's "
        "lifetime, not just each engine's latest cycle.",
    )
    lifetime_rows_healthy: int = Field(default=0, ge=0)
    lifetime_rows_degrading: int = Field(default=0, ge=0)
    lifetime_rows_warning: int = Field(default=0, ge=0)
    lifetime_rows_critical: int = Field(default=0, ge=0)
    lifetime_rows_failure_risk: int = Field(default=0, ge=0)


class DashboardResponse(BaseResponse):
    """The full dashboard payload."""

    fleet: FleetStatistics = Field(default_factory=FleetStatistics)
    engines: List[EngineSummary] = Field(default_factory=list)
    alerts: List[AlertItem] = Field(default_factory=list)
    timeline: Dict[str, List[TimelinePoint]] = Field(
        default_factory=dict, description="Per-engine timelines keyed by engine id."
    )
    dashboard_csv: str = Field(default="")
    exchange_csv: str = Field(
        default="",
        description="The integration handoff file consumed by the Maintenance "
        "Supervisor.",
    )
    metrics: Dict[str, Any] = Field(default_factory=dict)
