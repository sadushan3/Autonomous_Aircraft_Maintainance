"""
Schemas for the root-cause reasoning layer.

The reasoning layer answers eight questions about every fault-type anomaly:
why it occurred, what sequence of events caused it, which sensors contributed,
whether it is operational or fault-related, whether something similar has been
seen before, how it evolved, what the most likely root cause is, and what
maintenance action to recommend.

Causal language is deliberately soft.  Every record carries
``hard_causal_claim=False``: the sensor dependency graph encodes thermodynamic
coupling between engine subsystems, not proven causation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from app.schemas.Anomaly_Health_Monitoring.common_schema import (
    AnomalyType,
    BaseResponse,
    EvolutionPattern,
    Severity,
    StrictModel,
)


class SensorContribution(StrictModel):
    """How much one sensor contributed to an anomaly."""

    sensor: str = Field(..., description="Sensor column name, e.g. 'sensor_4'.")
    residual: float = Field(default=0.0, description="Actual minus expected value.")
    residual_z: float = Field(
        default=0.0, description="Robust z-score of the residual."
    )
    contribution: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Share of total absolute residual attributable to this sensor.",
    )
    direction: str = Field(
        default="stable", description="'rising', 'falling' or 'stable'."
    )
    subsystem: str = Field(
        default="unknown", description="Engine subsystem this sensor observes."
    )


class CandidateCause(StrictModel):
    """One hypothesised root cause with its supporting evidence."""

    cause: str = Field(..., description="Root-cause description.")
    likelihood: float = Field(
        ..., ge=0.0, le=1.0, description="Relative plausibility among candidates."
    )
    subsystem: str = Field(default="unknown")
    evidence: List[str] = Field(
        default_factory=list, description="Facts that fired this hypothesis."
    )
    rule_id: str = Field(default="", description="Identifier of the rule that matched.")
    inspection_focus: str = Field(
        default="", description="What a technician should physically examine."
    )
    recommended_action: str = Field(default="")


class RootCauseRecord(StrictModel):
    """Complete reasoning output for one engine at one cycle."""

    engine_id: int = Field(...)
    cycle: int = Field(..., ge=1)
    anomaly_type: AnomalyType = Field(default=AnomalyType.NONE)
    severity: Severity = Field(default=Severity.NONE)

    top_sensors: List[str] = Field(
        default_factory=list, description="Ranked highest-contributing sensors."
    )
    sensor_contributions: List[SensorContribution] = Field(default_factory=list)

    subsystem: str = Field(
        default="unknown", description="Most implicated engine subsystem."
    )
    root_cause: str = Field(
        default="No fault indication", description="Highest-likelihood cause."
    )
    cause_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_causes: List[CandidateCause] = Field(default_factory=list)

    evolution_pattern: EvolutionPattern = Field(default=EvolutionPattern.STABLE)
    onset_cycle: Optional[int] = Field(
        default=None, description="Cycle at which the deviation began."
    )
    duration_cycles: int = Field(
        default=0, ge=0, description="Cycles the anomaly has persisted."
    )
    trend_slope: float = Field(
        default=0.0, description="Slope of the aggregate residual over time."
    )

    similar_past_anomaly_id: Optional[str] = Field(
        default=None, description="Identifier of a matching remembered anomaly."
    )
    similarity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    times_seen_before: int = Field(default=0, ge=0)
    previously_false_alarm: bool = Field(
        default=False,
        description="A closely matching pattern was previously rejected by an "
        "operator. Weighted into the confidence estimate.",
    )

    recommended_action: str = Field(default="Continue monitoring")
    inspection_focus: str = Field(default="")
    hard_causal_claim: bool = Field(
        default=False,
        description="Always False. Dependencies are physical couplings, not "
        "demonstrated causation.",
    )


class ReasoningRequest(StrictModel):
    """Parameters for the reasoning stage."""

    top_k_sensors: Optional[int] = Field(
        default=None, ge=1, le=21, description="How many sensors to attribute."
    )
    only_anomalies: bool = Field(
        default=True,
        description="Reason only about flagged rows. When False every row is "
        "reasoned about, which is far slower.",
    )
    use_memory: bool = Field(
        default=True, description="Match against remembered past anomalies."
    )


class ReasoningResponse(BaseResponse):
    """Result of the reasoning stage."""

    rows: int = Field(default=0, ge=0)
    reasoned_rows: int = Field(
        default=0, ge=0, description="Rows that received a root-cause attribution."
    )
    engines: int = Field(default=0, ge=0)
    cause_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Frequency of each inferred root cause."
    )
    subsystem_distribution: Dict[str, int] = Field(default_factory=dict)
    evolution_distribution: Dict[str, int] = Field(default_factory=dict)
    memory_matches: int = Field(
        default=0, ge=0, description="Anomalies matched to a remembered pattern."
    )
    root_cause_csv: str = Field(default="")
    records: List[RootCauseRecord] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)


class DependencyEdge(StrictModel):
    """One directed edge of the sensor dependency graph."""

    source: str = Field(..., description="Upstream sensor or subsystem.")
    target: str = Field(..., description="Downstream sensor or subsystem.")
    relation: str = Field(
        default="influences", description="Nature of the physical coupling."
    )
    weight: float = Field(default=1.0, ge=0.0, le=1.0)


class EngineRootCauseResponse(BaseResponse):
    """Root-cause history for a single engine, read from ``root_cause_analysis.csv``."""

    engine_id: int = Field(...)
    records: List[RootCauseRecord] = Field(default_factory=list)


class DependencyGraphResponse(BaseResponse):
    """The sensor dependency graph, for inspection and dashboard rendering."""

    nodes: List[str] = Field(default_factory=list)
    edges: List[DependencyEdge] = Field(default_factory=list)
    subsystems: Dict[str, List[str]] = Field(
        default_factory=dict, description="Sensor membership per engine subsystem."
    )
    n_nodes: int = Field(default=0, ge=0)
    n_edges: int = Field(default=0, ge=0)
