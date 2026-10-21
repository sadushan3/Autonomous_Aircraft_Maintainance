from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import (
    SUBSYSTEM_DESCRIPTIONS,
    SensorDependencyGraph,
)
from app.utils.Anomaly_Health_Monitoring.utils import get_logger


@dataclass
class RuleMatch:


    cause: str
    subsystem: str
    likelihood: float
    evidence: List[str]
    rule_id: str
    inspection_focus: str
    recommended_action: str


_SUBSYSTEM_CAUSE_TEMPLATES: Dict[str, Tuple[str, str, str]] = {
    "hpc": (
        "High-Pressure Compressor efficiency degradation",
        "HPC blade tip clearance, compressor blade fouling/erosion",
        "Schedule HPC borescope inspection; consider compressor wash",
    ),
    "lpc": (
        "Low-Pressure Compressor efficiency degradation",
        "LPC blade fouling, inlet distortion",
        "Inspect LPC inlet and blades; consider compressor wash",
    ),
    "hpt": (
        "High-Pressure Turbine degradation",
        "HPT blade erosion, nozzle guide vane condition",
        "Schedule HPT borescope inspection",
    ),
    "lpt": (
        "Low-Pressure Turbine degradation",
        "LPT blade wear, coolant bleed path condition",
        "Schedule LPT borescope inspection",
    ),
    "burner": (
        "Combustor fuel-air mixture drift",
        "Burner fuel nozzles, fuel-air ratio control",
        "Inspect fuel nozzles and burner liner",
    ),
    "bleed": (
        "Bleed air system anomaly",
        "Bleed valves, bleed air seals",
        "Inspect bleed valves and associated seals",
    ),
    "bypass": (
        "Bypass duct / fan-flow anomaly",
        "Bypass duct integrity, fan flow path",
        "Inspect bypass duct and fan flow path",
    ),
    "fan": (
        "Fan / inlet condition degradation",
        "Fan blades, inlet condition",
        "Inspect fan blades and inlet for damage or fouling",
    ),
    "general": (
        "Diffuse, multi-subsystem deviation",
        "Full gas-path review",
        "Perform a full borescope inspection across the gas path",
    ),
}


class ReasoningRuleEngine:


    def __init__(
        self,
        config: Optional[Config] = None,
        graph: Optional[SensorDependencyGraph] = None,
    ) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.graph = graph or SensorDependencyGraph(self.config)


    def _subsystem_concentration(
        self, top_sensors: List[str]
    ) -> Dict[str, float]:


        weights: Dict[str, float] = {}
        for rank, sensor in enumerate(top_sensors):
            subsystem = self.graph.subsystem_of(sensor)
            weights[subsystem] = weights.get(subsystem, 0.0) + 1.0 / (rank + 1)
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    def evaluate(
        self,
        top_sensors: List[str],
        trend_rising: bool,
        health_state: str = "Healthy",
    ) -> RuleMatch:


        if not top_sensors:
            return RuleMatch(
                cause="No fault indication",
                subsystem="none",
                likelihood=0.0,
                evidence=["No contributing sensors supplied."],
                rule_id="RULE_NONE",
                inspection_focus="",
                recommended_action="Continue monitoring",
            )

        concentration = self._subsystem_concentration(top_sensors)
        dominant_subsystem = max(concentration, key=concentration.get)
        dominant_score = concentration[dominant_subsystem]

        cause, inspection, action = _SUBSYSTEM_CAUSE_TEMPLATES.get(
            dominant_subsystem, _SUBSYSTEM_CAUSE_TEMPLATES["general"]
        )

        evidence = [
            f"Top contributing sensors {top_sensors[:5]} concentrate "
            f"{dominant_score:.0%} on the {SUBSYSTEM_DESCRIPTIONS.get(dominant_subsystem, dominant_subsystem)} subsystem."
        ]
        likelihood = 0.4 + 0.4 * dominant_score
        if trend_rising:
            evidence.append("Aggregate residual trend is rising, consistent with progressive degradation.")
            likelihood = min(likelihood + 0.15, 0.98)
        else:
            evidence.append("Residual trend is not clearly rising; may be an isolated excursion.")
            likelihood = max(likelihood - 0.1, 0.05)

        if health_state in ("Critical", "Failure Risk"):
            action = f"URGENT: {action}"

        rule_id = f"RULE_{dominant_subsystem.upper()}_CONCENTRATION"
        self.logger.debug(
            "Rule fired: %s (subsystem=%s, likelihood=%.2f)", rule_id, dominant_subsystem, likelihood
        )
        return RuleMatch(
            cause=cause,
            subsystem=dominant_subsystem,
            likelihood=round(float(likelihood), 4),
            evidence=evidence,
            rule_id=rule_id,
            inspection_focus=inspection,
            recommended_action=action,
        )

    def rank_candidates(
        self, top_sensors: List[str], trend_rising: bool, health_state: str = "Healthy"
    ) -> List[RuleMatch]:


        if not top_sensors:
            return [self.evaluate(top_sensors, trend_rising, health_state)]

        concentration = self._subsystem_concentration(top_sensors)
        candidates: List[RuleMatch] = []
        for subsystem, score in sorted(concentration.items(), key=lambda item: item[1], reverse=True):
            cause, inspection, action = _SUBSYSTEM_CAUSE_TEMPLATES.get(
                subsystem, _SUBSYSTEM_CAUSE_TEMPLATES["general"]
            )
            likelihood = 0.4 + 0.4 * score
            if trend_rising:
                likelihood = min(likelihood + 0.15, 0.98)
            else:
                likelihood = max(likelihood - 0.1, 0.05)
            candidates.append(
                RuleMatch(
                    cause=cause,
                    subsystem=subsystem,
                    likelihood=round(float(likelihood), 4),
                    evidence=[
                        f"{score:.0%} of top-sensor weight concentrates on "
                        f"{SUBSYSTEM_DESCRIPTIONS.get(subsystem, subsystem)}."
                    ],
                    rule_id=f"RULE_{subsystem.upper()}_CONCENTRATION",
                    inspection_focus=inspection,
                    recommended_action=action,
                )
            )
        return candidates


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    engine = ReasoningRuleEngine(config)

    match = engine.evaluate(
        top_sensors=["sensor_3", "sensor_11", "sensor_12", "sensor_2"],
        trend_rising=True,
        health_state="Warning",
    )
    logger.info("Best match: %s", match)

    candidates = engine.rank_candidates(
        top_sensors=["sensor_3", "sensor_11", "sensor_20"], trend_rising=True
    )
    for candidate in candidates:
        logger.info("Candidate: %s (%.2f)", candidate.cause, candidate.likelihood)


if __name__ == "__main__":
    main()
