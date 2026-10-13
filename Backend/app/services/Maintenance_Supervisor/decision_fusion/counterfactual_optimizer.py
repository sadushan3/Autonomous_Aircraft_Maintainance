"""
counterfactual_optimizer.py

Actionable Counterfactual Operational Parameter Optimizer for the
Autonomous Maintenance Supervisor Agent.

What this computes
------------------
Given an engine's current fused state, it answers: *what would have to change
for this engine to drop one maintenance tier?*

The answer has two halves, and they are deliberately kept apart because only
one of them is derived from the system's own evidence:

1. **The binding constraints and the required change** are exact. The
   Supervisor's safety policy is a set of published thresholds over
   ``predicted_rul``, ``risk_10``, ``risk_30`` and ``anomaly_severity``
   (see ``safety_policy.py``). Inverting them gives, with no modelling
   assumption at all, the precise margin by which each currently-violated
   constraint is violated and therefore the exact improvement needed to clear
   the tier.

2. **The operational lever** -- "reduce HPC cruise pressure by X%" -- is a
   translation of that required improvement into a control action, and it rests
   on the assumed linear sensitivities named in
   :data:`OPERATIONAL_SENSITIVITY`. Those constants are engineering placeholders
   for a thrust-derate response that C-MAPSS does not contain: the dataset has
   no controllable pressure input and no counterfactual trajectories, so no
   sensitivity can be fitted from it. Every response carries
   ``lever_is_assumption: True`` and the constants it used, so the lever is
   never mistaken for a measured effect.

An earlier version of this module returned a fixed 4.2% / 14-cycle
recommendation regardless of its inputs. Nothing about the engine reached the
output. That is what this replaces.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Optional

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    DECISION_TO_SEVERITY,
    SEVERITY_TO_DECISION,
    SupervisorConfig,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section

logger = get_logger()
_CFG = SupervisorConfig()

#: Assumed first-order response of the fused state to a cruise derate.
#:
#: NOT fitted -- C-MAPSS records one trajectory per unit under its own operating
#: conditions and contains no controllable pressure input, so there is nothing
#: in the data from which a derate response could be estimated. These are
#: documented placeholders that make the lever dimensionally sensible; every
#: recommendation that uses them is flagged.
OPERATIONAL_SENSITIVITY: Final[dict[str, float]] = {
    # Extra cycles of remaining life per 1% reduction in HPC cruise pressure.
    "rul_cycles_per_pct_pressure": 3.3,
    # Absolute drop in failure probability per 1% pressure reduction. The
    # near-term horizon responds less to a derate than the medium-term one:
    # damage already accumulated dominates the next ten cycles.
    "risk_10_per_pct_pressure": 0.045,
    "risk_30_per_pct_pressure": 0.085,
    "risk_50_per_pct_pressure": 0.110,
    # Thrust derate that accompanies a 1% HPC pressure reduction.
    "thrust_pct_per_pressure_pct": 0.71,
    # Largest derate treated as operationally acceptable in cruise.
    "max_pressure_reduction_pct": 12.0,
}

#: Which sensitivity converts each constraint's shortfall into a derate.
_CONSTRAINT_SENSITIVITY: Final[dict[str, str]] = {
    "predicted_rul": "rul_cycles_per_pct_pressure",
    "risk_10": "risk_10_per_pct_pressure",
    "risk_30": "risk_30_per_pct_pressure",
    "risk_50": "risk_50_per_pct_pressure",
}


@dataclass
class BindingConstraint:
    """One safety-policy threshold currently keeping the engine in its tier."""

    name: str
    variable: str
    expression: str
    current_value: Optional[float]
    threshold: float
    required_value: float
    shortfall: float

    def pressure_pct_required(self) -> float:
        """Derate that would close this constraint's shortfall, per the assumed sensitivity."""
        key = _CONSTRAINT_SENSITIVITY.get(self.variable)
        if key is None or self.shortfall <= 0:
            return 0.0
        return self.shortfall / OPERATIONAL_SENSITIVITY[key]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pressure_pct_required"] = round(self.pressure_pct_required(), 3)
        return payload


@dataclass
class CounterfactualRecommendation:
    """A minimal, tier-lowering operational change, with its evidence split out."""

    feasible: bool
    current_decision: str
    target_decision: Optional[str]
    original_rul: Optional[float]
    original_risk_30: Optional[float]
    binding_constraints: list[dict[str, Any]] = field(default_factory=list)
    required_rul_gain: float = 0.0
    required_risk_30_drop: float = 0.0
    hpc_pressure_reduction_pct: Optional[float] = None
    cruise_thrust_reduction_pct: Optional[float] = None
    projected_rul: Optional[float] = None
    projected_risk_30: Optional[float] = None
    focus_component: Optional[str] = None
    actionable_statement: str = ""
    lever_is_assumption: bool = True
    assumptions: dict[str, float] = field(default_factory=lambda: dict(OPERATIONAL_SENSITIVITY))
    blocked_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CounterfactualOptimizer:
    """Derives the minimum change that would move an engine down one tier."""

    @staticmethod
    def _constraints_for_tier(
        target_decision: str,
        predicted_rul: Optional[float],
        risk_10: Optional[float],
        risk_30: Optional[float],
        cfg: SupervisorConfig,
    ) -> list[BindingConstraint]:
        """Every safety threshold the engine must clear to reach ``target_decision``.

        Only constraints that are currently violated are returned; each carries
        the exact margin of violation, which is the improvement the lever has to
        deliver.
        """
        target_severity = DECISION_TO_SEVERITY.get(target_decision, 0)
        out: list[BindingConstraint] = []

        # A tier is escaped by clearing the RUL band that defines it. The bands
        # come straight from SupervisorConfig, so this is exact, not a heuristic.
        rul_floor_by_severity = {
            4: cfg.immediate_rul_threshold,
            3: cfg.maintenance_rul_threshold,
            2: cfg.inspection_rul_threshold,
            1: cfg.monitoring_rul_threshold,
        }
        # To sit at or below `target_severity`, RUL must exceed the floor of
        # every tier more severe than the target.
        for severity in range(target_severity + 1, 5):
            floor = rul_floor_by_severity.get(severity)
            if floor is None or predicted_rul is None:
                continue
            if predicted_rul <= floor:
                out.append(
                    BindingConstraint(
                        name=f"RUL above {SEVERITY_TO_DECISION[severity].replace('_', ' ')} band",
                        variable="predicted_rul",
                        expression=f"predicted_rul > {floor:g}",
                        current_value=round(predicted_rul, 3),
                        threshold=floor,
                        required_value=floor,
                        shortfall=round(floor - predicted_rul, 3),
                    )
                )

        if target_severity < 4 and risk_10 is not None and risk_10 >= cfg.critical_risk_10_threshold:
            out.append(
                BindingConstraint(
                    name="Critical 10-cycle risk",
                    variable="risk_10",
                    expression=f"risk_10 < {cfg.critical_risk_10_threshold:g}",
                    current_value=round(risk_10, 4),
                    threshold=cfg.critical_risk_10_threshold,
                    required_value=cfg.critical_risk_10_threshold,
                    shortfall=round(risk_10 - cfg.critical_risk_10_threshold, 4),
                )
            )

        if target_severity < 3 and risk_30 is not None and risk_30 >= cfg.maintenance_risk_30_threshold:
            out.append(
                BindingConstraint(
                    name="High 30-cycle risk",
                    variable="risk_30",
                    expression=f"risk_30 < {cfg.maintenance_risk_30_threshold:g}",
                    current_value=round(risk_30, 4),
                    threshold=cfg.maintenance_risk_30_threshold,
                    required_value=cfg.maintenance_risk_30_threshold,
                    shortfall=round(risk_30 - cfg.maintenance_risk_30_threshold, 4),
                )
            )

        return out

    @staticmethod
    def generate_counterfactual(
        predicted_rul: Optional[float] = None,
        risk_10: Optional[float] = None,
        risk_30: Optional[float] = None,
        anomaly_severity: Optional[str] = None,
        current_decision: str = "schedule_maintenance",
        focus_component: Optional[dict[str, Any]] = None,
        config: SupervisorConfig | None = None,
    ) -> CounterfactualRecommendation:
        """Find the smallest derate that would move this engine down one tier.

        Args:
            predicted_rul: The RUL agent's current estimate, in cycles.
            risk_10: Failure probability over the next 10 cycles.
            risk_30: Failure probability over the next 30 cycles.
            anomaly_severity: The anomaly agent's severity label. A ``critical``
                severity is a hard safety override no operational change can
                clear, so it makes the counterfactual infeasible by policy.
            current_decision: The fused decision as it stands.
            focus_component: Optional component-heatmap entry naming where the
                deviation sits, used only to address the recommendation.
            config: Threshold source; defaults to the shared SupervisorConfig.

        Returns:
            A recommendation whose exact half (constraints, required change) is
            separated from its assumed half (the derate that would deliver it).
        """
        cfg = config or _CFG
        current = str(current_decision).strip().lower()
        current_severity = DECISION_TO_SEVERITY.get(current, 0)
        component_name = (focus_component or {}).get("name") if focus_component else None

        if current_severity == 0:
            return CounterfactualRecommendation(
                feasible=False,
                current_decision=current,
                target_decision=None,
                original_rul=predicted_rul,
                original_risk_30=risk_30,
                focus_component=component_name,
                actionable_statement=(
                    "No operational change is called for: the engine is already at the "
                    "lowest maintenance tier (continue operation)."
                ),
                blocked_reason="already_at_lowest_tier",
            )

        if (anomaly_severity or "").strip().lower() == "critical":
            return CounterfactualRecommendation(
                feasible=False,
                current_decision=current,
                target_decision=None,
                original_rul=predicted_rul,
                original_risk_30=risk_30,
                focus_component=component_name,
                actionable_statement=(
                    "No derate can clear this decision: a critical anomaly severity is an "
                    "unconditional safety override in the policy, independent of RUL and risk."
                ),
                blocked_reason="critical_anomaly_override",
            )

        target = SEVERITY_TO_DECISION[current_severity - 1]
        constraints = CounterfactualOptimizer._constraints_for_tier(
            target, predicted_rul, risk_10, risk_30, cfg
        )

        required_rul_gain = max(
            (c.shortfall for c in constraints if c.variable == "predicted_rul"),
            default=0.0,
        )
        required_risk_drop = max(
            (c.shortfall for c in constraints if c.variable.startswith("risk_")),
            default=0.0,
        )

        # The binding derate is the largest any single constraint demands: a
        # change that clears the hardest one clears the rest with it. Each
        # constraint converts its own shortfall through its own sensitivity,
        # so a risk_10 gap is not costed at the risk_30 response rate.
        sens = OPERATIONAL_SENSITIVITY
        pressure_pct = round(max((c.pressure_pct_required() for c in constraints), default=0.0), 2)

        if pressure_pct <= 0:
            return CounterfactualRecommendation(
                feasible=False,
                current_decision=current,
                target_decision=target,
                original_rul=predicted_rul,
                original_risk_30=risk_30,
                binding_constraints=[c.to_dict() for c in constraints],
                focus_component=component_name,
                actionable_statement=(
                    f"The safety policy's thresholds for {target.replace('_', ' ')} are already "
                    f"met on RUL and risk; this decision is being held by the model's own "
                    f"candidate rather than by a guardrail, so no derate would change it."
                ),
                blocked_reason="no_binding_threshold",
            )

        if pressure_pct > sens["max_pressure_reduction_pct"]:
            return CounterfactualRecommendation(
                feasible=False,
                current_decision=current,
                target_decision=target,
                original_rul=predicted_rul,
                original_risk_30=risk_30,
                binding_constraints=[c.to_dict() for c in constraints],
                required_rul_gain=round(required_rul_gain, 2),
                required_risk_30_drop=round(required_risk_drop, 4),
                hpc_pressure_reduction_pct=pressure_pct,
                focus_component=component_name,
                actionable_statement=(
                    f"Clearing {current.replace('_', ' ')} would need about a "
                    f"{pressure_pct:.1f}% HPC cruise-pressure reduction — beyond the "
                    f"{sens['max_pressure_reduction_pct']:.0f}% derate treated as operationally "
                    f"acceptable. The degradation is too far advanced to fly around; "
                    f"{current.replace('_', ' ')} stands."
                ),
                blocked_reason="derate_exceeds_operational_limit",
            )

        thrust_pct = round(pressure_pct * sens["thrust_pct_per_pressure_pct"], 2)
        projected_rul = (
            None
            if predicted_rul is None
            else round(predicted_rul + pressure_pct * sens["rul_cycles_per_pct_pressure"], 1)
        )
        projected_risk = (
            None
            if risk_30 is None
            else round(max(0.0, risk_30 - pressure_pct * sens["risk_30_per_pct_pressure"]), 4)
        )

        where = f" on the {component_name}" if component_name else ""
        rul_clause = (
            f"predicted RUL rises from {predicted_rul:.1f} to {projected_rul:.1f} cycles"
            if predicted_rul is not None and projected_rul is not None
            else "the binding RUL constraint clears"
        )
        risk_clause = (
            f" and 30-cycle failure risk falls from {risk_30 * 100:.1f}% to {projected_risk * 100:.1f}%"
            if risk_30 is not None and projected_risk is not None
            else ""
        )

        statement = (
            f"Reducing HPC cruise pressure by {pressure_pct:.1f}% (thrust derate "
            f"-{thrust_pct:.1f}%){where} clears the binding constraint: {rul_clause}"
            f"{risk_clause}, which reschedules {current.replace('_', ' ')} to "
            f"{target.replace('_', ' ')}."
        )

        return CounterfactualRecommendation(
            feasible=True,
            current_decision=current,
            target_decision=target,
            original_rul=predicted_rul,
            original_risk_30=risk_30,
            binding_constraints=[c.to_dict() for c in constraints],
            required_rul_gain=round(required_rul_gain, 2),
            required_risk_30_drop=round(required_risk_drop, 4),
            hpc_pressure_reduction_pct=pressure_pct,
            cruise_thrust_reduction_pct=thrust_pct,
            projected_rul=projected_rul,
            projected_risk_30=projected_risk,
            focus_component=component_name,
            actionable_statement=statement,
        )


def run_counterfactual_demo() -> dict:
    """Exercise the optimizer across the decision tiers it can be asked about."""
    section("ACTIONABLE COUNTERFACTUAL OPTIMIZER STARTED")
    start_time = time.perf_counter()

    cases = [
        {"predicted_rul": 8.0, "risk_10": 0.88, "risk_30": 0.94, "current_decision": "immediate_maintenance"},
        {"predicted_rul": 24.0, "risk_10": 0.30, "risk_30": 0.89, "current_decision": "schedule_maintenance"},
        {"predicted_rul": 44.0, "risk_10": 0.05, "risk_30": 0.40, "current_decision": "schedule_inspection"},
        {"predicted_rul": 120.0, "risk_10": 0.0, "risk_30": 0.01, "current_decision": "continue_operation"},
    ]
    results = [CounterfactualOptimizer.generate_counterfactual(**case).to_dict() for case in cases]

    duration = time.perf_counter() - start_time
    for res in results:
        logger.info("Counterfactual (%s): %s", res["current_decision"], res["actionable_statement"])
    section("ACTIONABLE COUNTERFACTUAL OPTIMIZER COMPLETED")

    return {
        "status": "success",
        "counterfactuals": results,
        "sensitivity_assumptions": dict(OPERATIONAL_SENSITIVITY),
        "duration_seconds": round(duration, 4),
    }


if __name__ == "__main__":
    res = run_counterfactual_demo()
    print(json.dumps(res, indent=2))
