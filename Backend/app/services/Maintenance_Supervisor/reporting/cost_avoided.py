"""
cost_avoided.py

Estimated Cost-Avoidance Calculator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Maps a supervisor `final_decision` to its associated action cost (as configured in
`SupervisorConfig`) and estimates the cost avoided by acting on that decision now
instead of letting the asset run to an unplanned failure.

Scaling by live risk/RUL magnitude
-----------------------------------
`estimated_cost_avoided` used to be `cost_unplanned_failure - action_cost`
unconditionally -- the same number for every engine that received the same
decision class, whether its risk_10 was 0.83 or 0.99. The unplanned-failure
cost is only realised with some probability, not with certainty, and that
probability is exactly what the Failure-Risk agent already estimates per
engine per cycle. `risk_10`/`risk_30`, when supplied, now weight the
failure cost by that probability instead of treating it as certain; when
they are not supplied (e.g. a bare decision-string call with no upstream
context), this falls back to the previous flat-cost behaviour, clearly
labelled as such in the response.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import SupervisorConfig

# Maps each DECISION_CLASSES value to the SupervisorConfig cost field that represents
# the cost of taking that action. `monitor_closely` and `schedule_inspection` do not
# have identically-named cost constants, so they are mapped to the closest-matching
# configured cost (cost_monitoring, cost_inspection respectively).
DECISION_TO_COST_ATTR: Final[dict[str, str]] = {
    "continue_operation": "cost_continue_operation",
    "monitor_closely": "cost_monitoring",
    "schedule_inspection": "cost_inspection",
    "schedule_maintenance": "cost_scheduled_maintenance",
    "immediate_maintenance": "cost_immediate_maintenance",
}


def _failure_probability(
    risk_10: float | None,
    risk_30: float | None,
    predicted_rul: float | None,
) -> tuple[float | None, str]:
    """Best available estimate of P(unplanned failure | no action taken), and its source."""
    if risk_10 is not None:
        return max(0.0, min(1.0, float(risk_10))), "risk_10"
    if risk_30 is not None:
        return max(0.0, min(1.0, float(risk_30))), "risk_30"
    if predicted_rul is not None:
        # No risk estimate at all: fall back to a coarse RUL-based proxy
        # (shorter RUL -> higher implied near-term failure probability).
        # This is explicitly a proxy, not a model output -- see "source".
        proxy = max(0.0, min(1.0, 1.0 - float(predicted_rul) / 100.0))
        return proxy, "predicted_rul_proxy"
    return None, "unavailable"


def estimate_cost_avoided(
    final_decision: str,
    config: SupervisorConfig | None = None,
    risk_10: float | None = None,
    risk_30: float | None = None,
    predicted_rul: float | None = None,
) -> dict:
    """Cost of the recommended action tier vs. the probability-weighted cost of unplanned failure.

    `risk_10` (preferred), `risk_30`, or `predicted_rul` -- in that order --
    weight `cost_unplanned_failure` by how likely failure actually is for
    this specific engine/cycle. With none of the three supplied, the
    failure cost is treated as certain (the old flat-cost behaviour),
    and `failure_probability_source` reports "unavailable" so a caller
    can tell the two cases apart.
    """
    cfg = config or SupervisorConfig()

    decision_key = str(final_decision).strip().lower()
    cost_attr = DECISION_TO_COST_ATTR.get(decision_key)
    if cost_attr is None:
        raise ValueError(
            f"Unknown final_decision '{final_decision}'. "
            f"Expected one of: {sorted(DECISION_TO_COST_ATTR)}"
        )

    action_cost = float(getattr(cfg, cost_attr))
    unplanned_failure_cost = float(cfg.cost_unplanned_failure)

    p_failure, source = _failure_probability(risk_10, risk_30, predicted_rul)
    effective_p_failure = 1.0 if p_failure is None else p_failure
    expected_failure_cost = unplanned_failure_cost * effective_p_failure

    return {
        "decision": decision_key,
        "action_cost": action_cost,
        "unplanned_failure_cost": unplanned_failure_cost,
        "failure_probability": p_failure,
        "failure_probability_source": source,
        "expected_failure_cost": expected_failure_cost,
        "estimated_cost_avoided": expected_failure_cost - action_cost,
    }
