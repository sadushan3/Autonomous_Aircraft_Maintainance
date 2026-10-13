"""
adaptive_policy.py

Adaptive Policy Updater & Continuous Learning Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Analyzes historical human engineer feedback logs to dynamically tune decision thresholds
and safety policies without needing full model retraining.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import (
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    DECISION_TO_SEVERITY,
    SupervisorConfig,
)
from app.services.Maintenance_Supervisor.feedback.feedback_store import SupervisorFeedbackStore
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_POLICY_PATH: Final[Path] = ARTIFACT_ROOT / "adaptive_policy_manifest.json"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "adaptive_policy_report.json"

# Below this many logged feedback entries, any threshold nudge would be
# fitting noise from a handful of engineer decisions. Below the minimum,
# the manifest reports the unadjusted config defaults and says so, rather
# than silently pretending a real adjustment was computed.
_MIN_FEEDBACK_FOR_ADJUSTMENT: Final[int] = 5

# Largest fractional nudge applied to a threshold per run, regardless of how
# lopsided the override signal is. Keeps the guardrails from swinging on a
# single noisy batch of feedback.
_MAX_ADJUSTMENT_FRACTION: Final[float] = 0.20

# Per unit of net override signal (escalations minus de-escalations, scaled
# by sample size), the fraction of _MAX_ADJUSTMENT_FRACTION applied.
_ADJUSTMENT_STEP: Final[float] = 0.05


def _compute_threshold_adjustments(
    entries: list[dict],
    cfg: SupervisorConfig,
) -> tuple[dict[str, float], dict[str, object]]:
    """Nudge the RUL and risk_10 immediate-maintenance thresholds from feedback.

    An "escalation" is an engineer overriding to a MORE severe decision than
    the one the supervisor proposed -- evidence the guardrails under-reacted
    and should fire more readily. A "de-escalation" is the reverse -- the
    guardrails were too trigger-happy. The net of the two, normalised by
    sample size, sets the nudge direction and magnitude; the result is
    clamped to +/- _MAX_ADJUSTMENT_FRACTION of the base threshold.

    immediate_rul_threshold rises to catch cases earlier when escalations
    dominate (the guard fires whenever rul <= threshold, so a higher
    threshold fires sooner). critical_risk_10_threshold falls in the same
    direction (the guard fires whenever risk_10 >= threshold, so a lower
    threshold fires sooner).
    """

    scored = [
        (e, DECISION_TO_SEVERITY.get(str(e.get("original_decision", "")).strip().lower()),
         DECISION_TO_SEVERITY.get(str(e.get("engineer_decision", "")).strip().lower()))
        for e in entries
    ]
    scored = [(e, o, n) for e, o, n in scored if o is not None and n is not None]

    escalations = sum(1 for _, o, n in scored if n > o)
    de_escalations = sum(1 for _, o, n in scored if n < o)
    agreements = sum(1 for _, o, n in scored if n == o)

    if len(scored) < _MIN_FEEDBACK_FOR_ADJUSTMENT:
        return (
            {
                "immediate_rul_threshold_adjusted": cfg.immediate_rul_threshold,
                "critical_risk_10_threshold_adjusted": cfg.critical_risk_10_threshold,
                "policy_version": "v1.1-adapted",
            },
            {
                "scored_entries": len(scored),
                "escalations": escalations,
                "de_escalations": de_escalations,
                "agreements": agreements,
                "adjustment_applied": False,
                "reason": (
                    f"Only {len(scored)} usable feedback entries (minimum "
                    f"{_MIN_FEEDBACK_FOR_ADJUSTMENT}); returning unadjusted "
                    "config defaults rather than fitting noise."
                ),
            },
        )

    net_signal = (escalations - de_escalations) / len(scored)
    nudge_fraction = max(-1.0, min(1.0, net_signal)) * _ADJUSTMENT_STEP
    nudge_fraction = max(-_MAX_ADJUSTMENT_FRACTION, min(_MAX_ADJUSTMENT_FRACTION, nudge_fraction))

    rul_adjusted = round(cfg.immediate_rul_threshold * (1.0 + nudge_fraction), 2)
    # risk_10 moves opposite the RUL threshold's direction (see docstring).
    risk_adjusted = round(
        min(0.99, max(0.5, cfg.critical_risk_10_threshold * (1.0 - nudge_fraction))), 4
    )

    return (
        {
            "immediate_rul_threshold_adjusted": rul_adjusted,
            "critical_risk_10_threshold_adjusted": risk_adjusted,
            "policy_version": "v1.1-adapted",
        },
        {
            "scored_entries": len(scored),
            "escalations": escalations,
            "de_escalations": de_escalations,
            "agreements": agreements,
            "net_signal": round(net_signal, 4),
            "nudge_fraction": round(nudge_fraction, 4),
            "adjustment_applied": True,
        },
    )


class AdaptivePolicyEngine:
    """Adapts decision policy thresholds based on human feedback trends."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG
        self.feedback_store = SupervisorFeedbackStore()

    def update_policy(self) -> dict:
        return run_adaptive_policy()


def run_adaptive_policy() -> dict:
    section("ADAPTIVE POLICY ENGINE STARTED")
    start_time = time.perf_counter()

    store = SupervisorFeedbackStore()
    entries = store.load_all_feedback()

    # Calculate override rate per class
    override_counts = {}
    for e in entries:
        orig = e.get("original_decision")
        override_counts[orig] = override_counts.get(orig, 0) + 1

    policy_updates, adjustment_stats = _compute_threshold_adjustments(entries, _CFG)

    _POLICY_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(policy_updates, _POLICY_PATH)

    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "feedback_analyzed_count": len(entries),
        "override_counts": override_counts,
        "adjustment_stats": adjustment_stats,
        "adapted_policy": policy_updates,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Adapted policy written to: %s", _POLICY_PATH)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("ADAPTIVE POLICY ENGINE COMPLETED")

    return report


def main() -> int:
    try:
        res = run_adaptive_policy()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in adaptive_policy: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
