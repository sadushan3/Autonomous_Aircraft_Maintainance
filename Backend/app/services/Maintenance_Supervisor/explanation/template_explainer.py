"""
template_explainer.py

Natural Language Explanation Template Generator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Generates human-readable, domain-specific justification narratives explaining WHY a particular
maintenance decision, priority, and urgency level was assigned.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def generate_decision_narrative(row: pd.Series, decision: str) -> str:
    """Generate domain narrative text for a single prediction."""
    rul = float(row.get("predicted_rul", 100.0))
    risk = float(row.get("risk_30", 0.0))
    anom = float(row.get("anomaly_score", 0.0))
    anom_sev = str(row.get("anomaly_severity", "none")).strip().lower()

    dec_clean = str(decision).strip().lower()

    if dec_clean == "immediate_maintenance":
        return (
            f"CRITICAL SAFETY ACTION: Immediate maintenance required. Equipment displays critical "
            f"degradation with predicted RUL of {rul:.1f} cycles, 30-cycle failure risk of {risk:.1%}, "
            f"and anomaly severity rated '{anom_sev}' (score: {anom:.2f})."
        )
    elif dec_clean == "schedule_maintenance":
        return (
            f"SCHEDULED MAINTENANCE REQUIRED: Recommended within next 10-30 cycles. RUL is {rul:.1f} cycles "
            f"with elevated 30-cycle failure probability of {risk:.1%}."
        )
    elif dec_clean == "schedule_inspection":
        return (
            f"INSPECTION ADVISED: High-priority engineering inspection recommended within 30 cycles. "
            f"Detected anomaly severity '{anom_sev}' (score: {anom:.2f}) and RUL of {rul:.1f} cycles."
        )
    elif dec_clean == "monitor_closely":
        return (
            f"INCREASED MONITORING: Equipment operating under elevated risk/anomaly parameters. "
            f"Increase telemetry sampling frequency. Current RUL: {rul:.1f} cycles, Risk: {risk:.1%}."
        )
    else:
        return (
            f"NORMAL OPERATION: All upstream health, RUL, and failure risk metrics remain within nominal "
            f"operating thresholds. Predicted RUL: {rul:.1f} cycles."
        )
