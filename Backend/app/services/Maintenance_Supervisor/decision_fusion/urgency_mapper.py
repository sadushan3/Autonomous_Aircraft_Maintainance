"""
urgency_mapper.py

Maintenance Urgency & Priority Mapper for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Maps final maintenance decision classes to operational priority ratings and recommended
time horizons for engineering execution.

Mappings:
---------
- "continue_operation"     -> Priority: "Low",      Urgency: "none"
- "monitor_closely"        -> Priority: "Medium",   Urgency: "observe_next_cycles"
- "schedule_inspection"    -> Priority: "High",     Urgency: "within_30_cycles"
- "schedule_maintenance"   -> Priority: "High",     Urgency: "within_10_to_30_cycles"
- "immediate_maintenance"  -> Priority: "Critical", Urgency: "immediate"
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

from app.config.Maintenance_Supervisor.supervisor_config import PRIORITY_MAPPING, URGENCY_MAPPING


def map_priority(decision: str) -> str:
    """Map decision string to operational priority level."""
    return PRIORITY_MAPPING.get(str(decision).strip().lower(), "Low")


def map_urgency(decision: str) -> str:
    """Map decision string to recommended operational urgency time horizon."""
    return URGENCY_MAPPING.get(str(decision).strip().lower(), "none")


def map_priorities_and_urgencies(decisions: list[str] | pd.Series) -> tuple[list[str], list[str]]:
    """Batch map decisions to priority and urgency lists."""
    dec_clean = [str(d).strip().lower() for d in decisions]
    priorities = [map_priority(d) for d in dec_clean]
    urgencies = [map_urgency(d) for d in dec_clean]
    return priorities, urgencies
