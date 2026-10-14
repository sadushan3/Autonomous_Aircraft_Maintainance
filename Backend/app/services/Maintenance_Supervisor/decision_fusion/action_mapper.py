"""
action_mapper.py

Actionable Recommendation & Engineering Directive Mapper for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Translates formal maintenance decision classes into human-readable, domain-specific
engineering directives and action recommendations for maintenance personnel.
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

from app.config.Maintenance_Supervisor.supervisor_config import ACTION_MAPPING


def map_action(decision: str) -> str:
    """Map decision string to actionable engineering directive."""
    return ACTION_MAPPING.get(
        str(decision).strip().lower(),
        "Continue normal operation and routine monitoring."
    )


def map_action_recommendations(decisions: list[str] | pd.Series) -> list[str]:
    """Batch map decisions to actionable recommendation strings."""
    return [map_action(d) for d in decisions]
