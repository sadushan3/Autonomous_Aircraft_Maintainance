from __future__ import annotations

from typing import Iterable, List

import numpy as np


def to_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(numeric) or np.isinf(numeric):
        return default
    return numeric


def clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, to_float(value)))


def recent_velocity(values: Iterable[float], current: float, increasing: bool = True) -> float:
    history: List[float] = [to_float(v) for v in values]
    if not history:
        return 0.0
    if increasing:
        return current - history[-1]
    return history[-1] - current


def recent_acceleration(
    values: Iterable[float], current: float, increasing: bool = True
) -> float:
    history: List[float] = [to_float(v) for v in values]
    if len(history) < 2:
        return 0.0

    previous_velocity = history[-1] - history[-2] if increasing else history[-2] - history[-1]
    current_velocity = current - history[-1] if increasing else history[-1] - current
    return current_velocity - previous_velocity
