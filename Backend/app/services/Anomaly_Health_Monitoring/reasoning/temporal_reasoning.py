"""
Temporal reasoning: how an anomaly evolved over time.

Given an engine's residual/anomaly-score history up to and including the
current cycle, this module answers three questions:

* When did the deviation begin (``onset_cycle``)?
* How has it developed since -- ``stable``, ``gradual``, ``abrupt`` or
  ``intermittent``?
* What is its current slope?

The evolution pattern matters for root-cause plausibility: an *abrupt* jump is
more consistent with a discrete event (foreign object damage, a seal failure)
than with wear, while a *gradual* rise over many cycles is the classic
signature of progressive degradation (blade erosion, fouling).
``intermittent`` -- an anomaly that repeatedly appears and reverts -- often
points to a marginal component or sensor rather than sustained damage, and is
flagged as worth watching rather than acting on immediately.

All slope computation is trailing and per-engine, matching the causal-window
discipline used throughout the pipeline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, rolling_slope


class TemporalReasoner:
    """Characterises how an anomaly has evolved across an engine's history."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the reasoner.

        Args:
            config: Configuration supplying the evolution window and slope
                threshold.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.window = int(self.config.reasoning.evolution_window)
        self.slope_eps = float(self.config.reasoning.trend_slope_eps)

    # ------------------------------------------------------------------ #
    # Onset detection
    # ------------------------------------------------------------------ #
    def find_onset(
        self, engine_history: pd.DataFrame, score_column: str, threshold: float
    ) -> Optional[int]:
        """Find the earliest recent cycle where the score first crossed threshold.

        Searches backward from the current (last) row through the engine's
        history, within the evolution window, for the first cycle where the
        score exceeded the threshold and stayed elevated through to now.

        Args:
            engine_history: This engine's rows, sorted by cycle, ending at the
                cycle being reasoned about.
            score_column: Column to threshold, e.g. ``anomaly_score``.
            threshold: The value above which a cycle counts as deviating.

        Returns:
            The onset cycle, or None if no crossing is found in the window.
        """
        if engine_history.empty or score_column not in engine_history.columns:
            return None

        recent = engine_history.tail(self.window).reset_index(drop=True)
        above = recent[score_column].astype(float) > threshold
        if not above.any():
            return None

        # Walk backward from the end to find the start of the current run.
        onset_index = len(above) - 1
        for index in range(len(above) - 1, -1, -1):
            if above.iloc[index]:
                onset_index = index
            else:
                break
        return int(recent.iloc[onset_index]["cycle"])

    # ------------------------------------------------------------------ #
    # Evolution pattern
    # ------------------------------------------------------------------ #
    def classify_evolution(
        self, engine_history: pd.DataFrame, score_column: str
    ) -> Dict[str, Any]:
        """Classify how the score has evolved over the trailing window.

        Args:
            engine_history: This engine's rows, sorted by cycle, ending at the
                cycle being reasoned about.
            score_column: Column to analyse, e.g. ``anomaly_score`` or
                ``residual_abs_mean``.

        Returns:
            A mapping with ``pattern``, ``slope`` and ``duration_cycles``.
        """
        recent = engine_history.tail(self.window)
        if len(recent) < 3 or score_column not in recent.columns:
            return {"pattern": "stable", "slope": 0.0, "duration_cycles": 0}

        series = recent[score_column].astype(float).reset_index(drop=True)
        slope_series = rolling_slope(series, min(len(series), self.window))
        slope = float(slope_series.iloc[-1])

        diffs = series.diff().dropna()
        volatility = float(diffs.std()) if len(diffs) > 1 else 0.0
        mean_level = float(series.mean()) or 1e-9
        relative_volatility = volatility / max(abs(mean_level), 1e-9)

        # A large single-step jump relative to the trailing spread => abrupt.
        max_step = float(diffs.abs().max()) if len(diffs) else 0.0
        is_abrupt = max_step > 3.0 * max(volatility, 1e-9) and max_step > 0.1

        sign_changes = int((np.sign(diffs.replace(0, np.nan).dropna())).diff().abs().gt(0).sum())
        is_intermittent = sign_changes >= max(len(diffs) // 3, 2) and relative_volatility > 0.5

        if is_abrupt:
            pattern = "abrupt"
        elif is_intermittent:
            pattern = "intermittent"
        elif abs(slope) > self.slope_eps:
            pattern = "gradual"
        else:
            pattern = "stable"

        duration = int((series > series.median()).sum()) if pattern != "stable" else 0

        return {
            "pattern": pattern,
            "slope": round(slope, 6),
            "duration_cycles": duration,
            "relative_volatility": round(relative_volatility, 4),
        }

    # ------------------------------------------------------------------ #
    # Orchestration
    # ------------------------------------------------------------------ #
    def analyze(
        self,
        engine_history: pd.DataFrame,
        score_column: str,
        threshold: float,
    ) -> Dict[str, Any]:
        """Run onset detection and evolution classification together.

        Args:
            engine_history: This engine's rows, sorted by cycle, ending at the
                cycle being reasoned about.
            score_column: Column to analyse.
            threshold: Deviation threshold for onset detection.

        Returns:
            A combined mapping of onset and evolution facts.
        """
        onset = self.find_onset(engine_history, score_column, threshold)
        evolution = self.classify_evolution(engine_history, score_column)

        current_cycle = (
            int(engine_history["cycle"].iloc[-1]) if len(engine_history) else None
        )
        duration = (
            int(current_cycle - onset) if onset is not None and current_cycle is not None else 0
        )

        return {
            "onset_cycle": onset,
            "duration_cycles": max(duration, evolution["duration_cycles"]),
            "evolution_pattern": evolution["pattern"],
            "trend_slope": evolution["slope"],
        }


def main() -> None:
    """Demonstrate temporal reasoning on a synthetic degrading trajectory."""
    config = get_config()
    logger = get_logger(__name__, config)
    reasoner = TemporalReasoner(config)

    cycles = list(range(1, 41))
    scores = [0.05] * 25 + list(np.linspace(0.1, 0.9, 15))
    history = pd.DataFrame({"cycle": cycles, "anomaly_score": scores})

    result = reasoner.analyze(history, "anomaly_score", threshold=0.5)
    logger.info("Gradual-rise result: %s", result)

    abrupt_scores = [0.05] * 30 + [0.95] * 10
    abrupt_history = pd.DataFrame({"cycle": cycles, "anomaly_score": abrupt_scores})
    logger.info("Abrupt-jump result: %s", reasoner.analyze(abrupt_history, "anomaly_score", threshold=0.5))


if __name__ == "__main__":
    main()
