from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, rolling_slope


class TemporalReasoner:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.window = int(self.config.reasoning.evolution_window)
        self.slope_eps = float(self.config.reasoning.trend_slope_eps)


    def find_onset(
        self, engine_history: pd.DataFrame, score_column: str, threshold: float
    ) -> Optional[int]:


        if engine_history.empty or score_column not in engine_history.columns:
            return None

        recent = engine_history.tail(self.window).reset_index(drop=True)
        above = recent[score_column].astype(float) > threshold
        if not above.any():
            return None


        onset_index = len(above) - 1
        for index in range(len(above) - 1, -1, -1):
            if above.iloc[index]:
                onset_index = index
            else:
                break
        return int(recent.iloc[onset_index]["cycle"])


    def classify_evolution(
        self, engine_history: pd.DataFrame, score_column: str
    ) -> Dict[str, Any]:


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


    def analyze(
        self,
        engine_history: pd.DataFrame,
        score_column: str,
        threshold: float,
    ) -> Dict[str, Any]:


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
