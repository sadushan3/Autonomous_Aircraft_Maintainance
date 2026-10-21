from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_score_engine import HealthScoreEngine
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import HealthStateClassifier
from app.utils.Anomaly_Health_Monitoring.model_utils import sensor_from_residual
from app.utils.Anomaly_Health_Monitoring.utils import read_json


class HealthExplainer:
    def __init__(self, config: Any) -> None:
        self.config = config

    def explain(self, residuals: pd.DataFrame, health: pd.DataFrame) -> Dict[Tuple[int, int], Dict[str, Any]]:

        keys = ["unit", "cycle"]
        if health.empty or residuals.empty:
            return {}
        if residuals.duplicated(keys).any() or health.duplicated(keys).any():
            raise ValueError("Health explanations require unique engine/cycle identities.")
        calculator = HealthIndexCalculator(self.config)
        calibration = read_json(calculator.calibration_path(), default={})
        columns = [c for c in calibration.get("informative_columns", []) if c in residuals]
        if not columns or "healthy_reference_deviation" not in calibration:
            return {}
        ordered = residuals.sort_values(keys).reset_index(drop=True)

        computed = calculator.compute(ordered)
        scored = HealthScoreEngine(self.config).compute(computed)
        classifier = HealthStateClassifier(self.config)
        thresholds = read_json(classifier.thresholds_path, default={})
        classifier.thresholds = thresholds.get("thresholds", classifier.thresholds)
        health_by_id = health.set_index(keys)
        joined = computed.merge(health[keys + ["health_index"]], on=keys,
                                how="left", suffixes=("_reconstructed", "_saved"), validate="one_to_one")
        policy = calibration.get("sensor_weights") or {}
        weights = np.array([policy.get(sensor_from_residual(c), 1.) for c in columns], dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(len(columns))
        weights /= weights.sum()
        window = int(calibration.get("residual_window", 1))
        sensor_values = calculator.sensor_values(ordered, columns, window)
        weighted = np.abs(sensor_values) * weights
        deviation = weighted.sum(axis=1)
        previous = joined.groupby("unit").health_index_reconstructed.shift(1)
        alpha = float(calibration.get("ewma_alpha", self.config.health.ewma_alpha))
        reference = max(float(calibration["healthy_reference_deviation"]), 1e-9)
        tolerance = float(self.config.explainability.fidelity_tolerance)
        result = {}
        for i, row in joined.iterrows():
            if not np.isfinite(weighted[i]).all() or pd.isna(row.health_index_saved):
                continue
            error = abs(float(row.health_index_saved) - float(row.health_index_reconstructed))
            saved_row = health_by_id.loc[(row.unit, row.cycle)]
            health_score = float(scored.iloc[i].health_score)
            state = classifier._classify_scalar(health_score)
            state_details = {}
            if "health_score" in saved_row and pd.notna(saved_row.health_score):
                score_error = abs(health_score - float(saved_row.health_score))
                error = max(error, score_error)
                state_matches = "health_state" not in saved_row or saved_row.health_state == state
                if not state_matches:
                    error = max(error, tolerance * 2)
                boundaries = classifier.thresholds
                intervals = {
                    "Healthy": (boundaries.get("healthy", 80.), None),
                    "Degrading": (boundaries.get("degrading", 60.), boundaries.get("healthy", 80.)),
                    "Warning": (boundaries.get("warning", 40.), boundaries.get("degrading", 60.)),
                    "Critical": (boundaries.get("critical", 20.), boundaries.get("warning", 40.)),
                    "Failure Risk": (0., boundaries.get("critical", 20.)),
                }
                lower, upper = intervals[state]
                state_details = {
                    "health_score": float(saved_row.health_score), "reconstructed_health_score": health_score,
                    "health_score_error": score_error, "health_score_trend_adjustment": health_score - float(row.health_index_reconstructed),
                    "health_trend": float(scored.iloc[i].health_trend), "health_state": state,
                    "state_matches": bool(state_matches), "state_lower_bound": float(lower),
                    "state_upper_bound": float(upper) if upper is not None else None,
                }
            first = pd.isna(previous.iloc[i])
            current_term = float(row.health_index_raw) * (1. if first else alpha)
            history_term = 0. if first else (1. - alpha) * float(previous.iloc[i])
            top = [{"sensor": sensor_from_residual(columns[j]), "weight": float(weights[j]),
                    "absolute_residual": float(abs(ordered.iloc[i][columns[j]])),
                    "smoothed_absolute_residual": float(abs(sensor_values[i, j])),
                    "weighted_deviation": float(weighted[i, j]),
                    "deviation_share": float(weighted[i, j] / deviation[i]) if deviation[i] else 0.}
                   for j in np.argsort(-weighted[i])[:5]]
            result[(int(row.unit), int(row.cycle))] = {
                "health_explanation_status": "computed" if error <= tolerance else "reference_mismatch",
                "health_evidence": {
                    "health_index": float(row.health_index_saved),
                    "reconstructed_health_index": float(row.health_index_reconstructed),
                    "reconstruction_error": error, "raw_health_index": float(row.health_index_raw),
                    "previous_health_index": None if first else float(previous.iloc[i]),
                    "current_cycle_term": current_term, "history_term": history_term,
                    "ewma_alpha": 1. if first else alpha, "weighted_deviation": float(deviation[i]),
                    "healthy_reference_deviation": reference,
                    "excess_over_healthy": float(max(deviation[i] / reference - 1., 0.)),
                    "decay_lambda": float(calibration.get("decay_lambda", self.config.health.decay_lambda)),
                    "residual_window": window, "top_sensors": top,
                    **state_details,
                    "attribution_scope": "Shares of weighted residual deviation before nonlinear decay; smoothing terms are health points.",
                },
            }
        return result
