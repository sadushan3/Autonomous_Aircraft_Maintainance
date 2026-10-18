from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.feedback.feedback_store import FeedbackStore
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    read_json,
    utc_now_iso,
)


_MIN_TAIL_MASS = 1e-4

_MAX_HEADROOM_FRACTION: float = 0.8


class ThresholdAdapter:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.path = self.config.paths.model_path(
            self.config.paths.residual_thresholds_json
        )
        self.feedback_store = FeedbackStore(self.config)


    def load_state(self) -> Dict[str, Any]:


        default = {
            "scale": 1.0,
            "fusion_threshold": float(self.config.anomaly.fusion_threshold),
            "feedback_applied": 0,
            "last_updated": None,
        }
        payload = read_json(self.path, default=default)
        return payload or default

    def _save_state(self, state: Dict[str, Any]) -> Path:


        return atomic_write_json(state, self.path)


    def _step_for_verdict(self, verdict: str) -> float:


        settings = self.config.feedback
        if verdict == "false_alarm":
            return float(settings.false_alarm_step)
        if verdict == "missed":
            return float(settings.missed_alarm_step)
        return float(settings.accepted_step)

    def _base_threshold(self) -> float:


        meta = read_json(self.config.paths.model_path("anomaly_fusion_meta.json"), default=None)
        if meta and meta.get("calibrated_fusion_threshold") is not None:
            return float(meta["calibrated_fusion_threshold"])
        return float(self.config.anomaly.fusion_threshold)

    def adapt(self, new_verdicts: Optional[List[str]] = None) -> Dict[str, Any]:


        records = self.feedback_store.load()
        settings = self.config.feedback

        if len(records) < int(settings.min_feedback_records):
            self.logger.info(
                "Only %d feedback record(s); below min_feedback_records=%d, "
                "no adaptation applied.",
                len(records),
                settings.min_feedback_records,
            )
            state = self.load_state()
            return state

        scale = 1.0
        for record in records:
            scale *= self._step_for_verdict(str(record.get("verdict", "accepted")))
        scale = max(min(scale, float(settings.max_threshold_scale)), float(settings.min_threshold_scale))

        base_threshold = self._base_threshold()


        tail_mass = max(1.0 - base_threshold, _MIN_TAIL_MASS)
        adapted_threshold = 1.0 - (tail_mass / scale)


        floor = base_threshold * float(settings.min_threshold_scale)
        adapted_threshold = max(adapted_threshold, floor)


        ceiling = base_threshold + (1.0 - base_threshold) * _MAX_HEADROOM_FRACTION

        state = {
            "scale": round(scale, 6),
            "fusion_threshold": round(adapted_threshold, 6),
            "base_threshold": base_threshold,
            "effective_scale": round(adapted_threshold / base_threshold, 6)
            if base_threshold > 0
            else 1.0,
            "clamped": bool(adapted_threshold <= floor),
            "tail_mass_base": round(tail_mass, 6),
            "tail_mass_adapted": round(max(1.0 - adapted_threshold, 0.0), 6),
            "reachable_ceiling": round(ceiling, 6),
            "feedback_applied": len(records),
            "last_updated": utc_now_iso(),
        }
        self._save_state(state)
        self.logger.info(
            "Threshold adapted from %d feedback record(s): scale=%.4f -> threshold=%.4f",
            len(records),
            scale,
            adapted_threshold,
        )
        return state

    def current_threshold(self) -> float:


        state = self.load_state()
        return float(state.get("fusion_threshold", self.config.anomaly.fusion_threshold))


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    adapter = ThresholdAdapter(config)
    state = adapter.adapt()
    logger.info("Adaptation state: %s", state)


if __name__ == "__main__":
    main()
