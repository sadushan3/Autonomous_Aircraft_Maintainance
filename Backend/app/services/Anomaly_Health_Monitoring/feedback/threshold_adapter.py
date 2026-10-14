"""
Threshold adaptation from operator feedback.

This is where the system actually learns from operators, without retraining
any model.  Every verdict nudges the fusion threshold multiplicatively:

* ``accepted``   -> scale stays at 1.0 (no change; the current threshold was right)
* ``false_alarm``-> scale *= false_alarm_step (>1)  -- raise the bar, fewer false alarms
* ``missed``     -> scale *= missed_alarm_step (<1) -- lower the bar, catch it next time

The scale is clamped to ``[min_threshold_scale, max_threshold_scale]`` so
repeated feedback cannot push the threshold to a degenerate extreme, and
adaptation only engages once at least ``min_feedback_records`` verdicts exist,
so a single early data point cannot swing the system.

The result is written to ``models/residual_thresholds.json`` as a scale factor
applied to ``config.anomaly.fusion_threshold`` -- :mod:`anomaly_fusion` reads
this file on every run, which is the entire feedback loop: no code path
outside this module and that read needs to know adaptation happened at all.
"""

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


#: How far the adapted threshold may travel from the calibrated base toward
#: the score ceiling of 1.0. At 0.8 a base of 0.9687 can rise to ~0.9937 --
#: still meaningfully stricter, but always reachable by a real anomaly score,
#: so feedback can desensitise the detector without ever disabling it.
#: Smallest exceedance mass the adapter will work with, so a degenerate
#: base threshold of exactly 1.0 cannot produce a zero-width tail.
_MIN_TAIL_MASS = 1e-4

_MAX_HEADROOM_FRACTION: float = 0.8


class ThresholdAdapter:
    """Derives an adapted fusion threshold from accumulated operator feedback."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the adapter.

        Args:
            config: Configuration supplying the nudge steps and clamp bounds.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.path = self.config.paths.model_path(
            self.config.paths.residual_thresholds_json
        )
        self.feedback_store = FeedbackStore(self.config)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def load_state(self) -> Dict[str, Any]:
        """Read the persisted threshold-adaptation state.

        Returns:
            The state document, defaulting to scale=1.0 when none exists yet.
        """
        default = {
            "scale": 1.0,
            "fusion_threshold": float(self.config.anomaly.fusion_threshold),
            "feedback_applied": 0,
            "last_updated": None,
        }
        payload = read_json(self.path, default=default)
        return payload or default

    def _save_state(self, state: Dict[str, Any]) -> Path:
        """Persist the threshold-adaptation state atomically.

        Args:
            state: The state document to write.

        Returns:
            The path written.
        """
        return atomic_write_json(state, self.path)

    # ------------------------------------------------------------------ #
    # Adaptation
    # ------------------------------------------------------------------ #
    def _step_for_verdict(self, verdict: str) -> float:
        """Return the multiplicative nudge for one verdict.

        Args:
            verdict: ``'accepted'``, ``'false_alarm'`` or ``'missed'``.

        Returns:
            The scale multiplier to apply.
        """
        settings = self.config.feedback
        if verdict == "false_alarm":
            return float(settings.false_alarm_step)
        if verdict == "missed":
            return float(settings.missed_alarm_step)
        return float(settings.accepted_step)

    def _base_threshold(self) -> float:
        """Return the threshold feedback scaling should be applied on top of.

        Prefers the fusion threshold calibrated from the healthy population
        (``anomaly_fusion_meta.json``, written by
        :meth:`AnomalyFusion._calibrate_fusion_threshold`) over the flat
        ``config.anomaly.fusion_threshold`` constant, since that constant is
        only a pre-calibration fallback and scaling it directly would apply
        operator feedback on top of a number with no relationship to what the
        detectors currently consider anomalous.

        Returns:
            The base threshold feedback scaling multiplies.
        """
        meta = read_json(self.config.paths.model_path("anomaly_fusion_meta.json"), default=None)
        if meta and meta.get("calibrated_fusion_threshold") is not None:
            return float(meta["calibrated_fusion_threshold"])
        return float(self.config.anomaly.fusion_threshold)

    def adapt(self, new_verdicts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Recompute the adapted threshold from all stored feedback.

        Recomputing from the full feedback history (rather than only
        incrementally applying ``new_verdicts``) keeps the state file
        idempotent and simple to audit: the scale is always a pure function of
        the feedback store's contents.

        Args:
            new_verdicts: Unused directly; accepted for API symmetry with the
                caller, which may want to report how many new verdicts drove
                this recomputation.

        Returns:
            The updated adaptation state.
        """
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

        # Adapt the TAIL MASS, not the threshold.
        #
        # The fused anomaly score is a percentile rank bounded in [0, 1], and
        # the quantity feedback is really about is the exceedance mass above
        # the cut -- roughly the flag rate. Multiplying the *threshold* by a
        # scale is meaningless once the threshold is already near 1.0: at
        # base=0.9542 a single 'false_alarm' verdict (scale=1.05) gives
        # 1.0019, which is above the maximum achievable score, so the old
        # ceiling clamp fired on the very first verdict and pinned the cut at
        # 0.9908 -- 80% of the way to 1.0. Every subsequent verdict landed on
        # the same clamped value, so 1, 2, 10 and 20 verdicts were
        # indistinguishable and the loop had no resolution at all. Measured
        # cost on FD001: flag rate fell 15.5% -> 7.3%, 24 of 100 engines were
        # never flagged, and median lead time collapsed from 50 to 14 cycles.
        #
        # Scaling the tail instead is well-defined everywhere: a false alarm
        # divides the exceedance mass by ``scale`` (flag proportionally fewer
        # rows), a missed detection multiplies it (flag more). The result is
        # strictly inside (0, 1) for any positive scale, so it approaches 1.0
        # asymptotically and can never disable detection -- no ceiling clamp is
        # needed, and every verdict moves the cut by a distinguishable amount.
        tail_mass = max(1.0 - base_threshold, _MIN_TAIL_MASS)
        adapted_threshold = 1.0 - (tail_mass / scale)

        # A floor guards the opposite failure: sustained 'missed' verdicts
        # driving the cut so low that everything is flagged.
        floor = base_threshold * float(settings.min_threshold_scale)
        adapted_threshold = max(adapted_threshold, floor)

        # Retained for reporting continuity; the tail rule cannot exceed it.
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
        """Return the threshold anomaly_fusion should currently apply.

        Returns:
            The adapted fusion threshold, falling back to the config default
            when no adaptation has run yet.
        """
        state = self.load_state()
        return float(state.get("fusion_threshold", self.config.anomaly.fusion_threshold))


def main() -> None:
    """Recompute the adapted threshold from whatever feedback currently exists."""
    config = get_config()
    logger = get_logger(__name__, config)
    adapter = ThresholdAdapter(config)
    state = adapter.adapt()
    logger.info("Adaptation state: %s", state)


if __name__ == "__main__":
    main()
