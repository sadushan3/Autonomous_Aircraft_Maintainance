from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict

EWA_ALPHA = 0.2
NUDGE_STEP = 0.06
MIN_THRESHOLD = 0.05
MAX_THRESHOLD = 0.99
HORIZONS = ("10", "30", "50")


class AdaptiveThresholdService:
    """Exponentially-weighted adaptive alert threshold per FD subset / horizon (Module 5).

    Starts from the static default threshold (Settings.FAILURE_RISK_DEFAULT_THRESHOLDS)
    and nudges it via operator feedback on past alerts: a false alarm pushes the
    threshold up (fewer future false alarms), a confirmed real event pushes it
    down (earlier future detection). Each nudge is blended into the running
    threshold with an EWA (alpha=0.2) so no single feedback event swings it too
    far. Persisted to a local JSON file, same pattern as MemoryService.
    """

    def __init__(self, storage_path: Path, default_thresholds: Dict[str, Dict[str, float]]):
        self.storage_path = storage_path
        self.default_thresholds = default_thresholds
        self._lock = Lock()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("{}", encoding="utf-8")

    def _read(self) -> Dict[str, Any]:
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, state: Dict[str, Any]) -> None:
        self.storage_path.write_text(json.dumps(state), encoding="utf-8")

    def _original(self, fd_subset: str, horizon: str) -> float:
        thresholds = self.default_thresholds.get(fd_subset, {})
        return float(thresholds.get(horizon, thresholds.get(int(horizon), 0.5)))

    def _entry(
        self,
        state: Dict[str, Any],
        fd_subset: str,
        horizon: str,
        seed: float | None = None,
    ) -> Dict[str, Any]:
        """Fetch this fd_subset/horizon's entry, creating it on first feedback.

        A new entry must start from the threshold the system is *actually*
        predicting against -- the model-calibrated value in the artifact
        metadata -- not from ``default_thresholds``, which is a flat 0.5
        placeholder. Seeding from the placeholder means the very first verdict
        silently replaces a calibrated threshold with 0.5: on FD002 that took
        the 30-cycle cut from 0.031 to 0.512 and collapsed recall from 0.80 to
        roughly 0.49, which is the opposite of what a single "false alarm"
        click should do. ``get_threshold`` already documents that it must never
        override a calibrated threshold; this is what makes that true once
        feedback exists.
        """
        fd_state = state.setdefault(fd_subset, {})
        start = self._original(fd_subset, horizon) if seed is None else float(seed)
        return fd_state.setdefault(
            horizon,
            {
                "current": start,
                "previous": start,
                # Provenance of the starting point. ``get_threshold`` refuses to
                # serve an entry that was seeded from the flat config placeholder
                # rather than a calibrated threshold -- see the note there.
                "seeded_from": "calibrated" if seed is not None else "config_default",
                "false_alarm_count": 0,
                "confirmed_count": 0,
                "last_updated": None,
            },
        )

    def get_threshold(self, fd_subset: str, horizon: str, static_fallback: float) -> float:
        """The live threshold failure_risk_service should actually predict against.

        Returns the EWA-adapted value once feedback exists for this fd_subset/horizon;
        otherwise defers to `static_fallback` (the caller's existing metadata-or-config
        baseline), so this never silently overrides a model-calibrated threshold that
        simply hasn't received feedback yet.
        """
        state = self._read()
        entry = state.get(fd_subset, {}).get(horizon)
        if entry is None:
            return static_fallback
        # An entry seeded from the flat 0.5 config placeholder is not an
        # adaptation of the calibrated threshold -- it replaced it. That is how
        # FD001's 30-cycle cut ended up serving 0.488 in place of its calibrated
        # 0.8 after three feedback clicks. Entries written before seed tracking
        # existed carry no provenance, so they are treated the same way: defer
        # to the calibrated value until genuine feedback rebuilds the entry.
        if entry.get("seeded_from") != "calibrated":
            return static_fallback
        return float(entry["current"])

    def record_feedback(
        self,
        fd_subset: str,
        horizon: str,
        was_false_alarm: bool,
        seed: float | None = None,
    ) -> Dict[str, Any]:
        """Nudge one horizon's threshold from an operator verdict.

        Args:
            fd_subset: Subset the alert belonged to.
            horizon: ``'10'``, ``'30'`` or ``'50'``.
            was_false_alarm: True raises the bar, False lowers it.
            seed: Threshold to start adaptation from when no entry exists yet.
                Callers should pass the currently-served (model-calibrated)
                threshold; omitting it falls back to the flat config default.
        """
        with self._lock:
            state = self._read()
            entry = self._entry(state, fd_subset, horizon, seed=seed)

            entry["previous"] = entry["current"]
            target = entry["current"] + (NUDGE_STEP if was_false_alarm else -NUDGE_STEP)
            target = max(MIN_THRESHOLD, min(MAX_THRESHOLD, target))
            entry["current"] = round(EWA_ALPHA * target + (1 - EWA_ALPHA) * entry["current"], 4)

            if was_false_alarm:
                entry["false_alarm_count"] += 1
            else:
                entry["confirmed_count"] += 1
            entry["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            self._write(state)
            return self._status_entry(fd_subset, horizon, entry)

    def _status_entry(self, fd_subset: str, horizon: str, entry: Dict[str, Any]) -> Dict[str, Any]:
        feedback_count = entry["false_alarm_count"] + entry["confirmed_count"]
        trend = "flat"
        if entry["current"] > entry["previous"]:
            trend = "up"
        elif entry["current"] < entry["previous"]:
            trend = "down"
        return {
            "fd_subset": fd_subset,
            "horizon": horizon,
            "original_threshold": self._original(fd_subset, horizon),
            "current_threshold": entry["current"],
            "trend": trend,
            "false_alarm_count": entry["false_alarm_count"],
            "confirmed_count": entry["confirmed_count"],
            "feedback_count": feedback_count,
            "false_alarm_rate": round(entry["false_alarm_count"] / feedback_count, 4) if feedback_count else None,
            "last_updated": entry["last_updated"],
        }

    def status(self, fd_subset: str) -> Dict[str, Dict[str, Any]]:
        state = self._read()
        fd_state = state.get(fd_subset, {})
        result: Dict[str, Any] = {}
        for horizon in HORIZONS:
            entry = fd_state.get(horizon)
            if entry is None:
                original = self._original(fd_subset, horizon)
                result[horizon] = {
                    "fd_subset": fd_subset,
                    "horizon": horizon,
                    "original_threshold": original,
                    "current_threshold": original,
                    "trend": "flat",
                    "false_alarm_count": 0,
                    "confirmed_count": 0,
                    "feedback_count": 0,
                    "false_alarm_rate": None,
                    "last_updated": None,
                }
            else:
                result[horizon] = self._status_entry(fd_subset, horizon, entry)
        return result
