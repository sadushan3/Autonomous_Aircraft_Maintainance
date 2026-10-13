"""
Learning updater: the feedback-layer orchestrator.

Applies one operator verdict end to end:

1. Record it durably (:mod:`feedback_store`).
2. Update the matching anomaly memory's verdict, if it can be matched
   (:mod:`alert_memory` / :mod:`reasoning.root_cause_tracker`).
3. Recompute the adapted anomaly threshold from the full feedback history
   (:mod:`threshold_adapter`).
4. When enough ``false_alarm``/``missed`` evidence exists for a specific
   health-state boundary, nudge that boundary too
   (:mod:`health_monitoring.health_state_classifier`), on the same
   multiplicative philosophy as the anomaly threshold.

This module is the only place feedback -> system-behaviour actually happens;
every other module in the feedback package is either a store or a pure
calculator that this orchestrator calls.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.feedback.alert_memory import AlertMemory
from app.services.Anomaly_Health_Monitoring.feedback.feedback_store import FeedbackStore
from app.services.Anomaly_Health_Monitoring.feedback.threshold_adapter import (
    ThresholdAdapter,
)
from app.services.Anomaly_Health_Monitoring.health_monitoring.health_state_classifier import (
    HealthStateClassifier,
)
from app.utils.Anomaly_Health_Monitoring.model_utils import get_identity_columns
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_csv,
    atomic_write_json,
    get_logger,
)


#: Smallest width the Critical band may be squeezed to by threshold nudging.
#: Critical spans ``[critical, warning)``; without a floor, repeated nudges to
#: 'warning' collapse it and engines jump Warning -> Failure Risk directly.
MIN_CRITICAL_BAND = 10.0


class LearningUpdater:
    """Applies operator feedback across the memory, threshold and state layers."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the updater and its collaborators.

        Args:
            config: Configuration passed through to every collaborator.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.feedback_store = FeedbackStore(self.config)
        self.alert_memory = AlertMemory(self.config)
        self.threshold_adapter = ThresholdAdapter(self.config)
        self.state_classifier = HealthStateClassifier(self.config)

    # ------------------------------------------------------------------ #
    # Single-verdict application
    # ------------------------------------------------------------------ #
    def apply_feedback(
        self,
        engine_id: int,
        cycle: int,
        verdict: str,
        operator: str = "unknown",
        comment: str = "",
        confirmed_root_cause: Optional[str] = None,
        anomaly_id: Optional[str] = None,
        system_root_cause: Optional[str] = None,
        severity: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record one verdict and propagate it through memory and thresholds.

        Args:
            engine_id: Engine the alert concerned.
            cycle: Cycle the alert concerned.
            verdict: ``'accepted'``, ``'false_alarm'`` or ``'missed'``.
            operator: Identifier of the submitting operator.
            comment: Free-text note.
            confirmed_root_cause: The cause actually found, if different.
            anomaly_id: The memory record id to update, if known.
            system_root_cause: The cause the system had proposed.
            severity: The severity the system had assigned.
            anomaly_score: The fused anomaly score at the time of the alert.

        Returns:
            A summary of every effect this feedback had.
        """
        feedback_record = self.feedback_store.add(
            engine_id=engine_id,
            cycle=cycle,
            verdict=verdict,
            operator=operator,
            comment=comment,
            confirmed_root_cause=confirmed_root_cause,
            system_root_cause=system_root_cause,
            anomaly_id=anomaly_id,
            severity=severity,
            anomaly_score=anomaly_score,
        )

        memory_updated = None
        if anomaly_id:
            memory_updated = self.alert_memory.update_verdict(
                anomaly_id, verdict, confirmed_root_cause
            )

        threshold_before = self.threshold_adapter.current_threshold()
        threshold_state = self.threshold_adapter.adapt()
        threshold_after = float(threshold_state.get("fusion_threshold", threshold_before))

        state_thresholds_before = dict(self.state_classifier.load_thresholds())
        state_thresholds_after = self._maybe_adapt_state_thresholds(verdict)

        summary: Dict[str, Any] = {
            "feedback_id": feedback_record["feedback_id"],
            "memory_updated": memory_updated is not None,
            "threshold_before": threshold_before,
            "threshold_after": threshold_after,
            "state_thresholds_before": state_thresholds_before,
            "state_thresholds_after": state_thresholds_after,
        }
        self.logger.info("Applied feedback: %s", summary)
        return summary

    def _maybe_adapt_state_thresholds(self, verdict: str) -> Dict[str, float]:
        """Nudge health-state thresholds after a run of consistent false-alarm feedback.

        A conservative rule: only adjust once the false-alarm rate among all
        recorded feedback exceeds 30%, and only nudge the 'warning' boundary,
        which is the one most exposed to over-eager escalation. This keeps the
        threshold classifier auditable -- it moves rarely and by a small,
        explained amount.

        The nudge is floored so it cannot collapse the band beneath it. Nudging
        'warning' alone repeatedly squeezes the Critical band, because Critical
        is defined as ``critical <= HI < warning`` and 'critical' never moves:
        eleven nudges took 'warning' from 40.0 to 22.75, leaving Critical just
        2.75 points wide against Warning's 37.25. Engines then skip straight
        from Warning to Failure Risk -- observed as 432 Critical rows versus
        701 Failure Risk on a run where Critical should be the wider band.
        ``MIN_CRITICAL_BAND`` keeps that band meaningful; once the floor is
        reached the nudge stops rather than degrading the taxonomy further.

        Args:
            verdict: The verdict just recorded (informational; the decision is
                based on the full history, not this single verdict).

        Returns:
            The resulting health-state thresholds.
        """
        summary = self.feedback_store.summary()
        total = summary["total"]
        false_alarms = summary["by_verdict"].get("false_alarm", 0)
        if total < int(self.config.feedback.min_feedback_records) or total == 0:
            return self.state_classifier.thresholds

        false_alarm_rate = false_alarms / total
        if false_alarm_rate > 0.3:
            current = self.state_classifier.load_thresholds()
            nudged = dict(current)
            critical = float(current.get("critical", 20.0))
            proposed = min(
                float(current.get("warning", 40.0)) * 0.95,
                float(current.get("degrading", 60.0)) - 1.0,
            )
            floor = critical + MIN_CRITICAL_BAND
            if proposed < floor:
                self.logger.info(
                    "False-alarm rate %.1f%% exceeds 30%%, but lowering 'warning' to "
                    "%.2f would leave the Critical band only %.2f wide (floor %.1f); "
                    "holding at %.2f.",
                    100.0 * false_alarm_rate,
                    proposed,
                    proposed - critical,
                    MIN_CRITICAL_BAND,
                    max(float(current.get("warning", 40.0)), floor),
                )
                nudged["warning"] = max(float(current.get("warning", 40.0)), floor)
                self.state_classifier.update_thresholds(nudged)
                return nudged

            nudged["warning"] = proposed
            self.state_classifier.update_thresholds(nudged)
            self.logger.info(
                "False-alarm rate %.1f%% exceeds 30%%; lowered 'warning' threshold to "
                "%.2f (Critical band %.2f wide)",
                100.0 * false_alarm_rate,
                nudged["warning"],
                nudged["warning"] - critical,
            )
            return nudged
        return self.state_classifier.thresholds

    # ------------------------------------------------------------------ #
    # Batch stage entry point
    # ------------------------------------------------------------------ #
    def run(self, root_causes: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Execute the feedback-learning pipeline stage.

        This does not require new feedback to be submitted in the same call;
        it (a) ingests any freshly reasoned anomalies into memory so they are
        matchable in future, and (b) reports the current adaptation state so
        the pipeline's ``feedback_updates.csv`` reflects the latest picture.

        Args:
            root_causes: Optional in-memory root-cause analysis output to
                ingest into memory; otherwise ``root_cause_analysis.csv`` is
                read from disk when present.

        Returns:
            A report of the current feedback and adaptation state.
        """
        self.config.ensure_directories()
        source = root_causes
        if source is None:
            path = self.config.paths.processed_path(self.config.paths.root_cause_analysis_csv)
            if path.is_file():
                from app.utils.Anomaly_Health_Monitoring.utils import read_csv

                source = read_csv(path, stage="learning_updater")

        ingested = self.alert_memory.ingest_root_causes(source) if source is not None else 0

        feedback_records = self.feedback_store.load()
        threshold_state = self.threshold_adapter.adapt()
        memory_summary = self.alert_memory.summary()

        changes = pd.DataFrame(
            [
                {
                    "feedback_id": r.get("feedback_id"),
                    # Carry-forward CSVs are keyed on 'unit'; 'engine_id' is
                    # the alias used only at the API / exchange boundary. The
                    # feedback *store* speaks the API's language because that
                    # is where operator verdicts arrive, so translate here.
                    "unit": r.get("engine_id"),
                    "cycle": r.get("cycle"),
                    "verdict": r.get("verdict"),
                    "threshold_before": None,
                    "threshold_after": threshold_state.get("fusion_threshold"),
                    "adjustment_reason": "recomputed from full feedback history",
                    "recorded_at": r.get("recorded_at"),
                }
                for r in feedback_records
            ]
        )
        if changes.empty:
            changes = pd.DataFrame(
                columns=[
                    "feedback_id", "unit", "cycle", "verdict",
                    "threshold_before", "threshold_after", "adjustment_reason", "recorded_at",
                ]
            )
        output = atomic_write_csv(
            changes, self.config.paths.processed_path(self.config.paths.feedback_updates_csv), config=self.config
        )

        report: Dict[str, Any] = {
            "total_feedback_records": len(feedback_records),
            "accepted_count": sum(1 for r in feedback_records if r.get("verdict") == "accepted"),
            "false_alarm_count": sum(1 for r in feedback_records if r.get("verdict") == "false_alarm"),
            "missed_count": sum(1 for r in feedback_records if r.get("verdict") == "missed"),
            "memories_ingested_this_run": ingested,
            "memory_summary": memory_summary,
            "threshold_state": threshold_state,
            "output_path": str(output),
        }
        atomic_write_json(report, self.config.paths.metrics / "feedback_learning.json")
        self.logger.info("Feedback learning report: %s", report)
        return report


def main() -> None:
    """Run the feedback-learning stage standalone."""
    config = get_config()
    logger = get_logger(__name__, config)
    updater = LearningUpdater(config)
    report = updater.run()
    logger.info("Feedback report: %s", report)


if __name__ == "__main__":
    main()
