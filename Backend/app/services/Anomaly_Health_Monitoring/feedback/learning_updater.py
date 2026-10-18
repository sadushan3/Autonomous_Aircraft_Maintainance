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


MIN_CRITICAL_BAND = 10.0


class LearningUpdater:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.feedback_store = FeedbackStore(self.config)
        self.alert_memory = AlertMemory(self.config)
        self.threshold_adapter = ThresholdAdapter(self.config)
        self.state_classifier = HealthStateClassifier(self.config)


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


    def run(self, root_causes: Optional[pd.DataFrame] = None) -> Dict[str, Any]:


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

    config = get_config()
    logger = get_logger(__name__, config)
    updater = LearningUpdater(config)
    report = updater.run()
    logger.info("Feedback report: %s", report)


if __name__ == "__main__":
    main()
