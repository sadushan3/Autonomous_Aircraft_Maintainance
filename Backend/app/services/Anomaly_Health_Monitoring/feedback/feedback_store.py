"""
Feedback store: durable, append-only persistence for operator verdicts.

Every time an operator accepts, rejects (false alarm) or reports a miss on an
alert, that verdict is recorded here as one immutable entry keyed by a
generated feedback id.  This is the raw material the rest of the feedback
layer works from: :mod:`alert_memory` derives the anomaly-signature memory
from it, and :mod:`threshold_adapter` derives threshold nudges from it.

Persistence is a single JSON document (atomic write, so a crash mid-write
cannot corrupt prior feedback), capped at ``feedback.memory_capacity`` records
on a FIFO basis so the store cannot grow without bound across a long-running
deployment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    new_id,
    read_json,
    utc_now_iso,
)


class FeedbackStore:
    """Append-only store of operator feedback on raised alerts."""

    def __init__(self, config: Optional[Config] = None) -> None:
        """Initialise the store.

        Args:
            config: Configuration supplying the storage path and capacity.
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.path = self.config.paths.model_path(self.config.paths.feedback_store_json)
        self.capacity = int(self.config.feedback.memory_capacity)

    # ------------------------------------------------------------------ #
    # Read / write
    # ------------------------------------------------------------------ #
    def load(self) -> List[Dict[str, Any]]:
        """Read every stored feedback record.

        Returns:
            A list of feedback records, oldest first.
        """
        payload = read_json(self.path, default={"records": []})
        return payload.get("records", []) if payload else []

    def add(
        self,
        engine_id: int,
        cycle: int,
        verdict: str,
        operator: str = "unknown",
        comment: str = "",
        confirmed_root_cause: Optional[str] = None,
        system_root_cause: Optional[str] = None,
        anomaly_id: Optional[str] = None,
        severity: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Record one operator verdict.

        Args:
            engine_id: Engine the alert concerned.
            cycle: Cycle the alert concerned.
            verdict: ``'accepted'``, ``'false_alarm'``, ``'missed'`` or
                ``'unknown'``.
            operator: Identifier of whoever submitted the feedback.
            comment: Free-text note.
            confirmed_root_cause: The cause actually found, if different from
                the system's hypothesis.
            system_root_cause: The cause the system had proposed, for
                comparison.
            anomaly_id: Identifier of the matched anomaly record, if known.
            severity: The severity the system had assigned.
            anomaly_score: The fused anomaly score at the time of the alert.

        Returns:
            The stored record, including its generated ``feedback_id``.
        """
        records = self.load()
        record: Dict[str, Any] = {
            "feedback_id": new_id("fb"),
            "engine_id": int(engine_id),
            "cycle": int(cycle),
            "verdict": verdict,
            "operator": operator,
            "comment": comment,
            "confirmed_root_cause": confirmed_root_cause,
            "system_root_cause": system_root_cause,
            "anomaly_id": anomaly_id,
            "severity": severity,
            "anomaly_score": anomaly_score,
            "recorded_at": utc_now_iso(),
        }
        records.append(record)

        if len(records) > self.capacity:
            overflow = len(records) - self.capacity
            records = records[overflow:]
            self.logger.info("Feedback store at capacity; dropped %d oldest record(s)", overflow)

        self._save(records)
        self.logger.info(
            "Recorded feedback %s: engine=%d cycle=%d verdict=%s",
            record["feedback_id"],
            engine_id,
            cycle,
            verdict,
        )
        return record

    def _save(self, records: List[Dict[str, Any]]) -> None:
        """Persist the full record list atomically.

        Args:
            records: The complete list of feedback records to write.
        """
        atomic_write_json({"records": records}, self.path)

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def summary(self) -> Dict[str, Any]:
        """Summarise verdict counts across all stored feedback.

        Returns:
            A mapping with total count and per-verdict counts.
        """
        records = self.load()
        counts: Dict[str, int] = {}
        for record in records:
            verdict = record.get("verdict", "unknown")
            counts[verdict] = counts.get(verdict, 0) + 1
        return {"total": len(records), "by_verdict": counts}

    def for_engine(self, engine_id: int) -> List[Dict[str, Any]]:
        """Return all feedback recorded for one engine.

        Args:
            engine_id: Engine identifier.

        Returns:
            Matching feedback records, oldest first.
        """
        return [r for r in self.load() if int(r.get("engine_id", -1)) == int(engine_id)]


def main() -> None:
    """Record a demo feedback entry and print the store summary."""
    config = get_config()
    logger = get_logger(__name__, config)
    store = FeedbackStore(config)
    store.add(
        engine_id=1,
        cycle=142,
        verdict="accepted",
        operator="demo_operator",
        comment="Confirmed HPC efficiency loss on inspection.",
        system_root_cause="High-Pressure Compressor efficiency degradation",
    )
    logger.info("Feedback summary: %s", store.summary())


if __name__ == "__main__":
    main()
