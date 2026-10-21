from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import pandas as pd

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.feedback.feedback_store import FeedbackStore
from app.utils.Anomaly_Health_Monitoring.utils import (
    atomic_write_json,
    get_logger,
    new_id,
    read_json,
    utc_now_iso,
)


_DEFAULT_CAPACITY = 5000


class AlertMemory:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.path = self.config.paths.model_path(self.config.paths.alert_memory_json)
        self.capacity = int(self.config.feedback.memory_capacity or _DEFAULT_CAPACITY)


    def load(self) -> List[Dict[str, Any]]:


        payload = read_json(self.path, default={"records": []})
        return payload.get("records", []) if payload else []

    def _save(self, records: List[Dict[str, Any]]) -> None:


        atomic_write_json({"records": records}, self.path)

    def remember(
        self,
        engine_id: int,
        cycle: int,
        signature: List[str],
        subsystem: str,
        root_cause: str,
        severity: str,
        anomaly_score: float,
    ) -> Dict[str, Any]:


        records = self.load()
        record: Dict[str, Any] = {
            "anomaly_id": new_id("mem"),
            "engine_id": int(engine_id),
            "cycle": int(cycle),
            "signature": list(signature),
            "subsystem": subsystem,
            "root_cause": root_cause,
            "confirmed_root_cause": None,
            "severity": severity,
            "anomaly_score": float(anomaly_score),
            "operator_verdict": "unknown",
            "recorded_at": utc_now_iso(),
            "times_matched": 0,
        }
        records.append(record)
        if len(records) > self.capacity:
            records = records[len(records) - self.capacity :]
        self._save(records)
        return record

    def update_verdict(
        self,
        anomaly_id: str,
        verdict: str,
        confirmed_root_cause: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:


        records = self.load()
        updated: Optional[Dict[str, Any]] = None
        for record in records:
            if record.get("anomaly_id") == anomaly_id:
                record["operator_verdict"] = verdict
                if confirmed_root_cause:
                    record["confirmed_root_cause"] = confirmed_root_cause
                updated = record
                break
        if updated is not None:
            self._save(records)
            self.logger.info("Updated memory %s verdict -> %s", anomaly_id, verdict)
        else:
            self.logger.warning("No memory record found for anomaly_id=%s", anomaly_id)
        return updated


    def ingest_root_causes(self, root_causes: pd.DataFrame) -> int:


        if root_causes.empty:
            return 0

        records = self.load()
        count = 0
        for _, row in root_causes.iterrows():
            signature = self._parse_list_column(row.get("top_sensors", "[]"))
            records.append(
                {
                    "anomaly_id": new_id("mem"),
                    "engine_id": int(row["unit"]),
                    "cycle": int(row["cycle"]),
                    "signature": signature,
                    "subsystem": str(row.get("subsystem", "unknown")),
                    "root_cause": str(row.get("root_cause", "")),
                    "confirmed_root_cause": None,
                    "severity": str(row.get("severity", "none")),
                    "anomaly_score": 0.0,
                    "operator_verdict": "unknown",
                    "recorded_at": utc_now_iso(),
                    "times_matched": 0,
                }
            )
            count += 1

        if len(records) > self.capacity:
            records = records[len(records) - self.capacity :]
        self._save(records)
        self.logger.info("Ingested %d anomaly signature(s) into memory", count)
        return count

    @staticmethod
    def _parse_list_column(value: Any) -> List[str]:


        if isinstance(value, list):
            return value
        if not isinstance(value, str):
            return []
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []

    def summary(self) -> Dict[str, Any]:


        records = self.load()
        false_alarms = sum(1 for r in records if r.get("operator_verdict") == "false_alarm")
        confirmed = sum(1 for r in records if r.get("operator_verdict") == "accepted")
        return {
            "total_memories": len(records),
            "false_alarm_patterns": false_alarms,
            "confirmed_fault_patterns": confirmed,
        }


def main() -> None:

    from app.utils.Anomaly_Health_Monitoring.utils import read_csv

    config = get_config()
    logger = get_logger(__name__, config)
    memory = AlertMemory(config)

    path = config.paths.processed_path(config.paths.root_cause_analysis_csv)
    if path.is_file():
        root_causes = read_csv(path, stage="alert_memory")
        count = memory.ingest_root_causes(root_causes)
        logger.info("Ingested %d record(s)", count)
    logger.info("Memory summary: %s", memory.summary())


if __name__ == "__main__":
    main()
