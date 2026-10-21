from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.utils.Anomaly_Health_Monitoring.utils import get_logger, read_json


class RootCauseTracker:


    def __init__(self, config: Optional[Config] = None) -> None:


        self.config = config or get_config()
        self.logger = get_logger(__name__, self.config)
        self.memory_path = self.config.paths.model_path(
            self.config.paths.alert_memory_json
        )
        self.threshold = float(self.config.reasoning.memory_match_threshold)
        self._memories: Optional[List[Dict[str, Any]]] = None


    def load_memories(self, force: bool = False) -> List[Dict[str, Any]]:


        if self._memories is not None and not force:
            return self._memories
        payload = read_json(self.memory_path, default={"records": []})
        self._memories = payload.get("records", []) if payload else []
        return self._memories


    @staticmethod
    def _jaccard(a: List[str], b: List[str]) -> float:


        set_a, set_b = set(a), set(b)
        if not set_a and not set_b:
            return 0.0
        union = set_a | set_b
        if not union:
            return 0.0
        return len(set_a & set_b) / len(union)

    def find_best_match(
        self, signature: List[str], subsystem: Optional[str] = None
    ) -> Tuple[Optional[Dict[str, Any]], float]:


        memories = self.load_memories()
        if subsystem:
            memories = [m for m in memories if m.get("subsystem") == subsystem] or memories

        best_record: Optional[Dict[str, Any]] = None
        best_score = 0.0
        for record in memories:
            score = self._jaccard(signature, record.get("signature", []))
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is not None and best_score >= self.threshold:
            return best_record, best_score
        return None, best_score

    def match(self, signature: List[str], subsystem: Optional[str] = None) -> Dict[str, Any]:


        record, similarity = self.find_best_match(signature, subsystem)
        if record is None:
            return {
                "matched": False,
                "anomaly_id": None,
                "similarity": round(similarity, 4),
                "root_cause": None,
                "previously_false_alarm": False,
                "times_seen_before": 0,
            }

        verdict = str(record.get("operator_verdict", "unknown"))
        result = {
            "matched": True,
            "anomaly_id": record.get("anomaly_id"),
            "similarity": round(similarity, 4),
            "root_cause": record.get("confirmed_root_cause") or record.get("root_cause"),
            "previously_false_alarm": verdict == "false_alarm",
            "times_seen_before": int(record.get("times_matched", 0)) + 1,
        }
        self.logger.info(
            "Anomaly signature matched memory %s (similarity=%.2f, false_alarm=%s)",
            result["anomaly_id"],
            result["similarity"],
            result["previously_false_alarm"],
        )
        return result

    def similar_history_count(self, subsystem: str) -> int:


        return sum(1 for m in self.load_memories() if m.get("subsystem") == subsystem)


def main() -> None:

    config = get_config()
    logger = get_logger(__name__, config)
    tracker = RootCauseTracker(config)
    result = tracker.match(["sensor_3", "sensor_11", "sensor_12"], subsystem="hpc")
    logger.info("Match result: %s", result)


if __name__ == "__main__":
    main()
