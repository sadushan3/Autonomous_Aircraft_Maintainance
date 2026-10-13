from __future__ import annotations

import json
import math
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


class MemoryService:
    """First-version Agent Memory (design doc Module 3).

    Persists a compact record of every combined prediction to a local JSON
    file and answers similarity queries over it with plain Euclidean
    distance. This intentionally does not implement adaptive threshold
    learning yet -- it is the storage + retrieval layer that future
    reasoning work builds on.
    """

    def __init__(self, storage_path: Path, max_records: int = 2000):
        self.storage_path = storage_path
        self.max_records = max_records
        self._lock = Lock()
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.storage_path.exists():
            self.storage_path.write_text("[]", encoding="utf-8")

    def _read(self) -> List[Dict[str, Any]]:
        try:
            return json.loads(self.storage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def _write(self, records: List[Dict[str, Any]]) -> None:
        trimmed = records[-self.max_records :]
        self.storage_path.write_text(json.dumps(trimmed), encoding="utf-8")

    def record(
        self,
        *,
        engine_id: Optional[str],
        fd_subset: str,
        cycle: int,
        predicted_rul: float,
        degradation_speed: float,
        degradation_acceleration: float,
        risk_30: float,
        lifecycle_state: str,
        risk_state: str,
    ) -> Dict[str, Any]:
        with self._lock:
            records = self._read()
            entry = {
                "id": (records[-1]["id"] + 1) if records else 1,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "engine_id": engine_id,
                "fd_subset": fd_subset,
                "cycle": cycle,
                "predicted_rul": predicted_rul,
                "degradation_speed": degradation_speed,
                "degradation_acceleration": degradation_acceleration,
                "risk_30": risk_30,
                "lifecycle_state": lifecycle_state,
                "risk_state": risk_state,
            }
            records.append(entry)
            self._write(records)
            return entry

    def status(self) -> Dict[str, Any]:
        records = self._read()
        engines = {r["engine_id"] for r in records if r.get("engine_id")}
        by_fd: Dict[str, int] = {}
        for r in records:
            by_fd[r["fd_subset"]] = by_fd.get(r["fd_subset"], 0) + 1
        return {
            "total_memories": len(records),
            "distinct_engines": len(engines),
            "memories_by_fd_subset": by_fd,
            "last_updated": records[-1]["timestamp"] if records else None,
            "memory_confidence": round(min(1.0, len(records) / 200.0), 4),
        }

    def similar(
        self,
        *,
        fd_subset: str,
        degradation_speed: float,
        degradation_acceleration: float,
        risk_30: float,
        exclude_engine_id: Optional[str] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        candidates = [r for r in self._read() if r.get("fd_subset") == fd_subset]
        if exclude_engine_id:
            candidates = [r for r in candidates if r.get("engine_id") != exclude_engine_id]

        scored = []
        for record in candidates:
            distance = math.sqrt(
                (record["degradation_speed"] - degradation_speed) ** 2
                + (record["degradation_acceleration"] - degradation_acceleration) ** 2
                + ((record["risk_30"] - risk_30) * 10) ** 2
            )
            scored.append({**record, "similarity": round(1.0 / (1.0 + distance), 4)})

        scored.sort(key=lambda item: item["similarity"], reverse=True)
        return scored[:limit]
