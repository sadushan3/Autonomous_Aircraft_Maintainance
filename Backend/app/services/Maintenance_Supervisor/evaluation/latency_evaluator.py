"""
latency_evaluator.py

Inference Latency & Throughput Benchmark Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Measures real-time decision inference latency per sample (p50, p95, p99) and throughput
(samples per second) under single-row and batch streaming conditions.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT
from app.services.Maintenance_Supervisor.decision_fusion.decision_fusion_engine import DecisionFusionEngine
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "latency_evaluator_report.json"


def run_latency_evaluation(input_path: Path | None = None, n_iterations: int = 100) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("LATENCY & THROUGHPUT BENCHMARK STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)
    engine = DecisionFusionEngine()

    # Single-sample streaming latency benchmark
    sample_latencies_ms: list[float] = []
    test_slice = df.head(min(n_iterations, len(df)))

    for _, row in test_slice.iterrows():
        single_df = pd.DataFrame([row])
        t0 = time.perf_counter()
        _ = engine.fuse_decisions(single_df)
        t1 = time.perf_counter()
        sample_latencies_ms.append((t1 - t0) * 1000.0)

    # Batch throughput benchmark
    t_batch_0 = time.perf_counter()
    _ = engine.fuse_decisions(df)
    t_batch_1 = time.perf_counter()
    batch_duration = t_batch_1 - t_batch_0
    throughput = len(df) / batch_duration if batch_duration > 0 else 0.0

    duration = time.perf_counter() - start_time

    p50 = float(np.percentile(sample_latencies_ms, 50))
    p95 = float(np.percentile(sample_latencies_ms, 95))
    p99 = float(np.percentile(sample_latencies_ms, 99))

    report = {
        "status": "success",
        "input_path": str(input_path),
        "single_sample_benchmarks_count": len(sample_latencies_ms),
        "latency_p50_ms": round(p50, 4),
        "latency_p95_ms": round(p95, 4),
        "latency_p99_ms": round(p99, 4),
        "batch_sample_count": len(df),
        "batch_throughput_samples_per_sec": round(throughput, 2),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Single Sample Latency (p50): %.2f ms | (p95): %.2f ms", p50, p95)
    logger.info("Batch Throughput: %.2f samples/sec", throughput)
    logger.info("Report written to: %s", _REPORT_PATH)
    section("LATENCY & THROUGHPUT BENCHMARK COMPLETED")

    return report


def main() -> int:
    try:
        res = run_latency_evaluation()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in latency_evaluator: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
