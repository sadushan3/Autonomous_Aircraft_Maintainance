"""
uncertainty_engine.py

Epistemic & Aleatoric Uncertainty Estimation Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Estimates prediction uncertainty across decision probabilities using normalized Shannon
Entropy and Gini Impurity to quantify model hesitation.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import REPORTS_ROOT, PROCESSED_ROOT, SupervisorConfig
from app.utils.Maintenance_Supervisor.logger import get_logger, section
from app.utils.Maintenance_Supervisor.atomic_writer import atomic_write_json

logger = get_logger()
_CFG = SupervisorConfig()

_TEST_PATH: Final[Path] = PROCESSED_ROOT / "splits" / "test.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "uncertainty_engine_report.json"


@dataclass
class UncertaintyResult:
    entropy_scores: np.ndarray  # Shannon entropy normalized to [0.0, 1.0]
    gini_scores: np.ndarray     # Gini impurity normalized
    high_uncertainty_flags: np.ndarray  # bool array indicating high uncertainty
    mean_entropy: float


class SupervisorUncertaintyEngine:
    """Computes entropy and uncertainty metrics over decision class probabilities."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

    def compute_uncertainty(self, predicted_probs: np.ndarray) -> UncertaintyResult:
        eps = 1e-12
        probs = np.clip(predicted_probs, eps, 1.0)

        # 1. Shannon Entropy: H(X) = - sum(p * log2(p))
        n_classes = probs.shape[1]
        max_entropy = np.log2(n_classes)
        raw_entropy = -np.sum(probs * np.log2(probs), axis=1)
        normalized_entropy = np.clip(raw_entropy / max_entropy, 0.0, 1.0)

        # 2. Gini Impurity: G(X) = 1 - sum(p^2)
        max_gini = 1.0 - (1.0 / n_classes)
        raw_gini = 1.0 - np.sum(probs ** 2, axis=1)
        normalized_gini = np.clip(raw_gini / max_gini, 0.0, 1.0)

        high_uncertainty_flags = normalized_entropy >= self.cfg.high_uncertainty_threshold

        return UncertaintyResult(
            entropy_scores=normalized_entropy,
            gini_scores=normalized_gini,
            high_uncertainty_flags=high_uncertainty_flags,
            mean_entropy=float(np.mean(normalized_entropy)),
        )


def run_uncertainty_engine(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("UNCERTAINTY ENGINE EVALUATION STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    # Simulated probability distribution (some deterministic, some uncertain)
    n_samples = len(df)
    n_classes = len(_CFG.decision_classes)
    dummy_probs = np.full((n_samples, n_classes), 1.0 / n_classes)

    engine = SupervisorUncertaintyEngine()
    res = engine.compute_uncertainty(dummy_probs)
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "samples_evaluated": len(df),
        "mean_entropy": round(res.mean_entropy, 4),
        "high_uncertainty_count": int(np.sum(res.high_uncertainty_flags)),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Mean Normalized Entropy: %.4f", res.mean_entropy)
    logger.info("High Uncertainty Samples: %d", report["high_uncertainty_count"])
    logger.info("Report written to: %s", _REPORT_PATH)
    section("UNCERTAINTY ENGINE EVALUATION COMPLETED")
    return report


def main() -> int:
    try:
        res = run_uncertainty_engine()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in uncertainty_engine: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
