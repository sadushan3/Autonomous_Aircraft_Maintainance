"""
agreement_engine.py

Multi-Agent Consensus & Conflict Detection Engine for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Quantifies agreement and measures decision conflict across independent upstream
components (RUL, Failure Risk, and Anomaly Monitoring).

It calculates:
1. `agreement_score`: Degree of alignment between RUL, Risk, and Anomaly signals in [0.0, 1.0].
2. `conflict_score`: Degree of contradiction across upstream agent outputs.
3. `high_conflict_flag`: Boolean flag indicating severe inter-agent disagreement.
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
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "agreement_engine_report.json"

# Fixed modelling bug (unit mismatch): anomaly_score is not itself a
# calibrated [0, 1] severity value -- on a live run it averages 0.77 even
# for rows the Anomaly & Health component itself labels severity "none"
# (vs. 0.99+ for "critical"), because the underlying detector score
# saturates near 1.0 regardless of true severity. Feeding that raw score
# into np.clip(anomaly_score, 0, 1) alongside genuinely-scaled rul_norm/
# risk_norm (which both sit near 0 for a healthy engine) made "conflict"
# spuriously high for almost every row -- 99.97% of rows the health monitor
# itself calls Healthy scored >= the 0.30 high-conflict threshold, driving
# ~97% of live decisions to a mandatory-review flag on "Agent Conflict"
# alone. anomaly_severity is the field the Anomaly & Health component
# already calibrates for exactly this purpose; map it to the same [0, 1]
# severity scale rul_norm/risk_norm use instead of the raw score.
_ANOMALY_SEVERITY_TO_NORM: Final[dict[str, float]] = {
    "none": 0.0,
    "low": 0.25,
    "medium": 0.5,
    "high": 0.75,
    "critical": 1.0,
}


@dataclass
class AgreementResult:
    agreement_scores: np.ndarray
    conflict_scores: np.ndarray
    high_conflict_flags: np.ndarray
    mean_agreement: float
    mean_conflict: float


class SupervisorAgreementEngine:
    """Computes inter-agent agreement and conflict metrics."""

    def __init__(self, config: SupervisorConfig | None = None):
        self.cfg = config or _CFG

    def compute_agreement(self, df_features: pd.DataFrame) -> AgreementResult:
        rul = df_features.get("predicted_rul", pd.Series(100.0, index=df_features.index)).values
        r30 = df_features.get("risk_30", pd.Series(0.0, index=df_features.index)).values

        # Fixed modelling bug: 1.0 - rul/150.0 is an arbitrary divisor
        # unrelated to any threshold the rest of the system treats as
        # meaningful. On real data it put a comfortably healthy RUL
        # (~80-100 cycles, at or above monitoring_rul_threshold) at ~0.35-0.47
        # severity -- roughly HALFWAY to critical -- while risk_norm/anom_norm
        # correctly sat near 0.0 for the same rows, which alone produced
        # "conflict" for 99%+ of rows regardless of anomaly_score's fix
        # above. Map predicted_rul through this component's own severity
        # bands instead (immediate/maintenance/inspection/monitoring
        # thresholds, already defined in SupervisorConfig and used
        # everywhere else a RUL severity decision is made), so a RUL at or
        # above monitoring_rul_threshold reads as truly healthy (~0.0) and
        # severity rises through the same four bands the safety guardrails
        # and rule cascade already use -- one severity scale for the whole
        # component, not a second, disconnected one just for this engine.
        rul_norm = np.interp(
            rul,
            [self.cfg.immediate_rul_threshold, self.cfg.maintenance_rul_threshold,
             self.cfg.inspection_rul_threshold, self.cfg.monitoring_rul_threshold],
            [1.0, 2.0 / 3.0, 1.0 / 3.0, 0.0],
        )
        risk_norm = np.clip(r30, 0.0, 1.0)

        # Fixed modelling bug: anomaly_severity (calibrated categorical),
        # not the raw anomaly_score (uncalibrated, saturates near 1.0
        # regardless of true severity -- see the module-level note above),
        # is what belongs on the same severity scale as rul_norm/risk_norm.
        # An unrecognized/missing category defaults to "none" (0.0) rather
        # than silently raising, matching the permissive .get() pattern
        # used for the other two signals above.
        anom_severity = (
            df_features.get("anomaly_severity", pd.Series("none", index=df_features.index))
            .astype(str).str.strip().str.lower()
        )
        anom_norm = anom_severity.map(_ANOMALY_SEVERITY_TO_NORM).fillna(0.0).values

        # Standard deviation across signals per sample
        stacked = np.column_stack([rul_norm, risk_norm, anom_norm])
        std_devs = np.std(stacked, axis=1)

        # Scale conflict to [0.0, 1.0]
        conflict_scores = np.clip(std_devs * 2.0, 0.0, 1.0)
        agreement_scores = np.clip(1.0 - conflict_scores, 0.0, 1.0)

        high_conflict_flags = conflict_scores >= self.cfg.high_conflict_threshold

        return AgreementResult(
            agreement_scores=agreement_scores,
            conflict_scores=conflict_scores,
            high_conflict_flags=high_conflict_flags,
            mean_agreement=float(np.mean(agreement_scores)),
            mean_conflict=float(np.mean(conflict_scores)),
        )


def run_agreement_engine(input_path: Path | None = None) -> dict:
    input_path = Path(input_path or _TEST_PATH).resolve()

    section("AGREEMENT ENGINE EVALUATION STARTED")
    start_time = time.perf_counter()

    if not input_path.exists():
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        input_path = SAMPLE_DATASET_PATH

    df = pd.read_csv(input_path, low_memory=False)

    engine = SupervisorAgreementEngine()
    res = engine.compute_agreement(df)
    duration = time.perf_counter() - start_time

    report = {
        "status": "success",
        "input_path": str(input_path),
        "samples_evaluated": len(df),
        "mean_agreement": round(res.mean_agreement, 4),
        "mean_conflict": round(res.mean_conflict, 4),
        "high_conflict_count": int(np.sum(res.high_conflict_flags)),
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Mean Agreement: %.4f | Mean Conflict: %.4f", res.mean_agreement, res.mean_conflict)
    logger.info("High Conflict Samples: %d", report["high_conflict_count"])
    logger.info("Report written to: %s", _REPORT_PATH)
    section("AGREEMENT ENGINE EVALUATION COMPLETED")
    return report


def main() -> int:
    try:
        res = run_agreement_engine()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in agreement_engine: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
