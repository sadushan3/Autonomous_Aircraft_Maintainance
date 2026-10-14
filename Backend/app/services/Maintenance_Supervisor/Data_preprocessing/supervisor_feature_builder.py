"""
supervisor_feature_builder.py

Feature Builder Utilities & Domain Feature Transformer for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module provides high-level domain feature extraction functions for constructing
interaction terms, multi-horizon pressure indicators, and agreement metrics from
raw upstream outputs.

It serves as both:
1. An operational helper module for feature_engineering.py and real-time inference.
2. A standalone feature transformer that can build features on arbitrary input dataframes
   (e.g., streaming inference or synthetic validation data).

Engineered Features
-------------------
1. Risk & Horizon Pressures:
   - short_horizon_risk_pressure (risk_10 / threshold_10)
   - medium_horizon_risk_pressure (risk_30 / threshold_30)
   - long_horizon_risk_pressure (risk_50 / threshold_50)

2. Domain Interactions:
   - risk_anomaly_interaction (risk_30 * anomaly_score)
   - rul_risk_interaction (degradation_speed * risk_30)
   - rul_anomaly_interaction (degradation_speed * anomaly_score)
   - uncertainty_risk_interaction (uncertainty_width * uncertainty_30)

3. Multi-Agent Agreement & Conflicts:
   - agent_agreement_score (alignment between RUL, Risk, and Anomaly signals)
   - decision_conflict_score (degree of contradiction across agent outputs)

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.Data_preprocessing.supervisor_feature_builder

Expected inputs
---------------
- processed/Maintenance_Supervisor/supervisor_sanitized_input.csv (or specified path)

Expected outputs
---------------
- processed/Maintenance_Supervisor/supervisor_builder_output.csv
- reports/Maintenance_Supervisor/supervisor_feature_builder_report.json

Exit codes
----------
0 — feature building completed successfully
1 — internal failure
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

# ---------------------------------------------------------------------------
# Bootstrap: ensure Backend/ is on sys.path
# ---------------------------------------------------------------------------

_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]

if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------

from app.config.Maintenance_Supervisor.supervisor_config import (  # noqa: E402
    PROCESSED_ROOT,
    REPORTS_ROOT,
    ENGINEERED_FEATURES,
)
from app.utils.Maintenance_Supervisor.logger import get_logger, section  # noqa: E402
from app.utils.Maintenance_Supervisor.atomic_writer import (  # noqa: E402
    atomic_write_json,
    atomic_write_csv,
)

# ---------------------------------------------------------------------------
# Logger — project singleton
# ---------------------------------------------------------------------------

logger = get_logger()

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

_INPUT_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_sanitized_input.csv"
_OUTPUT_PATH: Final[Path] = PROCESSED_ROOT / "supervisor_builder_output.csv"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "supervisor_feature_builder_report.json"


# ---------------------------------------------------------------------------
# Feature Engineering Functions
# ---------------------------------------------------------------------------

def compute_horizon_pressures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk pressure relative to decision thresholds across horizons."""
    res = pd.DataFrame(index=df.index)
    eps = 1e-6

    r10 = df.get("risk_10", pd.Series(0.0, index=df.index))
    t10 = df.get("threshold_10", pd.Series(0.8, index=df.index)).replace(0.0, eps)
    res["short_horizon_risk_pressure"] = (r10 / t10).clip(upper=10.0)

    r30 = df.get("risk_30", pd.Series(0.0, index=df.index))
    t30 = df.get("threshold_30", pd.Series(0.7, index=df.index)).replace(0.0, eps)
    res["medium_horizon_risk_pressure"] = (r30 / t30).clip(upper=10.0)

    r50 = df.get("risk_50", pd.Series(0.0, index=df.index))
    t50 = df.get("threshold_50", pd.Series(0.5, index=df.index)).replace(0.0, eps)
    res["long_horizon_risk_pressure"] = (r50 / t50).clip(upper=10.0)

    return res


def compute_domain_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Compute non-linear interaction features across agent disciplines."""
    res = pd.DataFrame(index=df.index)

    r30 = df.get("risk_30", pd.Series(0.0, index=df.index))
    anom = df.get("anomaly_score", pd.Series(0.0, index=df.index))
    speed = df.get("degradation_speed", pd.Series(0.0, index=df.index))
    u_width = df.get("uncertainty_width", pd.Series(0.0, index=df.index))
    u30 = df.get("uncertainty_30", pd.Series(0.0, index=df.index))

    res["risk_anomaly_interaction"] = r30 * anom
    res["rul_risk_interaction"] = speed * r30
    res["rul_anomaly_interaction"] = speed * anom
    res["uncertainty_risk_interaction"] = u_width * u30
    res["uncertainty_anomaly_interaction"] = u_width * anom

    return res


def compute_agreement_and_conflict(df: pd.DataFrame) -> pd.DataFrame:
    """Compute consensus and disagreement metrics between agent predictions."""
    res = pd.DataFrame(index=df.index)

    rul = df.get("predicted_rul", pd.Series(100.0, index=df.index))
    r30 = df.get("risk_30", pd.Series(0.0, index=df.index))
    anom = df.get("anomaly_score", pd.Series(0.0, index=df.index))

    # Normalized signals (0 = safe, 1 = severe)
    rul_norm = (1.0 - (rul / 150.0)).clip(0.0, 1.0)
    risk_norm = r30.clip(0.0, 1.0)
    anom_norm = anom.clip(0.0, 1.0)

    # Agreement = 1.0 - std_dev across normalized severe signals
    signal_matrix = np.column_stack([rul_norm.values, risk_norm.values, anom_norm.values])
    std_devs = np.std(signal_matrix, axis=1)

    res["agent_agreement_score"] = (1.0 - std_devs * 2.0).clip(0.0, 1.0)
    res["decision_conflict_score"] = (std_devs * 2.0).clip(0.0, 1.0)

    return res


def build_supervisor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Main transformer pipeline executing all feature generators."""
    pressures = compute_horizon_pressures(df)
    interactions = compute_domain_interactions(df)
    conflicts = compute_agreement_and_conflict(df)

    out_df = pd.concat([df, pressures, interactions, conflicts], axis=1)
    return out_df


# ---------------------------------------------------------------------------
# CLI & Standalone Orchestrator
# ---------------------------------------------------------------------------

def run_feature_builder(
    input_path: Path | None = None,
    output_path: Path | None = None,
) -> dict:
    """Run standalone feature builder pipeline."""
    if input_path is None:
        input_path = _INPUT_PATH
    if output_path is None:
        output_path = _OUTPUT_PATH

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    section("SUPERVISOR FEATURE BUILDER STARTED")
    logger.info("Input dataset : %s", input_path)
    logger.info("Output dataset: %s", output_path)

    start_time = time.perf_counter()

    if not input_path.exists():
        # Fallback to sample input if sanitized input isn't generated yet
        from app.config.Maintenance_Supervisor.supervisor_config import SAMPLE_DATASET_PATH
        if SAMPLE_DATASET_PATH.exists():
            input_path = SAMPLE_DATASET_PATH
            logger.info("Sanitized input not found, falling back to sample input: %s", input_path)
        else:
            raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    initial_cols = df.shape[1]

    built_df = build_supervisor_features(df)
    final_cols = built_df.shape[1]
    features_added = final_cols - initial_cols

    duration = time.perf_counter() - start_time

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(built_df, output_path)

    report = {
        "status": "success",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "rows": len(built_df),
        "initial_columns": initial_cols,
        "final_columns": final_cols,
        "features_added": features_added,
        "duration_seconds": round(duration, 4),
    }

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("Features added: %d", features_added)
    logger.info("Output written to: %s", output_path)
    section("SUPERVISOR FEATURE BUILDER COMPLETED")

    return report


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_feature_builder()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error in supervisor_feature_builder: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
