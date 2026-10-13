"""
sample_output_generator.py

Synthetic Upstream Output Generator for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
This module generates realistic, physics-informed synthetic outputs representing the
predictions produced by three upstream research components:
1. RUL (Remaining Useful Life) Prediction component
2. Failure Risk Prediction component
3. Anomaly & Health Monitoring component

It simulates multi-engine degradation trajectories across different operating sub-datasets
(FD001, FD002, FD003, FD004) to allow end-to-end testing, benchmarking, and rule-validation
of the Supervisor Agent without requiring live upstream model execution.

Physics-Informed Synthetic Generation Rules:
--------------------------------------------
- RUL Degradation:
  Actual RUL starts at lifetime L ~ Uniform(150, 300) cycles and decreases monotonically to 0.
  Predicted RUL is actual RUL + heteroscedastic noise (noise increases at higher RUL).
  Degradation speed accelerates as RUL decreases.

- Failure Risk Probabilities:
  risk_10, risk_30, risk_50 increase sigmoidally as actual RUL approaches 0.
  Guarantees risk_10 <= risk_30 <= risk_50 per row.

- Anomaly Severity & Scores:
  Anomaly score stays low during healthy cycles and spikes exponentially near failure.
  Anomaly severity transitions: none -> low -> medium -> high -> critical.

- Model Trace Identifiers:
  Populates model metadata strings (rul_model_used, risk_model_used, anomaly_model_used).

Execution
---------
Run from the Backend/ directory:

    "C:\\Python313\\python.exe" -m app.services.Maintenance_Supervisor.data_generation.sample_output_generator

Expected outputs
----------------
- data/Maintenance_Supervisor/sample/supervisor_sample_dataset.csv
- artifacts/Maintenance_Supervisor/sample_generation_manifest.json
- reports/Maintenance_Supervisor/sample_generation_report.json

Exit codes
----------
0 — sample generation completed successfully
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
    SAMPLE_DATASET_PATH,
    ARTIFACT_ROOT,
    REPORTS_ROOT,
    SupervisorConfig,
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
_CFG = SupervisorConfig()

# ---------------------------------------------------------------------------
# Output Paths
# ---------------------------------------------------------------------------

_MANIFEST_PATH: Final[Path] = ARTIFACT_ROOT / "sample_generation_manifest.json"
_REPORT_PATH: Final[Path] = REPORTS_ROOT / "sample_generation_report.json"

# ---------------------------------------------------------------------------
# Generator Parameters
# ---------------------------------------------------------------------------

DEFAULT_SUBSETS: Final[tuple[str, ...]] = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_ENGINES_PER_SUBSET: Final[int] = 50  # Total 200 engines ~ 40k-50k cycles
RUL_MODELS: Final[tuple[str, ...]] = ("LSTM_RUL_v2", "Transformer_RUL_v1", "XGBoost_RUL_v3")
RISK_MODELS: Final[tuple[str, ...]] = ("DeepSurvival_v1", "RandomForest_Risk_v2", "LightGBM_Risk_v1")
ANOMALY_MODELS: Final[tuple[str, ...]] = ("Autoencoder_v2", "IsolationForest_v1", "Mahalanobis_v1")
SENSOR_NAMES: Final[tuple[str, ...]] = (
    "s2_inlet_temp", "s3_inlet_press", "s4_bypass_ratio", "s7_fan_speed",
    "s11_core_speed", "s12_burner_fuel", "s15_bleed_enthalpy", "s20_hpc_outlet_temp"
)


# ---------------------------------------------------------------------------
# Core Generation Logic
# ---------------------------------------------------------------------------

def generate_engine_trajectory(
    engine_id: int,
    subset: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate a realistic lifetime trajectory for a single engine."""
    max_lifetime = rng.integers(140, 260)
    cycles = np.arange(1, max_lifetime + 1)
    true_rul = max_lifetime - cycles

    # Model selection metadata
    rul_model = rng.choice(RUL_MODELS)
    risk_model = rng.choice(RISK_MODELS)
    anomaly_model = rng.choice(ANOMALY_MODELS)

    # 1. RUL Prediction simulation (predicted_rul = true_rul + heteroscedastic noise)
    noise_std = 3.0 + 0.1 * true_rul
    noise = rng.normal(0.0, noise_std, size=len(cycles))
    pred_rul = np.maximum(0.0, true_rul + noise)

    unc_width = 8.0 + 0.15 * pred_rul + rng.normal(0.0, 1.5, size=len(cycles))
    unc_width = np.maximum(2.0, unc_width)
    lower_bound = np.maximum(0.0, pred_rul - (unc_width / 2.0))
    upper_bound = pred_rul + (unc_width / 2.0)

    deg_speed = 0.5 + 1.5 * np.exp(-true_rul / 40.0) + rng.normal(0.0, 0.05, size=len(cycles))
    deg_accel = 0.01 + 0.05 * np.exp(-true_rul / 30.0) + rng.normal(0.0, 0.005, size=len(cycles))

    conf_collapse = (pred_rul < 15.0) & (unc_width > 25.0)
    health_drop = (deg_speed > 1.8) & (true_rul < 40)
    instability = (rng.uniform(0, 1, len(cycles)) < 0.05) | health_drop

    # 2. Failure Risk simulation (sigmoidal curves)
    risk_10 = 1.0 / (1.0 + np.exp((true_rul - 12.0) / 4.0))
    risk_30 = 1.0 / (1.0 + np.exp((true_rul - 28.0) / 8.0))
    risk_50 = 1.0 / (1.0 + np.exp((true_rul - 45.0) / 12.0))

    # Add minor random jitter while preserving risk_10 <= risk_30 <= risk_50
    jitter = rng.uniform(0.0, 0.02, size=len(cycles))
    risk_10 = np.clip(risk_10 + jitter * 0.5, 0.0, 1.0)
    risk_30 = np.clip(np.maximum(risk_10, risk_30 + jitter), 0.0, 1.0)
    risk_50 = np.clip(np.maximum(risk_30, risk_50 + jitter * 1.5), 0.0, 1.0)

    u_10 = np.clip(0.05 + 0.3 * (1.0 - np.abs(risk_10 - 0.5) * 2.0), 0.01, 0.5)
    u_30 = np.clip(0.08 + 0.3 * (1.0 - np.abs(risk_30 - 0.5) * 2.0), 0.01, 0.5)
    u_50 = np.clip(0.10 + 0.3 * (1.0 - np.abs(risk_50 - 0.5) * 2.0), 0.01, 0.5)

    t_10 = np.full(len(cycles), _CFG.critical_risk_10_threshold)
    t_30 = np.full(len(cycles), _CFG.maintenance_risk_30_threshold)
    t_50 = np.full(len(cycles), _CFG.monitoring_risk_50_threshold)

    ctrl_risk = 0.5 * risk_10 + 0.3 * risk_30 + 0.2 * risk_50
    risk_vel = np.gradient(risk_30)
    risk_accel = np.gradient(risk_vel)

    fail_10 = risk_10 >= t_10
    fail_30 = risk_30 >= t_30
    fail_50 = risk_50 >= t_50

    risk_state = np.where(
        risk_10 >= 0.8, "critical",
        np.where(risk_30 >= 0.6, "high",
                 np.where(risk_50 >= 0.4, "elevated", "nominal"))
    )

    action_hint = np.where(
        risk_10 >= 0.8, "immediate_maintenance",
        np.where(risk_30 >= 0.6, "schedule_maintenance",
                 np.where(risk_50 >= 0.4, "schedule_inspection", "continue_operation"))
    )

    # 3. Anomaly simulation
    anom_score = 0.05 + 0.85 * np.exp(-true_rul / 25.0) + rng.normal(0.0, 0.02, size=len(cycles))
    anom_score = np.clip(anom_score, 0.0, 1.0)
    anom_thresh = np.full(len(cycles), 0.45)

    res_anom = np.clip(anom_score * 0.9 + rng.normal(0.0, 0.02, size=len(cycles)), 0.0, 1.0)
    if_score = np.clip(anom_score * 0.95 + rng.normal(0.0, 0.02, size=len(cycles)), 0.0, 1.0)
    mah_score = np.clip(anom_score * 1.05 + rng.normal(0.0, 0.03, size=len(cycles)), 0.0, 1.0)

    is_anom = anom_score >= anom_thresh
    anom_sev = np.where(
        anom_score >= 0.8, "critical",
        np.where(anom_score >= 0.65, "high",
                 np.where(anom_score >= 0.45, "medium",
                          np.where(anom_score >= 0.25, "low", "none")))
    )

    top_s1 = rng.choice(SENSOR_NAMES, size=len(cycles))
    top_s2 = rng.choice(SENSOR_NAMES, size=len(cycles))
    top_s3 = rng.choice(SENSOR_NAMES, size=len(cycles))

    s1_score = np.clip(anom_score * 0.8 + rng.uniform(0, 0.1, len(cycles)), 0, 1)
    s2_score = np.clip(anom_score * 0.6 + rng.uniform(0, 0.1, len(cycles)), 0, 1)
    s3_score = np.clip(anom_score * 0.4 + rng.uniform(0, 0.1, len(cycles)), 0, 1)

    ctx_drift = (rng.uniform(0, 1, len(cycles)) < 0.04) | (cycles > max_lifetime * 0.85)
    ctx_conf = np.clip(1.0 - (anom_score * 0.4) - (ctx_drift.astype(float) * 0.3), 0.2, 1.0)

    df_engine = pd.DataFrame({
        "engine_id": engine_id,
        "fd_subset": subset,
        "cycle": cycles,
        "predicted_rul": np.round(pred_rul, 2),
        "lower_bound": np.round(lower_bound, 2),
        "upper_bound": np.round(upper_bound, 2),
        "uncertainty_width": np.round(unc_width, 2),
        "degradation_speed": np.round(deg_speed, 4),
        "degradation_acceleration": np.round(deg_accel, 4),
        "lifecycle_state": np.where(true_rul < 20, "critical", np.where(true_rul < 50, "degrading", np.where(true_rul < 120, "nominal", "early"))),
        "confidence_collapse_flag": conf_collapse,
        "sudden_health_drop_flag": health_drop,
        "instability_flag": instability,
        "risk_10": np.round(risk_10, 4),
        "risk_30": np.round(risk_30, 4),
        "risk_50": np.round(risk_50, 4),
        "uncertainty_10": np.round(u_10, 4),
        "uncertainty_30": np.round(u_30, 4),
        "uncertainty_50": np.round(u_50, 4),
        "threshold_10": t_10,
        "threshold_30": t_30,
        "threshold_50": t_50,
        "control_risk": np.round(ctrl_risk, 4),
        "risk_velocity": np.round(risk_vel, 4),
        "risk_acceleration": np.round(risk_accel, 4),
        "risk_state": risk_state,
        "action_hint_for_supervisor": action_hint,
        "predicted_fail_in_10": fail_10,
        "predicted_fail_in_30": fail_30,
        "predicted_fail_in_50": fail_50,
        "anomaly_score": np.round(anom_score, 4),
        "anomaly_threshold": anom_thresh,
        "residual_anomaly_score": np.round(res_anom, 4),
        "isolation_forest_score": np.round(if_score, 4),
        "mahalanobis_score": np.round(mah_score, 4),
        "anomaly_severity": anom_sev,
        "top_sensor_1": top_s1,
        "top_sensor_2": top_s2,
        "top_sensor_3": top_s3,
        "top_sensor_1_score": np.round(s1_score, 4),
        "top_sensor_2_score": np.round(s2_score, 4),
        "top_sensor_3_score": np.round(s3_score, 4),
        "is_anomalous": is_anom,
        "context_drift_flag": ctx_drift,
        "context_confidence": np.round(ctx_conf, 4),
        "rul_model_used": rul_model,
        "risk_model_used": risk_model,
        "anomaly_model_used": anomaly_model,
    })

    return df_engine


def generate_sample_dataset(
    subsets: tuple[str, ...] = DEFAULT_SUBSETS,
    engines_per_subset: int = DEFAULT_ENGINES_PER_SUBSET,
    seed: int = 42,
) -> pd.DataFrame:
    """Orchestrate multi-subset synthetic dataset generation."""
    rng = np.random.default_rng(seed)
    all_engine_dfs: list[pd.DataFrame] = []

    global_engine_id = 1
    for subset in subsets:
        logger.info("Generating %d engines for sub-dataset %s...", engines_per_subset, subset)
        for _ in range(engines_per_subset):
            df_e = generate_engine_trajectory(global_engine_id, subset, rng)
            all_engine_dfs.append(df_e)
            global_engine_id += 1

    full_df = pd.concat(all_engine_dfs, ignore_index=True)
    return full_df


# ---------------------------------------------------------------------------
# CLI & Standalone Orchestrator
# ---------------------------------------------------------------------------

def run_sample_generator(
    output_path: Path | None = None,
    seed: int = 42,
) -> dict:
    """Run standalone synthetic dataset generation pipeline."""
    if output_path is None:
        output_path = SAMPLE_DATASET_PATH

    output_path = Path(output_path).resolve()

    section("SYNTHETIC SAMPLE OUTPUT GENERATOR STARTED")
    logger.info("Output dataset: %s", output_path)
    logger.info("Random seed   : %d", seed)

    start_time = time.perf_counter()

    full_df = generate_sample_dataset(seed=seed)
    total_rows = len(full_df)
    total_engines = full_df["engine_id"].nunique()

    duration = time.perf_counter() - start_time

    # Save CSV atomically
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(full_df, output_path)
    logger.info("Sample dataset written to: %s (%d rows)", output_path, total_rows)

    # Manifest and Report
    manifest = {
        "generated_by": "sample_output_generator.py",
        "output_path": str(output_path),
        "seed": seed,
        "total_rows": total_rows,
        "total_engines": total_engines,
        "subsets": list(DEFAULT_SUBSETS),
        "columns_generated": list(full_df.columns),
    }

    report = {
        "status": "success",
        "output_path": str(output_path),
        "total_rows": total_rows,
        "total_engines": total_engines,
        "duration_seconds": round(duration, 4),
    }

    _MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest, _MANIFEST_PATH)

    _REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(report, _REPORT_PATH)

    logger.info("-" * 78)
    logger.info("Generated rows        : %d", total_rows)
    logger.info("Generated engines     : %d", total_engines)
    logger.info("Duration              : %.3f seconds", duration)
    section("SYNTHETIC SAMPLE OUTPUT GENERATOR COMPLETED")

    return report


def main() -> int:
    """CLI Entrypoint."""
    try:
        res = run_sample_generator()
        print(json.dumps(res, indent=2))
        return 0
    except Exception as exc:
        logger.error("Error generating sample output: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
