"""
Pipeline stage 10: the full end-to-end orchestrator.

Runs every stage in order, from raw C-MAPSS ingestion through to the
dashboard data and the integration exchange file:

    01 preprocessing -> 02 context -> 03 digital twin -> 04 health monitoring
    -> 05 anomaly detection -> 06 reasoning -> 07 explainability
    -> 08 uncertainty -> 09 feedback learning -> dashboard + exchange file

Each stage is already independently crash-safe (see ``stage_guard`` in
``utils.py``): a failure in any one stage logs the error, leaves every prior
stage's output untouched, and this orchestrator stops and reports exactly
which stage failed rather than silently continuing with stale data.

Because stage filenames carry a leading digit (``01_preprocessing.py`` etc.)
they are not importable as ordinary dotted modules; this script loads each one
via :mod:`importlib.util` from its file path, which is the standard approach
for numerically-prefixed script files.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
_PIPELINE_DIR = Path(__file__).resolve().parent

from app.config.Anomaly_Health_Monitoring.config import Config, get_config  # noqa: E402
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_api import (  # noqa: E402
    DashboardApiService,
)
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_data_generator import (  # noqa: E402
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_anomaly import (  # noqa: E402
    AnomalyEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_context import (  # noqa: E402
    ContextEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_digital_twin import (  # noqa: E402
    DigitalTwinEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_explainability import (  # noqa: E402
    ExplainabilityEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_health import (  # noqa: E402
    HealthEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_reasoning import (  # noqa: E402
    ReasoningEvaluator,
)
from app.utils.Anomaly_Health_Monitoring.utils import (  # noqa: E402
    AnomalyHealthError,
    StageExecutionError,
    atomic_write_json,
    get_logger,
)

STAGE_NAME = "10_full_pipeline"

#: Numbered stage modules in execution order.
_STAGE_FILES: List[str] = [
    "01_preprocessing.py",
    "02_context_modeling.py",
    "03_digital_twin.py",
    "04_health_monitoring.py",
    "05_anomaly_detection.py",
    "06_reasoning.py",
    "07_explainability.py",
    "08_uncertainty.py",
    "09_feedback_learning.py",
]


def _load_stage_module(filename: str) -> ModuleType:
    """Load a numerically-prefixed stage script as an importable module.

    Args:
        filename: The stage script's filename, e.g. ``'01_preprocessing.py'``.

    Returns:
        The loaded module, exposing its ``run_stage`` function.

    Raises:
        AnomalyHealthError: If the module cannot be loaded.
    """
    path = _PIPELINE_DIR / filename
    module_name = f"ahm_pipeline_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AnomalyHealthError(f"Could not load pipeline stage module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_full_pipeline(
    config: Optional[Config] = None,
    retrain: bool = True,
    include_shap: bool = True,
    stop_on_failure: bool = True,
    source_csv: Optional[str] = None,
) -> Dict[str, Any]:
    """Run every stage in order and assemble the combined pipeline report.

    Args:
        config: Configuration to run under; defaults to the active singleton.
        retrain: Whether context/twin/detector stages retrain from scratch.
        include_shap: Whether the explainability stage computes global SHAP.
        stop_on_failure: Stop at the first failing stage (recommended). When
            False, later stages still run against whatever earlier stages
            managed to produce, useful for smoke-testing partial pipelines.
        source_csv: Optional already-parsed CSV to ingest instead of the
            configured raw subsets -- forwarded to stage 01 only, which
            already supports it (see ``DataLoader.run``). Every other stage
            operates on whatever stage 01 produced, unchanged.

    Returns:
        A report with per-stage results, in ``StageResponse``-like shape, plus
        a top-level summary and the exchange-file path.
    """
    cfg = config or get_config()
    logger = get_logger(__name__, cfg)
    cfg.ensure_directories()
    pipeline_started = time.perf_counter()

    stage_results: List[Dict[str, Any]] = []
    failed_stage: Optional[str] = None

    for filename in _STAGE_FILES:
        stage_id = filename.replace(".py", "")
        module = _load_stage_module(filename)
        stage_started = time.perf_counter()
        try:
            if stage_id == "01_preprocessing":
                result = module.run_stage(cfg, source_csv=source_csv, fit_policy=retrain)
            elif stage_id == "02_context_modeling":
                result = module.run_stage(cfg, retrain=retrain)
            elif stage_id == "03_digital_twin":
                result = module.run_stage(cfg, retrain=retrain)
            elif stage_id == "04_health_monitoring":
                # Stage 04 was previously called with no arguments, so its
                # `recalibrate=True` default held even under retrain=False: an
                # inference run re-fit the health index's healthy reference
                # from whatever engines happened to be in the batch and
                # overwrote `models/health_index_calibration.json`. That both
                # broke the inference-only contract and made the index
                # batch-relative -- a batch of only degraded engines
                # recalibrates onto their degraded level and reports them as
                # healthy.
                result = module.run_stage(cfg, recalibrate=retrain)
            elif stage_id == "05_anomaly_detection":
                result = module.run_stage(cfg, retrain=retrain)
            elif stage_id == "07_explainability":
                result = module.run_stage(cfg, include_shap=include_shap)
            else:
                result = module.run_stage(cfg)
            status = "success"
        except StageExecutionError as exc:
            result = {"stage": stage_id, "error": str(exc)}
            status = "failed"
            failed_stage = stage_id
            logger.error("Pipeline halted: stage '%s' failed: %s", stage_id, exc)
        stage_results.append(
            {
                "stage": stage_id,
                "status": status,
                "duration_seconds": round(time.perf_counter() - stage_started, 3),
                "details": result,
            }
        )
        if status == "failed" and stop_on_failure:
            break

    dashboard_result: Dict[str, Any] = {}
    if failed_stage is None:
        try:
            generator = DashboardDataGenerator(cfg)
            _, dashboard_report = generator.run()
            api_service = DashboardApiService(cfg)
            exchange_report = api_service.run()
            dashboard_result = {"dashboard": dashboard_report, "exchange": exchange_report}
        except AnomalyHealthError as exc:
            failed_stage = "dashboard"
            logger.error("Dashboard/exchange stage failed: %s", exc)
            dashboard_result = {"error": str(exc)}

    evaluation_summary: Dict[str, Any] = {}
    if failed_stage is None:
        evaluation_summary = _run_evaluations(cfg, logger)

    total_duration = time.perf_counter() - pipeline_started
    report: Dict[str, Any] = {
        "stage": STAGE_NAME,
        "status": "success" if failed_stage is None else "failed",
        "failed_stage": failed_stage,
        "total_duration_seconds": round(total_duration, 3),
        "stages": stage_results,
        "dashboard": dashboard_result,
        "evaluation": evaluation_summary,
        "exchange_file": dashboard_result.get("exchange", {}).get("exchange_path"),
    }
    atomic_write_json(report, cfg.paths.reports / "full_pipeline_report.json")
    logger.info(
        "Full pipeline %s in %.1fs (failed_stage=%s)",
        report["status"],
        total_duration,
        failed_stage,
    )
    return report


def _run_evaluations(config: Config, logger: Any) -> Dict[str, Any]:
    """Run every evaluation module and collect a pass/fail summary.

    Args:
        config: Active configuration.
        logger: Logger to report through.

    Returns:
        A mapping of evaluation name to its acceptance verdict.
    """
    evaluators = {
        "digital_twin": DigitalTwinEvaluator(config),
        "context": ContextEvaluator(config),
        "health": HealthEvaluator(config),
        "anomaly": AnomalyEvaluator(config),
        "reasoning": ReasoningEvaluator(config),
        "explainability": ExplainabilityEvaluator(config),
    }
    summary: Dict[str, Any] = {}
    for name, evaluator in evaluators.items():
        try:
            summary[name] = evaluator.run()
        except Exception as exc:  # noqa: BLE001 - evaluation must not crash the pipeline
            logger.warning("Evaluation '%s' failed to run: %s", name, exc)
            summary[name] = {"error": str(exc)}
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for standalone execution.

    Args:
        argv: Optional argument vector for testing.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description="Stage 10: full pipeline")
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for standalone execution."""
    args = parse_args()
    config = get_config()
    report = run_full_pipeline(
        config,
        retrain=not args.no_retrain,
        include_shap=not args.no_shap,
        stop_on_failure=not args.continue_on_failure,
    )
    if report["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
