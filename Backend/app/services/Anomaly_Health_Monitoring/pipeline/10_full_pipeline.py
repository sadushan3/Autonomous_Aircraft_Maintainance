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

from app.config.Anomaly_Health_Monitoring.config import Config, get_config
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_api import (
    DashboardApiService,
)
from app.services.Anomaly_Health_Monitoring.dashboard.dashboard_data_generator import (
    DashboardDataGenerator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_anomaly import (
    AnomalyEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_context import (
    ContextEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_digital_twin import (
    DigitalTwinEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_explainability import (
    ExplainabilityEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_health import (
    HealthEvaluator,
)
from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_reasoning import (
    ReasoningEvaluator,
)
from app.utils.Anomaly_Health_Monitoring.utils import (
    AnomalyHealthError,
    StageExecutionError,
    atomic_write_json,
    get_logger,
)

STAGE_NAME = "10_full_pipeline"


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


                result = module.run_stage(cfg, recalibrate=retrain)
            elif stage_id == "05_anomaly_detection":
                result = module.run_stage(cfg, retrain=retrain)
            elif stage_id == "07_explainability":
                result = module.run_stage(cfg, include_shap=include_shap)
            elif stage_id == "09_feedback_learning":
                result = {"stage": stage_id, "skipped": True,
                          "reason": "Scoring and offline retraining keep the selected cutoff frozen; apply explicit feedback separately."}
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
        except Exception as exc:
            logger.warning("Evaluation '%s' failed to run: %s", name, exc)
            summary[name] = {"error": str(exc)}
    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:


    parser = argparse.ArgumentParser(description="Stage 10: full pipeline")
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--no-shap", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_args(argv)


def main() -> None:

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
