from __future__ import annotations

import ast
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.reasoning.sensor_dependency_graph import SENSOR_SUBSYSTEMS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/study_guide"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    files = {}
    for base in ("app/services/Anomaly_Health_Monitoring", "app/utils/Anomaly_Health_Monitoring",
                 "app/config/Anomaly_Health_Monitoring", "app/routers/Anomaly_Health_Monitoring",
                 "app/schemas/Anomaly_Health_Monitoring", "scripts/Anomaly_Health_Monitoring",
                 "tests/Anomaly_Health_Monitoring"):
        for path in sorted((ROOT / base).rglob("*.py")):
            raw = path.read_bytes()
            tree = ast.parse(raw.decode("utf-8-sig"))
            files[path.relative_to(ROOT).as_posix()] = {"sha256": hashlib.sha256(raw).hexdigest(),
                "lines": len(raw.splitlines()), "classes": [node.name for node in tree.body if isinstance(node, ast.ClassDef)]}
    routes = []
    tree = ast.parse((ROOT / "app/routers/Anomaly_Health_Monitoring/routes.py").read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for deco in node.decorator_list:
                if isinstance(deco, ast.Call) and isinstance(deco.func, ast.Attribute) and deco.func.attr in {"get", "post", "put", "delete"}:
                    routes.append({"method": deco.func.attr.upper(), "path": cfg.api.router_prefix + ast.literal_eval(deco.args[0]),
                                   "handler": node.name, "line": node.lineno,
                                   "description": (ast.get_docstring(node) or "").split("\n")[0]})
    documents = {}
    paths = ["models/anomaly_fusion_meta.json", "models/twin_ensemble_weights.json", "models/health_index_calibration.json",
             "models/health_state_thresholds.json", "models/severity_bands.json", "models/training_split.json",
             "models/training_provenance.json", "metrics/holdout_anomaly_evaluation.json", "metrics/holdout_test_evaluation.json",
             "reports/health_evaluation.json", "metrics/evaluate_explainability.json", "metrics/residual_validation.json",
             "experiments/Anomaly_Health_Monitoring/run_20261015_linear_baseline_verified/evaluation.json"]
    for name in paths:
        path = ROOT / name
        documents[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "data": json.loads(path.read_text())}
    startup = joblib.load(cfg.paths.models / "startup_profile_twin.pkl")
    correction = startup.get("correction")
    startup_summary = {"keys": list(startup), "correction_type": type(correction).__name__,
                       "correction_parameters": correction.get_params(deep=False) if hasattr(correction, "get_params") else None}
    startup_summary.update({k: v for k, v in startup.items() if isinstance(v, (str, int, float, bool, type(None)))})
    counts = {}
    for subset in cfg.data.subsets:
        counts[subset] = {}
        for split in ("train", "test"):
            matches = list(cfg.paths.data_raw.rglob(f"{split}_{subset}.txt"))
            if len(matches) != 1:
                raise ValueError(f"Expected one {split}_{subset}.txt; found {len(matches)}")
            units = np.loadtxt(matches[0], usecols=0, dtype=int)
            counts[subset][split] = {"engines": len(np.unique(units)), "rows": len(units),
                                    "file": matches[0].relative_to(ROOT).as_posix()}
    versions = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "tensorflow", "tensorflow-cpu", "tensorflow-intel", "keras", "shap", "fastapi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not available under this distribution name"
    payload = {"generated_utc": datetime.now(timezone.utc).isoformat(), "config_defaults": asdict(cfg),
               "documents": documents, "source_files": files, "api_routes": routes,
               "sensor_subsystems_as_coded": SENSOR_SUBSYSTEMS, "startup_profile": startup_summary,
               "local_dataset_counts": counts, "installed_versions": versions}
    (OUT / "evidence_snapshot.json").write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"source_files": len(files), "routes": len(routes), "dataset_counts": counts,
                      "startup_profile": startup_summary, "versions": versions}, indent=2, default=str))


if __name__ == "__main__":
    main()
