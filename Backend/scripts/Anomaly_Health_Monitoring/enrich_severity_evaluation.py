from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.config.Anomaly_Health_Monitoring.config import Config
from app.services.Anomaly_Health_Monitoring.anomaly_detection.severity_classifier import SeverityClassifier
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import boolean_values
from scripts.Anomaly_Health_Monitoring.evaluate_holdout import load_frames
from app.utils.Anomaly_Health_Monitoring.utils import atomic_write_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    target = ROOT / "metrics/holdout_anomaly_evaluation.json"
    archive = ROOT / "experiments/Anomaly_Health_Monitoring/run_20261011_strengthening"
    scores = archive / "official_operational_scored.csv"
    labels = archive / "official_labels.csv"
    bands = ROOT / "models/severity_bands.json"
    report = json.loads(target.read_text(encoding="utf-8"))
    original_metrics = copy.deepcopy(report["metrics"])
    provenance = report["provenance"]
    if digest(scores) != provenance["scores_sha256"] or digest(labels) != provenance["labels_sha256"]:
        raise ValueError("Archived data does not match the published evaluation provenance")
    if digest(bands) != provenance["trained_models"]["severity_bands.json"]:
        raise ValueError("Severity cuts changed since this benchmark; refusing to mix calibrations")

    frame = load_frames(scores, labels)
    frame = frame.loc[frame["split"].eq("test") & boolean_values(frame["calibration_ready"])].copy()
    if len(frame) != report["metrics"]["test"]["rows"]:
        raise ValueError("Ready-row count disagrees with published benchmark")
    if not np.isfinite(frame[["anomaly_score", "threshold_used", "RUL"]].to_numpy(float)).all():
        raise ValueError("Nonfinite evaluation evidence")
    cfg = Config()
    cfg.mlops.log_to_file = False
    classified = SeverityClassifier(cfg).classify(frame)
    means = classified.groupby("severity")["RUL"].mean()
    counts = classified.groupby("severity").size()
    summary = {str(k): round(float(v), 6) for k, v in means.items()}
    row_counts = {str(k): int(v) for k, v in counts.items()}
    if sum(row_counts.values()) != len(frame):
        raise ValueError("Severity summary lost evaluated rows")
    report["metrics"]["test"]["severity_mean_rul"] = summary
    report["metrics"]["test"]["severity_row_counts"] = row_counts
    report["metrics"]["test"]["severity_summary_note"] = (
        "Mean true RUL per cycle on ready held-out rows, using frozen severity cuts. "
        "This is a descriptive association, not a calibrated remaining-life estimate or confirmed fault severity."
    )
    provenance["severity_summary"] = {
        "producer": "scripts/Anomaly_Health_Monitoring/enrich_severity_evaluation.py",
        "bands_sha256": digest(bands),
        "rows": len(frame),
        "method": "Existing SeverityClassifier.classify over archived ready test scores; no fitting or model inference",
    }

    unchanged = copy.deepcopy(report["metrics"])
    for key in ("severity_mean_rul", "severity_row_counts", "severity_summary_note"):
        unchanged["test"].pop(key, None)
        original_metrics["test"].pop(key, None)
    if unchanged != original_metrics:
        raise ValueError("Existing evaluation measurements changed")
    atomic_write_json(report, target)
    print(json.dumps({"severity_mean_rul": summary, "severity_row_counts": row_counts,
                      "rows": len(frame), "existing_metrics_unchanged": True,
                      "pipeline_run": False}, indent=2))


if __name__ == "__main__":
    main()
