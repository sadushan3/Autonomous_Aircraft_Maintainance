"""Recompute paper evidence from saved predictions; never fit or modify models."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent / "Backend"
sys.path.insert(0, str(BACKEND))
from app.services.Anomaly_Health_Monitoring.evaluation.monitoring_metrics import condition_metrics

RUN = BACKEND / "experiments/Anomaly_Health_Monitoring/run_20261018_four_detectors"
FIGURES = HERE / "figures"
EVIDENCE = HERE / "evidence"
SUBSETS = [f"FD00{i}" for i in range(1, 5)]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def savefig(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(FIGURES / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    scores_path = RUN / "official_scores.csv"
    frame = pd.read_csv(scores_path)
    keys = ["unit", "cycle"]
    assert not frame.duplicated(keys).any()
    reference = read_json(BACKEND / "metrics/holdout_anomaly_evaluation.json")
    metadata = read_json(BACKEND / "models/anomaly_fusion_meta.json")
    assert reference["provenance"]["saved_scores_sha256"] == digest(scores_path)
    assert metadata["score_version"] == "four_detector_ensemble_v1"
    score_columns = ["isolation_forest_score", "mahalanobis_score", "lstm_ae_score", "residual_rule_score"]
    assert np.isfinite(frame[score_columns].to_numpy()).all()
    fusion_error = float(np.abs(frame[score_columns].mean(axis=1) - frame.anomaly_score).max())
    assert fusion_error < 1e-12
    assert np.allclose(frame.threshold_used, metadata["calibrated_fusion_threshold"], rtol=0, atol=1e-15)
    assert metadata["fusion_weights"] == dict.fromkeys(["isolation_forest", "mahalanobis", "lstm_autoencoder", "residual_rule"], 0.25)

    # Reconstruct every evaluation label directly from the original test files.
    labels = []
    dataset = []
    for index, subset in enumerate(SUBSETS):
        train = pd.read_csv(BACKEND / f"data/raw/train_{subset}.txt", sep=r"\s+", header=None)
        test = pd.read_csv(BACKEND / f"data/raw/test_{subset}.txt", sep=r"\s+", header=None)
        terminal = np.loadtxt(BACKEND / f"data/raw/RUL_{subset}.txt").reshape(-1)
        last = test.groupby(0)[1].transform("max")
        labels.append(pd.DataFrame({"unit": test[0] + index * 10000, "cycle": test[1],
                                    "RUL": last - test[1] + test[0].map(dict(enumerate(terminal, 1))),
                                    "subset_id": subset}))
        dataset.append({"subset": subset, "train_engines": int(train[0].nunique()), "train_rows": len(train),
                        "test_engines": int(test[0].nunique()), "test_rows": len(test)})
    labels = pd.concat(labels, ignore_index=True)
    aligned = frame.merge(labels, on=keys, validate="one_to_one", suffixes=("", "_independent"), indicator=True)
    assert len(aligned) == len(labels) == len(frame)
    assert aligned._merge.eq("both").all()
    assert np.array_equal(aligned.RUL, aligned.RUL_independent)
    assert aligned.subset_id.eq(aligned.subset_id_independent).all()
    overall = condition_metrics(frame)
    saved = reference["metrics"]["test"]["condition_proxy_metrics"]
    for key in ("true_positive", "true_negative", "false_positive", "false_negative", "rows", "healthy_rows", "critical_rows"):
        assert overall[key] == saved[key], (key, overall[key], saved[key])
    for key in ("accuracy", "precision", "recall", "f1", "auc", "average_precision", "false_alarm_rate"):
        assert abs(overall[key] - saved[key]) < 1e-12, key
    per_subset = {name: condition_metrics(block) for name, block in frame.groupby("subset_id", sort=True)}
    plan = read_json(BACKEND / "models/training_split.json")
    role_sets = [set(values) for values in plan["engine_roles"].values()]
    assert sum(map(len, role_sets)) == len(set.union(*role_sets)) == 709
    assert plan["plan_id"] == metadata["training_plan_id"]

    baseline_path = BACKEND / "experiments/Anomaly_Health_Monitoring/run_20261015_linear_baseline_verified/comparison.csv"
    comparisons = pd.read_csv(baseline_path)
    # Only the separately measured fixed sensor baseline belongs in this comparison.
    baseline = comparisons.loc[comparisons.model.eq("logistic_sensor_baseline")].copy()
    assert len(baseline) == 5
    baseline.to_csv(EVIDENCE / "linear_baseline.csv", index=False)
    health_path = BACKEND / "metrics/holdout_test_evaluation.json"
    saved_health = read_json(health_path)
    # Use the same official residual inputs as the active anomaly experiment.
    from app.config.Anomaly_Health_Monitoring.config import Config
    from app.services.Anomaly_Health_Monitoring.health_monitoring.health_index_calculator import HealthIndexCalculator
    from app.services.Anomaly_Health_Monitoring.evaluation.evaluate_health import HealthEvaluator
    cfg = Config()
    cfg.mlops.log_to_file = False
    cfg.mlops.enable_mlflow = False
    cfg.paths.logs = HERE / "tools/logs"
    official_residuals = joblib.load(RUN / "official_inputs.joblib")["residuals"]
    health_labels = official_residuals[keys + ["RUL"]].merge(labels[keys + ["RUL"]], on=keys, validate="one_to_one", suffixes=("", "_independent"))
    assert len(health_labels) == len(labels)
    assert np.array_equal(health_labels.RUL, health_labels.RUL_independent)
    health_frame = HealthIndexCalculator(cfg).compute(official_residuals, recalibrate=False)
    extra_columns = [c for c in ("RUL", "subset_id", "split") if c not in health_frame]
    health_frame = health_frame.merge(official_residuals[keys + extra_columns], on=keys, validate="one_to_one")
    evaluator = HealthEvaluator(cfg)
    health = evaluator.evaluate(health_frame)
    health["evaluation_scope"] = "Recomputed from the four-detector experiment's official residuals with the current frozen health reference."
    health["per_subset"] = {name: evaluator.evaluate(block) for name, block in health_frame.groupby("subset_id", sort=True)}
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    (EVIDENCE / "health_recheck.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
    explanation_path = BACKEND / "metrics/evaluate_explainability.json"
    explain = read_json(explanation_path)
    rows = []
    for name, metrics in {"All": overall, **per_subset}.items():
        rows.append({"population": name, **{k: metrics[k] for k in ["input_engines", "evaluated_engines", "rows", "healthy_rows", "critical_rows", "true_positive", "false_positive", "false_negative", "true_negative", "accuracy", "precision", "recall", "f1", "auc", "average_precision", "false_alarm_rate"]}})
    pd.DataFrame(rows).to_csv(EVIDENCE / "anomaly_results.csv", index=False)

    sources = [scores_path, baseline_path, health_path, explanation_path,
               BACKEND / "models/anomaly_fusion_meta.json", BACKEND / "models/training_split.json",
               BACKEND / "models/health_index_calibration.json", BACKEND / "models/health_state_thresholds.json",
               BACKEND / "models/twin_ensemble_weights.json", RUN / "protocol.json", RUN / "activation_verification.json",
               RUN / "official_inputs.joblib", BACKEND / "processed/health_index.csv", BACKEND / "processed/raw_data.csv",
               BACKEND / "experiments/Anomaly_Health_Monitoring/run_20261015_linear_baseline_verified/protocol.json"]
    for name in ("anomaly_fusion.py", "mahalanobis_detector.py", "lstm_autoencorder.py", "isolation_forest_detector.py", "residual_anomaly_detector.py"):
        sources.append(BACKEND / "app/services/Anomaly_Health_Monitoring/anomaly_detection" / name)
    for subset in SUBSETS:
        sources.extend(BACKEND / f"data/raw/{prefix}_{subset}.txt" for prefix in ("train", "test", "RUL"))
    for relative in ("app/config/Anomaly_Health_Monitoring/config.py", "app/utils/Anomaly_Health_Monitoring/training_split.py",
                     "app/services/Anomaly_Health_Monitoring/health_monitoring/health_index_calculator.py",
                     "app/services/Anomaly_Health_Monitoring/health_monitoring/health_score_engine.py",
                     "app/services/Anomaly_Health_Monitoring/evaluation/evaluate_health.py",
                     "app/services/Anomaly_Health_Monitoring/evaluation/monitoring_metrics.py"):
        sources.append(BACKEND / relative)
    evidence = {"scope": "Anomaly_Health_Monitoring only; retrospective condition proxies, not verified faults",
                "model_version": metadata["score_version"], "training_plan_id": plan["plan_id"],
                "engine_roles": {k: len(v) for k, v in plan["engine_roles"].items()},
                "dataset": dataset, "independent_raw_label_check": True, "maximum_fusion_reconstruction_error": fusion_error,
                "overall": overall, "per_subset": per_subset, "health_recomputed_evaluation": health,
                "health_saved_evaluation_for_audit_only": saved_health,
                "health_difference_from_saved": {k: health[k] - saved_health[k] for k in ("rul_correlation", "monotonicity_score", "mean_health_index")},
                "explanation_saved_batch_evaluation": explain,
                "sources_sha256": {str(p.relative_to(BACKEND)): digest(p) for p in sources},
                "analysis_environment": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib")}}
    (EVIDENCE / "paper_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.05, 2.55))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.5); ax.axis("off")
    def box(x, y, width, height, title, color="#edf3fa"):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.035", linewidth=.7, edgecolor="#354e69", facecolor=color))
        ax.text(x + width/2, y + height/2, title, ha="center", va="center", fontsize=8)
    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "lw": .9, "color": "#354e69"})
    box(.1, 1.3, 1.55, .8, "Sensor settings\nPast-only cleaning\nMode scaling")
    box(2.0, 1.3, 1.65, .8, "Operating modes\nHealthy reference\nCommissioning")
    box(4.05, 1.3, 1.0, .8, "Sensor\nresiduals")
    box(5.5, 1.82, 2.3, 1.16, "Isolation Forest\nCovariance | LSTM AE\nResidual rule\nEqual fusion + gate")
    box(5.5, .12, 2.3, 1.03, "Residual deviation\nHealth index + EWMA\nTrend + five states")
    box(8.25, 1.14, 1.62, 1.18, "Alerts + dashboard\nResidual evidence\nReference SHAP\nAgreement", "#ecf4ee")
    arrow((1.65, 1.7), (2, 1.7)); arrow((3.65, 1.7), (4.05, 1.7))
    arrow((5.05, 1.87), (5.5, 2.3)); arrow((5.05, 1.5), (5.5, .7))
    arrow((7.8, 2.3), (8.25, 1.95)); arrow((7.8, .7), (8.25, 1.4))
    ax.text(4.9, 3.45, "Shared engine roles: fit 396 | calibration 85 | selection 86 | retrospective confirmation 142", ha="center", va="top", fontsize=8)
    savefig(fig, "architecture")

    ready = frame.loc[frame.calibration_ready.astype(bool)].copy()
    binary = ready.loc[ready.RUL.ge(125) | ready.RUL.le(30)]
    y = binary.RUL.le(30).astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55))
    for label, col in [("Four-detector fusion", "anomaly_score"), ("Isolation Forest", score_columns[0]),
                       ("Covariance", score_columns[1]), ("LSTM autoencoder", score_columns[2]), ("Residual rule", score_columns[3])]:
        precision, recall, _ = precision_recall_curve(y, binary[col])
        axes[0].plot(recall, precision, label=f"{label} ({average_precision_score(y, binary[col]):.3f})", lw=1.1)
    axes[0].scatter([overall["recall"]], [overall["precision"]], c="black", marker="x", zorder=5, s=24)
    axes[0].set(xlabel="Critical-condition recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03))
    axes[0].legend(loc="lower left", frameon=False, title="Average precision", title_fontsize=7)
    names = list(per_subset)
    axes[1].bar(np.arange(4) - .17, [per_subset[n]["recall"] for n in names], .34, label="Critical-cycle recall", color="#386b9a")
    axes[1].bar(np.arange(4) + .17, [per_subset[n]["engine_events"]["detection_rate_of_reaching"] for n in names], .34, label="Critical-engine detection", color="#7ba18d")
    axes[1].set(xticks=np.arange(4), xticklabels=names, ylabel="Fraction detected", ylim=(0, 1.06))
    axes[1].legend(loc="lower left", frameon=True, facecolor="white", framealpha=.95, edgecolor="none")
    for axis in axes:
        axis.grid(alpha=.2); axis.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    savefig(fig, "detection_results")

    # Plot all current-batch trajectories rather than selecting a favorable example.
    current_health = pd.read_csv(BACKEND / "processed/health_index.csv")
    current_raw = pd.read_csv(BACKEND / "processed/raw_data.csv", usecols=["unit", "cycle", "RUL"])
    current = current_health.drop(columns=["RUL"], errors="ignore").merge(current_raw, on=keys, validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.15))
    for _, block in current.groupby("unit"):
        axes[0].plot(block.cycle, block.health_index, alpha=.18, lw=.55, color="#386b9a")
    axes[0].set(xlabel="Observed cycle", ylabel="Health index", ylim=(0, 102), title=f"Current batch: all {current.unit.nunique()} engines")
    hp = health["per_subset"]
    axes[1].bar(np.arange(4) - .17, [hp[n]["rul_correlation"] for n in names], .34, label="Pooled Spearman correlation", color="#386b9a")
    axes[1].bar(np.arange(4) + .17, [hp[n]["monotonicity_score"] for n in names], .34, label="Monotonicity pass fraction", color="#7ba18d")
    axes[1].set(xticks=np.arange(4), xticklabels=names, ylabel="Correlation or fraction", ylim=(0, 1.05), title="Recomputed official-test health evaluation")
    axes[1].legend(loc="lower left", frameon=True, facecolor="white", framealpha=.95, edgecolor="none", fontsize=6.5)
    for axis in axes:
        axis.grid(alpha=.2); axis.set_axisbelow(True)
    fig.tight_layout(w_pad=2)
    savefig(fig, "health_results")
    pd.DataFrame(dataset).to_csv(EVIDENCE / "dataset_counts.csv", index=False)
    print(json.dumps({"label_check": "all original test labels match", "counts_and_metrics": "reproduce saved evaluation",
                      "fusion_error": fusion_error, "overall": {k: overall[k] for k in ["f1", "recall", "precision", "auc"]},
                      "by_subset": {s: {k: m[k] for k in ["f1", "recall", "false_positive", "false_negative", "auc"]} for s, m in per_subset.items()}}, indent=2))


if __name__ == "__main__":
    main()
