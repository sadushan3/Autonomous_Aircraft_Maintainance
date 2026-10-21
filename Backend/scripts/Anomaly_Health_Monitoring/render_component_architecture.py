from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as PlotPath

ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "reports/architecture"
COLORS = {
    "input": ("#EDF2F7", "#52647A"),
    "prep": ("#EAF2FF", "#2861AF"),
    "twin": ("#EDF0FF", "#5258B5"),
    "health": ("#E5F5EF", "#14725B"),
    "anomaly": ("#FFF0E5", "#AF571C"),
    "evidence": ("#F3EDFC", "#73509A"),
    "output": ("#E8F3F7", "#256479"),
    "control": ("#FFF7E8", "#96702B"),
}


def render(destination):
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / "anomaly_health_monitoring_architecture"
    if any(stem.with_suffix(ext).exists() for ext in (".png", ".svg", ".pdf")):
        raise FileExistsError("Architecture files already exist; choose another --output-dir")
    config_path = ROOT / "app/config/Anomaly_Health_Monitoring/config.py"
    from app.config.Anomaly_Health_Monitoring.config import Config
    config = Config()
    fusion = json.loads((config.paths.models / "anomaly_fusion_meta.json").read_text())
    plan = json.loads((config.paths.models / "training_split.json").read_text())
    weights = json.loads((config.paths.models / "twin_ensemble_weights.json").read_text())["weights"]
    active_detectors = [name for name, value in config.anomaly.fusion_weights.items() if value > 0]
    if active_detectors != ["condition_consensus"] or set(weights) != {"random_forest", "xgboost", "lightgbm", "tensorflow"}:
        raise ValueError("The active configuration changed; update diagram labels before rendering")
    roles = {role: len(units) for role, units in plan["engine_roles"].items()}
    if roles != {"fit": 396, "calibration": 85, "selection": 86, "confirmation": 142}:
        raise ValueError("Update partition labels for the new training plan")

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "pdf.fonttype": 42, "text.color": "#15273B"})
    fig = plt.figure(figsize=(24, 19.2), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 100), ylim=(80, 0))
    ax.axis("off")
    labels, boxes = [], []

    def text(x, y, value, size=10.6, color="#273B50", weight="normal", **kwargs):
        item = ax.text(x, y, value, fontsize=size, color=color, fontweight=weight,
                       va="top", ha="left", zorder=5, **kwargs)
        labels.append(item)
        return item

    def panel(x, y, w, h, fill, edge, title):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.6",
                     facecolor=fill, edgecolor=edge, linewidth=1.15, zorder=0))
        text(x + 1, y + .9, title, size=14, weight="bold",
             color="#52647A" if fill == "#FAFCFE" else "#96702B")

    def box(x, y, w, h, title, lines, kind, small=False):
        fill, edge = COLORS[kind]
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=0.48",
                     facecolor=fill, edgecolor=edge, linewidth=1.35, zorder=3))
        title_text = text(x + .8, y + .65, title, size=13.2 if small else 14, weight="bold", color=edge)
        contained = [title_text]
        for index, line in enumerate(lines):
            contained.append(text(x + .8, y + 2.15 + index * 1.06, line,
                                   size=11.8 if small else 13))
        boxes.append(((x, y, w, h), contained))

    def arrow(points, color="#52647A", style="solid", width=1.65, zorder=2):
        path = PlotPath(points, [PlotPath.MOVETO] + [PlotPath.LINETO] * (len(points) - 1))
        ax.add_patch(PathPatch(path, fill=False, color=color, linewidth=width, linestyle=style, zorder=zorder))
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=13,
                                     color=color, linewidth=width, linestyle=style, zorder=zorder))

    def caption(x, y, value, color="#52647A", size=9.1):
        text(x, y, value, size=size, color=color,
             bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})

    text(2.2, 1.5, "ANOMALY & HEALTH MONITORING", size=25, weight="bold", color="#162F48")
    text(2.3, 4.05, "System architecture  |  Active implementation  |  Data flow and offline control", size=12.8, color="#53677A")
    panel(2, 7, 73, 63.5, "#FAFCFE", "#B8C7D6", "INFERENCE / MONITORING FLOW  ·  Frozen learned artifacts")
    panel(78, 7, 19.5, 63.5, "#FFFCF6", "#CDB78B", "OFFLINE & CONTROL")

    box(4, 11, 20, 6, "Data inputs", ["C-MAPSS FD001–FD004 / CSV", "Engine ID, cycle, 21 sensors", "3 operating settings"], "input")
    box(28, 11, 22, 6, "01  Preprocessing", ["Validate, deduplicate, causal fill", "Frozen medians + sensor policy", "Past-only features; fixed age scale"], "prep")
    box(54, 11, 19, 6, "02  Operating context", ["6 operating modes (K-means)", "Transitions + drift checks", "Frozen per-mode scaling"], "prep")
    arrow([(24, 14), (28, 14)])
    arrow([(50, 14), (54, 14)])

    box(4, 24, 20, 9, "03  Digital twin ensemble", ["Random Forest • XGBoost", "LightGBM • TensorFlow MLP", "Healthy-reference sensor models", "Inverse-validation-RMSE weights", "Expected sensors + twin spread", "Prior-only commissioning offsets"], "twin")
    box(28, 24, 22, 9, "03  Residual calculation", ["Scaled observed − reference expected", "Monitoring residuals per sensor", "Frozen residual z-reference", "Aggregate deviation + readiness", "First 30 cycles: commissioning", "Residual validation / twin comparison"], "twin")
    arrow([(63.5, 17), (63.5, 20), (14, 20), (14, 24)], color="#2861AF")
    arrow([(39, 20), (39, 24)], color="#2861AF")
    ax.plot(39, 20, "o", color="#2861AF", markersize=4, zorder=4)
    caption(6, 18.25, "Context → twin features; scaled observations → residuals", color="#2861AF")
    arrow([(24, 28.5), (28, 28.5)], color="#5258B5")

    box(54, 24, 19, 9, "Optional display refinement", ["Startup-profile correction", "Uses first 30 cycles only", "Improves displayed expectation", "Monitoring keeps the separate", "context-ensemble reference", "Inside stage 03; not a detector"], "input", small=True)

    box(4, 39, 20, 11, "04  Health monitoring", ["Residual magnitude / healthy reference", "Exponential decay + causal EWMA", "Health index: 0–100", "Trend → adjusted health score", "Threshold-based health state", "Healthy → Degrading → Warning", "→ Critical → Failure Risk", "State-based health alerts"], "health")
    box(28, 39, 22, 11, "05  Anomaly detection", ["ACTIVE: residual-history consensus", "2 feature families × 3 ExtraTrees", "Causal 5 / 20 / 40-cycle histories", "Covariance-distance feature reference", f"Cutoff {fusion['calibrated_fusion_threshold']:.1f}; persisted margin {fusion['margin']:.1f}", "Transition gate + 3-cycle persistence", "None / operational / fault", "Severity + early-warning score"], "anomaly")
    box(54, 39, 19, 11, "08  Confidence estimation", ["Twin disagreement", "Detector coverage / agreement", "Score reliability + data quality", "Weighted confidence: 0–100", "Reliability + uncertainty scores", "Inputs from stages 01, 03, 05", "Executed after stage 07"], "evidence", small=True)
    arrow([(39, 33), (39, 36), (14, 36), (14, 39)], color="#14725B")
    arrow([(39, 36), (39, 39)], color="#AF571C")
    ax.plot(39, 36, "o", color="#52647A", markersize=4, zorder=4)
    caption(15.5, 36.9, "Shared residual evidence")
    arrow([(24, 45.5), (28, 45.5)], color="#14725B", width=1.2)
    caption(24.3, 42.9, "trend", color="#14725B", size=8.7)
    arrow([(50, 45.5), (54, 45.5)], color="#AF571C", width=1.2)
    arrow([(22, 33), (22, 34.8), (52, 34.8), (52, 41), (54, 41)], color="#7D8797", width=1.05)
    caption(43, 33.5, "twin spread", size=8.6)

    box(4, 56, 20, 9, "06  Root-cause reasoning", ["Fault events + residual history", "Sensor dependency graph + rules", "Subsystem / sensor attribution", "Temporal pattern + anomaly memory", "Suggested cause + recommended action", "Evidence-based, not causal proof"], "evidence")
    box(28, 56, 22, 9, "07  Explainability", ["Ranked sensor residual evidence", "Monitoring-reference SHAP (sampled)", "Health-index contribution details", "Why / sensors / pattern / severity", "Uses residuals, health and twin outputs", "Local explanations with fidelity checks"], "evidence")
    box(54, 56, 19, 9, "Dashboard / API / export", ["Merge stage outputs by engine/cycle", "Health + anomaly + confidence", "Explanations + recommended action", "FastAPI /anomaly-health", "anomaly_health_predictions.csv", "→ Maintenance Supervisor"], "output", small=True)
    arrow([(39, 50), (39, 53), (14, 53), (14, 56)], color="#AF571C")
    caption(16.5, 51.4, "Fault-type events", color="#AF571C")
    arrow([(24, 60.5), (28, 60.5)], color="#73509A")
    arrow([(50, 60.5), (54, 60.5)], color="#73509A")
    arrow([(63.5, 50), (63.5, 56)], color="#73509A")
    caption(65, 52.4, "confidence", color="#73509A", size=8.8)
    arrow([(4, 47), (3, 47), (3, 67), (39, 67), (39, 65)], color="#14725B", width=1.1)
    caption(9, 65.65, "Health index, state and trend → explanations / output merge", color="#14725B", size=9)

    box(79, 11, 17.5, 6, "Training engine fleet", ["709 training engines", "RUL proxies for healthy regions", "RUL labels remain offline"], "control", small=True)
    box(79, 19, 17.5, 8, "Partition BEFORE fitting", ["396 fit engines", "85 healthy-calibration engines", "86 development-selection engines", "142 retrospective-confirmation", "Shared saved partition / provenance"], "control", small=True)
    arrow([(87.75, 17), (87.75, 19)], color="#96702B")
    box(79, 29, 17.5, 10, "Fit, calibrate, select", ["396: preprocessing + model fitting", "Healthy fit rows: twin references", "396: covariance + health reference", "85: healthy-tail calibration", "86: early stopping / ensemble weights", "86: consensus margin + promotion", "No confirmation / official-test fit"], "control", small=True)
    arrow([(87.75, 27), (87.75, 29)], color="#96702B")
    box(79, 41, 17.5, 8, "Versioned artifacts", ["Models, scalers, references, cutoffs", "Shared plan + provenance metadata", "Load for scoring; no batch refitting", "Used by monitoring stages 01–05"], "control", small=True)
    arrow([(87.75, 39), (87.75, 41)], color="#96702B")
    arrow([(79, 44), (75.3, 44)], color="#96702B", style="dotted", width=1.4)
    caption(75.8, 42.2, "load", color="#96702B", size=8.3)
    box(79, 51, 17.5, 7, "Evaluation boundary", ["142 confirmation + 707 official engines", "Report only; no test-driven promotion", "Previously inspected benchmark", "F1 / recall + health trajectory checks"], "input", small=True)
    box(79, 60, 17.5, 8, "09  Explicit feedback", ["Operator verdict → feedback store", "Anomaly memory / threshold updates", "Health-state boundary adaptation", "Explicit API or CLI action only", "Skipped in the automatic pipeline"], "control", small=True)
    arrow([(73, 62), (76.3, 62), (76.3, 64), (79, 64)], color="#B97820", style="dashed", width=1.35)
    arrow([(96.5, 64), (98.7, 64), (98.7, 45), (96.5, 45)], color="#B97820", style="dashed", width=1.35)
    text(99.05, 59.5, "Explicit saved updates", size=9, color="#96702B", rotation=90)

    ax.add_patch(FancyBboxPatch((2, 72), 95.5, 3.9, boxstyle="round,pad=0.0,rounding_size=0.45",
                 facecolor="#F5F6F8", edgecolor="#BBC3CD", linestyle="dashed", linewidth=1.0))
    text(3, 72.65, "AVAILABLE BUT INACTIVE IN THE CURRENT ANOMALY FUSION", size=10.6, weight="bold", color="#5F6978")
    text(3, 74.05, "Isolation Forest  •  standalone Mahalanobis  •  LSTM autoencoder  •  residual rules  (weights = 0).   Linear baseline: evaluation only.", size=10.5, color="#5F6978")
    text(2.3, 77, "Solid arrows: data flow    ·    Dotted: frozen-artifact loading    ·    Dashed amber: explicit feedback control", size=10.5, color="#52647A")
    text(2.3, 78.5, "Numbers match pipeline stages. Health (04) runs before anomaly (05); confidence (08) runs after explainability (07). No RUL forecast is produced here.", size=10.1, color="#52647A")


    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    problems = []
    for (x, y, w, h), contained in boxes:
        for item in contained:
            bounds = item.get_window_extent(renderer).transformed(ax.transData.inverted())
            xs, ys = (bounds.x0, bounds.x1), (bounds.y0, bounds.y1)
            if min(xs) < x or max(xs) > x + w - .25 or min(ys) < y or max(ys) > y + h - .25:
                problems.append(item.get_text())
    if problems:
        raise ValueError(f"Node labels overflow their boxes: {problems}")
    for item in labels:
        bounds = item.get_window_extent(renderer).transformed(ax.transData.inverted())
        if min(bounds.x0, bounds.x1) < 0 or max(bounds.x0, bounds.x1) > 100 or min(bounds.y0, bounds.y1) < 0 or max(bounds.y0, bounds.y1) > 80:
            raise ValueError(f"Canvas label overflow: {item.get_text()}")
    fig.savefig(stem.with_suffix(".png"), dpi=220, facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    metadata = {"source_config": str(config_path), "training_plan_id": plan["plan_id"],
                "active_detectors": active_detectors, "twin_models": list(weights), "engine_roles": roles,
                "fusion_cutoff": fusion["calibrated_fusion_threshold"], "consensus_margin": fusion["margin"],
                "node_label_overflow_check": "passed", "canvas_label_overflow_check": "passed",
                "renderer": "matplotlib; deterministic source labels and arrows"}
    stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(str(stem.with_suffix(".png")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DESTINATION)
    render(parser.parse_args().output_dir)
