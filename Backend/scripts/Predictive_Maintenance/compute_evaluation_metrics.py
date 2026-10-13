"""Compute real held-out RUL metrics and emit ``evaluation_metrics.json``.

Closes the gap the audit flagged: no accuracy figure existed anywhere in the
repo for the RUL models, and the ``/evaluation/metrics`` endpoint (and the
dashboard's PICP/MPIW cells) had no file to read.

What it measures
----------------
The standard C-MAPSS held-out protocol: for every engine in ``test_FDxxx.txt``
take the final observed window, predict, and compare against that engine's true
remaining life from ``RUL_FDxxx.txt``.

Reported per subset and pooled:

* ``mae`` / ``rmse`` / ``r2`` -- point-accuracy of the predicted RUL.
* ``nasa_score``  -- the C-MAPSS asymmetric score, which punishes late
  predictions (over-estimating remaining life) far harder than early ones.
* ``picp`` -- Prediction Interval Coverage Probability: the fraction of engines
  whose true RUL fell inside ``[lower_bound, upper_bound]``. With a fixed
  ``q_hat`` this is the honest test of whether that constant band is calibrated.
* ``mpiw`` -- Mean Prediction Interval Width, the average of those bands.

The op_cluster question
-----------------------
The models were trained on raw sensors plus one-hot operating-condition
columns (``op_cluster_*``) derived by clustering the three op settings. The
serving path never reconstructs them, so ``PreprocessingService`` zero-fills
them -- meaning the deployed pipeline feeds an all-zero one-hot where training
saw exactly one active bit.

``--op-cluster`` selects which pipeline is measured:

* ``zero``    -- reproduces deployed behaviour exactly. This is what the API
                 returns today, so it is the number the dashboard should show.
* ``derived`` -- reconstructs the one-hot. FD001/FD003 are single-condition so
                 the bit is unambiguous; FD002/FD004 use KMeans over the op
                 settings with labels aligned to the training fingerprint (see
                 ``add_op_cluster``).
* ``both``    -- runs both and reports the gap.

Measured outcome (2026-08-24, 627 held-out engines)
---------------------------------------------------
Reconstructing the one-hot does **not** improve accuracy, so the serving
pipeline was deliberately left zero-filling:

===== ================= ================
subset MAE zero-filled   MAE reconstructed
===== ================= ================
FD001 11.639             11.628  (-0.1%, noise)
FD002 30.083             30.832  (worse)
FD003 12.191             12.374  (worse)
FD004 39.788             42.519  (worse)
===== ================= ================

Why: for FD001/FD003 the column was constant at training time (scaler
``mean_=1.0, scale_=1.0``), so it scaled to a constant 0.0 for every training
row and the network never learned a meaningful weight for it. Restoring the
"correct" 1.0 only shifts predictions by an arbitrary learned bias -- helping
one subset and hurting another, which is what the table shows. For FD002/FD004
the five ~15% clusters cannot be told apart from the scaler statistics, so the
remaining label permutation is unrecoverable and costs more than it saves.

An earlier run of this script used raw KMeans labels with no alignment and
appeared to improve FD004 by 20%. That was permutation luck, not signal; the
aligned mapping above reverses it.

Usage
-----
    python -m scripts.Predictive_Maintenance.compute_evaluation_metrics
    python -m scripts.Predictive_Maintenance.compute_evaluation_metrics --op-cluster both
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Predictive_Maintenance.config import get_settings  # noqa: E402
from app.services.Predictive_Maintenance.artifact_loader import ArtifactLoader  # noqa: E402

RAW_COLUMNS = (
    ["unit", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

#: C-MAPSS operating-condition count per subset.
N_CONDITIONS = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}

OUT_PATH = _BACKEND_ROOT / "metrics" / "Predictive_Maintenance" / "evaluation_metrics.json"


def load_subset(fd: str, data_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Read one subset's test frame and its per-engine true RUL."""
    test = pd.read_csv(
        data_root / f"test_{fd}.txt", sep=r"\s+", header=None, engine="python"
    )
    test = test.iloc[:, : len(RAW_COLUMNS)]
    test.columns = RAW_COLUMNS
    truth = pd.read_csv(
        data_root / f"RUL_{fd}.txt", sep=r"\s+", header=None, engine="python"
    ).iloc[:, 0].to_numpy(dtype=float)
    return test, truth


def add_op_cluster(frame: pd.DataFrame, fd: str, mode: str) -> pd.DataFrame:
    """Attach an ``op_cluster`` column so the one-hots can be reconstructed."""
    out = frame.copy()
    if mode == "zero":
        return out  # leave absent; PreprocessingService zero-fills the one-hots
    k = N_CONDITIONS.get(fd, 1)
    if k == 1:
        out["op_cluster"] = 0
        return out
    from sklearn.cluster import KMeans

    ops = out[["op_setting_1", "op_setting_2", "op_setting_3"]].to_numpy(dtype=float)
    labels = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(ops)
    # KMeans label order is arbitrary, so align it to the training fingerprint
    # recoverable from the scaler: FD002/FD004 were trained with one condition
    # at ~25% of rows and the other five at ~15% each, and the test split
    # reproduces that split to within 0.3%. Mapping the largest cluster to
    # index 0 therefore pins the one distinguishable column; the remaining five
    # differ by <0.03 in scaled value and are effectively interchangeable.
    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    out["op_cluster"] = remap[labels]
    return out


def nasa_score(pred: np.ndarray, true: np.ndarray) -> float:
    """C-MAPSS asymmetric score. Late predictions (pred > true) cost more."""
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def evaluate(fd: str, mode: str, loader: ArtifactLoader, settings: Any,
             data_root: Path) -> Optional[Dict[str, Any]]:
    from app.services.Predictive_Maintenance.preprocessing import PreprocessingService

    test, truth = load_subset(fd, data_root)
    artifacts = loader.load("rul", fd)
    window = settings.RUL_WINDOW_SIZES.get(fd, 50)
    q_hat = settings.DEFAULT_RUL_Q_HAT.get(fd, 20.0)
    pre = PreprocessingService()

    test = add_op_cluster(test, fd, mode)

    tensors: List[np.ndarray] = []
    trues: List[float] = []
    skipped = 0
    for unit, grp in test.groupby("unit"):
        grp = grp.sort_values("cycle")
        if len(grp) < window:
            skipped += 1
            continue
        aligned = pre._align_features(grp, artifacts.feature_columns)
        scaled = artifacts.scaler.transform(aligned)
        tensors.append(scaled[-window:])
        trues.append(float(truth[int(unit) - 1]))

    if not tensors:
        return None

    X = np.asarray(tensors, dtype=np.float32)
    pred = np.maximum(0.0, np.asarray(
        artifacts.model.predict(X, verbose=0)).reshape(-1)[: len(trues)])
    true = np.asarray(trues, dtype=float)

    lower = np.maximum(0.0, pred - q_hat)
    upper = pred + q_hat
    err = pred - true
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((true - true.mean()) ** 2))

    return {
        "name": f"TCN_Conformal {fd}",
        "fd_subset": fd,
        "engines_scored": int(len(true)),
        "engines_skipped_short": int(skipped),
        "window_size": int(window),
        "q_hat": float(q_hat),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
        "nasa_score": round(nasa_score(pred, true), 2),
        "picp": round(float(np.mean((true >= lower) & (true <= upper))), 4),
        "mpiw": round(float(np.mean(upper - lower)), 4),
        "mean_predicted_rul": round(float(pred.mean()), 4),
        "mean_true_rul": round(float(true.mean()), 4),
        "op_cluster_mode": mode,
        "_pred": pred.tolist(),
        "_true": true.tolist(),
    }


def pool(rows: List[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    """Pool per-subset results into one fleet-wide figure."""
    pred = np.concatenate([np.asarray(r["_pred"]) for r in rows])
    true = np.concatenate([np.asarray(r["_true"]) for r in rows])
    lower = np.concatenate(
        [np.maximum(0.0, np.asarray(r["_pred"]) - r["q_hat"]) for r in rows])
    upper = np.concatenate(
        [np.asarray(r["_pred"]) + r["q_hat"] for r in rows])
    err = pred - true
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {
        "name": "TCN_Conformal (all subsets)",
        "fd_subset": "ALL",
        "engines_scored": int(len(true)),
        "mae": round(float(np.mean(np.abs(err))), 4),
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "r2": round(1.0 - float(np.sum(err ** 2)) / ss_tot, 4) if ss_tot > 0 else None,
        "nasa_score": round(nasa_score(pred, true), 2),
        "picp": round(float(np.mean((true >= lower) & (true <= upper))), 4),
        "mpiw": round(float(np.mean(upper - lower)), 4),
        "mean_predicted_rul": round(float(pred.mean()), 4),
        "mean_true_rul": round(float(true.mean()), 4),
        "op_cluster_mode": mode,
    }


def run(mode: str, data_root: Path) -> Dict[str, Any]:
    settings = get_settings()
    loader = ArtifactLoader(settings)
    rows: List[Dict[str, Any]] = []
    for fd in settings.SUPPORTED_FDS:
        print(f"  scoring {fd} (op_cluster={mode}) ...", flush=True)
        r = evaluate(fd, mode, loader, settings, data_root)
        if r:
            rows.append(r)
            print(
                f"    engines={r['engines_scored']:>3}  MAE={r['mae']:>8.3f}  "
                f"RMSE={r['rmse']:>8.3f}  PICP={r['picp']:.3f}  MPIW={r['mpiw']:.1f}"
            )
    overall = pool(rows, mode)
    for r in rows:
        r.pop("_pred", None)
        r.pop("_true", None)
    return {"overall": overall, "per_subset": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--op-cluster", choices=["zero", "derived", "both"], default="zero")
    ap.add_argument("--data-root", default=str(_BACKEND_ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(OUT_PATH))
    args = ap.parse_args()
    data_root = Path(args.data_root)

    modes = ["zero", "derived"] if args.op_cluster == "both" else [args.op_cluster]
    results = {m: run(m, data_root) for m in modes}

    # models[0] is what the dashboard reads, so the deployed ("zero") pipeline
    # leads -- the UI must report what the API actually returns.
    primary = results.get("zero") or results[modes[0]]
    models = [primary["overall"]] + primary["per_subset"]
    if "derived" in results and "zero" in results:
        d = results["derived"]["overall"]
        d["name"] = "TCN_Conformal (all subsets, op_cluster reconstructed)"
        models.append(d)

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "scripts/Predictive_Maintenance/compute_evaluation_metrics.py",
        "protocol": (
            "C-MAPSS held-out: final observed window per test engine, compared "
            "against RUL_FDxxx.txt ground truth."
        ),
        "op_cluster_mode": primary["overall"]["op_cluster_mode"],
        "notes": [
            "picp/mpiw describe the fixed-width interval predicted_rul +/- q_hat, "
            "with the lower bound clamped at zero.",
            "q_hat is a per-subset constant from Settings.DEFAULT_RUL_Q_HAT; it is "
            "not calibrated per prediction, so picp is the honest check on it.",
            "op_cluster one-hots are zero-filled by the serving pipeline, so these "
            "figures measure the deployed path rather than the training-time one.",
            "Reconstructing those one-hots was measured and rejected: it is worse on "
            "FD002/FD003/FD004 and within noise on FD001. See the module docstring.",
        ],
        "models": models,
    }
    if "derived" in results:
        payload["comparison_op_cluster_derived"] = results["derived"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    o = models[0]
    print(f"  overall: MAE={o['mae']} RMSE={o['rmse']} R2={o['r2']} "
          f"NASA={o['nasa_score']} PICP={o['picp']} MPIW={o['mpiw']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
