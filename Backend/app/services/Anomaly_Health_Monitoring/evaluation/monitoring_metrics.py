"""Auditable monitoring metrics; RUL condition bands are explicitly proxies.

Intervals resample whole engines, preserving serial dependence within each
trajectory. No threshold is fitted by these evaluation functions.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def engine_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in ("subset_id", "split", "unit") if c in frame]


def boolean_values(series: pd.Series) -> np.ndarray:
    """Parse CSV flags without interpreting the string 'False' as true."""
    converted = series.map({True: True, False: False, 1: True, 0: False,
                            "True": True, "False": False, "true": True, "false": False})
    if converted.isna().any():
        raise ValueError(f"{series.name} must contain explicit non-missing boolean flags")
    return converted.to_numpy(bool)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, flags: np.ndarray) -> dict[str, Any]:
    y, p = np.asarray(labels, dtype=bool), np.asarray(flags, dtype=bool)
    tp, fp = int((y & p).sum()), int((~y & p).sum())
    fn, tn = int((y & ~p).sum()), int((~y & ~p).sum())
    ratio = lambda a, b: float(a / b) if b else None
    return {"rows": len(y), "true_positive": tp, "false_positive": fp,
            "false_negative": fn, "true_negative": tn,
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "false_alarm_rate": ratio(fp, fp + tn),
            "auc": float(roc_auc_score(y, scores)) if y.any() and (~y).any() else None,
            "average_precision": float(average_precision_score(y, scores)) if y.any() and (~y).any() else None}


def engine_intervals(frame: pd.DataFrame, *, samples: int = 300, seed: int = 20261011) -> dict[str, Any]:
    """Percentile intervals from engine-level confusion-count bootstrap."""
    counts = frame.groupby(engine_columns(frame), sort=True)[["tp", "fp", "fn", "tn"]].sum().to_numpy(float)
    if len(counts) < 2:
        return {"available": False, "reason": "At least two engines are required."}
    rng = np.random.default_rng(seed)
    sums = counts[rng.integers(0, len(counts), size=(samples, len(counts)))].sum(axis=1)
    tp, fp, fn, tn = sums.T
    result: dict[str, Any] = {"available": True, "method": "whole-engine percentile bootstrap",
                              "samples": samples, "seed": seed, "confidence": 0.95}
    for name, numerator, denominator in (("precision", tp, tp + fp), ("recall", tp, tp + fn),
                                         ("f1", 2 * tp, 2 * tp + fp + fn),
                                         ("false_alarm_rate", fp, fp + tn)):
        values = np.divide(numerator, denominator, out=np.full(samples, np.nan), where=denominator > 0)
        valid = values[np.isfinite(values)]
        result[name] = np.quantile(valid, [.025, .975]).tolist() if len(valid) else None
    return result


def condition_metrics(frame: pd.DataFrame, healthy_rul: int = 125, critical_rul: int = 30,
                      *, bootstrap: bool = True, sustained_cycles: int = 3) -> dict[str, Any]:
    if not 0 <= critical_rul < healthy_rul:
        raise ValueError("Condition bands require 0 <= critical_rul < healthy_rul")
    required = {"unit", "cycle", "RUL", "anomaly_score", "anomaly_flag"}
    missing = sorted(required - set(frame.columns))
    if missing:
        return {"available": False, "missing_columns": missing}
    data = frame.copy()
    valid = np.isfinite(data[["RUL", "anomaly_score", "cycle"]].to_numpy(float)).all(axis=1)
    valid &= data.anomaly_score.between(0, 1).to_numpy()
    ready = boolean_values(data.calibration_ready) if "calibration_ready" in data else np.ones(len(data), bool)
    data = data.loc[valid & ready].copy()
    score = data.anomaly_score.to_numpy(float)
    raw_flag = boolean_values(data.anomaly_flag)
    flag = data.anomaly_type.eq("fault").to_numpy() if "anomaly_type" in data else raw_flag
    rul = data.RUL.to_numpy(float)
    healthy, critical = rul >= healthy_rul, rul <= critical_rul
    bands = healthy | critical
    report: dict[str, Any] = {
        "available": bool(len(data)), "label_basis": "RUL condition proxies; not confirmed fault labels",
        "healthy_rul_min": healthy_rul, "critical_rul_max": critical_rul,
        "flag_policy": "escalated fault" if "anomaly_type" in data else "raw detector flag",
        "input_rows": len(frame), "evaluated_rows": len(data),
        "input_engines": int(frame.groupby(engine_columns(frame)).ngroups),
        "evaluated_engines": int(data.groupby(engine_columns(data)).ngroups),
        "engines_without_ready_prediction": int(frame.groupby(engine_columns(frame)).ngroups - data.groupby(engine_columns(data)).ngroups),
        "warmup_or_unready_rows_excluded": int((~ready).sum()),
        "invalid_rows": int((~valid).sum()), "middle_band_rows_excluded_from_binary_metrics": int((~bands).sum()),
        "healthy_rows": int(healthy.sum()), "critical_rows": int(critical.sum()),
        **binary_metrics(critical[bands], score[bands], flag[bands]),
    }
    data["_flag"] = flag
    data["tp"], data["fp"] = critical & flag, healthy & flag
    data["fn"], data["tn"] = critical & ~flag, healthy & ~flag
    reaching = detected = healthy_engines = false_alarm_engines = episodes = 0
    leads: list[float] = []
    sustained_leads: list[float] = []
    for _, group in data.sort_values(engine_columns(data) + ["cycle"]).groupby(engine_columns(data), sort=True):
        flags = group._flag.to_numpy(bool)
        cycles = group.cycle.to_numpy(float)
        remaining = group.RUL.to_numpy(float)
        in_critical = remaining <= critical_rul
        in_healthy = remaining >= healthy_rul
        reaching += bool(in_critical.any())
        detected += bool((flags & in_critical).any())
        healthy_engines += bool(in_healthy.any())
        false_alarm_engines += bool((flags & in_healthy).any())
        if flags.any():
            leads.append(float(remaining[np.flatnonzero(flags)[0]]))
        run = 0
        confirmed = False
        for i, raised in enumerate(flags):
            contiguous = i > 0 and cycles[i] - cycles[i - 1] == 1
            if raised and in_healthy[i] and (not contiguous or not flags[i - 1]):
                episodes += 1
            run = (run + 1 if contiguous else 1) if raised else 0
            if run >= sustained_cycles and not confirmed:
                sustained_leads.append(float(remaining[i]))
                confirmed = True
    report["engine_events"] = {
        "engines": int(data.groupby(engine_columns(data)).ngroups),
        "engines_reaching_critical": reaching, "critical_engines_detected": detected,
        "detection_rate_of_reaching": detected / reaching if reaching else None,
        "engines_with_healthy_observations": healthy_engines,
        "healthy_engines_with_false_alert": false_alarm_engines,
        "healthy_engine_false_alert_rate": false_alarm_engines / healthy_engines if healthy_engines else None,
        "false_alert_episodes": episodes,
        "false_alert_episodes_per_1000_healthy_cycles": episodes * 1000 / healthy.sum() if healthy.any() else None,
        "first_alert_rul_median": float(np.median(leads)) if leads else None,
        "sustained_alert_required_cycles": sustained_cycles,
        "engines_with_sustained_alert": len(sustained_leads),
        "first_sustained_alert_rul_median": float(np.median(sustained_leads)) if sustained_leads else None,
        "lead_time_note": "RUL at the alert, including early false alerts; not proof of correct fault onset detection.",
    }
    if bootstrap and len(data):
        report["confidence_intervals"] = engine_intervals(data)
    if "fault_label" in data:
        observed = data.fault_label.notna().to_numpy()
        labels = boolean_values(data.loc[observed, "fault_label"])
        report["confirmed_label_metrics"] = binary_metrics(labels, score[observed], flag[observed])
        report["confirmed_label_metrics"]["label_basis"] = "Caller-supplied fault_label; label quality requires independent verification."
    else:
        report["confirmed_label_metrics"] = {"available": False, "reason": "No independently verified fault_label column supplied."}
    return report
