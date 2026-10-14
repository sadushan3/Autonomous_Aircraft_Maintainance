"""
maintenance_metrics.py

Custom Domain-Specific Predictive Maintenance Metrics for the
Autonomous Maintenance Supervisor Agent.

Purpose
-------
Calculates research-grade domain metrics for evaluating predictive maintenance supervisor decisions:
1. False Critical Rate (FCR): Unnecessary immediate shutdowns on healthy equipment (False Positives on critical).
2. Missed Critical Rate (MCR): Failing to order immediate maintenance when equipment is about to fail (Unsafe False Negatives).
3. Cost Penalty Matrix: Weighted economic loss penalizing missed critical events much more heavily than false alarms.
4. Standard Classification Metrics: Accuracy, Macro Precision, Recall, Macro F1, Log Loss.
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

# Bootstrap
_CURRENT_FILE: Final[Path] = Path(__file__).resolve()
_BACKEND_ROOT: Final[Path] = _CURRENT_FILE.parents[4]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from app.config.Maintenance_Supervisor.supervisor_config import DECISION_CLASSES, DECISION_TO_SEVERITY
from app.utils.Maintenance_Supervisor.logger import get_logger

logger = get_logger()

# Asymmetric Economic Cost Matrix (Row=True, Col=Pred)
# Missing an immediate maintenance event is 100x worse than false alarm
COST_MATRIX: Final[dict[tuple[int, int], float]] = {
    (4, 0): 100.0, (4, 1): 50.0, (4, 2): 20.0, (4, 3): 5.0, (4, 4): 0.0,
    (3, 0): 30.0,  (3, 1): 15.0, (3, 2): 5.0,  (3, 3): 0.0, (3, 4): 2.0,
    (2, 0): 10.0,  (2, 1): 5.0,  (2, 2): 0.0,  (2, 3): 2.0, (2, 4): 5.0,
    (1, 0): 2.0,   (1, 1): 0.0,  (1, 2): 1.0,  (1, 3): 3.0, (1, 4): 5.0,
    (0, 0): 0.0,   (0, 1): 1.0,  (0, 2): 2.0,  (0, 3): 5.0, (0, 4): 10.0,
}


@dataclass
class MaintenanceMetricsResult:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    false_critical_rate: float
    missed_critical_rate: float
    total_cost_penalty: float
    per_class_f1: dict[str, float]
    confusion_mat: list[list[int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "false_critical_rate": round(self.false_critical_rate, 4),
            "missed_critical_rate": round(self.missed_critical_rate, 4),
            "total_cost_penalty": round(self.total_cost_penalty, 2),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "confusion_matrix": self.confusion_mat,
        }


def compute_maintenance_metrics(
    y_true: pd.Series | np.ndarray | list[str],
    y_pred: pd.Series | np.ndarray | list[str],
) -> MaintenanceMetricsResult:
    y_true_clean = np.array([DECISION_TO_SEVERITY.get(str(v).strip().lower(), 0) for v in y_true])
    y_pred_clean = np.array([DECISION_TO_SEVERITY.get(str(v).strip().lower(), 0) for v in y_pred])

    acc = float(accuracy_score(y_true_clean, y_pred_clean))
    prec = float(precision_score(y_true_clean, y_pred_clean, average="macro", zero_division=0))
    rec = float(recall_score(y_true_clean, y_pred_clean, average="macro", zero_division=0))
    f1 = float(f1_score(y_true_clean, y_pred_clean, average="macro", zero_division=0))

    class_f1_arr = f1_score(y_true_clean, y_pred_clean, average=None, labels=list(range(len(DECISION_CLASSES))), zero_division=0)
    per_class_f1 = {cls_name: float(score) for cls_name, score in zip(DECISION_CLASSES, class_f1_arr)}

    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=list(range(len(DECISION_CLASSES))))

    # False Critical Rate: Pred == Immediate (4) when True != Immediate (4)
    true_non_crit = (y_true_clean != 4)
    pred_crit = (y_pred_clean == 4)
    fcr = float(np.sum(true_non_crit & pred_crit) / np.sum(true_non_crit)) if np.sum(true_non_crit) > 0 else 0.0

    # Missed Critical Rate: True == Immediate (4) when Pred != Immediate (4)
    true_crit = (y_true_clean == 4)
    pred_non_crit = (y_pred_clean != 4)
    mcr = float(np.sum(true_crit & pred_non_crit) / np.sum(true_crit)) if np.sum(true_crit) > 0 else 0.0

    # Total Cost Penalty
    total_cost = 0.0
    for t_val, p_val in zip(y_true_clean, y_pred_clean):
        total_cost += COST_MATRIX.get((t_val, p_val), 0.0)

    return MaintenanceMetricsResult(
        accuracy=acc,
        macro_precision=prec,
        macro_recall=rec,
        macro_f1=f1,
        false_critical_rate=fcr,
        missed_critical_rate=mcr,
        total_cost_penalty=total_cost,
        per_class_f1=per_class_f1,
        confusion_mat=cm.tolist(),
    )
