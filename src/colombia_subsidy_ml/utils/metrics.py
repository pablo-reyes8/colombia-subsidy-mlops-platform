from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


def compute_binary_metrics(y_true, y_pred) -> Dict[str, Any]:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
    }


def compute_anomaly_metrics(y_true, y_pred_anom) -> Dict[str, Any]:
    """Assumes y_pred_anom is 1 for predicted anomaly (subsidy) and 0 for normal."""
    return compute_binary_metrics(y_true, y_pred_anom)
