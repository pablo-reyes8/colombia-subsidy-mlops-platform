from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(y_true, y_pred, *, y_proba: Optional[np.ndarray] = None) -> Dict[str, Any]:
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    out: Dict[str, Any] = {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm,
        "report": report,
    }

    if y_proba is not None:
        y_proba = np.asarray(y_proba)
        if y_proba.ndim != 1:
            raise ValueError("y_proba must be a 1-D probability array for positive class")

        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = None

        try:
            out["average_precision"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            out["average_precision"] = None

    return out


def compute_anomaly_metrics(y_true, y_pred_anom, *, y_score: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Assumes y_pred_anom is 1 for predicted anomaly (subsidy) and 0 for normal."""
    return compute_binary_metrics(y_true, y_pred_anom, y_proba=y_score)
