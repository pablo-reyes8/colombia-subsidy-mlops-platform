from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


def make_classifier(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "random_forest":
        return RandomForestClassifier(**params)
    if name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed")
        return XGBClassifier(**params)
    raise ValueError(f"Unsupported classifier: {name}")


def make_anomaly_model(name: str, params: Dict[str, Any]):
    name = name.lower()
    if name == "isolation_forest":
        return IsolationForest(**params)
    if name == "one_class_svm":
        return OneClassSVM(**params)
    raise ValueError(f"Unsupported anomaly model: {name}")
