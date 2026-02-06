from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class CascadeConfig:
    threshold_stage1: float = 0.5
    threshold_stage2: float = 0.5


class CascadeClassifier:
    """Two-stage cascade classifier using predict_proba outputs."""

    def __init__(self, model_stage1, model_stage2, *, threshold_stage1: float = 0.5, threshold_stage2: float = 0.5):
        self.model_stage1 = model_stage1
        self.model_stage2 = model_stage2
        self.threshold_stage1 = threshold_stage1
        self.threshold_stage2 = threshold_stage2

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model_stage1.fit(X, y)
        proba1 = self.model_stage1.predict_proba(X)[:, 1]
        X2 = np.hstack([X, proba1.reshape(-1, 1)])
        self.model_stage2.fit(X2, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba1 = self.model_stage1.predict_proba(X)[:, 1]
        mask1 = proba1 >= self.threshold_stage1
        X2 = np.hstack([X, proba1.reshape(-1, 1)])
        proba2 = np.zeros_like(proba1)
        if mask1.any():
            proba2[mask1] = self.model_stage2.predict_proba(X2[mask1])[:, 1]
        final = np.vstack([1 - proba2, proba2]).T
        return final

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold_stage2).astype(int)
