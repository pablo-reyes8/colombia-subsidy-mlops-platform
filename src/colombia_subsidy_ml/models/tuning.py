from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import ParameterSampler, StratifiedKFold

from colombia_subsidy_ml.models.factory import make_classifier
from colombia_subsidy_ml.utils.metrics import compute_binary_metrics


@dataclass
class SearchResult:
    model: Any
    params: Dict[str, Any]
    metrics: Dict[str, Any]


def sample_param_sets(
    base_params: Dict[str, Any],
    param_distributions: Optional[Dict[str, List[Any]]],
    *,
    n_iter: int,
    random_state: int,
) -> List[Dict[str, Any]]:
    if not param_distributions:
        return [dict(base_params)]

    sampled = list(
        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=max(1, n_iter),
            random_state=random_state,
        )
    )
    return [{**base_params, **params} for params in sampled]


def _score_tuple(metrics: Dict[str, Any], objective_metric: str, recall_min: float) -> Optional[Tuple[float, float, float]]:
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    f1 = float(metrics.get("f1", 0.0))

    if recall < recall_min:
        return None

    objective_metric = objective_metric.lower()
    if objective_metric == "precision":
        return precision, recall, f1
    if objective_metric == "f1":
        return f1, recall, precision
    return recall, precision, f1


def _mean_metric_dict(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    keys = ["precision", "recall", "f1", "roc_auc", "average_precision"]
    out: Dict[str, float] = {}
    for key in keys:
        vals = [m[key] for m in metrics_list if key in m and m[key] is not None]
        if vals:
            out[key] = float(np.mean(vals))
    return out


def _stage1_cv_metrics(
    *,
    model_name: str,
    params: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    threshold: float,
    cv_folds: int,
    random_state: int,
) -> Optional[Dict[str, float]]:
    y_train = np.asarray(y_train)
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    max_folds = int(class_counts.min())
    effective_folds = min(int(cv_folds), max_folds)
    if effective_folds < 2:
        return None

    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=random_state)
    fold_metrics: List[Dict[str, Any]] = []

    for tr_idx, va_idx in splitter.split(X_train, y_train):
        model = make_classifier(model_name, params)
        model.fit(X_train[tr_idx], y_train[tr_idx])

        y_proba = model.predict_proba(X_train[va_idx])[:, 1]
        y_pred = (y_proba >= threshold).astype(int)
        fold_metrics.append(compute_binary_metrics(y_train[va_idx], y_pred, y_proba=y_proba))

    return _mean_metric_dict(fold_metrics)


def tune_stage1_classifier(
    *,
    model_name: str,
    base_params: Dict[str, Any],
    param_distributions: Optional[Dict[str, List[Any]]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    stage_threshold: float,
    objective_metric: str,
    recall_min: float,
    n_iter: int,
    random_state: int,
    cv_folds: int = 1,
) -> SearchResult:
    candidates = sample_param_sets(
        base_params,
        param_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )

    best_score = None
    best_params: Optional[Dict[str, Any]] = None
    best_cv_metrics: Optional[Dict[str, float]] = None

    for params in candidates:
        cv_metrics = None
        if int(cv_folds) > 1:
            cv_metrics = _stage1_cv_metrics(
                model_name=model_name,
                params=params,
                X_train=X_train,
                y_train=y_train,
                threshold=stage_threshold,
                cv_folds=cv_folds,
                random_state=random_state,
            )

        if cv_metrics is not None:
            score = _score_tuple(cv_metrics, objective_metric, recall_min)
        else:
            model = make_classifier(model_name, params)
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_val)[:, 1]
            y_pred = (y_proba >= stage_threshold).astype(int)
            metrics = compute_binary_metrics(y_val, y_pred, y_proba=y_proba)
            score = _score_tuple(metrics, objective_metric, recall_min)

        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_params = params
            best_cv_metrics = cv_metrics

    if best_params is None:
        best_params = dict(base_params)

    best_model = make_classifier(model_name, best_params)
    best_model.fit(X_train, y_train)
    holdout_proba = best_model.predict_proba(X_val)[:, 1]
    holdout_pred = (holdout_proba >= stage_threshold).astype(int)
    holdout_metrics = compute_binary_metrics(y_val, holdout_pred, y_proba=holdout_proba)

    if best_cv_metrics is not None:
        holdout_metrics["cv_mean"] = best_cv_metrics

    return SearchResult(model=best_model, params=best_params, metrics=holdout_metrics)


def tune_stage2_classifier(
    *,
    model_name: str,
    base_params: Dict[str, Any],
    param_distributions: Optional[Dict[str, List[Any]]],
    stage1_model: Any,
    threshold_stage1: float,
    threshold_stage2: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    objective_metric: str,
    recall_min: float,
    n_iter: int,
    random_state: int,
) -> SearchResult:
    proba1_train = stage1_model.predict_proba(X_train)[:, 1]
    X2_train = np.hstack([X_train, proba1_train.reshape(-1, 1)])

    proba1_val = stage1_model.predict_proba(X_val)[:, 1]
    X2_val = np.hstack([X_val, proba1_val.reshape(-1, 1)])

    candidates = sample_param_sets(
        base_params,
        param_distributions,
        n_iter=n_iter,
        random_state=random_state,
    )

    best_score = None
    best_result = None

    for params in candidates:
        model = make_classifier(model_name, params)
        model.fit(X2_train, y_train)

        mask1 = proba1_val >= threshold_stage1
        proba2 = np.zeros_like(proba1_val)
        if mask1.any():
            proba2[mask1] = model.predict_proba(X2_val[mask1])[:, 1]

        y_pred = (proba2 >= threshold_stage2).astype(int)
        metrics = compute_binary_metrics(y_val, y_pred, y_proba=proba2)

        score = _score_tuple(metrics, objective_metric, recall_min)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_result = SearchResult(model=model, params=params, metrics=metrics)

    if best_result is None:
        fallback_params = dict(base_params)
        fallback_model = make_classifier(model_name, fallback_params)
        fallback_model.fit(X2_train, y_train)

        mask1 = proba1_val >= threshold_stage1
        fallback_proba2 = np.zeros_like(proba1_val)
        if mask1.any():
            fallback_proba2[mask1] = fallback_model.predict_proba(X2_val[mask1])[:, 1]

        fallback_pred = (fallback_proba2 >= threshold_stage2).astype(int)
        fallback_metrics = compute_binary_metrics(y_val, fallback_pred, y_proba=fallback_proba2)
        best_result = SearchResult(model=fallback_model, params=fallback_params, metrics=fallback_metrics)

    return best_result
