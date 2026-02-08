from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.model_selection import ParameterSampler, train_test_split

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import build_feature_pipeline, prepare_xy, split_features
from colombia_subsidy_ml.models.factory import make_anomaly_model
from colombia_subsidy_ml.tracking.mlflow_utils import log_artifacts, log_metrics, log_params, start_mlflow_run
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_anomaly_metrics


def _split_indices(y, *, test_size: float, val_size: float, random_state: int, stratify: bool):
    idx = np.arange(len(y))
    strat = y if stratify else None
    train_val_idx, test_idx = train_test_split(
        idx,
        test_size=test_size,
        random_state=random_state,
        stratify=strat,
    )
    strat2 = y[train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        random_state=random_state,
        stratify=strat2,
    )
    return train_idx, val_idx, test_idx


def _anomaly_outputs(
    model,
    X: np.ndarray,
    *,
    score_threshold: Optional[float] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    y_score = None
    if hasattr(model, "decision_function"):
        try:
            # Higher score should represent "more anomalous" for metric curves.
            y_score = -np.asarray(model.decision_function(X))
        except Exception:
            y_score = None

    if score_threshold is not None and y_score is not None:
        y_pred_anomaly = (y_score >= score_threshold).astype(int)
        return y_pred_anomaly, y_score

    raw_pred = model.predict(X)
    y_pred_anomaly = (raw_pred == -1).astype(int)
    return y_pred_anomaly, y_score


def _score_tuple(metrics: Dict[str, float], objective_metric: str, recall_min: float):
    recall = float(metrics.get("recall", 0.0))
    precision = float(metrics.get("precision", 0.0))
    f1 = float(metrics.get("f1", 0.0))

    if recall < recall_min:
        return None

    objective_metric = objective_metric.lower()
    if objective_metric == "precision":
        return precision, recall, f1
    if objective_metric == "recall":
        return recall, precision, f1
    return f1, recall, precision


def _find_best_score_threshold(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    objective_metric: str,
    recall_min: float,
    grid_size: int,
) -> Tuple[float, Dict[str, float]]:
    if y_score.size == 0:
        raise ValueError("y_score cannot be empty for threshold tuning")

    quantiles = np.linspace(0.01, 0.99, max(10, int(grid_size)))
    thresholds = np.unique(np.quantile(y_score, quantiles))

    best_threshold = float(np.median(y_score))
    best_metrics = compute_anomaly_metrics(y_true, (y_score >= best_threshold).astype(int), y_score=y_score)
    best_score = _score_tuple(best_metrics, objective_metric, recall_min)

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        metrics = compute_anomaly_metrics(y_true, y_pred, y_score=y_score)
        score = _score_tuple(metrics, objective_metric, recall_min)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def _search_anomaly_params(
    *,
    model_type: str,
    base_params: Dict[str, object],
    param_distributions: Optional[Dict[str, List[object]]],
    n_iter: int,
    random_state: int,
    objective_metric: str,
    recall_min: float,
    X_train_norm: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    tune_score_threshold: bool,
    threshold_grid_size: int,
):
    candidates = [dict(base_params)]
    if param_distributions:
        sampled = list(
            ParameterSampler(
                param_distributions=param_distributions,
                n_iter=max(1, n_iter),
                random_state=random_state,
            )
        )
        candidates = [{**base_params, **params} for params in sampled]

    best = None
    best_score = None

    for params in candidates:
        model = make_anomaly_model(model_type, params)
        model.fit(X_train_norm)

        y_val_pred, y_val_score = _anomaly_outputs(model, X_val)
        score_threshold = None
        metrics = compute_anomaly_metrics(y_val, y_val_pred, y_score=y_val_score)

        if tune_score_threshold and y_val_score is not None:
            score_threshold, metrics = _find_best_score_threshold(
                y_true=y_val,
                y_score=y_val_score,
                objective_metric=objective_metric,
                recall_min=recall_min,
                grid_size=threshold_grid_size,
            )

        score = _score_tuple(metrics, objective_metric, recall_min)
        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best = {
                "params": params,
                "metrics": metrics,
                "score_threshold": score_threshold,
            }

    if best is None:
        model = make_anomaly_model(model_type, base_params)
        model.fit(X_train_norm)
        y_val_pred, y_val_score = _anomaly_outputs(model, X_val)
        score_threshold = None
        metrics = compute_anomaly_metrics(y_val, y_val_pred, y_score=y_val_score)

        if tune_score_threshold and y_val_score is not None:
            score_threshold, metrics = _find_best_score_threshold(
                y_true=y_val,
                y_score=y_val_score,
                objective_metric=objective_metric,
                recall_min=recall_min,
                grid_size=threshold_grid_size,
            )

        best = {
            "params": dict(base_params),
            "metrics": metrics,
            "score_threshold": score_threshold,
        }

    return best


def train_anomaly(config_path: Union[str, Path]):
    config = load_yaml(config_path)
    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = config.get("target_col", "Subsidio")
    X, y = prepare_xy(df, target_col)

    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.2))
    val_size = float(split_cfg.get("val_size", 0.2))
    random_state = int(split_cfg.get("random_state", 13))
    stratify = bool(split_cfg.get("stratify", True))

    train_idx, val_idx, test_idx = _split_indices(
        y.values,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=stratify,
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    spec = split_features(X_train)
    feature_cfg = config.get("feature_engineering") or config.get("preprocessing", {})
    preprocessor = build_feature_pipeline(spec, config=feature_cfg)

    X_train_pp = to_dense(preprocessor.fit_transform(X_train))
    X_val_pp = to_dense(preprocessor.transform(X_val))
    X_test_pp = to_dense(preprocessor.transform(X_test))

    model_cfg = config["model"]
    model_type = str(model_cfg["type"])
    base_params = dict(model_cfg.get("params", {}))

    normal_mask_train = (y_train.values == 0)
    X_train_norm = X_train_pp[normal_mask_train]

    search_cfg = config.get("search", {})
    search_enabled = bool(search_cfg.get("enabled", False))

    mlflow_cfg = config.get("mlflow", {})
    run_name = str(config.get("run_name", "anomaly_train"))

    with start_mlflow_run(mlflow_cfg, run_name=run_name) as mlflow_enabled:
        if search_enabled:
            best = _search_anomaly_params(
                model_type=model_type,
                base_params=base_params,
                param_distributions=search_cfg.get("param_distributions"),
                n_iter=int(search_cfg.get("n_iter", 20)),
                random_state=int(search_cfg.get("random_state", random_state)),
                objective_metric=str(search_cfg.get("objective_metric", "f1")),
                recall_min=float(search_cfg.get("recall_min", 0.0)),
                X_train_norm=X_train_norm,
                X_val=X_val_pp,
                y_val=y_val.values,
                tune_score_threshold=bool(search_cfg.get("tune_score_threshold", True)),
                threshold_grid_size=int(search_cfg.get("threshold_grid_size", 40)),
            )
            selected_params = dict(best["params"])
            val_metrics = best["metrics"]
            selected_score_threshold = best.get("score_threshold")
        else:
            model = make_anomaly_model(model_type, base_params)
            model.fit(X_train_norm)
            y_val_pred, y_val_score = _anomaly_outputs(model, X_val_pp)
            selected_params = dict(base_params)
            val_metrics = compute_anomaly_metrics(y_val.values, y_val_pred, y_score=y_val_score)
            selected_score_threshold = None

        # Refit with train+val (normal class only) before final test evaluation.
        X_trainval_pp = np.vstack([X_train_pp, X_val_pp])
        y_trainval = np.concatenate([y_train.values, y_val.values])
        normal_mask_trainval = (y_trainval == 0)
        X_trainval_norm = X_trainval_pp[normal_mask_trainval]

        model = make_anomaly_model(model_type, selected_params)
        model.fit(X_trainval_norm)

        y_test_pred, y_test_score = _anomaly_outputs(
            model,
            X_test_pp,
            score_threshold=selected_score_threshold,
        )
        test_metrics = compute_anomaly_metrics(y_test.values, y_test_pred, y_score=y_test_score)

        metrics = {
            "validation": val_metrics,
            "test": test_metrics,
            "selected_params": selected_params,
            "selected_score_threshold": selected_score_threshold,
        }

        artifacts_cfg = config.get("artifacts", {})
        artifacts_dir = Path(artifacts_cfg.get("dir", "artifacts/anomaly"))
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
        joblib.dump(model, artifacts_dir / "model.joblib")
        np.savez(artifacts_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

        metadata = {
            "target_col": target_col,
            "feature_columns": list(X.columns),
            "config_path": str(config_path),
            "model_type": model_type,
            "selected_params": selected_params,
            "selected_score_threshold": selected_score_threshold,
            "feature_engineering": feature_cfg,
            "model_version": str(config.get("model_version", "v1")),
        }
        (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        log_params(
            mlflow_enabled,
            {
                "pipeline": "anomaly",
                "config_path": str(config_path),
                "model_type": model_type,
                "search.enabled": search_enabled,
                "selected_params": selected_params,
                "selected_score_threshold": selected_score_threshold,
            },
        )
        log_metrics(mlflow_enabled, val_metrics, prefix="val_")
        log_metrics(mlflow_enabled, test_metrics, prefix="test_")
        log_artifacts(mlflow_enabled, artifacts_dir, artifact_path="anomaly_artifacts")

    return {
        "artifacts_dir": artifacts_dir,
        "metrics": metrics,
        "metadata": metadata,
    }
