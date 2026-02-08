from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import build_feature_pipeline, prepare_xy, split_features
from colombia_subsidy_ml.models.cascade import CascadeClassifier
from colombia_subsidy_ml.models.factory import make_classifier
from colombia_subsidy_ml.models.io import save_cascade_artifacts
from colombia_subsidy_ml.models.tuning import tune_stage1_classifier, tune_stage2_classifier
from colombia_subsidy_ml.tracking.mlflow_utils import log_artifacts, log_metrics, log_params, start_mlflow_run
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_binary_metrics


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


def _tune_thresholds(
    model: CascadeClassifier,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    recall_min: float,
    grid: np.ndarray,
) -> Tuple[float, float, Dict[str, float]]:
    best = {
        "precision": -1.0,
        "recall": 0.0,
        "f1": 0.0,
        "thr1": model.threshold_stage1,
        "thr2": model.threshold_stage2,
    }

    proba1 = model.model_stage1.predict_proba(X_val)[:, 1]
    X2 = np.hstack([X_val, proba1.reshape(-1, 1)])

    for thr1 in grid:
        mask1 = proba1 >= thr1
        if not mask1.any():
            continue

        proba2 = np.zeros_like(proba1)
        proba2[mask1] = model.model_stage2.predict_proba(X2[mask1])[:, 1]

        for thr2 in grid:
            y_pred = (proba2 >= thr2).astype(int)
            metrics = compute_binary_metrics(y_val, y_pred, y_proba=proba2)
            if metrics["recall"] < recall_min:
                continue
            if metrics["precision"] > best["precision"]:
                best = {
                    "precision": float(metrics["precision"]),
                    "recall": float(metrics["recall"]),
                    "f1": float(metrics["f1"]),
                    "thr1": float(thr1),
                    "thr2": float(thr2),
                }

    return best["thr1"], best["thr2"], best


def _fit_cascade_with_params(
    *,
    stage1_name: str,
    stage1_params: Dict[str, object],
    stage2_name: str,
    stage2_params: Dict[str, object],
    threshold_stage1: float,
    threshold_stage2: float,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> CascadeClassifier:
    model_stage1 = make_classifier(stage1_name, stage1_params)
    model_stage2 = make_classifier(stage2_name, stage2_params)

    cascade = CascadeClassifier(
        model_stage1,
        model_stage2,
        threshold_stage1=threshold_stage1,
        threshold_stage2=threshold_stage2,
    )
    cascade.fit(X_train, y_train)
    return cascade


def train_cascade(config_path: Union[str, Path]) -> Dict[str, object]:
    config = load_yaml(config_path)
    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = config.get("target_col", "Subsidio")
    X, y = prepare_xy(df, target_col)

    split_cfg = config.get("split", {})
    test_size = float(split_cfg.get("test_size", 0.10))
    val_size = float(split_cfg.get("val_size", 0.20))
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
    feature_pipeline = build_feature_pipeline(spec, config=feature_cfg)

    X_train_pp = to_dense(feature_pipeline.fit_transform(X_train))
    X_val_pp = to_dense(feature_pipeline.transform(X_val))
    X_test_pp = to_dense(feature_pipeline.transform(X_test))

    res_cfg = config.get("resampling", {})
    if bool(res_cfg.get("use_smote", False)):
        smote = SMOTE(random_state=int(res_cfg.get("random_state", random_state)))
        X_train_pp, y_train = smote.fit_resample(X_train_pp, y_train)

    y_train_np = np.asarray(y_train)
    y_val_np = y_val.values
    y_test_np = y_test.values

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    search_cfg = config.get("hyperparameter_search", {})
    search_enabled = bool(search_cfg.get("enabled", False))
    n_iter = int(search_cfg.get("n_iter", 20))
    search_random_state = int(search_cfg.get("random_state", random_state))

    threshold_stage1 = float(stage1_cfg.get("threshold", 0.5))
    threshold_stage2 = float(stage2_cfg.get("threshold", 0.5))

    mlflow_cfg = config.get("mlflow", {})
    run_name = str(config.get("run_name", "cascade_train"))

    with start_mlflow_run(mlflow_cfg, run_name=run_name) as mlflow_enabled:
        if search_enabled and bool(search_cfg.get("stage1", {}).get("enabled", True)):
            stage1_search_cfg = search_cfg.get("stage1", {})
            stage1_tuned = tune_stage1_classifier(
                model_name=str(stage1_cfg["model"]),
                base_params=dict(stage1_cfg.get("params", {})),
                param_distributions=stage1_search_cfg.get("param_distributions"),
                X_train=X_train_pp,
                y_train=y_train_np,
                X_val=X_val_pp,
                y_val=y_val_np,
                stage_threshold=threshold_stage1,
                objective_metric=str(stage1_search_cfg.get("objective_metric", "recall")),
                recall_min=float(stage1_search_cfg.get("recall_min", 0.0)),
                n_iter=int(stage1_search_cfg.get("n_iter", n_iter)),
                random_state=int(stage1_search_cfg.get("random_state", search_random_state)),
            )
            selected_stage1_params = stage1_tuned.params
            stage1_search_metrics = stage1_tuned.metrics
        else:
            selected_stage1_params = dict(stage1_cfg.get("params", {}))
            stage1_model = make_classifier(stage1_cfg["model"], selected_stage1_params)
            stage1_model.fit(X_train_pp, y_train_np)
            stage1_proba_val = stage1_model.predict_proba(X_val_pp)[:, 1]
            stage1_pred_val = (stage1_proba_val >= threshold_stage1).astype(int)
            stage1_search_metrics = compute_binary_metrics(y_val_np, stage1_pred_val, y_proba=stage1_proba_val)

        stage1_best_model = make_classifier(stage1_cfg["model"], selected_stage1_params)
        stage1_best_model.fit(X_train_pp, y_train_np)

        if search_enabled and bool(search_cfg.get("stage2", {}).get("enabled", True)):
            stage2_search_cfg = search_cfg.get("stage2", {})
            stage2_tuned = tune_stage2_classifier(
                model_name=str(stage2_cfg["model"]),
                base_params=dict(stage2_cfg.get("params", {})),
                param_distributions=stage2_search_cfg.get("param_distributions"),
                stage1_model=stage1_best_model,
                threshold_stage1=threshold_stage1,
                threshold_stage2=threshold_stage2,
                X_train=X_train_pp,
                y_train=y_train_np,
                X_val=X_val_pp,
                y_val=y_val_np,
                objective_metric=str(stage2_search_cfg.get("objective_metric", "precision")),
                recall_min=float(stage2_search_cfg.get("recall_min", 0.0)),
                n_iter=int(stage2_search_cfg.get("n_iter", n_iter)),
                random_state=int(stage2_search_cfg.get("random_state", search_random_state)),
            )
            selected_stage2_params = stage2_tuned.params
            stage2_search_metrics = stage2_tuned.metrics
        else:
            selected_stage2_params = dict(stage2_cfg.get("params", {}))
            proba1_train = stage1_best_model.predict_proba(X_train_pp)[:, 1]
            X2_train = np.hstack([X_train_pp, proba1_train.reshape(-1, 1)])
            stage2_model = make_classifier(stage2_cfg["model"], selected_stage2_params)
            stage2_model.fit(X2_train, y_train_np)

            proba1_val = stage1_best_model.predict_proba(X_val_pp)[:, 1]
            X2_val = np.hstack([X_val_pp, proba1_val.reshape(-1, 1)])
            mask1_val = proba1_val >= threshold_stage1
            proba2_val = np.zeros_like(proba1_val)
            if mask1_val.any():
                proba2_val[mask1_val] = stage2_model.predict_proba(X2_val[mask1_val])[:, 1]
            stage2_pred_val = (proba2_val >= threshold_stage2).astype(int)
            stage2_search_metrics = compute_binary_metrics(y_val_np, stage2_pred_val, y_proba=proba2_val)

        cascade = _fit_cascade_with_params(
            stage1_name=str(stage1_cfg["model"]),
            stage1_params=selected_stage1_params,
            stage2_name=str(stage2_cfg["model"]),
            stage2_params=selected_stage2_params,
            threshold_stage1=threshold_stage1,
            threshold_stage2=threshold_stage2,
            X_train=X_train_pp,
            y_train=y_train_np,
        )

        metrics_cfg = config.get("metrics", {})
        tune_thresholds_flag = bool(config.get("tune_thresholds", False))
        tuning_summary = None
        if tune_thresholds_flag:
            grid_cfg = config.get("threshold_search", {})
            grid_min = float(grid_cfg.get("min", 0.1))
            grid_max = float(grid_cfg.get("max", 0.9))
            grid_steps = int(grid_cfg.get("steps", 17))
            grid = np.linspace(grid_min, grid_max, grid_steps)

            thr1, thr2, tuning_summary = _tune_thresholds(
                cascade,
                X_val_pp,
                y_val_np,
                recall_min=float(metrics_cfg.get("recall_min", 0.0)),
                grid=grid,
            )
            cascade.threshold_stage1 = thr1
            cascade.threshold_stage2 = thr2

        y_val_proba = cascade.predict_proba(X_val_pp)[:, 1]
        y_test_proba = cascade.predict_proba(X_test_pp)[:, 1]
        y_val_pred = (y_val_proba >= cascade.threshold_stage2).astype(int)
        y_test_pred = (y_test_proba >= cascade.threshold_stage2).astype(int)

        metrics = {
            "stage1_validation": stage1_search_metrics,
            "stage2_validation": stage2_search_metrics,
            "val": compute_binary_metrics(y_val_np, y_val_pred, y_proba=y_val_proba),
            "test": compute_binary_metrics(y_test_np, y_test_pred, y_proba=y_test_proba),
            "selected_stage1_params": selected_stage1_params,
            "selected_stage2_params": selected_stage2_params,
        }
        if tuning_summary:
            metrics["threshold_tuning"] = tuning_summary

        artifacts_cfg = config.get("artifacts", {})
        artifacts_dir = Path(artifacts_cfg.get("dir", "artifacts/cascade"))

        metadata = {
            "target_col": target_col,
            "feature_columns": list(X.columns),
            "threshold_stage1": float(cascade.threshold_stage1),
            "threshold_stage2": float(cascade.threshold_stage2),
            "config_path": str(config_path),
            "stage1_model_name": str(stage1_cfg["model"]),
            "stage2_model_name": str(stage2_cfg["model"]),
            "selected_stage1_params": selected_stage1_params,
            "selected_stage2_params": selected_stage2_params,
            "feature_engineering": feature_cfg,
            "model_version": str(config.get("model_version", "v1")),
        }

        save_cascade_artifacts(
            artifacts_dir,
            preprocessor=feature_pipeline,
            cascade=cascade,
            metadata=metadata,
            metrics=metrics,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
        )

        log_params(
            mlflow_enabled,
            {
                "pipeline": "cascade",
                "config_path": str(config_path),
                "stage1.model": stage1_cfg["model"],
                "stage2.model": stage2_cfg["model"],
                "resampling.use_smote": bool(res_cfg.get("use_smote", False)),
                "hyperparameter_search.enabled": search_enabled,
                "selected_stage1_params": selected_stage1_params,
                "selected_stage2_params": selected_stage2_params,
            },
        )
        log_metrics(mlflow_enabled, metrics.get("val", {}), prefix="val_")
        log_metrics(mlflow_enabled, metrics.get("test", {}), prefix="test_")
        log_artifacts(mlflow_enabled, artifacts_dir, artifact_path="cascade_artifacts")

    return {
        "artifacts_dir": artifacts_dir,
        "metrics": metrics,
        "metadata": metadata,
    }
