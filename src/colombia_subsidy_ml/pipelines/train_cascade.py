from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import build_preprocessor, prepare_xy, split_features
from colombia_subsidy_ml.models.cascade import CascadeClassifier
from colombia_subsidy_ml.models.factory import make_classifier
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_binary_metrics


def _split_indices(y, *, test_size: float, val_size: float, random_state: int, stratify: bool):
    idx = np.arange(len(y))
    strat = y if stratify else None
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=random_state, stratify=strat
    )
    strat2 = y[train_val_idx] if stratify else None
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, random_state=random_state, stratify=strat2
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
    best = {"precision": -1.0, "recall": 0.0, "thr1": model.threshold_stage1, "thr2": model.threshold_stage2}
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
            metrics = compute_binary_metrics(y_val, y_pred)
            if metrics["recall"] < recall_min:
                continue
            if metrics["precision"] > best["precision"]:
                best = {
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "thr1": float(thr1),
                    "thr2": float(thr2),
                }
    return best["thr1"], best["thr2"], best


def train_cascade(config_path: Union[str, Path]) -> Dict[str, object]:
    config = load_yaml(config_path)
    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = config.get("target_col", "Subsidio")
    X, y = prepare_xy(df, target_col)

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.1)
    val_size = split_cfg.get("val_size", 0.2)
    random_state = split_cfg.get("random_state", 13)
    stratify = bool(split_cfg.get("stratify", True))

    train_idx, val_idx, test_idx = _split_indices(
        y.values, test_size=test_size, val_size=val_size, random_state=random_state, stratify=stratify
    )

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    spec = split_features(X_train)
    preproc_cfg = config.get("preprocessing", {})
    preprocessor = build_preprocessor(spec, scale_numeric=preproc_cfg.get("scale_numeric", "minmax"))

    X_train_pp = to_dense(preprocessor.fit_transform(X_train))
    X_val_pp = to_dense(preprocessor.transform(X_val))
    X_test_pp = to_dense(preprocessor.transform(X_test))

    res_cfg = config.get("resampling", {})
    if res_cfg.get("use_smote", False):
        smote = SMOTE(random_state=res_cfg.get("random_state", random_state))
        X_train_pp, y_train = smote.fit_resample(X_train_pp, y_train)

    stage1_cfg = config["stage1"]
    stage2_cfg = config["stage2"]

    model_stage1 = make_classifier(stage1_cfg["model"], stage1_cfg.get("params", {}))
    model_stage2 = make_classifier(stage2_cfg["model"], stage2_cfg.get("params", {}))

    cascade = CascadeClassifier(
        model_stage1,
        model_stage2,
        threshold_stage1=stage1_cfg.get("threshold", 0.5),
        threshold_stage2=stage2_cfg.get("threshold", 0.5),
    )
    cascade.fit(X_train_pp, y_train)

    metrics_cfg = config.get("metrics", {})
    tune = bool(config.get("tune_thresholds", False))
    tuning_summary = None
    if tune:
        grid = np.linspace(0.1, 0.9, 17)
        thr1, thr2, tuning_summary = _tune_thresholds(
            cascade,
            X_val_pp,
            y_val.values,
            recall_min=float(metrics_cfg.get("recall_min", 0.0)),
            grid=grid,
        )
        cascade.threshold_stage1 = thr1
        cascade.threshold_stage2 = thr2

    y_val_pred = cascade.predict(X_val_pp)
    y_test_pred = cascade.predict(X_test_pp)

    metrics = {
        "val": compute_binary_metrics(y_val.values, y_val_pred),
        "test": compute_binary_metrics(y_test.values, y_test_pred),
    }
    if tuning_summary:
        metrics["tuning"] = tuning_summary

    artifacts_cfg = config.get("artifacts", {})
    artifacts_dir = Path(artifacts_cfg.get("dir", "artifacts/cascade"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save artifacts
    import joblib

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    joblib.dump(cascade.model_stage1, artifacts_dir / "stage1_model.joblib")
    joblib.dump(cascade.model_stage2, artifacts_dir / "stage2_model.joblib")

    np.savez(artifacts_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)

    metadata = {
        "target_col": target_col,
        "feature_columns": list(X.columns),
        "threshold_stage1": cascade.threshold_stage1,
        "threshold_stage2": cascade.threshold_stage2,
        "config_path": str(config_path),
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "artifacts_dir": artifacts_dir,
        "metrics": metrics,
        "metadata": metadata,
    }
