from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import prepare_xy
from colombia_subsidy_ml.models.io import load_cascade_artifacts
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_binary_metrics


def evaluate_cascade(config_path: Union[str, Path], *, artifacts_dir: Optional[Union[str, Path]] = None):
    config = load_yaml(config_path)
    artifacts_dir = Path(artifacts_dir or config.get("artifacts", {}).get("dir", "artifacts/cascade"))

    preprocessor, cascade, metadata = load_cascade_artifacts(artifacts_dir)
    target_col = metadata["target_col"]

    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    X, y = prepare_xy(df, target_col)
    X_pp = to_dense(preprocessor.transform(X))

    metrics = {}
    split_path = artifacts_dir / "split_indices.npz"
    if split_path.exists():
        split = np.load(split_path)
        test_idx = split["test_idx"]

        y_test_proba = cascade.predict_proba(X_pp[test_idx])[:, 1]
        y_test_pred = (y_test_proba >= cascade.threshold_stage2).astype(int)
        metrics["test"] = compute_binary_metrics(y.values[test_idx], y_test_pred, y_proba=y_test_proba)
    else:
        y_all_proba = cascade.predict_proba(X_pp)[:, 1]
        y_all_pred = (y_all_proba >= cascade.threshold_stage2).astype(int)
        metrics["all"] = compute_binary_metrics(y.values, y_all_pred, y_proba=y_all_proba)

    (artifacts_dir / "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
