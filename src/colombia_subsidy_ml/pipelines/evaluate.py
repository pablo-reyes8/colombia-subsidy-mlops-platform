from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import prepare_xy
from colombia_subsidy_ml.models.cascade import CascadeClassifier
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_binary_metrics


def evaluate_cascade(
    config_path: Union[str, Path], *, artifacts_dir: Optional[Union[str, Path]] = None
):
    config = load_yaml(config_path)
    artifacts_dir = Path(artifacts_dir or config.get("artifacts", {}).get("dir", "artifacts/cascade"))

    import joblib

    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    stage1 = joblib.load(artifacts_dir / "stage1_model.joblib")
    stage2 = joblib.load(artifacts_dir / "stage2_model.joblib")

    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))
    target_col = metadata["target_col"]

    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    X, y = prepare_xy(df, target_col)
    X_pp = to_dense(preprocessor.transform(X))

    cascade = CascadeClassifier(
        stage1,
        stage2,
        threshold_stage1=metadata.get("threshold_stage1", 0.5),
        threshold_stage2=metadata.get("threshold_stage2", 0.5),
    )

    metrics = {}
    split_path = artifacts_dir / "split_indices.npz"
    if split_path.exists():
        split = np.load(split_path)
        test_idx = split["test_idx"]
        y_pred = cascade.predict(X_pp[test_idx])
        metrics["test"] = compute_binary_metrics(y.values[test_idx], y_pred)
    else:
        y_pred = cascade.predict(X_pp)
        metrics["all"] = compute_binary_metrics(y.values, y_pred)

    (artifacts_dir / "metrics_eval.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
