from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.model_selection import train_test_split

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.features.preprocess import build_preprocessor, prepare_xy, split_features
from colombia_subsidy_ml.models.factory import make_anomaly_model
from colombia_subsidy_ml.utils.arrays import to_dense
from colombia_subsidy_ml.utils.metrics import compute_anomaly_metrics


def train_anomaly(config_path: Union[str, Path]):
    config = load_yaml(config_path)
    df = read_dataset(config["dataset_path"])
    df = df.loc[:, ~df.columns.duplicated()]

    target_col = config.get("target_col", "Subsidio")
    X, y = prepare_xy(df, target_col)

    split_cfg = config.get("split", {})
    test_size = split_cfg.get("test_size", 0.2)
    random_state = split_cfg.get("random_state", 13)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    spec = split_features(X_train)
    preproc_cfg = config.get("preprocessing", {})
    preprocessor = build_preprocessor(spec, scale_numeric=preproc_cfg.get("scale_numeric", "minmax"))

    X_train_pp = to_dense(preprocessor.fit_transform(X_train))
    X_test_pp = to_dense(preprocessor.transform(X_test))

    # Train only on normal class (0)
    normal_mask = (y_train.values == 0)
    X_train_norm = X_train_pp[normal_mask]

    model_cfg = config["model"]
    model = make_anomaly_model(model_cfg["type"], model_cfg.get("params", {}))
    model.fit(X_train_norm)

    # Predict anomalies: -1 => anomaly => subsidy class (1)
    y_pred = model.predict(X_test_pp)
    y_pred_anom = (y_pred == -1).astype(int)

    metrics = compute_anomaly_metrics(y_test.values, y_pred_anom)

    artifacts_cfg = config.get("artifacts", {})
    artifacts_dir = Path(artifacts_cfg.get("dir", "artifacts/anomaly"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    joblib.dump(model, artifacts_dir / "model.joblib")

    metadata = {
        "target_col": target_col,
        "feature_columns": list(X.columns),
        "config_path": str(config_path),
        "model_type": model_cfg["type"],
    }
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "artifacts_dir": artifacts_dir,
        "metrics": metrics,
        "metadata": metadata,
    }
