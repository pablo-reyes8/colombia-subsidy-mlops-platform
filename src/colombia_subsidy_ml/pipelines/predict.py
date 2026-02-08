from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset, write_dataset
from colombia_subsidy_ml.models.io import load_cascade_artifacts
from colombia_subsidy_ml.utils.arrays import to_dense


def predict_cascade(
    config_path: Union[str, Path],
    *,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    artifacts_dir: Optional[Union[str, Path]] = None,
) -> Path:
    config = load_yaml(config_path)
    artifacts_dir = Path(artifacts_dir or config.get("artifacts", {}).get("dir", "artifacts/cascade"))

    preprocessor, cascade, metadata = load_cascade_artifacts(artifacts_dir)
    target_col = metadata["target_col"]
    feature_cols = metadata["feature_columns"]

    df = read_dataset(input_path)
    df = df.loc[:, ~df.columns.duplicated()]

    if target_col in df.columns:
        df_features = df.drop(columns=[target_col])
    else:
        df_features = df

    missing = [c for c in feature_cols if c not in df_features.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    X = df_features[feature_cols]
    X_pp = to_dense(preprocessor.transform(X))

    proba = cascade.predict_proba(X_pp)[:, 1]
    pred = (proba >= cascade.threshold_stage2).astype(int)

    output = df.copy()
    output["pred_subsidio"] = pred
    output["proba_subsidio"] = proba

    output_path = Path(output_path)
    write_dataset(output, output_path)
    return output_path
