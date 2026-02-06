from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


@dataclass
class PreprocessSpec:
    numeric: List[str]
    categorical: List[str]


def split_features(df: pd.DataFrame) -> PreprocessSpec:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()
    return PreprocessSpec(numeric=numeric, categorical=categorical)


def _make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(spec: PreprocessSpec, *, scale_numeric: str = "minmax") -> ColumnTransformer:
    if scale_numeric == "standard":
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()

    transformers = []
    if spec.numeric:
        transformers.append(("num", scaler, spec.numeric))
    if spec.categorical:
        transformers.append(("cat", _make_onehot(), spec.categorical))

    return ColumnTransformer(transformers)


def prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
