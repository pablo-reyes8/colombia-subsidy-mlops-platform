from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, PolynomialFeatures, RobustScaler, StandardScaler


@dataclass
class PreprocessSpec:
    numeric: List[str]
    categorical: List[str]


@dataclass
class FeatureEngineeringConfig:
    scale_numeric: str = "minmax"
    numeric_imputer_strategy: str = "median"
    categorical_imputer_strategy: str = "most_frequent"
    polynomial_degree: int = 1
    polynomial_include_bias: bool = False
    pca_components: Optional[int] = None
    pca_random_state: int = 13


def split_features(df: pd.DataFrame) -> PreprocessSpec:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return PreprocessSpec(numeric=numeric, categorical=categorical)


def _make_onehot() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_numeric_scaler(scale_numeric: str):
    scale_numeric = (scale_numeric or "minmax").lower()
    if scale_numeric == "standard":
        return StandardScaler()
    if scale_numeric == "robust":
        return RobustScaler()
    return MinMaxScaler()


def _coerce_feature_config(
    config: Optional[Union[FeatureEngineeringConfig, Dict[str, object]]], *, scale_numeric: str
) -> FeatureEngineeringConfig:
    if config is None:
        return FeatureEngineeringConfig(scale_numeric=scale_numeric)
    if isinstance(config, FeatureEngineeringConfig):
        return config

    return FeatureEngineeringConfig(
        scale_numeric=str(config.get("scale_numeric", scale_numeric)),
        numeric_imputer_strategy=str(config.get("numeric_imputer_strategy", "median")),
        categorical_imputer_strategy=str(config.get("categorical_imputer_strategy", "most_frequent")),
        polynomial_degree=int(config.get("polynomial_degree", 1)),
        polynomial_include_bias=bool(config.get("polynomial_include_bias", False)),
        pca_components=(int(config["pca_components"]) if config.get("pca_components") is not None else None),
        pca_random_state=int(config.get("pca_random_state", 13)),
    )


def build_preprocessor(
    spec: PreprocessSpec,
    *,
    scale_numeric: str = "minmax",
    numeric_imputer_strategy: str = "median",
    categorical_imputer_strategy: str = "most_frequent",
) -> ColumnTransformer:
    transformers = []

    if spec.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=numeric_imputer_strategy)),
                ("scaler", _make_numeric_scaler(scale_numeric)),
            ]
        )
        transformers.append(("num", numeric_pipeline, spec.numeric))

    if spec.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy=categorical_imputer_strategy)),
                ("onehot", _make_onehot()),
            ]
        )
        transformers.append(("cat", categorical_pipeline, spec.categorical))

    return ColumnTransformer(transformers)


def build_feature_pipeline(
    spec: PreprocessSpec,
    *,
    config: Optional[Union[FeatureEngineeringConfig, Dict[str, object]]] = None,
    scale_numeric: str = "minmax",
) -> Pipeline:
    cfg = _coerce_feature_config(config, scale_numeric=scale_numeric)

    steps = [
        (
            "preprocessor",
            build_preprocessor(
                spec,
                scale_numeric=cfg.scale_numeric,
                numeric_imputer_strategy=cfg.numeric_imputer_strategy,
                categorical_imputer_strategy=cfg.categorical_imputer_strategy,
            ),
        )
    ]

    if cfg.polynomial_degree > 1:
        steps.append(
            (
                "polynomial",
                PolynomialFeatures(degree=cfg.polynomial_degree, include_bias=cfg.polynomial_include_bias),
            )
        )

    if cfg.pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(n_components=cfg.pca_components, random_state=cfg.pca_random_state),
            )
        )

    return Pipeline(steps=steps)


def prepare_xy(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: {target_col}")
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y
