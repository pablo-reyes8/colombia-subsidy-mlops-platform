import pandas as pd

from colombia_subsidy_ml.features.preprocess import build_feature_pipeline, build_preprocessor, split_features


def test_preprocess_builds():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3],
            "num2": [10.0, 11.0, 12.0],
            "cat": ["a", "b", "a"],
        }
    )
    spec = split_features(df)
    preprocessor = build_preprocessor(spec, scale_numeric="minmax")
    X = preprocessor.fit_transform(df)
    assert X.shape[0] == 3


def test_feature_pipeline_polynomial_pca():
    df = pd.DataFrame(
        {
            "num1": [1, 2, 3, 4],
            "num2": [2.0, 3.0, 4.0, 5.0],
            "cat": ["a", "b", "a", "b"],
        }
    )
    spec = split_features(df)
    pipeline = build_feature_pipeline(
        spec,
        config={
            "scale_numeric": "standard",
            "polynomial_degree": 2,
            "pca_components": 2,
            "pca_random_state": 13,
        },
    )
    X = pipeline.fit_transform(df)
    assert X.shape == (4, 2)
