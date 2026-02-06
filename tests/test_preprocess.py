import pandas as pd

from colombia_subsidy_ml.features.preprocess import build_preprocessor, split_features


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
