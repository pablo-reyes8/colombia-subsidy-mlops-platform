import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from colombia_subsidy_ml.models.cascade import CascadeClassifier


def test_cascade_predicts():
    X, y = make_classification(
        n_samples=200,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    model1 = LogisticRegression(max_iter=200)
    model2 = LogisticRegression(max_iter=200)

    cascade = CascadeClassifier(model1, model2, threshold_stage1=0.4, threshold_stage2=0.5)
    cascade.fit(X, y)
    proba = cascade.predict_proba(X)
    pred = cascade.predict(X)

    assert proba.shape == (200, 2)
    assert pred.shape == (200,)
    assert set(np.unique(pred)).issubset({0, 1})
