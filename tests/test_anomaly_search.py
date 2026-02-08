import numpy as np

from colombia_subsidy_ml.pipelines.train_anomaly import _find_best_score_threshold


def test_find_best_score_threshold_returns_metrics():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    threshold, metrics = _find_best_score_threshold(
        y_true=y_true,
        y_score=y_score,
        objective_metric="f1",
        recall_min=0.0,
        grid_size=20,
    )

    assert isinstance(threshold, float)
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
