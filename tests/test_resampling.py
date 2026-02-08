import numpy as np

from colombia_subsidy_ml.pipelines.train_cascade import _resample_training_data


def test_resample_training_data_smote_summary():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(100, 4))
    y = np.array([0] * 90 + [1] * 10)

    X_res, y_res, summary = _resample_training_data(
        X,
        y,
        config={"use_smote": True, "method": "smote", "random_state": 13},
        random_state=13,
    )

    assert summary["enabled"] is True
    assert summary["method"] == "smote"
    assert len(y_res) == X_res.shape[0]
    assert len(y_res) >= len(y)
