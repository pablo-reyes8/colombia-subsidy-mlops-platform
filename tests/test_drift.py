from colombia_subsidy_ml.pipelines.drift import build_retraining_decision, extract_drift_summary


def test_extract_drift_summary():
    report_dict = {
        "metrics": [
            {
                "metric": "DataDriftPreset",
                "result": {
                    "dataset_drift": True,
                    "number_of_drifted_columns": 4,
                    "number_of_columns": 10,
                },
            }
        ]
    }

    summary = extract_drift_summary(report_dict)

    assert summary["dataset_drift"] is True
    assert summary["drifted_columns"] == 4
    assert summary["total_columns"] == 10
    assert summary["drift_share"] == 0.4


def test_build_retraining_decision_uses_dataset_flag():
    summary = {
        "dataset_drift": True,
        "drifted_columns": 1,
        "total_columns": 10,
        "drift_share": 0.1,
    }
    decision = build_retraining_decision(
        summary,
        {
            "enabled": True,
            "retrain_on_dataset_drift": True,
            "drift_share_threshold": 0.5,
            "min_drifted_columns": 5,
        },
    )

    assert decision["should_retrain"] is True
    assert "dataset_drift_flag=true" in decision["reasons"]


def test_build_retraining_decision_uses_thresholds():
    summary = {
        "dataset_drift": False,
        "drifted_columns": 4,
        "total_columns": 10,
        "drift_share": 0.4,
    }
    decision = build_retraining_decision(
        summary,
        {
            "enabled": True,
            "retrain_on_dataset_drift": False,
            "drift_share_threshold": 0.3,
            "min_drifted_columns": 3,
        },
    )

    assert decision["should_retrain"] is True
    assert "drift_share>=0.3000" in decision["reasons"]
    assert "drifted_columns>=3" in decision["reasons"]


def test_build_retraining_decision_disabled_policy():
    summary = {
        "dataset_drift": True,
        "drifted_columns": 7,
        "total_columns": 10,
        "drift_share": 0.7,
    }
    decision = build_retraining_decision(summary, {"enabled": False})

    assert decision["should_retrain"] is False
    assert decision["reasons"] == ["retraining_policy_disabled"]
