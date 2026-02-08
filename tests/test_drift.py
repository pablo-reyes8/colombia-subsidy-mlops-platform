from colombia_subsidy_ml.pipelines.drift import extract_drift_summary


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
