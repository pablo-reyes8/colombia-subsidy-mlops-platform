from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.io import read_dataset
from colombia_subsidy_ml.tracking.mlflow_utils import log_artifacts, log_metrics, log_params, start_mlflow_run


def _import_evidently():
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset, DataQualityPreset, TargetDriftPreset
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Evidently is not installed. Install optional dependencies: "
            "pip install 'colombia-subsidy-ml[mlops]'"
        ) from exc
    return Report, DataDriftPreset, DataQualityPreset, TargetDriftPreset


def extract_drift_summary(report_dict: Dict[str, Any]) -> Dict[str, Any]:
    summary = {
        "dataset_drift": None,
        "drifted_columns": None,
        "total_columns": None,
        "drift_share": None,
    }

    for metric in report_dict.get("metrics", []):
        result = metric.get("result", {})
        if "dataset_drift" not in result:
            continue

        drifted = result.get("number_of_drifted_columns")
        total = result.get("number_of_columns")

        summary["dataset_drift"] = bool(result.get("dataset_drift"))
        summary["drifted_columns"] = int(drifted) if drifted is not None else None
        summary["total_columns"] = int(total) if total is not None else None

        if summary["drifted_columns"] is not None and summary["total_columns"]:
            summary["drift_share"] = float(summary["drifted_columns"] / summary["total_columns"])

        break

    return summary


def run_drift_check(config_path: Union[str, Path]) -> Dict[str, Any]:
    config = load_yaml(config_path)

    reference_path = config["reference_dataset_path"]
    current_path = config["current_dataset_path"]
    target_col = config.get("target_col")

    ref_df = read_dataset(reference_path)
    cur_df = read_dataset(current_path)

    exclude_columns = config.get("exclude_columns", [])
    if exclude_columns:
        ref_df = ref_df.drop(columns=[c for c in exclude_columns if c in ref_df.columns])
        cur_df = cur_df.drop(columns=[c for c in exclude_columns if c in cur_df.columns])

    Report, DataDriftPreset, DataQualityPreset, TargetDriftPreset = _import_evidently()

    metrics = [DataDriftPreset()]
    if bool(config.get("include_data_quality", True)):
        metrics.append(DataQualityPreset())
    if target_col and target_col in ref_df.columns and target_col in cur_df.columns:
        metrics.append(TargetDriftPreset())

    report = Report(metrics=metrics)
    report.run(reference_data=ref_df, current_data=cur_df)

    output_cfg = config.get("output", {})
    output_dir = Path(output_cfg.get("dir", "artifacts/drift"))
    output_dir.mkdir(parents=True, exist_ok=True)

    html_name = output_cfg.get("html_name", "drift_report.html")
    json_name = output_cfg.get("json_name", "drift_report.json")
    summary_name = output_cfg.get("summary_name", "drift_summary.json")

    html_path = output_dir / html_name
    json_path = output_dir / json_name
    summary_path = output_dir / summary_name

    report.save_html(str(html_path))
    report.save_json(str(json_path))

    try:
        report_dict = report.as_dict()
    except Exception:
        report_dict = json.loads(json_path.read_text(encoding="utf-8"))

    summary = extract_drift_summary(report_dict)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    mlflow_cfg = config.get("mlflow", {})
    run_name = str(config.get("run_name", "drift_check"))
    with start_mlflow_run(mlflow_cfg, run_name=run_name) as mlflow_enabled:
        log_params(
            mlflow_enabled,
            {
                "pipeline": "drift",
                "reference_dataset_path": reference_path,
                "current_dataset_path": current_path,
                "target_col": target_col,
            },
        )
        log_metrics(mlflow_enabled, summary, prefix="drift_")
        log_artifacts(mlflow_enabled, output_dir, artifact_path="drift_reports")

    return {
        "output_dir": output_dir,
        "summary": summary,
        "html_report": html_path,
        "json_report": json_path,
    }
