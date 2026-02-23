from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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


def build_retraining_decision(
    summary: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    policy = dict(policy or {})
    enabled = bool(policy.get("enabled", True))

    reasons: List[str] = []
    dataset_drift = summary.get("dataset_drift")
    drift_share = summary.get("drift_share")
    drifted_columns = summary.get("drifted_columns")

    if not enabled:
        reasons.append("retraining_policy_disabled")
    else:
        if bool(policy.get("retrain_on_dataset_drift", True)) and dataset_drift is True:
            reasons.append("dataset_drift_flag=true")

        drift_share_threshold = policy.get("drift_share_threshold")
        if drift_share_threshold is not None and drift_share is not None:
            if float(drift_share) >= float(drift_share_threshold):
                reasons.append(
                    f"drift_share>={float(drift_share_threshold):.4f}"
                )

        min_drifted_columns = policy.get("min_drifted_columns")
        if min_drifted_columns is not None and drifted_columns is not None:
            if int(drifted_columns) >= int(min_drifted_columns):
                reasons.append(f"drifted_columns>={int(min_drifted_columns)}")

    active_reasons = [reason for reason in reasons if reason != "retraining_policy_disabled"]
    should_retrain = bool(enabled and active_reasons)

    return {
        "should_retrain": should_retrain,
        "reasons": reasons,
        "observed": {
            "dataset_drift": dataset_drift,
            "drift_share": drift_share,
            "drifted_columns": drifted_columns,
            "total_columns": summary.get("total_columns"),
        },
        "policy": {
            "enabled": enabled,
            "retrain_on_dataset_drift": bool(policy.get("retrain_on_dataset_drift", True)),
            "drift_share_threshold": (
                float(policy["drift_share_threshold"])
                if policy.get("drift_share_threshold") is not None
                else None
            ),
            "min_drifted_columns": (
                int(policy["min_drifted_columns"])
                if policy.get("min_drifted_columns") is not None
                else None
            ),
        },
    }


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
    decision_name = output_cfg.get("decision_name", "drift_decision.json")

    html_path = output_dir / html_name
    json_path = output_dir / json_name
    summary_path = output_dir / summary_name
    decision_path = output_dir / decision_name

    report.save_html(str(html_path))
    report.save_json(str(json_path))

    try:
        report_dict = report.as_dict()
    except Exception:
        report_dict = json.loads(json_path.read_text(encoding="utf-8"))

    summary = extract_drift_summary(report_dict)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    retraining_policy = config.get("retraining_policy", {})
    decision = build_retraining_decision(summary, retraining_policy)
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")

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
        log_metrics(
            mlflow_enabled,
            {"should_retrain": int(bool(decision.get("should_retrain", False)))},
            prefix="retrain_",
        )
        log_artifacts(mlflow_enabled, output_dir, artifact_path="drift_reports")

    return {
        "output_dir": output_dir,
        "summary": summary,
        "decision": decision,
        "html_report": html_path,
        "json_report": json_path,
        "decision_file": decision_path,
    }
