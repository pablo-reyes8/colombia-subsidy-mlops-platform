from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


def _import_mlflow():
    try:
        import mlflow
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "MLflow is enabled in config but is not installed. Install optional dependencies: "
            "pip install 'colombia-subsidy-ml[mlops]'"
        ) from exc
    return mlflow


def _flatten_dict(data: Dict[str, Any], *, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, parent_key=full_key, sep=sep))
        else:
            flattened[full_key] = value
    return flattened


@contextmanager
def start_mlflow_run(mlflow_cfg: Dict[str, Any], *, run_name: str) -> Iterator[bool]:
    enabled = bool(mlflow_cfg.get("enabled", False))
    if not enabled:
        yield False
        return

    mlflow = _import_mlflow()

    tracking_uri = mlflow_cfg.get("tracking_uri")
    experiment_name = mlflow_cfg.get("experiment_name", "colombia-subsidy-ml")
    tags = mlflow_cfg.get("tags", {})

    if tracking_uri:
        mlflow.set_tracking_uri(str(tracking_uri))
    mlflow.set_experiment(str(experiment_name))

    with mlflow.start_run(run_name=run_name, tags=tags):
        yield True


def log_params(mlflow_enabled: bool, params: Dict[str, Any]) -> None:
    if not mlflow_enabled:
        return
    mlflow = _import_mlflow()
    for key, value in _flatten_dict(params).items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            mlflow.log_param(key, str(list(value)))
            continue
        mlflow.log_param(key, value)


def log_metrics(mlflow_enabled: bool, metrics: Dict[str, Any], *, prefix: str = "") -> None:
    if not mlflow_enabled:
        return
    mlflow = _import_mlflow()

    flat_metrics = _flatten_dict(metrics)
    for key, value in flat_metrics.items():
        if not isinstance(value, (int, float)):
            continue
        metric_name = f"{prefix}{key}" if prefix else key
        mlflow.log_metric(metric_name, float(value))


def log_artifacts(mlflow_enabled: bool, artifacts_dir: Path, *, artifact_path: Optional[str] = None) -> None:
    if not mlflow_enabled:
        return
    mlflow = _import_mlflow()
    mlflow.log_artifacts(str(artifacts_dir), artifact_path=artifact_path)
