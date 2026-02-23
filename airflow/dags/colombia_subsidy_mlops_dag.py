from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from docker.types import Mount


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _container_project_path(*parts: str) -> str:
    base = Path(os.getenv("AIRFLOW_PROJECT_MOUNT_PATH", "/opt/project"))
    return str(base.joinpath(*parts))


def _host_project_path(*parts: str) -> str:
    base = Path(os.getenv("PROJECT_ROOT_HOST_PATH", "."))
    return str(base.joinpath(*parts))


def _job_mounts() -> list[Mount]:
    return [
        Mount(source=_host_project_path("data"), target="/app/data", type="bind"),
        Mount(source=_host_project_path("artifacts"), target="/app/artifacts", type="bind"),
        Mount(source=_host_project_path("configs"), target="/app/configs", type="bind", read_only=True),
    ]


def _job_environment() -> dict[str, str]:
    env = {
        "SUBSIDY_MLOPS_ORCHESTRATOR": os.getenv("SUBSIDY_MLOPS_ORCHESTRATOR", "airflow"),
    }

    for key in ("SUBSIDY_MLFLOW_ENABLED", "SUBSIDY_MLFLOW_TRACKING_URI", "SUBSIDY_MLFLOW_EXPERIMENT_NAME"):
        value = os.getenv(key)
        if value is not None:
            env[key] = value

    return env


def _decide_retraining() -> str:
    decision_path = Path(
        os.getenv(
            "AIRFLOW_DRIFT_DECISION_PATH",
            _container_project_path("artifacts", "drift", "drift_decision.json"),
        )
    )

    if not decision_path.exists():
        force_retrain = _env_flag("AIRFLOW_FORCE_RETRAIN_ON_MISSING_DRIFT", True)
        print(f"Drift decision file not found at {decision_path}. force_retrain={force_retrain}")
        return "retrain_start" if force_retrain else "skip_retrain"

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    should_retrain = bool(decision.get("should_retrain", False))
    print(f"Drift decision loaded from {decision_path}: {decision}")
    return "retrain_start" if should_retrain else "skip_retrain"


def _docker_task(task_id: str, *, image: str, command: str) -> DockerOperator:
    return DockerOperator(
        task_id=task_id,
        image=image,
        command=command,
        api_version="auto",
        docker_url=os.getenv("DOCKER_HOST", "unix:///var/run/docker.sock"),
        network_mode=os.getenv("AIRFLOW_PIPELINE_DOCKER_NETWORK", "colombia-subsidy-ml_default"),
        mounts=_job_mounts(),
        environment=_job_environment(),
        mount_tmp_dir=False,
        do_xcom_push=False,
    )


DATASET_IMAGE = os.getenv("AIRFLOW_PIPELINE_DATASET_IMAGE", "colombia-subsidy-ml:dataset")
TRAIN_CASCADE_IMAGE = os.getenv("AIRFLOW_PIPELINE_TRAIN_CASCADE_IMAGE", "colombia-subsidy-ml:train-cascade")
TRAIN_ANOMALY_IMAGE = os.getenv("AIRFLOW_PIPELINE_TRAIN_ANOMALY_IMAGE", "colombia-subsidy-ml:train-anomaly")
EVALUATE_IMAGE = os.getenv("AIRFLOW_PIPELINE_EVALUATE_IMAGE", "colombia-subsidy-ml:evaluate")
DRIFT_IMAGE = os.getenv("AIRFLOW_PIPELINE_DRIFT_IMAGE", "colombia-subsidy-ml:drift")
KFP_IMAGE = os.getenv("AIRFLOW_PIPELINE_KFP_IMAGE", "colombia-subsidy-ml:kubeflow-compiler")

DATASET_CONFIG = os.getenv("AIRFLOW_PIPELINE_DATASET_CONFIG", "configs/dataset.yaml")
CASCADE_CONFIG = os.getenv("AIRFLOW_PIPELINE_CASCADE_CONFIG", "configs/train_cascade.yaml")
ANOMALY_CONFIG = os.getenv("AIRFLOW_PIPELINE_ANOMALY_CONFIG", "configs/train_anomaly.yaml")
DRIFT_CONFIG = os.getenv("AIRFLOW_PIPELINE_DRIFT_CONFIG", "configs/drift.yaml")

with DAG(
    dag_id="colombia_subsidy_mlops_orchestrator",
    description="Orquesta drift monitoring y reentrenamiento del pipeline de subsidios",
    schedule="@daily",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={
        "owner": "mlops",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["mlops", "drift", "kubeflow", "mlflow"],
) as dag:
    start = EmptyOperator(task_id="start")

    drift_check = _docker_task(
        "drift_check",
        image=DRIFT_IMAGE,
        command=f"subsidy-ml drift-check --config {DRIFT_CONFIG}",
    )

    branch_retrain = BranchPythonOperator(
        task_id="branch_retrain",
        python_callable=_decide_retraining,
    )

    retrain_start = EmptyOperator(task_id="retrain_start")
    skip_retrain = EmptyOperator(task_id="skip_retrain")

    compile_kubeflow_spec = _docker_task(
        "compile_kubeflow_spec",
        image=KFP_IMAGE,
        command="subsidy-ml compile-kubeflow --output artifacts/kubeflow/subsidy_pipeline.yaml",
    )

    build_dataset = _docker_task(
        "build_dataset",
        image=DATASET_IMAGE,
        command=f"subsidy-ml build-dataset --config {DATASET_CONFIG}",
    )

    train_cascade = _docker_task(
        "train_cascade",
        image=TRAIN_CASCADE_IMAGE,
        command=f"subsidy-ml train --config {CASCADE_CONFIG}",
    )

    train_anomaly = _docker_task(
        "train_anomaly",
        image=TRAIN_ANOMALY_IMAGE,
        command=f"subsidy-ml train-anomaly --config {ANOMALY_CONFIG}",
    )

    evaluate_cascade = _docker_task(
        "evaluate_cascade",
        image=EVALUATE_IMAGE,
        command=f"subsidy-ml evaluate --config {CASCADE_CONFIG}",
    )

    retrain_done = EmptyOperator(task_id="retrain_done")

    pipeline_done = EmptyOperator(
        task_id="pipeline_done",
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
    )

    start >> drift_check >> branch_retrain
    branch_retrain >> retrain_start
    branch_retrain >> skip_retrain

    retrain_start >> compile_kubeflow_spec
    retrain_start >> build_dataset
    build_dataset >> train_cascade >> evaluate_cascade
    build_dataset >> train_anomaly

    [compile_kubeflow_spec, evaluate_cascade, train_anomaly] >> retrain_done
    skip_retrain >> pipeline_done
    retrain_done >> pipeline_done
