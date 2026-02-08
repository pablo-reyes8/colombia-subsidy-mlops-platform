from __future__ import annotations

from pathlib import Path
from typing import Union


def _import_kfp():
    try:
        from kfp import compiler, dsl
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "Kubeflow Pipelines SDK (kfp) is not installed. Install optional dependencies: "
            "pip install 'colombia-subsidy-ml[mlops]'"
        ) from exc
    return compiler, dsl


def build_pipeline_definition():
    _, dsl = _import_kfp()

    @dsl.container_component
    def build_dataset_component(image: str, config_path: str):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "colombia_subsidy_ml", "build-dataset", "--config", config_path],
        )

    @dsl.container_component
    def train_cascade_component(image: str, config_path: str):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "colombia_subsidy_ml", "train", "--config", config_path],
        )

    @dsl.container_component
    def evaluate_component(image: str, config_path: str):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "colombia_subsidy_ml", "evaluate", "--config", config_path],
        )

    @dsl.container_component
    def drift_component(image: str, config_path: str):
        return dsl.ContainerSpec(
            image=image,
            command=["python", "-m", "colombia_subsidy_ml", "drift-check", "--config", config_path],
        )

    @dsl.pipeline(name="colombia-subsidy-mlops")
    def subsidy_pipeline(
        image: str = "colombia-subsidy-ml:latest",
        dataset_config: str = "configs/dataset.yaml",
        cascade_config: str = "configs/train_cascade.yaml",
        drift_config: str = "configs/drift.yaml",
    ):
        build_task = build_dataset_component(image=image, config_path=dataset_config)
        train_task = train_cascade_component(image=image, config_path=cascade_config).after(build_task)
        evaluate_component(image=image, config_path=cascade_config).after(train_task)
        drift_component(image=image, config_path=drift_config).after(train_task)

    return subsidy_pipeline


def compile_kubeflow_pipeline(output_path: Union[str, Path]) -> Path:
    compiler, _ = _import_kfp()
    pipeline_fn = build_pipeline_definition()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compiler.Compiler().compile(
        pipeline_func=pipeline_fn,
        package_path=str(output_path),
    )

    return output_path
