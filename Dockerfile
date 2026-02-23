# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11
ARG AIRFLOW_VERSION=2.9.3

FROM python:${PYTHON_VERSION}-slim AS python-base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"
WORKDIR /app
RUN python -m venv "$VIRTUAL_ENV" && \
    useradd --create-home --shell /bin/bash appuser

FROM python-base AS deps-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements.txt

FROM deps-base AS deps-mlops
COPY requirements-mlops.txt /tmp/requirements-mlops.txt
RUN python -m pip install -r /tmp/requirements-mlops.txt

FROM python-base AS app-core
COPY --from=deps-base /opt/venv /opt/venv
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY dvc.yaml /app/dvc.yaml
RUN python -m pip install . && \
    mkdir -p /app/artifacts /app/data && \
    chown -R appuser:appuser /app
USER appuser

FROM python-base AS app-mlops
COPY --from=deps-mlops /opt/venv /opt/venv
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY dvc.yaml /app/dvc.yaml
RUN python -m pip install . && \
    mkdir -p /app/artifacts /app/data && \
    chown -R appuser:appuser /app
USER appuser

FROM app-core AS dataset
CMD ["subsidy-ml", "build-dataset", "--config", "configs/dataset.yaml"]

FROM app-mlops AS train-cascade
CMD ["subsidy-ml", "train", "--config", "configs/train_cascade.yaml"]

FROM app-mlops AS train-anomaly
CMD ["subsidy-ml", "train-anomaly", "--config", "configs/train_anomaly.yaml"]

FROM app-mlops AS evaluate
CMD ["subsidy-ml", "evaluate", "--config", "configs/train_cascade.yaml"]

FROM app-mlops AS drift-check
CMD ["subsidy-ml", "drift-check", "--config", "configs/drift.yaml"]

FROM app-mlops AS compile-kubeflow
CMD ["subsidy-ml", "compile-kubeflow", "--output", "artifacts/kubeflow/subsidy_pipeline.yaml"]

FROM app-mlops AS api
EXPOSE 8000
CMD ["subsidy-ml", "serve-api", "--host", "0.0.0.0", "--port", "8000"]

FROM app-mlops AS mlops-shell
CMD ["bash"]

# Backward-compatible aliases used by the existing workflow.
FROM train-cascade AS train
FROM mlops-shell AS mlops

FROM apache/airflow:${AIRFLOW_VERSION}-python3.11 AS airflow
USER airflow
RUN pip install --no-cache-dir \
    "apache-airflow-providers-docker>=3.11" \
    "docker>=7"
