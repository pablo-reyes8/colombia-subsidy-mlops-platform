# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.11

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

FROM python-base AS train
COPY --from=deps-base /opt/venv /opt/venv
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY dvc.yaml /app/dvc.yaml
RUN python -m pip install .
USER appuser
CMD ["subsidy-ml", "train", "--config", "configs/train_cascade.yaml"]

FROM python-base AS api
COPY --from=deps-mlops /opt/venv /opt/venv
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
RUN python -m pip install .
USER appuser
EXPOSE 8000
CMD ["subsidy-ml", "serve-api", "--host", "0.0.0.0", "--port", "8000"]

FROM python-base AS mlops
COPY --from=deps-mlops /opt/venv /opt/venv
COPY pyproject.toml README.md /app/
COPY src /app/src
COPY configs /app/configs
COPY scripts /app/scripts
COPY dvc.yaml /app/dvc.yaml
RUN python -m pip install .
USER appuser
CMD ["bash"]
