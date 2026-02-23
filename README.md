# Colombia Subsidy MLOps Prediction

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/colombia-subsidy-ml-prediction)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/colombia-subsidy-ml-prediction)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/colombia-subsidy-ml-prediction)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/colombia-subsidy-ml-prediction)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/colombia-subsidy-ml-prediction?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/colombia-subsidy-ml-prediction?style=social)





A professional, reproducible MLOps-style project built on Colombia’s GEIH household survey to study whether subsidies reduce inequality and to predict potential subsidy candidates under extreme class imbalance. The original notebooks pipelines are preserved in `notebooks/`, while the production path is modularized into pipelines, configs, and CLI tools.

---

## Table of Contents
1. [Overview](#overview)
2. [Descriptive Analysis (Selected Figures)](#descriptive-analysis-selected-figures)
3. [MLOps Scope](#mlops-scope)
4. [Project Structure](#project-structure)
5. [End-to-End Pipeline](#end-to-end-pipeline)
6. [Model Results Summary](#model-results-summary)
7. [Configuration](#configuration)
8. [Local CLI Workflows](#local-cli-workflows)
9. [Docker Strategy](#docker-strategy)
10. [Docker Compose Profiles](#docker-compose-profiles)
11. [Orchestration & Reproducibility](#orchestration--reproducibility)
12. [Experiment Tracking](#experiment-tracking)
13. [API Deployment](#api-deployment)
14. [Monitoring & Drift](#monitoring--drift)
15. [Artifacts](#artifacts)
16. [Testing](#testing)
17. [Notebooks](#notebooks)
18. [Roadmap](#roadmap)

---

## 1. Overview
This repository delivers:
- A modular ML stack for subsidy prediction with severe class imbalance.
- A robust supervised cascade pipeline (XGBoost + RandomForest) with feature engineering, hyperparameter search, and threshold optimization.
- Unsupervised anomaly baselines (IsolationForest / OneClassSVM) with score-threshold tuning.
- Operational MLOps components: MLflow tracking, DVC pipelines, Kubeflow compilation, FastAPI serving, and Evidently drift checks.
- Container-first execution for both training jobs and serving workloads.

---

## 2. Descriptive Analysis (Selected Figures)
Below are selected figures extracted from the original descriptive notebook. These are intentionally curated (not all plots) and laid out for readability.

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 980px;">
  <tr>
    <td align="center" style="border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px;">
      <img src="results/Boxplot%20Subs.png" alt="Descriptive plot 1"
           style="width: 100%; max-width: 460px; height: auto; display: block;" />
    </td>
    <td align="center" style="border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px;">
      <img src="docs/assets/analysis_plot_2.png" alt="Descriptive plot 2"
           style="width: 100%; max-width: 460px; height: auto; display: block;" />
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="padding: 10px 6px 16px 6px;">
      <sub><b>Figure 2.1.</b> Summary distribution & key diagnostics (left) and complementary descriptive patterns (right).</sub>
    </td>
  </tr>
</table>

<br/>

<table align="center" style="border-collapse: collapse; width: 100%; max-width: 980px;">
  <tr>
    <td align="center" style="border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px;">
      <img src="docs/assets/analysis_plot_3.png" alt="Descriptive plot 3"
           style="width: 100%; max-width: 460px; height: auto; display: block;" />
    </td>
    <td align="center" style="border: 1px solid #e5e7eb; padding: 12px; border-radius: 10px;">
      <img src="docs/assets/analysis_plot_4.png" alt="Descriptive plot 4"
           style="width: 100%; max-width: 460px; height: auto; display: block;" />
    </td>
  </tr>
  <tr>
    <td colspan="2" align="center" style="padding: 10px 6px 16px 6px;">
      <sub><b>Figure 2.2.</b> Additional distributional comparisons and subgroup contrasts.</sub>
    </td>
  </tr>
</table>

<p align="center">
  <img src="results/Geographical%20distribution%20of%20subs.png" alt="Geographical distribution of subsidies" width="940" />
</p>

---

## 3. MLOps Scope
This project is organized as a production-oriented MLOps workflow:
- **Data layer**: deterministic data preparation with config-driven inputs.
- **Training layer**: supervised and unsupervised pipelines with reproducible splits and saved artifacts.
- **Evaluation layer**: metrics persistence (`metrics.json`, `metrics_eval.json`) and threshold-aware reporting.
- **Serving layer**: FastAPI API with schemas and model metadata endpoints.
- **Monitoring layer**: Evidently drift checks over reference vs current data.
- **Orchestration layer**: DVC stage graph + Kubeflow pipeline compilation.
- **Tracking layer**: optional MLflow logging for params, metrics, artifacts, and run tags.

---

## 4. Project Structure
```text
.
├─ artifacts/                     # model artifacts, predictions, drift reports, mlflow backend
├─ configs/                       # yaml configs for dataset, training, drift
├─ data/
│  ├─ raw/                        # GEIH raw sources
│  └─ processed/                  # training-ready tables
├─ docs/
├─ notebooks/                     # research / analysis history
├─ scripts/                       # thin wrappers for jobs
├─ src/colombia_subsidy_ml/
│  ├─ api/                        # FastAPI app, schemas, model loading
│  ├─ data/                       # ingestion and dataset building
│  ├─ features/                   # preprocessing and feature pipeline
│  ├─ mlops/                      # kubeflow pipeline compilation
│  ├─ models/                     # cascade model, factory, tuning, artifact io
│  ├─ pipelines/                  # train/evaluate/predict/drift workflows
│  ├─ tracking/                   # MLflow helpers
│  └─ utils/
├─ tests/
├─ .dockerignore
├─ docker-compose.yml
├─ dvc.yaml
├─ Dockerfile
└─ pyproject.toml
```

---

## 5. End-to-End Pipeline
```text
Raw GEIH Data
   -> build-dataset
   -> train (cascade / anomaly)
   -> evaluate + predict
   -> drift-check (Evidently)
   -> API serving (FastAPI)
```

Cross-cutting concerns:
- **Tracking**: MLflow (optional, config-driven).
- **Reproducibility**: DVC stage graph + deterministic configs.
- **Orchestration**: Kubeflow pipeline compilation for CI/CD or cluster execution.

---

## 6. Model Results Summary
The following table summarizes key reference results obtained in the modeling notebook (`notebooks/Full Maching Learning Modeling.ipynb`):

| Model / Strategy | Precision (Subsidio=1) | Recall (Subsidio=1) | F1 (Subsidio=1) | Main Trade-off |
|---|---:|---:|---:|---|
| Cascade (XGBoost + RF + threshold tuning) | 0.218 | 0.711 | ~0.33 | Meets recall target with moderate precision |
| One-Class SVM (anomaly framing) | 0.934 | 0.264 | 0.411 | Very high precision, low recall |
| IsolationForest (anomaly framing) | 0.926 | 0.303 | 0.457 | Very high precision, low recall |

For production runs, use the current artifacts (`artifacts/*/metrics.json`) as the source of truth.

---

## 7. Configuration
Core configs:
- `configs/dataset.yaml`: input raw tables and processed output path.
- `configs/train_cascade.yaml`: supervised cascade config (feature engineering, resampling, search, thresholds, MLflow).
- `configs/train_anomaly.yaml`: anomaly model config (search + score thresholding + MLflow).
- `configs/drift.yaml`: reference/current dataset and Evidently report output.

---

## 8. Local CLI Workflows
Install:
```bash
pip install -e .
pip install -e ".[mlops]"  # optional extras for full MLOps stack
```

Run pipelines:
```bash
python -m colombia_subsidy_ml build-dataset --config configs/dataset.yaml
python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
python -m colombia_subsidy_ml train-anomaly --config configs/train_anomaly.yaml
python -m colombia_subsidy_ml evaluate --config configs/train_cascade.yaml
python -m colombia_subsidy_ml predict --config configs/train_cascade.yaml --input data/processed/Base_Modelo_Subsidios.csv --output artifacts/predictions.csv
python -m colombia_subsidy_ml drift-check --config configs/drift.yaml
python -m colombia_subsidy_ml compile-kubeflow --output artifacts/kubeflow/subsidy_pipeline.yaml
```

---

## 9. Docker Strategy
The repository now ships a **multi-stage Dockerfile** with dedicated targets:

| Docker target | Purpose | Included dependencies | Default command |
|---|---|---|---|
| `train` | Lightweight training/evaluation image | Base ML stack (`requirements.txt`) | `subsidy-ml train --config configs/train_cascade.yaml` |
| `api` | Serving image | Full MLOps stack (`requirements.txt` + `requirements-mlops.txt`) | `subsidy-ml serve-api --host 0.0.0.0 --port 8000` |
| `mlops` | Jobs and orchestration tooling | Full MLOps stack | `bash` |

Build examples:
```bash
docker build --target train -t colombia-subsidy-ml:train .
docker build --target api -t colombia-subsidy-ml:api .
docker build --target mlops -t colombia-subsidy-ml:mlops .
```

---

## 10. Docker Compose Profiles
`docker-compose.yml` defines production-friendly profiles:
- `api`: FastAPI online inference service.
- `jobs`: one-off training/anomaly/drift jobs.
- `tracking`: local MLflow server.

Examples:
```bash
# API serving
docker compose up api

# Run training jobs
docker compose --profile jobs run --rm train-cascade
docker compose --profile jobs run --rm train-anomaly

# Drift monitoring job
docker compose --profile jobs run --rm drift-check

# MLflow tracking server
docker compose --profile tracking up mlflow
```

---

## 11. Orchestration & Reproducibility
### DVC pipeline
```bash
dvc repro
```

Stages:
- dataset build
- cascade training
- anomaly training
- cascade evaluation
- drift monitoring

### Kubeflow
```bash
python -m colombia_subsidy_ml compile-kubeflow --output artifacts/kubeflow/subsidy_pipeline.yaml
```

---

## 12. Experiment Tracking
MLflow is optional and controlled by each YAML config under `mlflow:`:
- `enabled`
- `experiment_name`
- `tracking_uri`
- `tags`

When enabled, runs log:
- flattened params
- validation/test metrics
- generated artifacts (models, metadata, reports)

---

## 13. API Deployment
Start API:
```bash
python -m colombia_subsidy_ml serve-api --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `GET /metadata`
- `POST /predict`

Docs:
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Optional artifact override:
```bash
export SUBSIDY_ARTIFACTS_DIR=artifacts/cascade
```

---

## 14. Monitoring & Drift
Run drift detection:
```bash
python -m colombia_subsidy_ml drift-check --config configs/drift.yaml
```

Outputs:
- `artifacts/drift/drift_report.html`
- `artifacts/drift/drift_report.json`
- `artifacts/drift/drift_summary.json`

---

## 15. Artifacts
Typical outputs:
- `artifacts/cascade/` (preprocessor, cascade model, metadata, metrics, split indices)
- `artifacts/anomaly/` (preprocessor, anomaly model, metadata, metrics, split indices)
- `artifacts/predictions.csv`
- `artifacts/drift/*`

---

## 16. Testing
```bash
pytest -q
```

---

## 17. Notebooks
Original notebooks are kept for traceability:
- `notebooks/Subsidy Analysis.ipynb`
- `notebooks/Full Maching Learning Modeling.ipynb`

---

## 18. Roadmap
- Add CI pipeline (lint, tests, build, security scan).
- Add model registry promotion rules per environment.
- Add scheduled drift checks and alerting integration.
- Add canary/champion-challenger deployment policy.

## License
Apache License 2.0. Feel free to use the code and all the pipelines 

