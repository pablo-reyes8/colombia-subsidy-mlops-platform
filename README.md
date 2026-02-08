# Colombia Subsidy ML (GEIH)

![Repo size](https://img.shields.io/github/repo-size/pablo-reyes8/colombia-subsidy-ml-prediction)
![Last commit](https://img.shields.io/github/last-commit/pablo-reyes8/colombia-subsidy-ml-prediction)
![Open issues](https://img.shields.io/github/issues/pablo-reyes8/colombia-subsidy-ml-prediction)
![Contributors](https://img.shields.io/github/contributors/pablo-reyes8/colombia-subsidy-ml-prediction)
![Forks](https://img.shields.io/github/forks/pablo-reyes8/colombia-subsidy-ml-prediction?style=social)
![Stars](https://img.shields.io/github/stars/pablo-reyes8/colombia-subsidy-ml-prediction?style=social)








A professional, reproducible MLOps-style project built on Colombia’s GEIH household survey to study whether subsidies reduce inequality and to predict potential subsidy candidates under extreme class imbalance. The original exploratory notebooks are preserved in `notebooks/`, while the production path is modularized into pipelines, configs, and CLI tools.

---

## Table of Contents
1. Overview
2. Descriptive Analysis (Selected Figures)
3. Project Structure
4. Data & Inputs
5. Pipelines & CLI
6. Modeling Approach
7. Configuration
8. Artifacts & Outputs
9. Tests
10. Docker
11. Notebooks
12. Roadmap

---

## 1. Overview
This repository delivers:
- A dataset build pipeline that consolidates raw GEIH tables into a modeling-ready file.
- Feature engineering and preprocessing in a reusable package.
- A two-stage cascade model optimized for recall/precision tradeoffs on the minority class.
- Anomaly detection baselines (Isolation Forest / One-Class SVM) for high-precision targeting.
- Config-driven training, evaluation, and inference from the command line.

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

## 3. Project Structure
```
.
├─ artifacts/
├─ configs/
├─ data/
├─ docs/
├─ notebooks/
├─ scripts/
├─ src/colombia_subsidy_ml/
│  ├─ api/
│  ├─ data/
│  ├─ features/
│  ├─ mlops/
│  ├─ models/
│  ├─ pipelines/
│  ├─ tracking/
│  └─ utils/
├─ tests/
├─ dvc.yaml
├─ Dockerfile
└─ pyproject.toml
```

## Installation
Base project:
```bash
pip install -e .
```

With MLOps extras (MLflow, Evidently, FastAPI, Kubeflow, DVC):
```bash
pip install -e ".[mlops]"
```

## CLI Workflows
Build dataset:
```bash
python -m colombia_subsidy_ml build-dataset --config configs/dataset.yaml
```

Train cascade:
```bash
python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
```

Train anomaly model:
```bash
python -m colombia_subsidy_ml train-anomaly --config configs/train_anomaly.yaml
```

Evaluate cascade:
```bash
python -m colombia_subsidy_ml evaluate --config configs/train_cascade.yaml
```

Batch predict:
```bash
python -m colombia_subsidy_ml predict \
  --config configs/train_cascade.yaml \
  --input data/processed/Base_Modelo_Subsidios.csv \
  --output artifacts/predictions.csv
```

Drift report with Evidently:
```bash
python -m colombia_subsidy_ml drift-check --config configs/drift.yaml
```

Compile Kubeflow pipeline:
```bash
python -m colombia_subsidy_ml compile-kubeflow --output artifacts/kubeflow/subsidy_pipeline.yaml
```

Serve API:
```bash
python -m colombia_subsidy_ml serve-api --host 0.0.0.0 --port 8000
```

## FastAPI Endpoints
- `GET /health`: service and model readiness.
- `GET /metadata`: required features, thresholds, and model metadata.
- `POST /predict`: score a list of records.

Interactive docs:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

Set a custom artifact location for API serving:
```bash
export SUBSIDY_ARTIFACTS_DIR=artifacts/cascade
```

## Configs
- `configs/train_cascade.yaml`: robust supervised pipeline with feature engineering, search, threshold tuning, and optional MLflow.
- `configs/train_anomaly.yaml`: anomaly model + search + optional MLflow.
- `configs/drift.yaml`: reference/current dataset paths and Evidently output.

## Reproducibility
Run DVC stages:
```bash
dvc repro
```

Stages included:
- dataset build
- cascade training
- anomaly training
- cascade evaluation
- drift monitoring

## Testing
```bash
pytest -q
```

## Docker
```bash
docker build -t colombia-subsidy-ml .
docker run --rm -v "$PWD":/app colombia-subsidy-ml \
  python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
```

## Notebooks
Original notebooks are still available for research traceability:
- `notebooks/Subsidy Analysis.ipynb`
- `notebooks/Full Maching Learning Modeling.ipynb`

## License
Apache License 2.0
