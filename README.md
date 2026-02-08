# Colombia Subsidy ML (GEIH)

MLOps-ready project for subsidy prediction with severe class imbalance. The exploratory notebooks are preserved in `notebooks/`, and the production implementation is in `src/` with configurable pipelines, artifact versioning, API serving, experiment tracking, and drift monitoring.

## Main Capabilities
- Modular data pipeline to build `Base_Modelo_Subsidios.csv`.
- Two-stage cascade model (XGBoost + Random Forest) with:
  - SMOTE support
  - polynomial feature expansion
  - optional PCA
  - hyperparameter search by stage
  - threshold tuning under recall constraints
- Anomaly baseline (Isolation Forest / One-Class SVM) with tunable search.
- Optional MLflow experiment tracking.
- Evidently drift reports (HTML + JSON summary).
- FastAPI inference service with documented schemas.
- Reproducibility via DVC (`dvc.yaml`) and Kubeflow pipeline compilation.

## Project Structure
```text
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
