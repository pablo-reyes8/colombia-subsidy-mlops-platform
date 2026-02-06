# Colombia Subsidy ML (GEIH)

Professionalized MLOps-style project for analyzing GEIH data and training models to identify subsidy-eligible households under extreme class imbalance. The original notebooks are preserved under `notebooks/`, while the production path is now organized into reproducible scripts, configs, and pipelines.

## What This Repo Now Provides
- Reproducible dataset build pipeline from raw GEIH tables.
- Modular feature engineering and preprocessing.
- Train/evaluate/inference pipelines runnable from CLI or scripts.
- Two-stage cascade model and anomaly detection support.
- Config-driven artifacts and outputs.
- Tests and Docker for portability.

## Project Layout
```
.
├─ artifacts/                 # model artifacts (ignored by git)
├─ configs/                   # yaml configs for data and models
├─ data/
│  ├─ raw/                    # raw GEIH tables (CSV.rar)
│  └─ processed/              # consolidated modeling dataset
├─ notebooks/                 # original exploratory notebooks
├─ scripts/                   # thin CLI wrappers
├─ src/colombia_subsidy_ml/    # package (pipelines, models, utils)
├─ tests/                     # pytest
├─ Dockerfile
├─ pyproject.toml
└─ README.md
```

## Quickstart (CLI)
Install deps and run any pipeline:

```bash
pip install -e .

# Build dataset from raw GEIH tables
python -m colombia_subsidy_ml build-dataset --config configs/dataset.yaml

# Train cascade model
python -m colombia_subsidy_ml train --config configs/train_cascade.yaml

# Evaluate on holdout
python -m colombia_subsidy_ml evaluate --config configs/train_cascade.yaml

# Predict using saved artifacts
python -m colombia_subsidy_ml predict \
  --config configs/train_cascade.yaml \
  --input data/processed/Base_Modelo_Subsidios.csv \
  --output artifacts/predictions.csv
```

## Quickstart (Scripts)
```bash
python scripts/build_dataset.py --config configs/dataset.yaml
python scripts/train.py --config configs/train_cascade.yaml
python scripts/evaluate.py --config configs/train_cascade.yaml
python scripts/predict.py --config configs/train_cascade.yaml \
  --input data/processed/Base_Modelo_Subsidios.csv \
  --output artifacts/predictions.csv
```

## Tests
```bash
pytest -q
```

## Docker
```bash
docker build -t colombia-subsidy-ml .
docker run --rm -v "$PWD":/app colombia-subsidy-ml \
  python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
```

## Notes
- The notebooks are kept intact for traceability.
- You can tune thresholds, models, and resampling via YAML configs in `configs/`.
- Artifacts are saved under `artifacts/`.

