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

Selected figures from the original descriptive notebook. These plots are intentionally curated (not exhaustive) and arranged for readability.

<p align="center">
  <img src="results\Boxplot Subs.png" alt="Descriptive plot 1" width="460" />
  &nbsp;&nbsp;&nbsp;
  <img src="docs/assets/analysis_plot_2.png" alt="Descriptive plot 2" width="460" />
</p>
<p align="center">
  <sub><b>Figure 2.1.</b> Summary distribution & key diagnostics (left) and complementary descriptive patterns (right).</sub>
</p>

<br/>

<p align="center">
  <img src="docs/assets/analysis_plot_3.png" alt="Descriptive plot 3" width="460" />
  &nbsp;&nbsp;&nbsp;
  <img src="docs/assets/analysis_plot_4.png" alt="Descriptive plot 4" width="460" />
</p>
<p align="center">
  <sub><b>Figure 2.2.</b> Additional distributional comparisons and subgroup contrasts.</sub>
</p>

<br/>

<!-- Full-width highlight figure -->
<p align="center">
  <!-- Option A (recommended): rename the file to avoid spaces -->
  <!-- <img src="results/geographical_distribution_of_subs.png" alt="Geographical distribution of subsidies" width="940" /> -->

  <!-- Option B: keep the original filename (URL-encoded space) -->
  <img src="results/Geographical%20distribution%20of%20subs.png" alt="Geographical distribution of subsidies" width="940" />
</p>
<p align="center">
  <sub><b>Figure 2.3.</b> Geographical distribution of subsidies across departments (counts and total value).</sub>
</p>

<!-- Optional: collapse extra plots if you want a cleaner main README -->
<!--
<details>
  <summary><b>More descriptive figures</b> (click to expand)</summary>
  <br/>
  <p align="center">
    <img src="docs/assets/analysis_plot_5.png" width="460" />
    &nbsp;&nbsp;&nbsp;
    <img src="docs/assets/analysis_plot_6.png" width="460" />
  </p>
</details>
-->
---

## 3. Project Structure
```
.
├─ artifacts/                 # model artifacts (ignored by git)
├─ configs/                   # YAML configs for data & models
├─ data/
│  ├─ raw/                    # raw GEIH tables (CSV.rar or extracted CSVs)
│  └─ processed/              # consolidated modeling dataset
├─ docs/assets/               # images used in README
├─ notebooks/                 # original exploratory notebooks
├─ scripts/                   # thin CLI wrappers
├─ src/colombia_subsidy_ml/    # package (pipelines, models, utils)
├─ tests/                     # pytest
├─ Dockerfile
├─ pyproject.toml
└─ README.md
```

---

## 4. Data & Inputs
- **Raw GEIH tables** should be placed in `data/raw/`.
- Expected file names (see `configs/dataset.yaml`):
  - `generales.csv`
  - `laborales.csv`
  - `hogar.csv`
  - `subsidios.csv`
  - `fuerza_trabajo.csv`
  - `desempleados.csv`
- The consolidated dataset is saved to `data/processed/Base_Modelo_Subsidios.csv`.

---

## 5. Pipelines & CLI
You can run everything from the CLI (no notebooks required).

### Install
```bash
pip install -e .
```

### Build dataset
```bash
python -m colombia_subsidy_ml build-dataset --config configs/dataset.yaml
```

### Train cascade model
```bash
python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
```

### Evaluate
```bash
python -m colombia_subsidy_ml evaluate --config configs/train_cascade.yaml
```

### Predict
```bash
python -m colombia_subsidy_ml predict \
  --config configs/train_cascade.yaml \
  --input data/processed/Base_Modelo_Subsidios.csv \
  --output artifacts/predictions.csv
```

---

## 6. Modeling Approach
### Two-Stage Cascade (Supervised)
- **Stage 1**: High-recall classifier (XGBoost) to filter candidates.
- **Stage 2**: Precision-focused classifier (Random Forest) refines candidates.
- Thresholds are configurable and can be tuned to meet recall constraints.

### Anomaly Detection (Unsupervised)
- **Isolation Forest** and **One-Class SVM** treat subsidy recipients as anomalies.
- Provides a high-precision option when false positives are costly.

---

## 7. Configuration
All training and inference behavior is controlled in YAML:
- `configs/dataset.yaml`: raw table mapping and processing output path.
- `configs/train_cascade.yaml`: train/val/test split, preprocessing, SMOTE, model params.
- `configs/train_anomaly.yaml`: anomaly model configuration.

---

## 8. Artifacts & Outputs
Artifacts are written to `artifacts/`:
- `preprocessor.joblib`
- `stage1_model.joblib`
- `stage2_model.joblib`
- `metadata.json`
- `metrics.json`
- `split_indices.npz`

---

## 9. Tests
```bash
pytest -q
```

---

## 10. Docker
```bash
docker build -t colombia-subsidy-ml .
docker run --rm -v "$PWD":/app colombia-subsidy-ml \
  python -m colombia_subsidy_ml train --config configs/train_cascade.yaml
```

---

## 11. Notebooks
Original notebooks remain available for reference:
- `notebooks/Subsidy Analysis.ipynb`
- `notebooks/Full Maching Learning Modeling.ipynb`

---

## 12. Roadmap
- Add experiment tracking (MLflow or Weights & Biases).
- Add data validation (Great Expectations).
- Add model registry and versioning.
- Add CI pipeline for tests and linting.

---

## License
Apache License 2.0
