# Subsidy Model API

## Run
```bash
python -m colombia_subsidy_ml serve-api --host 0.0.0.0 --port 8000
```

Optional artifact path override:
```bash
export SUBSIDY_ARTIFACTS_DIR=artifacts/cascade
```

## Endpoints

### `GET /health`
Returns service readiness and model load status.

### `GET /metadata`
Returns:
- model name and version
- thresholds for stage 1 and stage 2
- required feature columns

### `POST /predict`
Scores records using the cascade model.

Request body:
```json
{
  "records": [
    {
      "EDAD": 35,
      "SEXO": "M",
      "INGLABO": 1200000
    }
  ]
}
```

Response body:
```json
{
  "model_name": "cascade_classifier",
  "model_version": "v2",
  "thresholds": {
    "stage1": 0.5,
    "stage2": 0.5
  },
  "results": [
    {
      "pred_subsidio": 1,
      "proba_subsidio": 0.812
    }
  ]
}
```

Interactive docs:
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
