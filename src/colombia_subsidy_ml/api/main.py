from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from colombia_subsidy_ml.api.model_store import CascadeModelService
from colombia_subsidy_ml.api.schemas import (
    HealthResponse,
    MetadataResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)


def _default_artifacts_dir() -> Path:
    return Path(os.getenv("SUBSIDY_ARTIFACTS_DIR", "artifacts/cascade"))


app = FastAPI(
    title="Colombia Subsidy ML API",
    description=(
        "Inference API for the two-stage subsidy cascade model. "
        "Use /metadata to inspect required features and /predict to score records."
    ),
    version="0.2.0",
)

_model_service: Optional[CascadeModelService] = None


@app.on_event("startup")
def _startup_load_model() -> None:
    global _model_service
    artifacts_dir = _default_artifacts_dir()
    if not artifacts_dir.exists():
        _model_service = None
        return
    _model_service = CascadeModelService.load(artifacts_dir)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts_dir = _default_artifacts_dir()
    return HealthResponse(
        status="ok",
        artifacts_dir=str(artifacts_dir),
        model_loaded=(_model_service is not None),
    )


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    if _model_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model and ensure artifacts are present.")

    md = _model_service.metadata
    return MetadataResponse(
        model_name="cascade_classifier",
        model_version=md.get("model_version"),
        target_col=str(md.get("target_col", "Subsidio")),
        threshold_stage1=float(md.get("threshold_stage1", 0.5)),
        threshold_stage2=float(md.get("threshold_stage2", 0.5)),
        feature_columns=list(md.get("feature_columns", [])),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if _model_service is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Train model and ensure artifacts are present.")

    try:
        scored = _model_service.predict_records(payload.records)
    except KeyError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    items = [
        PredictionItem(
            pred_subsidio=int(row.pred_subsidio),
            proba_subsidio=float(row.proba_subsidio),
        )
        for row in scored.itertuples(index=False)
    ]

    return PredictionResponse(
        model_name="cascade_classifier",
        model_version=_model_service.metadata.get("model_version"),
        thresholds={
            "stage1": float(_model_service.cascade.threshold_stage1),
            "stage2": float(_model_service.cascade.threshold_stage2),
        },
        results=items,
    )
