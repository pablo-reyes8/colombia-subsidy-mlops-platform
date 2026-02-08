from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    records: List[Dict[str, Any]] = Field(
        ...,
        description="Rows to score. Each record must include all training feature columns.",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "records": [
                    {"EDAD": 35, "SEXO": "M", "INGLABO": 1200000},
                    {"EDAD": 27, "SEXO": "F", "INGLABO": 700000},
                ]
            }
        }


class PredictionItem(BaseModel):
    pred_subsidio: int
    proba_subsidio: float


class PredictionResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    thresholds: Dict[str, float]
    results: List[PredictionItem]


class MetadataResponse(BaseModel):
    model_name: str
    model_version: Optional[str] = None
    target_col: str
    threshold_stage1: float
    threshold_stage2: float
    feature_columns: List[str]


class HealthResponse(BaseModel):
    status: str
    artifacts_dir: str
    model_loaded: bool
