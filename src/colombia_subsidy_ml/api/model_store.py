from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

from colombia_subsidy_ml.models.io import load_cascade_artifacts
from colombia_subsidy_ml.utils.arrays import to_dense


@dataclass
class CascadeModelService:
    artifacts_dir: Path
    preprocessor: Any
    cascade: Any
    metadata: Dict[str, Any]

    @classmethod
    def load(cls, artifacts_dir: Path) -> "CascadeModelService":
        preprocessor, cascade, metadata = load_cascade_artifacts(artifacts_dir)
        return cls(
            artifacts_dir=Path(artifacts_dir),
            preprocessor=preprocessor,
            cascade=cascade,
            metadata=metadata,
        )

    @property
    def feature_columns(self) -> List[str]:
        return list(self.metadata.get("feature_columns", []))

    def validate_records(self, records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(list(records))
        if df.empty:
            raise ValueError("Request must contain at least one record")

        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required feature columns: {missing}")

        return df[self.feature_columns]

    def predict_records(self, records: Iterable[Dict[str, Any]]) -> pd.DataFrame:
        X = self.validate_records(records)
        X_pp = to_dense(self.preprocessor.transform(X))

        proba = self.cascade.predict_proba(X_pp)[:, 1]
        pred = (proba >= float(self.cascade.threshold_stage2)).astype(int)

        return pd.DataFrame(
            {
                "pred_subsidio": pred.astype(int),
                "proba_subsidio": proba.astype(float),
            }
        )
