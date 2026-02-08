from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np

from colombia_subsidy_ml.models.cascade import CascadeClassifier


def save_cascade_artifacts(
    artifacts_dir: Union[str, Path],
    *,
    preprocessor: Any,
    cascade: CascadeClassifier,
    metadata: Dict[str, Any],
    metrics: Dict[str, Any],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Path:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.joblib")
    joblib.dump(cascade, artifacts_dir / "cascade_model.joblib")
    joblib.dump(cascade.model_stage1, artifacts_dir / "stage1_model.joblib")
    joblib.dump(cascade.model_stage2, artifacts_dir / "stage2_model.joblib")

    np.savez(artifacts_dir / "split_indices.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx)
    (artifacts_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return artifacts_dir


def load_cascade_artifacts(artifacts_dir: Union[str, Path]) -> Tuple[Any, CascadeClassifier, Dict[str, Any]]:
    artifacts_dir = Path(artifacts_dir)
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    preprocessor = joblib.load(artifacts_dir / "preprocessor.joblib")
    metadata = json.loads((artifacts_dir / "metadata.json").read_text(encoding="utf-8"))

    cascade_path = artifacts_dir / "cascade_model.joblib"
    if cascade_path.exists():
        cascade = joblib.load(cascade_path)
    else:
        stage1 = joblib.load(artifacts_dir / "stage1_model.joblib")
        stage2 = joblib.load(artifacts_dir / "stage2_model.joblib")
        cascade = CascadeClassifier(
            stage1,
            stage2,
            threshold_stage1=float(metadata.get("threshold_stage1", 0.5)),
            threshold_stage2=float(metadata.get("threshold_stage2", 0.5)),
        )

    return preprocessor, cascade, metadata
