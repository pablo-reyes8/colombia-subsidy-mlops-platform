from __future__ import annotations

from pathlib import Path
from typing import Union

from colombia_subsidy_ml.config import load_yaml
from colombia_subsidy_ml.data.build import build_from_config, save_dataset


def run_dataset_pipeline(config_path: Union[str, Path]) -> Path:
    config = load_yaml(config_path)
    df = build_from_config(config)
    output_path = Path(config["processed_path"])
    save_dataset(df, output_path)
    return output_path
