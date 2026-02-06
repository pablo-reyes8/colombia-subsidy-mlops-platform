from __future__ import annotations

import argparse
from pathlib import Path

from colombia_subsidy_ml.pipelines.dataset import run_dataset_pipeline
from colombia_subsidy_ml.pipelines.evaluate import evaluate_cascade
from colombia_subsidy_ml.pipelines.predict import predict_cascade
from colombia_subsidy_ml.pipelines.train_anomaly import train_anomaly
from colombia_subsidy_ml.pipelines.train_cascade import train_cascade


def _path(p: str) -> str:
    return str(Path(p))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="subsidy-ml", description="GEIH subsidy ML pipelines")
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build-dataset", help="Build processed dataset from raw tables")
    p_build.add_argument("--config", required=True)

    p_train = sub.add_parser("train", help="Train cascade model")
    p_train.add_argument("--config", required=True)

    p_train_anom = sub.add_parser("train-anomaly", help="Train anomaly model")
    p_train_anom.add_argument("--config", required=True)

    p_eval = sub.add_parser("evaluate", help="Evaluate cascade model")
    p_eval.add_argument("--config", required=True)
    p_eval.add_argument("--artifacts-dir", default=None)

    p_pred = sub.add_parser("predict", help="Run predictions using cascade model")
    p_pred.add_argument("--config", required=True)
    p_pred.add_argument("--input", required=True)
    p_pred.add_argument("--output", required=True)
    p_pred.add_argument("--artifacts-dir", default=None)

    args = parser.parse_args(argv)

    if args.command == "build-dataset":
        output = run_dataset_pipeline(args.config)
        print(f"Dataset built: {output}")
        return 0

    if args.command == "train":
        result = train_cascade(args.config)
        print(f"Artifacts: {result['artifacts_dir']}")
        return 0

    if args.command == "train-anomaly":
        result = train_anomaly(args.config)
        print(f"Artifacts: {result['artifacts_dir']}")
        return 0

    if args.command == "evaluate":
        metrics = evaluate_cascade(args.config, artifacts_dir=args.artifacts_dir)
        print(metrics)
        return 0

    if args.command == "predict":
        output = predict_cascade(
            args.config,
            input_path=args.input,
            output_path=args.output,
            artifacts_dir=args.artifacts_dir,
        )
        print(f"Predictions saved: {output}")
        return 0

    parser.print_help()
    return 1
