from colombia_subsidy_ml.pipelines.evaluate import evaluate_cascade

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()

    metrics = evaluate_cascade(args.config, artifacts_dir=args.artifacts_dir)
    print(metrics)
