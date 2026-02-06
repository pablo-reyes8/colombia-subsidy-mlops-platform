from colombia_subsidy_ml.pipelines.predict import predict_cascade

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--artifacts-dir", default=None)
    args = parser.parse_args()

    output = predict_cascade(
        args.config,
        input_path=args.input,
        output_path=args.output,
        artifacts_dir=args.artifacts_dir,
    )
    print(f"Predictions saved: {output}")
