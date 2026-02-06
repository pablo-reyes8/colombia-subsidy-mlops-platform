from colombia_subsidy_ml.pipelines.train_cascade import train_cascade

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = train_cascade(args.config)
    print(f"Artifacts: {result['artifacts_dir']}")
