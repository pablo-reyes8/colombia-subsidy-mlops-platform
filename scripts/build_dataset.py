from colombia_subsidy_ml.pipelines.dataset import run_dataset_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    output = run_dataset_pipeline(args.config)
    print(f"Dataset built: {output}")
