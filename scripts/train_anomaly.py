from colombia_subsidy_ml.pipelines.train_anomaly import train_anomaly

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = train_anomaly(args.config)
    print(f"Artifacts: {result['artifacts_dir']}")
