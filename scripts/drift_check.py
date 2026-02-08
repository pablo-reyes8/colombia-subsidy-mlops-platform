from colombia_subsidy_ml.pipelines.drift import run_drift_check

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    result = run_drift_check(args.config)
    print(f"Drift report dir: {result['output_dir']}")
    print(result["summary"])
