from colombia_subsidy_ml.mlops.kubeflow_pipeline import compile_kubeflow_pipeline

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/kubeflow/subsidy_pipeline.yaml")
    args = parser.parse_args()

    output = compile_kubeflow_pipeline(args.output)
    print(f"Kubeflow pipeline compiled: {output}")
