"""
Threshold-based model validation (quality gate).
Exits with 1 if metrics are below thresholds; CI uses this to fail the pipeline.
"""
import argparse
import os
import sys

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.70"))
MIN_F1 = float(os.environ.get("MIN_F1", "0.65"))


def validate_run(run_id=None, min_accuracy=MIN_ACCURACY, min_f1=MIN_F1):
    """Load run from MLflow and check metrics against thresholds."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if run_id is None:
        # Use latest run from default experiment
        exp = client.get_experiment_by_name("milestone3_experiment")
        if exp is None:
            print("FAILED: No experiment 'milestone3_experiment' found.")
            sys.exit(1)
        runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=1)
        if not runs:
            print("FAILED: No runs found.")
            sys.exit(1)
        run = runs[0]
        run_id = run.info.run_id

    run = client.get_run(run_id)
    metrics = run.data.metrics
    accuracy = metrics.get("accuracy")
    f1 = metrics.get("f1_score")

    if accuracy is None:
        print("FAILED: No 'accuracy' metric in run.")
        sys.exit(1)
    if f1 is None:
        print("FAILED: No 'f1_score' metric in run.")
        sys.exit(1)

    failed = False
    if accuracy < min_accuracy:
        print(f"FAILED: Accuracy {accuracy:.4f} below threshold {min_accuracy}")
        failed = True
    if f1 < min_f1:
        print(f"FAILED: F1 score {f1:.4f} below threshold {min_f1}")
        failed = True

    if failed:
        sys.exit(1)
    print(f"PASSED: accuracy={accuracy:.4f}, f1={f1:.4f} (min_accuracy={min_accuracy}, min_f1={min_f1})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--min-accuracy", type=float, default=MIN_ACCURACY)
    parser.add_argument("--min-f1", type=float, default=MIN_F1)
    args = parser.parse_args()
    validate_run(run_id=args.run_id, min_accuracy=args.min_accuracy, min_f1=args.min_f1)
