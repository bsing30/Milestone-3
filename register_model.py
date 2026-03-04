"""
Register the trained model to MLflow Model Registry and transition to Staging.
Reads run_id from file written by train.py (or passed via env).
"""
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
RUN_ID_FILE = Path(os.environ.get("RUN_ID_FILE", "data/latest_run_id.txt"))
MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "milestone3_classifier")


def register_model(run_id=None):
    """Register model from run to registry and transition to Staging."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    if run_id is None and RUN_ID_FILE.exists():
        run_id = RUN_ID_FILE.read_text().strip()
    if not run_id:
        raise ValueError("No run_id provided and RUN_ID_FILE not found. Run train first.")

    run = client.get_run(run_id)
    artifact_uri = run.info.artifact_uri
    model_uri = f"{artifact_uri}/model"

    result = mlflow.register_model(model_uri, MODEL_NAME)
    version = result.version
    client.transition_model_version_stage(MODEL_NAME, version, "Staging")
    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description="Promoted from pipeline run.",
    )
    print(f"Registered model '{MODEL_NAME}' version {version} -> Staging")
    return version


if __name__ == "__main__":
    register_model()
