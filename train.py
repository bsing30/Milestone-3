"""
Training script with MLflow tracking.
Logs params, metrics, artifacts, and model. Used by DAG and CI.
"""
import argparse
import os
import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
PROCESSED_DIR = DATA_DIR / "processed"
RUN_ID_FILE = Path(os.environ.get("RUN_ID_FILE", "data/latest_run_id.txt"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")


def load_data(data_version=None):
    """Load preprocessed data by version."""
    if data_version is None:
        vf = PROCESSED_DIR / "data_version.txt"
        if not vf.exists():
            raise FileNotFoundError("Run preprocess.py first or pass --data-version")
        data_version = vf.read_text().strip()
    base = PROCESSED_DIR / data_version
    X_train = np.load(base / "X_train.npy")
    X_test = np.load(base / "X_test.npy")
    y_train = np.load(base / "y_train.npy")
    y_test = np.load(base / "y_test.npy")
    return X_train, X_test, y_train, y_test, data_version


def train(
    data_version=None,
    learning_rate=0.01,
    max_iter=100,
    C=1.0,
    experiment_name="milestone3_experiment",
):
    """Train model and log to MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    X_train, X_test, y_train, y_test, data_version = load_data(data_version)

    params = {
        "data_version": data_version,
        "max_iter": max_iter,
        "C": C,
        "solver": "lbfgs",
    }
    # sklearn LogisticRegression doesn't have learning_rate; log for lineage
    params["learning_rate_log"] = learning_rate

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        model = LogisticRegression(max_iter=max_iter, C=C, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metrics({"accuracy": accuracy, "f1_score": f1})
        mlflow.sklearn.log_model(model, "model")

        # Artifact hashing for reproducibility
        import pickle
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            pickle.dump(model, f)
            path = f.name
        try:
            with open(path, "rb") as f:
                model_hash = hashlib.sha256(f.read()).hexdigest()
            mlflow.set_tag("model_sha256", model_hash)
        finally:
            os.unlink(path)

        run_id = run.info.run_id
        RUN_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
        RUN_ID_FILE.write_text(run_id)
        print(f"Run ID: {run_id}, accuracy: {accuracy:.4f}, f1: {f1:.4f}")
        return run_id, accuracy, f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", type=str, default=None)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--experiment-name", type=str, default="milestone3_experiment")
    args = parser.parse_args()
    train(
        data_version=args.data_version,
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        C=args.C,
        experiment_name=args.experiment_name,
    )
