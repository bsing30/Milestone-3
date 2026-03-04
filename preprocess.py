"""
Data preprocessing for train pipeline.
Writes versioned outputs for idempotency (same inputs -> same outputs).
"""
import os
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
PROCESSED_DIR = DATA_DIR / "processed"
CONFIG = {"n_samples": 1000, "n_features": 20, "n_informative": 10, "random_state": 42}


def get_data_version():
    """Deterministic version from config for idempotency."""
    return hashlib.sha256(json.dumps(CONFIG, sort_keys=True).encode()).hexdigest()[:12]


def preprocess():
    """Generate and preprocess data. Safe to re-run (idempotent)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    version = get_data_version()
    version_file = PROCESSED_DIR / "data_version.txt"
    if version_file.exists() and version_file.read_text().strip() == version:
        print(f"Data already exists for version {version}, skipping.")
        return version

    X, y = make_classification(**CONFIG)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG["random_state"]
    )

    out = PROCESSED_DIR / version
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_test.npy", X_test)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_test.npy", y_test)
    version_file.write_text(version)
    print(f"Preprocessed data version: {version}")
    return version


if __name__ == "__main__":
    preprocess()
