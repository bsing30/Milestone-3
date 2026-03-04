"""
Generate 5+ MLflow runs with varying hyperparameters for Milestone 3.
Then register the best run and transition to Staging -> Production.
"""
import os
import subprocess
import sys

# Ensure we run from repo root
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def run(cmd, env=None):
    e = os.environ.copy()
    if env:
        e.update(env)
    r = subprocess.run(cmd, shell=True, env=e)
    if r.returncode != 0:
        sys.exit(r.returncode)

print("Preprocessing...")
run("python preprocess.py")

print("Running 5 experiments with different C...")
for C in [0.1, 0.5, 1.0, 2.0, 5.0]:
    run(f"python train.py --C {C} --max-iter 100")

print("Validating (quality gate)...")
run("python model_validation.py")

print("Registering model and transitioning to Staging...")
run("python register_model.py")

print("Done. Open MLflow UI (mlflow ui) to view runs and promote a version to Production.")
