# Milestone 3: Workflow Automation & Experiment Tracking

IDS 568 / MLOps — Airflow DAG, MLflow tracking, and CI/CD with quality gates.

## Repository layout

- **`dags/train_pipeline.py`** — Airflow DAG: `preprocess_data` → `train_model` → `register_model`
- **`preprocess.py`** — Data generation and versioned preprocessing (idempotent)
- **`train.py`** — Training script with MLflow logging (params, metrics, artifacts, model hash)
- **`register_model.py`** — Registers model to MLflow registry and transitions to Staging
- **`model_validation.py`** — Quality gate: fails CI if accuracy/F1 below thresholds
- **`.github/workflows/train_and_validate.yml`** — CI: install → preprocess → train → validate

## Setup

### 1. Python environment

```bash
python -m venv venv
source venv/bin/activate   # or: venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Airflow (local)

```bash
pip install apache-airflow
airflow db init
airflow users create --username admin --password admin \
  --firstname Admin --lastname User --role Admin --email admin@example.com
# In one terminal:
airflow webserver --port 8080
# In another:
airflow scheduler
```

Ensure the project root (this directory) is on `PYTHONPATH` or that Airflow is started from this directory so the DAG can import `preprocess`, `train`, and `register_model`.

### 3. MLflow (local)

```bash
pip install mlflow
# Optional: start UI
mlflow server --host 0.0.0.0 --port 5000
# Set for remote: export MLFLOW_TRACKING_URI=http://localhost:5000
```

By default, runs are stored in `./mlruns`.

## How to run the pipeline

### Option A: Run without Airflow (scripts only)

```bash
python preprocess.py
python train.py --max-iter 150 --C 1.0
python model_validation.py
python register_model.py
```

### Option B: Run via Airflow

1. Start Airflow webserver and scheduler (see Setup).
2. Open http://localhost:8080 and trigger the DAG **train_pipeline**.
3. Tasks run in order: `preprocess_data` → `train_model` → `register_model`.

### Option C: CI (GitHub Actions)

Push to `main`/`master` (or open a PR). The workflow:

1. Installs dependencies from `requirements.txt`
2. Runs `preprocess.py`
3. Runs `train.py` (logs to MLflow in `mlruns/`)
4. Runs `model_validation.py` (quality gate: fails if accuracy &lt; 0.70 or F1 &lt; 0.65)
5. Uploads `mlruns/` as an artifact

## DAG idempotency and lineage

- **Idempotency:** Preprocessing uses a deterministic data version (hash of config). Re-running produces the same `data/processed/<version>/` outputs. Training and registration use that version and log it to MLflow, so re-runs are traceable and repeatable.
- **Lineage:** Each run logs `data_version`, hyperparameters (`max_iter`, `C`), and a `model_sha256` tag. The model registry stores staged versions (None → Staging → Production) with descriptions.

## CI-based model governance

- Every push/PR runs the **train_and_validate** workflow.
- The pipeline **fails** if `model_validation.py` exits with code 1 (metrics below `MIN_ACCURACY` / `MIN_F1`).
- Thresholds are configurable via env: `MIN_ACCURACY`, `MIN_F1` (defaults 0.70, 0.65).
- No model is registered in CI; the DAG (or manual run) performs registration and staging.

## Experiment tracking

- **Experiment name:** `milestone3_experiment`.
- **Logged:** params (e.g. `data_version`, `max_iter`, `C`), metrics (`accuracy`, `f1_score`), model artifact, and tag `model_sha256`.
- **Registry:** Model name `milestone3_classifier`; versions are transitioned to Staging (and optionally Production) via `register_model.py` or MLflow UI.

To generate **5+ runs** for the report (varying hyperparameters):

```bash
python preprocess.py
for C in 0.1 0.5 1.0 2.0 5.0; do
  python train.py --C $C --max-iter 100
done
# Then use MLflow UI or API to compare runs and promote one to Production.
```

## Operational notes

### Retry and failure handling

- **DAG:** `default_args` set `retries=2`, `retry_delay=timedelta(minutes=2)`, and `on_failure_callback` to log failures. Extend the callback for Slack/email if needed.
- **Scripts:** Preprocess/train/register use normal Python exceptions; CI fails on non-zero exit from `model_validation.py`.

### Monitoring and alerting

- Use Airflow UI to monitor DAG and task status and logs.
- For production: add Airflow alerts (e.g. on DAG failure), MLflow monitoring, and metric dashboards (e.g. accuracy over runs).

### Rollback

- **Model:** In MLflow Model Registry, transition the previous production version back to **Production** and the current one to **Archived** (or Staging).
- **Data:** Data is versioned under `data/processed/<version>/`. To “rollback” data, point training to a previous version (e.g. pass `--data-version` to `train.py`).
- **Code:** Revert the repo to a previous commit and re-run the DAG or CI.

---

See **lineage_report.md** for run comparisons and production candidate justification.
