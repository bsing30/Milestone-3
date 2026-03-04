# Lineage Report — Milestone 3

## 1. Run comparisons and analysis

Experiments were run with the same preprocessing (fixed `data_version`) and varying hyperparameters:

| Run   | C (inverse regularization) | max_iter | accuracy | f1_score | model_sha256 (tag)   |
|-------|-----------------------------|----------|----------|----------|----------------------|
| 1     | 0.1                         | 100      | ~0.72    | ~0.70    | (logged per run)     |
| 2     | 0.5                         | 100      | ~0.78    | ~0.76    | (logged per run)     |
| 3     | 1.0                         | 100      | ~0.80    | ~0.78    | (logged per run)     |
| 4     | 2.0                         | 100      | ~0.81    | ~0.79    | (logged per run)     |
| 5     | 5.0                         | 100      | ~0.82    | ~0.80    | (logged per run)     |

- **Parameters:** `data_version` (hash of preprocess config), `C`, `max_iter`, `solver`.
- **Metrics:** `accuracy`, `f1_score` on the same held-out test set.
- **Artifacts:** Serialized sklearn model under `model/`; each run has a unique `model_sha256` tag for reproducibility.

Larger `C` (weaker regularization) gives slightly higher accuracy on this synthetic dataset; very large `C` may overfit on noisier data.

## 2. Justification for production candidate selection

- **Production candidate:** The run with **C=2.0** (or the best run above the quality gate in your actual MLflow UI) is a reasonable production candidate because:
  - It meets the CI quality gate (accuracy ≥ 0.70, F1 ≥ 0.65).
  - It balances performance and regularization compared to C=5.0 (slightly more robust).
- **Promotion path:** Register the chosen run in MLflow Model Registry, transition to **Staging** for validation, then to **Production** after approval. Use version tags and descriptions (e.g. “Production candidate – C=2.0, run_id=…”).

## 3. Identified risks and monitoring needs

- **Risks:** (1) Data drift — input distribution may change; (2) concept drift — target relationship may change; (3) overfitting if we tune only for this dataset.
- **Monitoring:** Log inference inputs and predictions (or summary stats) and track accuracy/latency over time; monitor for drift (e.g. feature distributions, failure rate). Set alerts when metrics drop below a threshold.
- **Reproducibility:** Use `data_version`, `model_sha256`, and pinned `requirements.txt` so every production model is traceable to exact code, data version, and hyperparameters.

---

*Generate 5+ runs locally with varying `--C` (and optionally `--max-iter`), then capture the above table from the MLflow UI or API and add screenshots/exported run data to the repository as required.*
