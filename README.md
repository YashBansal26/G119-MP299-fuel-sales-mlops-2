# MP299 — Fuel Station Daily Sales (Dehradun)

SCSE3040 MLOps mini project.

| | |
|---|---|
| **Group** | G119 |
| **Problem** | MP299 |
| **Faculty** | Pulkit Dwivedi |
| **Batch** | 59 |

## Members

| Enrolment no. | Name | Role | Owns |
|---|---|---|---|
| S24CSEU0231 | Yash Bansal | **Data lead** | DVC setup, the seven data checks (`src/fuel_sales/data_checks.py`), cleaning code, data part of `reports/findings.yaml` |
| S24CSEU0449 | Sakshi | Model lead | training, MLflow tracking and registry, holdout prediction |
| S24CSEU0076 | Aaryan | Service lead | FastAPI app, request rules, tests, Dockerfile |
| S24CSEU1801 | Aryan Bansal | Ops lead | GitHub Actions, Kubernetes files, logging, drift and retraining |

## The problem

The Dehradun Fuel Retailers' Federation wants to predict each station's
`litres_per_day` so tanker orders (placed two days ahead) are right-sized.
Regression, scored with MAE in litres.

## Layout

```
params.yaml                 settings (paths, columns, check thresholds)
src/fuel_sales/config.py    loads params.yaml, reads the data files
src/fuel_sales/data_checks.py   finds the seven data problems in train.csv
src/fuel_sales/clean.py     fixes them -> data/processed/train_clean.csv
tests/                      pytest tests (no CSVs needed)
data/raw/*.csv.dvc          DVC pointers; the CSVs are not in git
data/processed/train_clean.csv.dvc   DVC pointer to the cleaned training data
reports/findings.yaml       the filled-in findings form
reports/data_checks.json    evidence written by data_checks.py
reports/cleaning_report.json    what clean.py changed
src/fuel_sales/model.py     the model pipeline (impute + one-hot + regressor)
src/fuel_sales/train.py     trains v1 with MLflow, registers @champion
mlflow.db                   MLflow tracking + registry (SQLite)
models/model.joblib         the champion model the API loads
app/main.py                 FastAPI service: POST /predict, GET /health
app/logging_config.py       JSON logging to stdout
Dockerfile, .dockerignore   container image mp299-api:1.0
.github/workflows/ci.yml    runs pytest on every push
k8s/                        Deployment mp299-api (4 replicas) + Service mp299-svc
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

The CSVs are tracked by DVC, not git. After cloning, copy the three original
files from the problem zip into `data/raw/` (never open/save them in Excel),
then check they match:

```bash
dvc status          # should say "Data and pipelines are up to date."
```

## Stage 1 — data problems

```bash
python src/fuel_sales/data_checks.py
pytest
```

What the checks found in `train.csv` (1000 rows):

| Problem | Present | Details |
|---|---|---|
| leaking_column | no | every non-target column is also in `holdout_features.csv`; no feature correlates ≥ 0.98 with the target |
| unit_mixup | no | per-`zone` min/mean/max of every numeric column are on the same scale |
| sentinel_value | **yes** | `staff` = **999** in **66** rows (real values are 3–20) |
| duplicate_rows | no | 0 exact duplicates, 0 duplicates ignoring `record_id` |
| stuck_sensor | **yes** | `nearby_pumps` stuck at 6 for 100 rows, **2026-04-06 → 2026-04-24** |
| target_typos | **yes** | **12** rows above a gap (31,343 → 101,170); they are ~10× the normal level |
| spelling_variants | no | `ownership` and `zone` have the same 3 clean levels as `later.csv` |

`data_sha256` of the original `train.csv`:
`7d5e7fb87a385182fc2a2372e6cd4a5f8b48220316b37160299e895cc8e86711`

## Stage 2 — cleaning (data lead)

```bash
python src/fuel_sales/clean.py
dvc add data/processed/train_clean.csv
```

`clean.py` runs the seven checks, then fixes only what they found:

| Problem | Fix | Rows |
|---|---|---|
| `staff` = 999 | set to blank, filled with the median staff (12) | 66 |
| `nearby_pumps` stuck at 6 (2026-04-06 → 2026-04-24) | set to blank, filled with the median (5) | 100 |
| target typed 10× too big | divided by 10 (the cause is known, so the row is kept) | 12 |

No rows are dropped: 1000 in, 1000 out. `validate()` then confirms no
blanks, no sentinel, no target gap and no duplicates are left, and the
script stops with an error if any are.

For the model lead: training code should use

```python
from fuel_sales.clean import clean_train, clean_features
```

or read `data/processed/train_clean.csv`. `clean_features()` turns
`staff = 999` into a blank in new data (later.csv, holdout, API requests),
so the model pipeline should include an imputer.

## Stage 2 — training v1 with MLflow

```bash
python src/fuel_sales/train.py
mlflow ui --backend-store-uri sqlite:///mlflow.db     # optional: browse the runs
```

Each candidate in `params.yaml` is one MLflow run in experiment
`mp299-fuel-sales`. Every run logs `data_sha256` (SHA-256 of the original
`train.csv`), its settings, and `mae_cv` (5-fold cross-validated MAE on the
cleaned data, in litres). The model is fitted on all 1000 cleaned rows.

| Run | mae_cv (litres) |
|---|---|
| `v1_ridge` | 962 |
| `v1_random_forest_200` | 1,399 |
| `v1_random_forest_400_deep` | 1,383 |
| `v1_gradient_boosting` | 1,031 |
| `v1_hist_gradient_boosting` | 1,065 |

The lowest `mae_cv` wins: **ridge** (run `9f4de37350ea448b9aa8bb3d04e78f22`). It
is registered as `mp299-fuel-sales-model` version 1 with alias
`@champion`, and saved to `models/model.joblib` for the API.

The model pipeline imputes blanks with the median and ignores unseen
categories, so a request with `staff` missing or a new `zone` value still
gets a prediction.

The model files MLflow writes go to `mlruns/`, which is in `.gitignore`;
only `mlflow.db` is committed, as the handbook asks.

## Stage 3 — API, container, CI, Kubernetes

### Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to try it. Example request:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"highway_distance_km": 7.34, "nearby_pumps": 5, "price_rs": 94.25, "trucks_passing": 1868,
       "temp_c": 45.3, "staff": 9, "ownership": "public_sector", "zone": "City"}'
# {"prediction": 29194.3, "litres_per_day": 29194.3, "model_version": "1"}
```

The client's rules are in the pydantic schema (`StationDay` in `app/main.py`),
so a bad request gets **422** and never reaches the model:

| Rule | Field |
|---|---|
| `temp_c` above 75 is impossible | `Field(..., le=75)` |
| `highway_distance_km` below 0 is a typing mistake | `Field(..., ge=0)` |

`ownership` and `zone` only accept the three known values. A `staff` of 999
(the sentinel found in Stage 1) is treated as blank and imputed by the model.

Every prediction writes one JSON line to stdout, for example:

```json
{"time": "...", "level": "INFO", "logger": "mp299-api", "message": "prediction", "event": "prediction",
 "zone": "City", "ownership": "public_sector", "prediction": 29194.3, "model_version": "1", "latency_ms": 2.1}
```

### Docker

```bash
docker build -t mp299-api:1.0 .
docker run -p 8000:8000 mp299-api:1.0
```

### CI

`.github/workflows/ci.yml` runs on every push: Python 3.12, `pip install -r
requirements.txt`, `pytest`. The tests build their own small data and use the
committed `models/model.joblib`, so no CSVs are needed.

### Kubernetes

```bash
kubectl apply -f k8s/
```

| Thing | Value |
|---|---|
| Deployment | `mp299-api`, 4 replicas, image `mp299-api:1.0`, container port 8000 |
| Service | `mp299-svc`, port 80 → 8000, selector `app: mp299-api` |
| Probes | readiness and liveness on `/health`; liveness waits longer (20 s vs 5 s) |
