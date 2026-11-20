# MP299 — Fuel Station Daily Sales (Dehradun)

SCSE3040 MLOps mini project.

| | |
|---|---|
| **Group** | G119 |
| **Problem** | MP299 |
| **Faculty** | Pulkit Dwivedi |
| **Batch** | 59 |

## Members

| Enrolment no. | Name | Role |
|---|---|---|
| S24CSEU0449 | Sakshi | Data lead |
| S24CSEU0231 | Yash Bansal | Model lead |
| S24CSEU1801 | Aryan Bansal | Service lead |
| S24CSEU0076 | Aaryan | Ops lead |

(Roles are a suggestion — swap them to match how you split the work.)

## The problem

The Dehradun Fuel Retailers' Federation wants to predict each station's
`litres_per_day` so tanker orders (placed two days ahead) are right-sized.
Regression, scored with MAE in litres.

## Layout

```
params.yaml                 settings (paths, columns, check thresholds)
src/fuel_sales/config.py    loads params.yaml, reads the data files
src/fuel_sales/data_checks.py   finds the seven data problems in train.csv
tests/                      pytest tests (no CSVs needed)
data/raw/*.csv.dvc          DVC pointers; the CSVs are not in git
reports/findings.yaml       the filled-in findings form
reports/data_checks.json    evidence written by data_checks.py
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
