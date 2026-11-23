"""Train v1 on the cleaned train.csv, tracked with MLflow.

Run from the repo root:
    python src/fuel_sales/train.py

For every candidate in params.yaml it starts one MLflow run that logs
  - parameter data_sha256 (SHA-256 of the original data/raw/train.csv)
  - the model settings
  - metric mae_cv (5-fold cross-validated MAE on cleaned train.csv)
  - the fitted model (trained on all cleaned rows)
The run with the lowest mae_cv is registered as a new version of the
registered model and gets the alias `champion`. Its model is also saved to
models/model.joblib for the API.
"""

import hashlib
import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import KFold, cross_val_score

if __package__ in (None, ""):  # allow `python src/fuel_sales/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuel_sales.clean import clean_train  # noqa: E402
from fuel_sales.config import ROOT, load_params, read_csv  # noqa: E402
from fuel_sales.data_checks import run_all  # noqa: E402
from fuel_sales.model import build_pipeline  # noqa: E402


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_clean_train(params):
    train, later, holdout = (read_csv(k, params) for k in ("train", "later", "holdout"))
    findings = run_all(train, later, holdout, params)
    clean, _ = clean_train(train, findings, params, later)
    return clean


def cv_mae(pipe, X, y, folds, seed):
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error")
    return float(-scores.mean()), float(scores.std())


def main():
    params = load_params()
    mcfg, tcfg = params["mlflow"], params["train"]
    num, cat, target = params["numeric_features"], params["categorical_features"], params["target"]

    data_sha = sha256_of(ROOT / params["data"]["train"])
    df = load_clean_train(params)
    X, y = df[num + cat].astype({c: float for c in num}), df[target]  # floats: API input may have blanks

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])

    results = []
    for cand in tcfg["candidates"]:
        pipe = build_pipeline(num, cat, cand["model"], cand["params"], tcfg["random_state"])
        with mlflow.start_run(run_name=f"v1_{cand['name']}") as run:
            mlflow.set_tag("model_version_role", "v1")
            mlflow.log_param("data_sha256", data_sha)
            mlflow.log_param("model_type", cand["model"])
            mlflow.log_params({f"model__{k}": v for k, v in (cand["params"] or {}).items()})
            mlflow.log_param("train_rows", len(df))
            mlflow.log_param("features", ",".join(num + cat))

            mae, mae_std = cv_mae(pipe, X, y, tcfg["cv_folds"], tcfg["random_state"])
            pipe.fit(X, y)
            mae_train = float(np.mean(np.abs(pipe.predict(X) - y)))
            mlflow.log_metrics({"mae_cv": mae, "mae_cv_std": mae_std, "mae_train": mae_train})

            info = mlflow.sklearn.log_model(pipe, name="model",
                                     signature=infer_signature(X.head(), pipe.predict(X.head())),
                                     input_example=X.head(3),
                                     serialization_format="cloudpickle")
            results.append({"name": cand["name"], "run_id": run.info.run_id, "mae_cv": mae, "pipe": pipe,
                            "model_uri": info.model_uri})
            print(f"{cand['name']:24s} mae_cv={mae:8.1f}  run_id={run.info.run_id}")

    best = min(results, key=lambda r: r["mae_cv"])
    print(f"\nBest: {best['name']}  mae_cv={best['mae_cv']:.1f}")

    # register the best run's model and make it the champion
    mv = mlflow.register_model(best["model_uri"], mcfg["registered_model_name"])
    client = MlflowClient()
    client.set_registered_model_alias(mcfg["registered_model_name"], mcfg["champion_alias"], mv.version)
    client.set_model_version_tag(mcfg["registered_model_name"], mv.version, "role", "v1")
    print(f"Registered {mcfg['registered_model_name']} version {mv.version} as @{mcfg['champion_alias']}")

    # model file for the API
    model_path = ROOT / tcfg["model_path"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["pipe"], model_path)

    summary = {
        "experiment_name": mcfg["experiment_name"],
        "best_run_id": best["run_id"],
        "best_model": best["name"],
        "registered_model_name": mcfg["registered_model_name"],
        "champion_version": int(mv.version),
        "data_sha256": data_sha,
        "runs": [{k: r[k] for k in ("name", "run_id", "mae_cv")} for r in results],
    }
    (ROOT / "reports" / "training_v1.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                       encoding="utf-8", newline="\n")
    print(f"Saved {model_path.relative_to(ROOT)} and reports/training_v1.json")


if __name__ == "__main__":
    main()
