"""Stage 4: drift, per-group error, retrain v2, promotion rule, holdout predictions.

Run from the repo root (after train.py has registered v1):
    python src/fuel_sales/monitor.py

Steps (PROBLEM.md section 7 and 8):
  1. PSI of every numeric column, cleaned train.csv vs later.csv, 10 bins.
  2. v1's MAE on later.csv for each value of `zone`.
  3. Sort later.csv by date. v2 is trained on cleaned train.csv + the first
     70% of later.csv. Its settings are chosen from params.yaml
     (monitoring.v2_candidates) on an inner split of that 70%, so the last
     30% stays unseen. Compare v1 and v2 on the last 30%.
  4. Promote v2 only if its MAE there is at least 15% lower than v1's.
     v2 is registered as version 2 either way; @champion moves only if promoted.
  5. Predict holdout_features.csv with the champion.
"""

import json
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

if __package__ in (None, ""):  # allow `python src/fuel_sales/monitor.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuel_sales.clean import clean_features, clean_train  # noqa: E402
from fuel_sales.config import ROOT, load_params, read_csv  # noqa: E402
from fuel_sales.data_checks import run_all  # noqa: E402
from fuel_sales.model import build_pipeline  # noqa: E402
from fuel_sales.train import sha256_of  # noqa: E402


# --------------------------------------------------------------------------
def psi(expected, actual, bins=10, eps=1e-4):
    """Population Stability Index.

    Bin edges are the deciles of the reference (`expected`, cleaned train)
    so each bin holds ~10% of the training rows; the outer edges are opened
    to -inf/+inf so later values outside the training range still count.
    Repeated edges (whole-number columns) are merged.
    PSI = sum((a% - e%) * ln(a% / e%)); empty bins get a tiny share (eps).
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value):
    return "no drift" if value < 0.1 else "moderate drift" if value < 0.25 else "major drift"


def mae(y, pred):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(pred))))


def mae_by_group(model, df, features, target, group):
    out = {}
    for g, part in df.groupby(group):
        out[g] = mae(part[target], model.predict(part[features]))
    return dict(sorted(out.items()))


def split_by_date(df, date_col, id_col, fraction):
    """Sort by date (record_id breaks ties) and cut at `fraction`."""
    s = df.sort_values([date_col, id_col], kind="stable").reset_index(drop=True)
    cut = int(round(len(s) * fraction))
    return s.iloc[:cut].reset_index(drop=True), s.iloc[cut:].reset_index(drop=True)


def as_model_input(df, num, cat, findings):
    X = clean_features(df[num + cat], findings)
    return X.astype({c: float for c in num})


# --------------------------------------------------------------------------
def main():
    params = load_params()
    mcfg, tcfg, mon = params["mlflow"], params["train"], params["monitoring"]
    num, cat, target = params["numeric_features"], params["categorical_features"], params["target"]
    feats = num + cat
    date_col, id_col, group = params["date_column"], params["id_column"], params["group_column"]

    train_raw, later, holdout = (read_csv(k, params) for k in ("train", "later", "holdout"))
    findings = run_all(train_raw, later, holdout, params)
    train, _ = clean_train(train_raw, findings, params, later)
    data_sha = sha256_of(ROOT / params["data"]["train"])

    mlflow.set_tracking_uri(mcfg["tracking_uri"])
    mlflow.set_experiment(mcfg["experiment_name"])
    client = MlflowClient()
    name, alias = mcfg["registered_model_name"], mcfg["champion_alias"]

    # v1 = the current champion (version 1), trained on cleaned train.csv only
    v1_version = client.get_model_version_by_alias(name, alias)
    v1_run = client.get_run(v1_version.run_id)
    v1 = joblib.load(ROOT / tcfg["model_path"])

    # 1. PSI -----------------------------------------------------------------
    psi_values = {c: round(psi(train[c], later[c], mon["psi_bins"]), 4) for c in num}
    most_drifted = max(psi_values, key=psi_values.get)

    # 2. v1 per group on later.csv --------------------------------------------
    later_X = as_model_input(later, num, cat, findings)
    later_in = pd.concat([later_X, later[[target]]], axis=1)
    by_group = {g: round(v, 2) for g, v in mae_by_group(v1, later_in, feats, target, group).items()}
    worst = max(by_group, key=by_group.get)
    mae_v1_later = mae(later[target], v1.predict(later_X))

    # 3. retrain v2 on train + first 70% of later ------------------------------
    first, last = split_by_date(later, date_col, id_col, mon["retrain_fraction"])
    inner_fit, inner_val = split_by_date(first, date_col, id_col, mon["retrain_fraction"])

    def fit_v2(cand, recent):
        rows = pd.concat([train[feats + [target]], recent[feats + [target]]], ignore_index=True)
        weight = np.r_[np.ones(len(train)), np.full(len(recent), float(cand["recent_weight"]))]
        pipe = build_pipeline(num, cat, cand["model"], cand["params"], tcfg["random_state"])
        X = as_model_input(rows, num, cat, findings)
        return pipe.fit(X, rows[target], model__sample_weight=weight), X, len(rows)

    inner = {}
    for cand in mon["v2_candidates"]:
        pipe, _, _ = fit_v2(cand, inner_fit)
        inner[cand["name"]] = mae(inner_val[target], pipe.predict(as_model_input(inner_val, num, cat, findings)))
    chosen = next(c for c in mon["v2_candidates"] if c["name"] == min(inner, key=inner.get))
    v2, X2, v2_rows = fit_v2(chosen, first)

    X_check = as_model_input(last, num, cat, findings)
    mae_v1_check = mae(last[target], v1.predict(X_check))
    mae_v2_check = mae(last[target], v2.predict(X_check))
    improvement = 1 - mae_v2_check / mae_v1_check

    # 4. promotion rule -------------------------------------------------------
    promote = improvement >= mon["promote_min_improvement"]
    decision = "promote" if promote else "keep"

    with mlflow.start_run(run_name="v2_retrain_train_plus_later70") as run:
        mlflow.set_tag("model_version_role", "v2")
        mlflow.log_param("data_sha256", data_sha)
        mlflow.log_param("v2_choice", chosen["name"])
        mlflow.log_param("model_type", chosen["model"])
        mlflow.log_params({f"model__{k}": v for k, v in chosen["params"].items()})
        mlflow.log_param("recent_weight", chosen["recent_weight"])
        mlflow.log_param("train_rows", v2_rows)
        mlflow.log_param("later_rows_used", len(first))
        mlflow.log_param("check_rows", len(last))
        mlflow.log_param("check_dates", f"{last[date_col].min()}..{last[date_col].max()}")
        mlflow.log_param("decision", decision)
        mlflow.log_metrics({
            "mae_v1_check": mae_v1_check,
            "mae_v2_check": mae_v2_check,
            "mae_improvement_pct": 100 * improvement,
            "mae_v1_later": mae_v1_later,
            **{f"psi_{c}": v for c, v in psi_values.items()},
            **{f"mae_v1_later_{g}": v for g, v in by_group.items()},
            **{f"mae_inner_{n}": v for n, v in inner.items()},
        })
        info = mlflow.sklearn.log_model(v2, name="model",
                                        signature=infer_signature(X2.head(), v2.predict(X2.head())),
                                        input_example=X2.head(3),
                                        serialization_format="cloudpickle")
        v2_run_id = run.info.run_id

    mv2 = mlflow.register_model(info.model_uri, name)
    client.set_model_version_tag(name, mv2.version, "role", "v2")
    client.set_model_version_tag(name, mv2.version, "decision", decision)
    if promote:
        client.set_registered_model_alias(name, alias, mv2.version)
        joblib.dump(v2, ROOT / tcfg["model_path"])   # the API serves the champion
    champion_version = int(mv2.version) if promote else int(v1_version.version)
    champion = v2 if promote else v1

    # 5. holdout predictions with the champion --------------------------------
    pred = champion.predict(as_model_input(holdout, num, cat, findings))
    out = pd.DataFrame({id_col: holdout[id_col], "prediction": np.clip(pred, 0, None).round(1)})
    pred_path = ROOT / mon["predictions"]
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False, lineterminator="\n")

    report = {
        "psi": psi_values,
        "psi_label": {c: psi_label(v) for c, v in psi_values.items()},
        "most_drifted_feature": most_drifted,
        "mae_by_group": by_group,
        "worst_served_group": worst,
        "mae_v1_on_later": round(mae_v1_later, 2),
        "check_split": {"first_rows": len(first), "last_rows": len(last),
                        "last_dates": [last[date_col].min(), last[date_col].max()]},
        "v2_inner_validation_mae": {n: round(v, 1) for n, v in inner.items()},
        "v2_choice": chosen["name"],
        "mae_v1_on_check": round(mae_v1_check, 2),
        "mae_v2_on_check": round(mae_v2_check, 2),
        "improvement_pct": round(100 * improvement, 2),
        "rule": f"promote if v2 MAE <= {1 - mon['promote_min_improvement']:.2f} x v1 MAE",
        "decision": decision,
        "v2_run_id": v2_run_id,
        "v2_registered_version": int(mv2.version),
        "champion_version": champion_version,
        "holdout_rows_predicted": len(out),
    }
    rep = ROOT / mon["report"]
    rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    print("PSI (cleaned train vs later):")
    for c, v in psi_values.items():
        print(f"  {c:22s} {v:8.4f}  {psi_label(v)}")
    print(f"Most drifted: {most_drifted}\n")
    print("v1 MAE on later.csv by zone:", by_group, f"-> worst: {worst}\n")
    print("v2 candidates, inner-validation MAE:", {n: round(v) for n, v in inner.items()})
    print(f"v2 = {chosen['name']}\n")
    print(f"Last 30% of later.csv ({len(last)} rows): v1 MAE {mae_v1_check:.1f}, v2 MAE {mae_v2_check:.1f}, "
          f"improvement {100 * improvement:.1f}%")
    print(f"Decision: {decision.upper()}  (v2 = version {mv2.version}; @{alias} = version {champion_version})")
    print(f"Saved {pred_path.relative_to(ROOT)} ({len(out)} rows) and {rep.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
