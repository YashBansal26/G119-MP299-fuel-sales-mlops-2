"""Hunt the seven kinds of data problem in train.csv.

Run from the repo root:
    python -m fuel_sales.data_checks          (with src/ on PYTHONPATH)
or  python src/fuel_sales/data_checks.py

Every check takes a DataFrame and returns a dict shaped like the matching
block of reports/findings.yaml, plus an "evidence" key that shows why.
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):  # allow `python src/fuel_sales/data_checks.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuel_sales.config import ROOT, load_params, read_csv  # noqa: E402


# 1. leaking_column ----------------------------------------------------------
def check_leaking_column(train, holdout, target, threshold=0.98):
    """A leak is a column we cannot have at prediction time (missing from the
    holdout file) or one that tracks the target almost exactly."""
    missing = [c for c in train.columns if c not in holdout.columns and c != target]
    numeric = train.select_dtypes("number").drop(columns=[target], errors="ignore")
    corr = numeric.corrwith(train[target]).abs().round(4).to_dict()
    too_close = [c for c, r in corr.items() if r >= threshold]
    suspects = missing + [c for c in too_close if c not in missing]
    return {
        "present": bool(suspects),
        "column": suspects[0] if suspects else None,
        "evidence": {"not_in_holdout": missing, "abs_corr_with_target": corr},
    }


# 2. unit_mixup --------------------------------------------------------------
def check_unit_mixup(df, group_col, columns, factors, tolerance=0.15, sentinels=()):
    """For each numeric column, compare each group's mean with the mean of the
    other groups. A ratio close to a real unit factor (10, 12, 60, 100, 1000)
    means that group typed the column in another unit."""
    hits, table = [], {}
    for col in columns:
        clean = df[~df[col].isin(sentinels)]
        table[col] = clean.groupby(group_col)[col].agg(["min", "mean", "max"]).round(3).to_dict("index")
        for g in clean[group_col].unique():
            inside = clean.loc[clean[group_col] == g, col].mean()
            outside = clean.loc[clean[group_col] != g, col].mean()
            if not outside:
                continue
            ratio = inside / outside
            for f in factors:
                if abs(ratio / f - 1) <= tolerance:
                    hits.append({"column": col, "group_value": g, "factor": f, "ratio": round(ratio, 3)})
    best = hits[0] if hits else None
    return {
        "present": bool(hits),
        "column": best["column"] if best else None,
        "group_value": best["group_value"] if best else None,
        "factor": best["factor"] if best else None,
        "evidence": {"hits": hits, "per_group_min_mean_max": table},
    }


# 3. sentinel_value ----------------------------------------------------------
def check_sentinel_value(df, columns, candidates, min_count=10):
    """Look for a 'special' number (999, -1, ...) repeated many times."""
    hits = []
    for col in columns:
        counts = df[col].value_counts()
        for v in candidates:
            n = int(counts.get(v, 0))
            if n >= min_count:
                hits.append({"column": col, "value": v, "count": n,
                             "next_largest_value": float(df.loc[df[col] != v, col].max())})
    hits.sort(key=lambda h: -h["count"])
    best = hits[0] if hits else None
    return {
        "present": bool(hits),
        "column": best["column"] if best else None,
        "value": best["value"] if best else None,
        "count": best["count"] if best else None,
        "evidence": {"hits": hits},
    }


# 4. duplicate_rows ----------------------------------------------------------
def check_duplicate_rows(df, id_col):
    full = int(df.duplicated().sum())
    no_id = int(df.drop(columns=[id_col]).duplicated().sum())
    count = max(full, no_id)
    return {
        "present": count > 0,
        "count": count if count > 0 else None,
        "evidence": {"exact_duplicates": full, "duplicates_ignoring_record_id": no_id},
    }


# 5. stuck_sensor ------------------------------------------------------------
def longest_run(df, col, date_col):
    """Longest run of one repeated value when rows are sorted by date."""
    s = df.sort_values([date_col], kind="stable").reset_index(drop=True)
    run_id = (s[col] != s[col].shift()).cumsum()
    runs = s.groupby(run_id).agg(value=(col, "first"), rows=(col, "size"),
                                 start=(date_col, "first"), end=(date_col, "last"),
                                 days=(date_col, "nunique"))
    return runs.sort_values("rows").iloc[-1].to_dict()


def check_stuck_sensor(df, columns, date_col, min_days=7):
    runs = {c: longest_run(df, c, date_col) for c in columns}
    stuck = {c: r for c, r in runs.items() if r["days"] >= min_days}
    best = max(stuck, key=lambda c: stuck[c]["rows"]) if stuck else None
    return {
        "present": bool(stuck),
        "column": best,
        "start_date": stuck[best]["start"] if best else None,
        "end_date": stuck[best]["end"] if best else None,
        "evidence": {"longest_run_per_column": {c: {k: (v.item() if hasattr(v, "item") else v)
                                                     for k, v in r.items()} for c, r in runs.items()}},
    }


# 6. target_typos ------------------------------------------------------------
def check_target_typos(df, target, gap_ratio=2.5):
    """Sort the target; a typo group sits above a big empty gap."""
    y = df[target].sort_values().reset_index(drop=True)
    ratios = (y / y.shift()).fillna(1)
    i = int(ratios.idxmax())
    biggest = float(ratios.iloc[i])
    present = biggest >= gap_ratio
    count = int(len(y) - i) if present else None
    return {
        "present": present,
        "count": count,
        "evidence": {"value_below_gap": int(y.iloc[i - 1]), "value_above_gap": int(y.iloc[i]),
                     "gap_ratio": round(biggest, 2),
                     "typo_median_over_clean_median": round(float(y.iloc[i:].median() / y.iloc[:i].median()), 2)
                     if present else None},
    }


# 7. spelling_variants -------------------------------------------------------
def normalise(text):
    return re.sub(r"[^a-z0-9_]", "", str(text).strip().lower())


def check_spelling_variants(train, later, columns):
    hits, evidence = [], {}
    for col in columns:
        raw = sorted(train[col].dropna().unique().tolist())
        norm = sorted({normalise(v) for v in raw})
        clean_levels = sorted(later[col].dropna().unique().tolist())
        evidence[col] = {"train_levels": raw, "later_levels": clean_levels}
        if len(raw) > len(clean_levels) or len(raw) > len(norm):
            hits.append({"column": col, "true_levels": len(clean_levels)})
    best = hits[0] if hits else None
    return {
        "present": bool(hits),
        "column": best["column"] if best else None,
        "true_levels": best["true_levels"] if best else None,
        "evidence": evidence,
    }


# ---------------------------------------------------------------------------
def run_all(train, later, holdout, params):
    c = params["checks"]
    target = params["target"]
    num = params["numeric_features"]
    cat = params["categorical_features"]
    sentinel = check_sentinel_value(train, num, c["sentinel_candidates"], c["sentinel_min_count"])
    sentinel_values = [h["value"] for h in sentinel["evidence"]["hits"]]
    return {
        "leaking_column": check_leaking_column(train, holdout, target, c["leak_corr_threshold"]),
        "unit_mixup": check_unit_mixup(train, params["group_column"], num, c["unit_factors"],
                                       c["unit_ratio_tolerance"], sentinel_values),
        "sentinel_value": sentinel,
        "duplicate_rows": check_duplicate_rows(train, params["id_column"]),
        "stuck_sensor": check_stuck_sensor(train, num, params["date_column"], c["stuck_min_run_days"]),
        "target_typos": check_target_typos(train, target, c["target_gap_ratio"]),
        "spelling_variants": check_spelling_variants(train, later, cat),
    }


def main():
    params = load_params()
    train, later, holdout = (read_csv(k, params) for k in ("train", "later", "holdout"))
    results = run_all(train, later, holdout, params)

    out = ROOT / params["reports"]["data_checks"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print(f"train.csv: {len(train)} rows\n")
    for kind, r in results.items():
        answer = {k: v for k, v in r.items() if k != "evidence"}
        print(f"{kind:18s} {answer}")
    print(f"\nFull evidence written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
