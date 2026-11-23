"""Clean train.csv using what the seven data checks found.

Run from the repo root:
    python src/fuel_sales/clean.py

Writes data/processed/train_clean.csv and reports/cleaning_report.json.
Other code (training, monitoring) should call `clean_train()` / `clean_features()`
instead of reading the raw file.

What it fixes (only if the check says the problem is present):
  sentinel_value     the special number -> blank, then filled with the median
  stuck_sensor       the stuck readings -> blank, then filled with the median
  target_typos       targets typed 10x too big -> divided by 10
  duplicate_rows     extra copies dropped
  unit_mixup         the affected group's column divided by the factor
  spelling_variants  text trimmed / lower-cased and mapped to later.csv's spellings
  leaking_column     column dropped
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow `python src/fuel_sales/clean.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fuel_sales.config import ROOT, load_params, read_csv  # noqa: E402
from fuel_sales.data_checks import normalise, run_all  # noqa: E402


def clean_train(train, findings, params, later=None):
    """Return (cleaned DataFrame, report dict). `findings` is the output of
    data_checks.run_all(). The input frame is not changed."""
    df = train.copy()
    target, num = params["target"], params["numeric_features"]
    report = {"rows_in": len(df)}

    # leaking column: drop it
    leak = findings["leaking_column"]
    if leak["present"]:
        df = df.drop(columns=[leak["column"]])
        report["dropped_column"] = leak["column"]

    # duplicates: keep the first copy
    if findings["duplicate_rows"]["present"]:
        before = len(df)
        df = df.drop_duplicates(subset=[c for c in df.columns if c != params["id_column"]])
        report["duplicates_dropped"] = before - len(df)

    # spelling variants: map every spelling to the clean level from later.csv
    sp = findings["spelling_variants"]
    if sp["present"]:
        col = sp["column"]
        clean_levels = later[col].unique() if later is not None else df[col].unique()
        lookup = {normalise(v): v for v in clean_levels}
        df[col] = df[col].map(lambda v: lookup.get(normalise(v), v))
        report["spelling_fixed"] = {col: sorted(df[col].unique().tolist())}

    # unit mix-up: put the affected group back in the right unit
    um = findings["unit_mixup"]
    if um["present"]:
        mask = df[params["group_column"]] == um["group_value"]
        df.loc[mask, um["column"]] = df.loc[mask, um["column"]] / um["factor"]
        report["unit_fixed_rows"] = int(mask.sum())

    # sentinel: blank it
    sv = findings["sentinel_value"]
    if sv["present"]:
        mask = df[sv["column"]] == sv["value"]
        df[sv["column"]] = df[sv["column"]].astype(float)
        df.loc[mask, sv["column"]] = np.nan
        report["sentinel_blanked"] = {sv["column"]: int(mask.sum())}

    # stuck sensor: blank the stuck run (same value, inside the date window)
    st = findings["stuck_sensor"]
    if st["present"]:
        col, date = st["column"], params["date_column"]
        stuck_value = findings["stuck_sensor"]["evidence"]["longest_run_per_column"][col]["value"]
        mask = df[date].between(st["start_date"], st["end_date"]) & (df[col] == stuck_value)
        df[col] = df[col].astype(float)
        df.loc[mask, col] = np.nan
        report["stuck_blanked"] = {col: int(mask.sum())}

    # fill the blanks with each column's median (from the good rows)
    # (whole-number columns like staff get a whole-number median)
    medians = {}
    for c in num:
        if df[c].isna().any():
            m = float(df[c].median())
            if train[c].dtype.kind == "i":
                m = float(round(m))
                df[c] = df[c].fillna(m).astype(int)
            else:
                df[c] = df[c].fillna(m)
            medians[c] = m
    report["filled_with_median"] = medians

    # target typos: divide the too-big values by 10
    tt = findings["target_typos"]
    if tt["present"]:
        cutoff = tt["evidence"]["value_above_gap"]
        mask = df[target] >= cutoff
        df.loc[mask, target] = (df.loc[mask, target] / 10).round().astype(int)
        report["target_divided_by_10"] = int(mask.sum())

    report["rows_out"] = len(df)
    return df.reset_index(drop=True), report


def clean_features(df, findings):
    """Clean new data (later.csv, holdout, API requests) the same way as the
    features of train.csv: turn the sentinel into a blank. Missing values
    are left for the model pipeline's imputer."""
    df = df.copy()
    sv = findings["sentinel_value"]
    if sv["present"] and sv["column"] in df:
        df[sv["column"]] = df[sv["column"]].replace(sv["value"], np.nan)
    return df


def validate(df, params):
    """Checks the cleaned training data must pass. Returns a list of problems."""
    target, num = params["target"], params["numeric_features"]
    problems = []
    if df[num + [target]].isna().any().any():
        problems.append("blank values left")
    # a sentinel = one special number repeated many times (one or two 999s
    # in a column like trucks_passing can be real readings)
    min_count = params["checks"]["sentinel_min_count"]
    for c in num:
        for v in params["checks"]["sentinel_candidates"]:
            if (df[c] == v).sum() >= min_count:
                problems.append(f"sentinel {v} left in {c}")
    y = df[target].sort_values()
    if (y / y.shift()).max() >= params["checks"]["target_gap_ratio"]:
        problems.append("target still has a big gap (typos left)")
    if df.duplicated().any():
        problems.append("duplicate rows left")
    return problems


def main():
    params = load_params()
    train, later, holdout = (read_csv(k, params) for k in ("train", "later", "holdout"))
    findings = run_all(train, later, holdout, params)
    clean, report = clean_train(train, findings, params, later)

    problems = validate(clean, params)
    report["validation_problems"] = problems

    out = ROOT / params["data"]["train_clean"]
    out.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(out, index=False, lineterminator="\n")
    rep = ROOT / params["reports"]["cleaning"]
    rep.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    for k, v in report.items():
        print(f"{k:22s} {v}")
    print(f"\nSaved {out.relative_to(ROOT)}  ({len(clean)} rows)")
    if problems:
        raise SystemExit(f"Cleaning failed: {problems}")


if __name__ == "__main__":
    main()
