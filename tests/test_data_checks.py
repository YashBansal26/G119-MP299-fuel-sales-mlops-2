"""Tests for the seven data checks. They build tiny DataFrames in memory,
so they do not need the CSVs (which live in DVC, not git)."""

import pandas as pd

from fuel_sales import data_checks as dc
from fuel_sales.config import load_params


def make_df(n=40):
    dates = pd.date_range("2026-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    return pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": dates,
        "staff": [5 + i % 10 for i in range(n)],
        "nearby_pumps": [i % 7 for i in range(n)],
        "highway_distance_km": [1.0 + (i % 9) for i in range(n)],
        "zone": ["Highway", "City", "Industrial", "City"] * (n // 4),
        "litres_per_day": [15000 + 100 * i for i in range(n)],
    })


def test_params_load():
    p = load_params()
    assert p["target"] == "litres_per_day"
    assert "staff" in p["numeric_features"]


def test_sentinel_found():
    df = make_df()
    df.loc[:11, "staff"] = 999
    r = dc.check_sentinel_value(df, ["staff"], [999, -1], min_count=10)
    assert r["present"] and r["column"] == "staff" and r["value"] == 999 and r["count"] == 12


def test_duplicates_counted():
    df = make_df()
    df = pd.concat([df, df.iloc[[0, 1, 2]]], ignore_index=True)
    assert dc.check_duplicate_rows(df, "record_id")["count"] == 3
    assert dc.check_duplicate_rows(make_df(), "record_id")["present"] is False


def test_stuck_sensor_dates():
    df = make_df()
    df.loc[10:20, "nearby_pumps"] = 6
    r = dc.check_stuck_sensor(df, ["nearby_pumps"], "date", min_days=7)
    assert r["present"]
    assert (r["start_date"], r["end_date"]) == (df.loc[10, "date"], df.loc[20, "date"])


def test_target_typos():
    df = make_df()
    df.loc[[3, 7], "litres_per_day"] *= 10
    r = dc.check_target_typos(df, "litres_per_day")
    assert r["present"] and r["count"] == 2


def test_unit_mixup():
    df = make_df()
    df.loc[df.zone == "City", "highway_distance_km"] *= 1000
    r = dc.check_unit_mixup(df, "zone", ["highway_distance_km"], [10, 12, 60, 100, 1000])
    assert r["present"] and r["group_value"] == "City" and r["factor"] == 1000


def test_spelling_and_leak():
    later = make_df()
    train = later.copy()
    train.loc[0, "zone"] = "city "
    r = dc.check_spelling_variants(train, later, ["zone"])
    assert r["present"] and r["column"] == "zone" and r["true_levels"] == 3

    holdout = later.drop(columns=["litres_per_day", "staff"])
    leak = dc.check_leaking_column(later, holdout, "litres_per_day")
    assert leak["present"] and leak["column"] == "staff"
