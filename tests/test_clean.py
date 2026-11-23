"""Tests for the cleaning code. They build a small messy dataset in memory
(no CSVs needed), run the real checks on it, clean it, and check the result."""

import numpy as np
import pandas as pd

from fuel_sales.clean import clean_features, clean_train, validate
from fuel_sales.config import load_params
from fuel_sales.data_checks import run_all

PARAMS = load_params()


def make_clean(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n // 2, freq="D").repeat(2).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "record_id": [f"R{i:05d}" for i in range(n)],
        "date": dates,
        "highway_distance_km": rng.uniform(0, 15, n).round(2),
        "nearby_pumps": rng.integers(0, 11, n),
        "price_rs": rng.uniform(90, 110, n).round(2),
        "trucks_passing": rng.integers(5, 2000, n),
        "temp_c": rng.uniform(5, 46, n).round(1),
        "staff": rng.integers(3, 21, n),
        "ownership": rng.choice(["private", "cooperative", "public_sector"], n),
        "zone": rng.choice(["Highway", "City", "Industrial"], n),
        "litres_per_day": rng.integers(8000, 30000, n),
    })


def make_messy():
    clean = make_clean()
    messy = clean.copy()
    messy.loc[0:14, "staff"] = 999                      # 15 sentinels
    messy.loc[40:69, "nearby_pumps"] = 6                # stuck for 15 days
    messy.loc[[5, 90, 150], "litres_per_day"] *= 10     # 3 typos
    return clean, messy


def findings_for(df, later):
    holdout = later.drop(columns=[PARAMS["target"]])
    return run_all(df, later, holdout, PARAMS)


def test_checks_see_the_injected_problems():
    clean, messy = make_messy()
    f = findings_for(messy, clean)
    assert f["sentinel_value"]["present"] and f["sentinel_value"]["count"] == 15
    assert f["stuck_sensor"]["column"] == "nearby_pumps"
    assert f["target_typos"]["count"] == 3


def test_clean_train_fixes_everything():
    clean, messy = make_messy()
    out, report = clean_train(messy, findings_for(messy, clean), PARAMS, clean)
    assert validate(out, PARAMS) == []
    assert (out["staff"] != 999).all()
    assert out["litres_per_day"].max() < 50000
    assert report["target_divided_by_10"] == 3
    assert report["sentinel_blanked"] == {"staff": 15}


def test_clean_train_keeps_rows_and_good_values():
    clean, messy = make_messy()
    out, _ = clean_train(messy, findings_for(messy, clean), PARAMS, clean)
    assert len(out) == len(messy)
    # rows we did not touch are unchanged; typo rows are back to the true value
    assert out.loc[100:139, "staff"].tolist() == clean.loc[100:139, "staff"].tolist()
    assert out.loc[[5, 90, 150], "litres_per_day"].tolist() == clean.loc[[5, 90, 150], "litres_per_day"].tolist()
    # the input frame is not changed in place
    assert (messy.loc[0:14, "staff"] == 999).all()


def test_validate_catches_leftovers():
    clean, messy = make_messy()
    assert any("sentinel" in p for p in validate(messy, PARAMS))
    assert any("typos" in p for p in validate(messy, PARAMS))


def test_clean_features_blanks_sentinel_only():
    clean, messy = make_messy()
    f = findings_for(messy, clean)
    new = pd.DataFrame({"staff": [999, 7], "nearby_pumps": [6, 6]})
    out = clean_features(new, f)
    assert np.isnan(out.loc[0, "staff"]) and out.loc[1, "staff"] == 7
    assert out["nearby_pumps"].tolist() == [6, 6]
