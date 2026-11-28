"""Tests for the Stage 4 helpers (PSI, date split, MAE by group). In-memory data only."""

import numpy as np
import pandas as pd

from fuel_sales.model import build_pipeline
from fuel_sales.monitor import mae, mae_by_group, psi, psi_label, split_by_date


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)
    assert psi(a, b) < 0.02


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    a, b = rng.uniform(5, 46, 1000), rng.uniform(22, 63, 1000)   # like temp_c train vs later
    assert psi(a, b) > 0.25
    assert psi_label(psi(a, b)) == "major drift"


def test_psi_handles_whole_numbers_and_values_outside_training_range():
    a = np.repeat(np.arange(0, 11), 50)          # nearby_pumps-like: many repeated values
    b = np.r_[a, [50, 60]]                        # two values never seen in training
    value = psi(a, b)
    assert np.isfinite(value) and value >= 0


def test_split_by_date_is_sorted_and_70_30():
    df = pd.DataFrame({"date": ["2026-07-03", "2026-07-01", "2026-07-02"] * 10,
                       "record_id": [f"R{i:03d}" for i in range(30)], "x": range(30)})
    first, last = split_by_date(df, "date", "record_id", 0.7)
    assert len(first) == 21 and len(last) == 9
    assert first["date"].max() <= last["date"].min()


def test_mae_and_mae_by_group():
    assert mae([10, 20], [12, 17]) == 2.5
    num, cat = ["a"], ["zone"]
    df = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6], "zone": ["X", "X", "X", "Y", "Y", "Y"],
                       "y": [10.0, 20, 30, 40, 50, 60]})
    model = build_pipeline(num, cat, "ridge", {"alpha": 1e-6}).fit(df[num + cat], df["y"])
    groups = mae_by_group(model, df, num + cat, "y", "zone")
    assert set(groups) == {"X", "Y"} and all(v < 1 for v in groups.values())


def test_promotion_rule():
    rule = lambda v1, v2: (1 - v2 / v1) >= 0.15
    assert rule(1000, 850) and not rule(1000, 851)


def test_ridge_poly_learns_a_curve():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"t": rng.uniform(0, 60, 300), "zone": rng.choice(["A", "B"], 300)})
    y = 5 * df["t"] ** 2 + np.where(df["zone"] == "B", 3000, 0)
    lin = build_pipeline(["t"], ["zone"], "ridge", {"alpha": 1.0}).fit(df, y)
    poly = build_pipeline(["t"], ["zone"], "ridge_poly", {"alpha": 1.0, "degree": 2}).fit(df, y)
    assert mae(y, poly.predict(df)) < mae(y, lin.predict(df)) / 5
