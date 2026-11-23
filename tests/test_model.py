"""Tests for the model pipeline. Small in-memory data; no CSVs, no MLflow."""

import numpy as np
import pandas as pd
import pytest

from fuel_sales.config import load_params
from fuel_sales.model import REGRESSORS, build_pipeline
from fuel_sales.train import sha256_of

P = load_params()
NUM, CAT = P["numeric_features"], P["categorical_features"]


def toy(n=120, seed=1):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "highway_distance_km": rng.uniform(0, 15, n),
        "nearby_pumps": rng.integers(0, 11, n).astype(float),
        "price_rs": rng.uniform(90, 110, n),
        "trucks_passing": rng.integers(5, 2000, n).astype(float),
        "temp_c": rng.uniform(5, 46, n),
        "staff": rng.integers(3, 21, n).astype(float),
        "ownership": rng.choice(["private", "cooperative", "public_sector"], n),
        "zone": rng.choice(["Highway", "City", "Industrial"], n),
    })
    y = 20000 - 500 * X["highway_distance_km"] + 5 * X["trucks_passing"] + rng.normal(0, 300, n)
    return X, y


@pytest.mark.parametrize("kind", list(REGRESSORS))
def test_every_candidate_fits_and_predicts(kind):
    X, y = toy()
    pipe = build_pipeline(NUM, CAT, kind, {}).fit(X, y)
    pred = pipe.predict(X)
    assert pred.shape == (len(X),) and np.isfinite(pred).all()


def test_pipeline_handles_blanks_and_unknown_category():
    X, y = toy()
    pipe = build_pipeline(NUM, CAT, "ridge", {"alpha": 1.0}).fit(X, y)
    new = X.head(2).copy()
    new.loc[new.index[0], "staff"] = np.nan          # sentinel turned into a blank
    new.loc[new.index[1], "zone"] = "Rural"          # value never seen in training
    assert np.isfinite(pipe.predict(new)).all()


def test_ridge_learns_the_signal():
    X, y = toy()
    pipe = build_pipeline(NUM, CAT, "ridge", {"alpha": 1.0}).fit(X, y)
    assert np.mean(np.abs(pipe.predict(X) - y)) < 500


def test_candidates_in_params_are_valid():
    cands = P["train"]["candidates"]
    assert len(cands) >= 3
    assert all(c["model"] in REGRESSORS for c in cands)


def test_sha256_of(tmp_path):
    f = tmp_path / "x.csv"
    f.write_bytes(b"a,b\n1,2\n")
    assert sha256_of(f) == "492d5ea496056f1a6a6592241032fab764c321596317930b4fa0e1e8bc3b7470"
