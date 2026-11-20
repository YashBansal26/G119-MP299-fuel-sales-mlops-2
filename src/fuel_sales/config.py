"""Load project settings from params.yaml and read the raw data files."""

from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path: str | Path = ROOT / "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(key: str, params: dict | None = None) -> pd.DataFrame:
    """Read one of the data files named in params.yaml (train, later, holdout)."""
    params = params or load_params()
    return pd.read_csv(ROOT / params["data"][key])
