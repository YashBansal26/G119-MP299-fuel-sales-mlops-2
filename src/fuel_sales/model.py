"""Build the model pipeline: impute + one-hot encode + regressor."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

REGRESSORS = {
    "ridge": Ridge,
    "ridge_poly": Ridge,          # Ridge on degree-2 features (squares + interactions)
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
    "hist_gradient_boosting": HistGradientBoostingRegressor,
}


def make_regressor(kind, params, random_state=42):
    cls = REGRESSORS[kind]
    params = dict(params or {})
    if "random_state" in cls().get_params():
        params.setdefault("random_state", random_state)
    if kind == "random_forest":
        params.setdefault("n_jobs", -1)
    return cls(**params)


def build_pipeline(numeric, categorical, kind="random_forest", params=None, random_state=42):
    """Numeric: median imputer (new data may have blanks, e.g. staff=999 turned
    into a blank) + scaling. Categorical: one-hot, unknown values ignored.
    kind="ridge_poly" adds degree-`degree` features, so the model can learn
    curves (e.g. sales rising faster at high temp_c) and zone x feature effects."""
    params = dict(params or {})
    degree = params.pop("degree", 2) if kind == "ridge_poly" else None
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=degree is None), categorical),
    ])
    steps = [("prep", pre)]
    if degree:
        steps.append(("poly", PolynomialFeatures(degree, include_bias=False)))
    steps.append(("model", make_regressor(kind, params, random_state)))
    return Pipeline(steps)
