"""MP299 prediction service.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

POST /predict  one station-day -> predicted litres_per_day
GET  /health   {"status": "ok", ...}
"""

import os
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.logging_config import get_logger

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
STAFF_SENTINEL = 999  # found in train.csv: 999 was typed instead of a blank

FEATURES = ["highway_distance_km", "nearby_pumps", "price_rs", "trucks_passing",
            "temp_c", "staff", "ownership", "zone"]

log = get_logger()
model = joblib.load(MODEL_PATH)
app = FastAPI(title="MP299 Fuel Station Daily Sales", version="1.0")


class StationDay(BaseModel):
    """One request = one station on one day. The client's two rules are
    enforced here, so a bad request gets 422 and never reaches the model."""

    highway_distance_km: float = Field(..., ge=0, description="km to the highway; below 0 is a typing mistake")
    nearby_pumps: int = Field(..., description="other stations within 3 km")
    price_rs: float = Field(..., description="petrol price, rupees per litre")
    trucks_passing: int = Field(..., description="trucks passing per day")
    temp_c: float = Field(..., le=75, description="maximum temperature, deg C; above 75 is impossible")
    staff: int = Field(..., description="attendants on duty")
    ownership: Literal["public_sector", "private", "cooperative"]
    zone: Literal["Highway", "City", "Industrial"]

    model_config = {"json_schema_extra": {"example": {
        "highway_distance_km": 7.34, "nearby_pumps": 5, "price_rs": 94.25, "trucks_passing": 1868,
        "temp_c": 45.3, "staff": 9, "ownership": "public_sector", "zone": "City"}}}


class Prediction(BaseModel):
    prediction: float          # predicted litres_per_day
    litres_per_day: float      # same value, named after the target
    model_version: str


def to_frame(req: StationDay) -> pd.DataFrame:
    row = req.model_dump()
    if row["staff"] == STAFF_SENTINEL:   # same fix as cleaning: a blank, the model imputes it
        row["staff"] = np.nan
    df = pd.DataFrame([row], columns=FEATURES)
    num = FEATURES[:6]
    df[num] = df[num].astype(float)
    return df


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None, "model_version": MODEL_VERSION}


@app.post("/predict", response_model=Prediction)
def predict(req: StationDay):
    start = time.perf_counter()
    value = float(model.predict(to_frame(req))[0])
    value = max(value, 0.0)  # a station cannot sell negative litres
    log.info("prediction", extra={
        "event": "prediction",
        "zone": req.zone,
        "ownership": req.ownership,
        "prediction": round(value, 1),
        "model_version": MODEL_VERSION,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    })
    value = round(value, 1)
    return Prediction(prediction=value, litres_per_day=value, model_version=MODEL_VERSION)
