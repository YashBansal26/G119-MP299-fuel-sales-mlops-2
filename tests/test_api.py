"""API tests with FastAPI's TestClient. They use the committed
models/model.joblib, so no CSVs are needed (works in GitHub Actions)."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app.logging_config import JsonFormatter
from app.main import app

client = TestClient(app)

GOOD = {
    "highway_distance_km": 7.34, "nearby_pumps": 5, "price_rs": 94.25, "trucks_passing": 1868,
    "temp_c": 45.3, "staff": 9, "ownership": "public_sector", "zone": "City",
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_ok():
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == body["litres_per_day"]
    assert 0 < body["prediction"] < 100_000


def test_rule_temp_above_75_rejected():
    assert client.post("/predict", json={**GOOD, "temp_c": 75.1}).status_code == 422
    assert client.post("/predict", json={**GOOD, "temp_c": 75}).status_code == 200


def test_rule_negative_highway_distance_rejected():
    assert client.post("/predict", json={**GOOD, "highway_distance_km": -0.01}).status_code == 422
    assert client.post("/predict", json={**GOOD, "highway_distance_km": 0}).status_code == 200


@pytest.mark.parametrize("field", list(GOOD))
def test_every_field_is_required(field):
    body = {k: v for k, v in GOOD.items() if k != field}
    assert client.post("/predict", json=body).status_code == 422


def test_bad_category_and_bad_type_rejected():
    assert client.post("/predict", json={**GOOD, "zone": "Rural"}).status_code == 422
    assert client.post("/predict", json={**GOOD, "staff": "many"}).status_code == 422


def test_staff_sentinel_still_predicts():
    r = client.post("/predict", json={**GOOD, "staff": 999})
    assert r.status_code == 200
    assert 0 < r.json()["prediction"] < 100_000


def test_prediction_is_logged_as_one_json_line(capsys):
    client.post("/predict", json=GOOD)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    entry = json.loads(lines[-1])
    assert entry["event"] == "prediction" and entry["zone"] == "City"
    assert "prediction" in entry and "latency_ms" in entry


def test_json_formatter():
    rec = logging.LogRecord("t", logging.INFO, "", 0, "hello", None, None)
    rec.zone = "Highway"
    out = json.loads(JsonFormatter().format(rec))
    assert out["message"] == "hello" and out["zone"] == "Highway" and out["level"] == "INFO"
