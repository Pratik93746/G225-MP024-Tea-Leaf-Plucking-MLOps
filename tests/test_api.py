from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
VALID = {"worker_exp_yr": 10, "rain_mm_week": 120, "slope_deg": 15, "temp_c": 24.0,
         "bush_age_yr": 40, "hours_worked": 9.0, "flush": "first", "section": "River Side"}


def test_health_is_ok():
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "ok"


def test_predict_returns_a_sensible_number():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0 < body["prediction"] < 150


def test_client_rule_hours_above_15_is_rejected():
    assert client.post("/predict", json={**VALID, "hours_worked": 15.1}).status_code == 422
    assert client.post("/predict", json={**VALID, "hours_worked": 15}).status_code == 200   # boundary is allowed


def test_client_rule_negative_rain_is_rejected():
    assert client.post("/predict", json={**VALID, "rain_mm_week": -0.1}).status_code == 422
    assert client.post("/predict", json={**VALID, "rain_mm_week": 0}).status_code == 200


def test_missing_unknown_and_extra_fields_are_rejected():
    assert client.post("/predict", json={k: v for k, v in VALID.items() if k != "temp_c"}).status_code == 422
    assert client.post("/predict", json={**VALID, "section": "Nowhere"}).status_code == 422
    assert client.post("/predict", json={**VALID, "kg_weighed_at_factory": 30}).status_code == 422  # the leaking column


def test_messy_flush_spelling_is_understood():
    assert client.post("/predict", json={**VALID, "flush": " SECOND"}).status_code == 200
