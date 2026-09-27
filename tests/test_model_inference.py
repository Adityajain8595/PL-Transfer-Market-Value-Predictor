import pytest
from fastapi.testclient import TestClient

from src.serve.app import app


# Test API client fixture
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


sample_payload = {
    "age_at_valuation": 23.5,
    "position": "Attack",
    "sub_position": "Left Winger",
    "dominant_foot": "right",
    "club_name": "Arsenal FC",
    "last_known_value_eur": 50000000.0,
    "days_between_valuations": 180,
    "minutes_since_last_val": 1600,
    "european_minutes_played": 350,
    "goal_contributions_per_90": 0.75,
    "yellow_cards_since_val": 1,
    "minutes_prior_window": 1500,
    "contrib_per_90_prior": 0.50
}


# FastAPI endpoint validation tests
def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_single_prediction_endpoint(client):
    resp = client.post("/predict", json=sample_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "predicted_market_value_eur" in data
    assert data["predicted_market_value_eur"] > 0
    assert "log_market_value" in data


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text


def test_batch_prediction_endpoint(client):
    resp = client.post("/predict/batch", json={"players": [sample_payload, sample_payload]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["predictions"]) == 2


def test_hydrated_player_endpoint(client):
    resp = client.get("/predict/player/433177")
    assert resp.status_code == 200
    data = resp.json()
    assert data["player_id"] == 433177
    assert "player_name" in data
    assert "predicted_market_value_eur" in data
    assert data["predicted_market_value_eur"] > 0


def test_hydrated_player_not_found(client):
    resp = client.get("/predict/player/99999999")
    assert resp.status_code == 404


def test_scenario_simulation_endpoint(client):
    payload = {
        "player_id": 433177,
        "simulated_minutes": 2200,
        "simulated_goals": 8,
        "simulated_assists": 6,
        "simulated_yellow_cards": 2,
        "simulated_european_minutes": 360,
        "simulated_club": "Arsenal FC",
        "days_elapsed": 180
    }
    resp = client.post("/predict/simulate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["player_id"] == 433177
    assert "current_market_value_eur" in data
    assert "predicted_market_value_eur" in data
    assert "value_change_eur" in data
    assert "value_change_pct" in data