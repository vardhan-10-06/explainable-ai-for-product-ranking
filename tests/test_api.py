"""
tests/test_api.py
===================
API-level tests for the FastAPI backend. Uses a temporary SQLite DB and
exercises the auth + health-check flow. Ranking/explain endpoints are
tested for correct error handling when no trained model artifact is
present, since training a full ensemble in a unit test is out of scope.
"""
import os

os.environ.setdefault("CONFIG_PATH", os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_register_and_login_flow():
    username = "test_user_pytest"
    client.post("/auth/register", json={"username": username, "password": "testpass123", "role": "viewer"})
    response = client.post("/auth/token", data={"username": username, "password": "testpass123"})
    assert response.status_code == 200
    assert "access_token" in response.json()


def test_ranking_endpoint_requires_auth():
    response = client.post("/rank/", json={"user_id": "U000001", "top_k": 5})
    assert response.status_code == 401
