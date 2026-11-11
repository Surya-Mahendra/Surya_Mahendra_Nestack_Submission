from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_rejects_blank_query():
    response = client.post("/query", json={"query": "   ", "top_k": 3})
    assert response.status_code == 422


def test_query_rejects_top_k_above_limit():
    response = client.post("/query", json={"query": "test", "top_k": 100})
    assert response.status_code == 422
