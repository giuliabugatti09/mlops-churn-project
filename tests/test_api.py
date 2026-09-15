"""Testes automatizados da API de inferência."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info_returns_expected_fields():
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "churn-classifier"
    assert body["stage"] == "Production"

def test_predict_returns_valid_prediction():
    payload = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["churn_prediction"] in [0, 1]
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_rejects_invalid_payload():
    """SeniorCitizen só aceita 0 ou 1 — valores fora disso devem
    ser rejeitados pela validação automática do Pydantic.
    """
    payload = {
        "SeniorCitizen": 5,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "TotalCharges": 845.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422