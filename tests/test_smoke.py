from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "DepositWise"
    assert response.json()["dashboard"] == "/dashboard"


def test_dashboard_serves_workbench():
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "Campaign inputs" in response.text
    assert "prediction-form" in response.text
    assert "Estimate subscription likelihood" in response.text


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model_loaded" in response.json()


def test_predict_without_model_is_service_unavailable_when_artifact_missing():
    from backend.app.main import MODEL_PATH

    if not MODEL_PATH.exists():
        response = client.post("/predict", json={"customer": {"age": 35}})
        assert response.status_code == 503
