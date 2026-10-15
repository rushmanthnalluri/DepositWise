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


def test_model_info_is_explicit_when_report_is_missing():
    from backend.app.main import REPORT_PATH

    response = client.get("/model-info")
    if REPORT_PATH.exists():
        assert response.status_code == 200
        assert "selected_model" in response.json()
    else:
        assert response.status_code == 404


def test_dataset_profile_csv_is_in_memory_and_reports_quality():
    response = client.post(
        "/dataset/profile",
        files={"file": ("campaign.csv", b"age,y\n35,yes\n42,no\n,yes\n", "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "campaign.csv"
    assert data["rows"] == 3
    assert data["column_count"] == 2
    assert data["missing_cells"] == 1
    assert data["duplicate_rows"] == 0
    assert data["persisted"] is False
    assert data["training_triggered"] is False
    assert len(data["sample"]) == 3
    dataset_id = data["dataset_id"]

    features = client.get(f"/workflow/{dataset_id}/features")
    assert features.status_code == 200
    assert len(features.json()["columns"]) == 2

    statistics = client.get(f"/workflow/{dataset_id}/statistics")
    assert statistics.status_code == 200
    assert statistics.json()["numeric_column_count"] == 1

    missing = client.get(f"/workflow/{dataset_id}/missing")
    assert missing.status_code == 200
    assert missing.json()["missing_cells"] == 1

    visualization = client.get(f"/workflow/{dataset_id}/visualization")
    assert visualization.status_code == 200
    assert visualization.json()["target_distribution"]["column"] == "y"


def test_dataset_profile_rejects_unsupported_file_type():
    response = client.post(
        "/dataset/profile",
        files={"file": ("campaign.json", b'{"age": 35}', "application/json")},
    )
    assert response.status_code == 415


def test_dataset_profile_rejects_empty_csv():
    response = client.post(
        "/dataset/profile",
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 400


def test_sandbox_training_and_prediction_do_not_touch_production_model():
    rows = ["age,balance,y"]
    for i in range(60):
        label = "yes" if i % 2 else "no"
        rows.append(f"{20 + i},{100 + i * 3},{label}")
    upload = client.post(
        "/dataset/profile",
        files={"file": ("small-campaign.csv", ("\n".join(rows) + "\n").encode(), "text/csv")},
    )
    assert upload.status_code == 200
    dataset_id = upload.json()["dataset_id"]

    trained = client.post(
        f"/workflow/{dataset_id}/train",
        json={"target": "y", "models": ["logistic_regression"], "row_limit": 20000, "numeric_imputation": "mean", "categorical_imputation": "constant", "scale_numeric": False},
    )
    assert trained.status_code == 200
    result = trained.json()
    assert result["sandbox_only"] is True
    assert result["selected_model"] == "logistic_regression"
    assert result["preprocessing_config"]["numeric_imputation"] == "mean"
    assert result["preprocessing_config"]["categorical_imputation"] == "constant"
    assert result["preprocessing_config"]["scale_numeric"] is False
    assert "final_test_metrics" in result

    prediction = client.post(
        f"/workflow/{dataset_id}/predict",
        json={"run_id": result["run_id"], "customer": {"age": 37, "balance": 611}},
    )
    assert prediction.status_code == 200
    assert prediction.json()["sandbox_only"] is True
    assert 0 <= prediction.json()["subscription_probability"] <= 1


def test_metrics_endpoint_reports_aggregate_data_without_customer_features():
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "prediction_count" in metrics
    assert "mean_latency_ms" in metrics
    assert metrics["drift_monitoring"] == "not_configured"
    assert "raw customer features are not logged" in metrics["privacy_note"]


def test_prediction_threshold_override_is_range_validated():
    response = client.post(
        "/predict",
        json={"customer": {"age": 35}, "threshold": 1.5},
    )
    assert response.status_code == 422
