import io
import joblib
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from ml.train import build_pipeline

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
    history = client.get("/experiments")
    assert history.status_code == 200
    assert history.json()["persistent"] is False
    assert any(run["run_id"] == result["run_id"] for run in history.json()["runs"])

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
    assert metrics["drift_monitoring"] in {"not_configured", "numeric_psi"}
    assert "raw customer features are not logged" in metrics["privacy_note"]


def test_prediction_threshold_override_is_range_validated():
    response = client.post(
        "/predict",
        json={"customer": {"age": 35}, "threshold": 1.5},
    )
    assert response.status_code == 422


def test_predict_returns_baseline_sensitivity_and_honors_threshold(tmp_path, monkeypatch):
    import backend.app.main as main

    X = pd.DataFrame({
        "age": list(range(20, 40)),
        "balance": [100 + i * 10 for i in range(20)],
        "job": ["admin." if i % 2 == 0 else "technician" for i in range(20)],
    })
    y = pd.Series([0, 1] * 10)
    pipeline = build_pipeline(X, LogisticRegression(max_iter=500))
    pipeline.fit(X, y)
    model_path = tmp_path / "sandboxed-test-model.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "features": list(X.columns),
        "selected_model": "logistic_regression",
        "threshold": 0.5,
        "baseline_customer": {"age": 29.5, "balance": 195.0, "job": "admin."},
    }, model_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)

    response = client.post(
        "/predict",
        json={"customer": {"age": 35, "balance": 300, "job": "technician"}, "threshold": 0.9},
    )
    assert response.status_code == 200
    result = response.json()
    assert result["threshold"] == 0.9
    assert result["explanation_method"].startswith("single-feature baseline sensitivity")
    assert len(result["top_feature_impacts"]) <= 5
    assert all("impact_probability" in item for item in result["top_feature_impacts"])


def test_psi_detects_distribution_shift_without_storing_raw_rows():
    from backend.app.main import compute_psi

    assert compute_psi([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert compute_psi([0.8, 0.2], [0.2, 0.8]) > 0.2


def test_batch_scoring_exports_probabilities_and_classifications(tmp_path, monkeypatch):
    import backend.app.main as main

    X = pd.DataFrame({
        "age": list(range(20, 40)),
        "balance": [100 + i * 10 for i in range(20)],
        "job": ["admin." if i % 2 == 0 else "technician" for i in range(20)],
    })
    y = pd.Series([0, 1] * 10)
    pipeline = build_pipeline(X, LogisticRegression(max_iter=500))
    pipeline.fit(X, y)
    model_path = tmp_path / "batch-test-model.joblib"
    joblib.dump({"pipeline": pipeline, "features": list(X.columns), "selected_model": "logistic_regression", "threshold": 0.5}, model_path)
    monkeypatch.setattr(main, "MODEL_PATH", model_path)

    dataset_rows = ["age,balance,job,y"]
    for i in range(20):
        dataset_rows.append(f"{20 + i},{100 + i * 10},{'admin.' if i % 2 == 0 else 'technician'},{'yes' if i % 2 else 'no'}")
    upload = client.post(
        "/dataset/profile",
        files={"file": ("batch.csv", ("\n".join(dataset_rows) + "\n").encode(), "text/csv")},
    )
    assert upload.status_code == 200

    response = client.post(f"/workflow/{upload.json()['dataset_id']}/score-batch")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    scored = pd.read_csv(io.StringIO(response.text))
    assert len(scored) == 20
    assert "depositwise_subscription_probability" in scored.columns
    assert "depositwise_predicted_subscription" in scored.columns


def test_dataset_profile_detects_semicolon_delimited_bank_marketing_csv():
    response = client.post(
        "/dataset/profile",
        files={"file": ("bank-full.csv", b"age;job;y\n35;management;yes\n42;technician;no\n", "text/csv")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["column_count"] == 3
    assert [column["name"] for column in data["columns"]] == ["age", "job", "y"]
    assert data["target_summary"]["column"] == "y"


def test_uploaded_eda_reports_subscription_rates_and_feature_distributions():
    rows = ["age,job,y"]
    for i in range(40):
        job = "admin." if i < 20 else "technician"
        target = "yes" if i % 2 else "no"
        rows.append(f"{20 + i},{job},{target}")
    upload = client.post(
        "/dataset/profile",
        files={"file": ("segments.csv", ("\n".join(rows) + "\n").encode(), "text/csv")},
    )
    assert upload.status_code == 200
    dataset_id = upload.json()["dataset_id"]
    response = client.get(f"/workflow/{dataset_id}/visualization")
    assert response.status_code == 200
    rates = response.json()["categorical_target_rates"]
    assert len(rates) == 2
    assert all(item["count"] == 20 for item in rates)
    assert all(item["positive_rate"] == 0.5 for item in rates)
    payload = response.json()
    assert payload["numeric_distributions"]
    assert payload["numeric_distributions"][0]["feature"] == "age"
    assert sum(payload["numeric_distributions"][0]["counts"]) == 40
    assert payload["categorical_distributions"]
    assert payload["categorical_distributions"][0]["feature"] == "job"
    assert sum(payload["categorical_distributions"][0]["counts"]) == 40
    assert "correlation_matrix" in payload
    assert payload["numeric_target_summary"]
    assert {item["target_label"] for item in payload["numeric_target_summary"]} == {"yes", "no"}
