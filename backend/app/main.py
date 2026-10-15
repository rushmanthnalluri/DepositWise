"""DepositWise inference API and browser workbench."""
from __future__ import annotations
import io
import json
import os
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ml.train import choose_f1_threshold, evaluate

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
REPORT_PATH = ROOT / "reports/generated/model_comparison.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_CACHED_DATASETS = 8
PREDICTION_COUNT = 0
DRIFT_SAMPLE_COUNT = 0
FEATURE_REFERENCE: dict[str, dict[str, list[float]]] = {}
LIVE_FEATURE_COUNTS: dict[str, list[int]] = {}
PREDICTION_LATENCIES = deque(maxlen=500)
LAST_PREDICTION_AT: str | None = None
DATASET_CACHE: dict[str, pd.DataFrame] = {}
SANDBOX_MODELS: dict[str, dict[str, Any]] = {}
SANDBOX_RUN_HISTORY = deque(maxlen=25)
app = FastAPI(title="DepositWise API", description="Term-deposit subscription propensity prediction for educational use.", version="0.3.0")


class PredictionRequest(BaseModel):
    customer: dict[str, Any] = Field(..., description="Dataset feature names and values, excluding target y and post-call duration.")
    threshold: float | None = Field(default=None, ge=0.0, le=1.0, description="Optional decision threshold override for what-if analysis.")


class SandboxTrainingRequest(BaseModel):
    target: str = Field(..., min_length=1, max_length=128)
    models: list[str] = Field(default_factory=lambda: ["logistic_regression", "random_forest", "extra_trees"])
    row_limit: int = Field(default=20000, ge=100, le=20000)
    numeric_imputation: Literal["median", "mean"] = "median"
    categorical_imputation: Literal["most_frequent", "constant"] = "most_frequent"
    scale_numeric: bool = True


class SandboxPredictionRequest(BaseModel):
    run_id: str = Field(..., min_length=1, max_length=64)
    customer: dict[str, Any]


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "DepositWise", "dashboard": "/dashboard", "docs": "/docs", "health": "/health"}


@app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
def dashboard() -> HTMLResponse:
    page = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
    styles = (FRONTEND_DIR / "styles.css").read_text(encoding="utf-8")
    script = (FRONTEND_DIR / "app.js").read_text(encoding="utf-8")
    page = page.replace('<link rel="stylesheet" href="/static/styles.css">', f"<style>{styles}</style>")
    page = page.replace('<script src="/static/app.js" defer></script>', "")
    page = page.replace("</body>", f"<script>{script}</script></body>")
    return HTMLResponse(content=page)



def compute_psi(reference: list[float], observed: list[float]) -> float:
    """Population Stability Index from binned proportions, with numerical smoothing."""
    if len(reference) != len(observed) or not reference:
        return 0.0
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(observed, dtype=float)
    ref = np.clip(ref, 1e-6, None)
    cur = np.clip(cur, 1e-6, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    global FEATURE_REFERENCE
    if not FEATURE_REFERENCE and MODEL_PATH.exists():
        try:
            FEATURE_REFERENCE = joblib.load(MODEL_PATH).get("feature_reference", {})
        except Exception:
            FEATURE_REFERENCE = {}
    latencies = list(PREDICTION_LATENCIES)
    drift_features = []
    for feature, reference in FEATURE_REFERENCE.items():
        counts = LIVE_FEATURE_COUNTS.get(feature, [])
        total = sum(counts)
        if total < 50:
            continue
        observed = [count / total for count in counts]
        psi = compute_psi(reference.get("proportions", []), observed)
        drift_features.append({"feature": feature, "psi": psi, "samples": total})
    drift_features.sort(key=lambda item: item["psi"], reverse=True)
    max_psi = max((item["psi"] for item in drift_features), default=None)
    drift_status = "insufficient_data" if max_psi is None else "drift" if max_psi >= 0.2 else "warning" if max_psi >= 0.1 else "stable"
    return {
        "prediction_count": PREDICTION_COUNT,
        "recent_sample_count": len(latencies),
        "mean_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
        "last_prediction_at": LAST_PREDICTION_AT,
        "model_loaded": MODEL_PATH.exists(),
        "drift_monitoring": "numeric_psi" if FEATURE_REFERENCE else "not_configured",
        "drift_sample_count": DRIFT_SAMPLE_COUNT,
        "drift_status": drift_status,
        "drift_features": drift_features[:10],
        "psi_thresholds": {"warning": 0.1, "drift": 0.2, "minimum_feature_samples": 50},
        "privacy_note": "Only aggregate counts, binned numeric feature frequencies, and inference latency are retained in memory; raw customer features are not logged.",
    }



@app.get("/model-info")
def model_info() -> dict[str, Any]:
    """Expose the actual evaluation report generated during training."""
    if not REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Evaluation report is not available yet.")
    try:
        report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return {
            "selected_model": report.get("selected_model"),
            "rows": report.get("rows"),
            "positive_rate": report.get("positive_rate"),
            "positive_rate_full_dataset": report.get("positive_rate_full_dataset"),
            "selection_metric": report.get("selection_metric"),
            "validation_model_comparison": report.get("validation_model_comparison", []),
            "threshold_selection": report.get("threshold_selection"),
            "final_test_metrics_at_0_5": report.get("final_test_metrics_at_0_5"),
            "final_test_metrics": report.get("final_test_metrics"),
            "test_calibration_curve": report.get("test_calibration_curve", []),
            "top_encoded_feature_importance": report.get("top_encoded_feature_importance", []),
            "excluded_features": report.get("excluded_features", []),
            "limitations": report.get("limitations", []),
        }
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=500, detail="Evaluation report could not be read.") from exc


@app.post("/dataset/profile")
async def profile_dataset(file: UploadFile = File(...)) -> dict[str, Any]:
    """Profile an uploaded CSV/XLSX in memory; never saves or trains on the upload."""
    filename = Path(file.filename or "dataset").name
    suffix = Path(filename).suffix.lower()
    if suffix not in {".csv", ".xlsx"}:
        raise HTTPException(status_code=415, detail="Upload a CSV or XLSX file.")
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds the 10 MB upload limit.")
    try:
        if suffix == ".xlsx":
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                members = archive.infolist()
                if len(members) > 10_000:
                    raise HTTPException(status_code=422, detail="Excel file contains too many internal records.")
                if sum(member.file_size for member in members) > 100 * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Excel file expands beyond the 100 MB safety limit.")
            frame = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
        else:
            frame = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not parse this file. Check that it is a valid CSV or XLSX.") from exc
    if frame.empty or len(frame.columns) == 0:
        raise HTTPException(status_code=422, detail="The file contains no usable rows or columns.")
    if len(frame.columns) > 300:
        raise HTTPException(status_code=422, detail="The file has more than 300 columns; reduce it before profiling.")
    if frame.columns.duplicated().any():
        raise HTTPException(status_code=422, detail="The file contains duplicate column names. Rename them before profiling.")
    if len(frame) > 250_000:
        raise HTTPException(status_code=422, detail="The file has more than 250,000 rows; sample it before profiling.")
    if len(DATASET_CACHE) >= MAX_CACHED_DATASETS:
        DATASET_CACHE.pop(next(iter(DATASET_CACHE)))
    dataset_id = uuid4().hex
    DATASET_CACHE[dataset_id] = frame.copy(deep=True)
    columns = []
    for name in frame.columns:
        series = frame[name]
        columns.append({
            "name": str(name),
            "dtype": str(series.dtype),
            "missing": int(series.isna().sum()),
            "unique": int(series.nunique(dropna=True)),
        })
    sample_frame = frame.head(8).astype(object).where(pd.notna(frame.head(8)), None)
    sample = json.loads(sample_frame.to_json(orient="records", date_format="iso"))
    target_candidates = [name for name in frame.columns if str(name).strip().lower() in {"y", "target", "subscribed", "subscription"}]
    target_summary = None
    if target_candidates:
        target_name = target_candidates[0]
        target_summary = {
            "column": str(target_name),
            "counts": {str(key): int(value) for key, value in frame[target_name].fillna("<missing>").astype(str).value_counts(dropna=False).items()},
        }
    return {
        "filename": filename,
        "dataset_id": dataset_id,
        "rows": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "missing_cells": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
        "numeric_columns": int(len(frame.select_dtypes(include="number").columns)),
        "categorical_columns": int(len(frame.select_dtypes(exclude="number").columns)),
        "target_summary": target_summary,
        "columns": columns,
        "sample": sample,
        "persisted": False,
        "training_triggered": False,
    }


def _get_uploaded_dataset(dataset_id: str) -> pd.DataFrame:
    frame = DATASET_CACHE.get(dataset_id)
    if frame is None:
        raise HTTPException(status_code=404, detail="This temporary dataset profile has expired or was evicted. Upload the file again.")
    return frame


@app.get("/workflow/{dataset_id}/features")
def uploaded_feature_inventory(dataset_id: str) -> dict[str, Any]:
    frame = _get_uploaded_dataset(dataset_id)
    return {
        "dataset_id": dataset_id,
        "columns": [
            {
                "name": str(name),
                "dtype": str(frame[name].dtype),
                "role": "target candidate" if str(name).strip().lower() in {"y", "target", "subscribed", "subscription"} else "feature",
                "missing": int(frame[name].isna().sum()),
                "unique": int(frame[name].nunique(dropna=True)),
            }
            for name in frame.columns
        ],
    }


@app.get("/workflow/{dataset_id}/statistics")
def uploaded_descriptive_statistics(dataset_id: str) -> dict[str, Any]:
    frame = _get_uploaded_dataset(dataset_id)
    numeric = frame.select_dtypes(include="number")
    records = json.loads(numeric.describe().transpose().reset_index().rename(columns={"index": "column"}).to_json(orient="records"))
    return {"dataset_id": dataset_id, "numeric_column_count": int(len(numeric.columns)), "statistics": records}


@app.get("/workflow/{dataset_id}/missing")
def uploaded_missing_values(dataset_id: str) -> dict[str, Any]:
    frame = _get_uploaded_dataset(dataset_id)
    total = max(1, len(frame))
    missing = [
        {"column": str(name), "missing": int(frame[name].isna().sum()), "missing_fraction": float(frame[name].isna().sum() / total)}
        for name in frame.columns
    ]
    return {"dataset_id": dataset_id, "rows": int(len(frame)), "missing_cells": int(frame.isna().sum().sum()), "duplicate_rows": int(frame.duplicated().sum()), "columns": missing}


@app.get("/workflow/{dataset_id}/visualization")
def uploaded_visualization_summary(dataset_id: str) -> dict[str, Any]:
    frame = _get_uploaded_dataset(dataset_id)
    target_candidates = [name for name in frame.columns if str(name).strip().lower() in {"y", "target", "subscribed", "subscription"}]
    target = None
    categorical_target_rates = []
    if target_candidates:
        column = target_candidates[0]
        counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False)
        target = {"column": str(column), "labels": [str(key) for key in counts.index], "counts": [int(value) for value in counts.values]}
        raw_target = frame[column]
        labels = raw_target.astype(str)
        known = sorted(raw_target.dropna().astype(str).unique().tolist())
        positive_label = next((label for label in known if label.lower() in {"yes", "1", "true"}), known[-1] if known else None)
        positive = labels.eq(positive_label).astype(float).where(raw_target.notna()) if positive_label is not None else pd.Series(np.nan, index=frame.index)
        for feature in frame.select_dtypes(exclude="number").columns:
            if feature == column or frame[feature].nunique(dropna=True) > 20:
                continue
            grouped = pd.DataFrame({"group": frame[feature].fillna("<missing>").astype(str), "positive": positive}).groupby("group", dropna=False)
            for group, values in grouped:
                valid = values["positive"].dropna()
                if len(valid) >= 10:
                    categorical_target_rates.append({
                        "feature": str(feature),
                        "group": str(group),
                        "count": int(len(valid)),
                        "positive_rate": float(valid.mean()),
                    })
        categorical_target_rates.sort(key=lambda item: (item["feature"], -item["positive_rate"]))
    numeric = frame.select_dtypes(include="number")
    correlations = []
    numeric_distributions = []
    for column in list(numeric.columns)[:8]:
        values = numeric[column].dropna().to_numpy(dtype=float)
        if values.size:
            counts, edges = np.histogram(values, bins=10)
            numeric_distributions.append({
                "feature": str(column),
                "labels": [f"{edges[i]:.3g}–{edges[i + 1]:.3g}" for i in range(len(edges) - 1)],
                "counts": [int(value) for value in counts],
                "missing": int(numeric[column].isna().sum()),
            })
    categorical_distributions = []
    for column in list(frame.select_dtypes(exclude="number").columns)[:10]:
        counts = frame[column].fillna("<missing>").astype(str).value_counts(dropna=False).head(8)
        categorical_distributions.append({
            "feature": str(column),
            "labels": [str(value) for value in counts.index],
            "counts": [int(value) for value in counts.values],
            "other_count": max(0, int(frame[column].notna().sum()) - int(counts.sum())),
        })
    if len(numeric.columns) > 1:
        corr = numeric.corr(numeric_only=True).fillna(0)
        names = list(corr.columns)
        for i, column in enumerate(names):
            for other in names[i + 1:]:
                correlations.append({"feature_a": str(column), "feature_b": str(other), "correlation": float(corr.loc[column, other])})
        correlations.sort(key=lambda item: abs(item["correlation"]), reverse=True)
    return {
        "dataset_id": dataset_id,
        "target_distribution": target,
        "categorical_target_rates": categorical_target_rates[:200],
        "numeric_distributions": numeric_distributions,
        "categorical_distributions": categorical_distributions,
        "top_numeric_correlations": correlations[:30],
    }




def build_sandbox_pipeline(X: pd.DataFrame, classifier, numeric_imputation: str, categorical_imputation: str, scale_numeric: bool) -> Pipeline:
    """Build a configurable preprocessing pipeline for sandbox runs only."""
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy=numeric_imputation))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical_imputer = SimpleImputer(strategy="constant", fill_value="missing") if categorical_imputation == "constant" else SimpleImputer(strategy="most_frequent")
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), numeric),
        ("categorical", Pipeline([("imputer", categorical_imputer), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("classifier", classifier)])




@app.get("/experiments")
def experiments() -> dict[str, Any]:
    """Return recent in-memory sandbox experiment summaries."""
    return {"runs": list(SANDBOX_RUN_HISTORY), "persistent": False, "limit": 25}


@app.post("/workflow/{dataset_id}/score-batch")
def score_uploaded_batch(dataset_id: str, threshold: float | None = Query(default=None, ge=0.0, le=1.0)) -> StreamingResponse:
    """Score up to 20,000 uploaded rows with the production model and return a CSV download."""
    frame = _get_uploaded_dataset(dataset_id).copy(deep=True)
    if len(frame) > 20000:
        raise HTTPException(status_code=413, detail="Batch scoring is limited to 20,000 rows per request.")
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Production model artifact is unavailable.")
    try:
        artifact = joblib.load(MODEL_PATH)
        expected = artifact["features"]
        pipeline = artifact["pipeline"]
        target_candidates = [name for name in frame.columns if str(name).strip().lower() in {"y", "target", "subscribed", "subscription"}]
        source_features = frame.drop(columns=target_candidates + ["duration"], errors="ignore")
        missing_features = [name for name in expected if name not in source_features.columns]
        if missing_features:
            raise HTTPException(status_code=422, detail="Dataset is missing required production features: " + ", ".join(missing_features[:10]))
        features = source_features.reindex(columns=expected)
        probabilities = pipeline.predict_proba(features)[:, 1]
        selected_threshold = float(threshold if threshold is not None else artifact.get("threshold", 0.5))
        output = frame.copy()
        output["depositwise_subscription_probability"] = probabilities
        output["depositwise_predicted_subscription"] = np.where(probabilities >= selected_threshold, "yes", "no")
        for column in output.select_dtypes(include=["object", "string"]).columns:
            output[column] = output[column].map(lambda value: "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value)
        buffer = io.StringIO()
        output.to_csv(buffer, index=False)
        content = buffer.getvalue()
        return StreamingResponse(
            iter([content]),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="depositwise_scored.csv"',
                "X-Scored-Rows": str(len(output)),
                "X-Decision-Threshold": str(selected_threshold),
            },
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Could not batch-score this dataset. Check that its feature schema matches the production model.") from exc


@app.post("/workflow/{dataset_id}/train")
def train_sandbox_models(dataset_id: str, request: SandboxTrainingRequest) -> dict[str, Any]:
    """Train temporary binary classifiers on an upload without replacing the production model."""
    frame = _get_uploaded_dataset(dataset_id).copy(deep=True)
    if request.target not in frame.columns:
        raise HTTPException(status_code=422, detail="The selected target column does not exist in this dataset.")
    frame = frame.dropna(subset=[request.target])
    if len(frame) > request.row_limit:
        raise HTTPException(status_code=413, detail=f"Sandbox training is limited to {request.row_limit:,} rows. Upload a smaller sample.")
    labels = frame[request.target].astype(str)
    classes = sorted(labels.unique().tolist())
    if len(classes) != 2:
        raise HTTPException(status_code=422, detail="Sandbox training currently supports binary classification targets only.")
    if int(labels.value_counts().min()) < 10:
        raise HTTPException(status_code=422, detail="Each target class needs at least 10 rows for a reliable split.")
    X = frame.drop(columns=[request.target]).drop(columns=["duration"], errors="ignore")
    X = X.drop(columns=[column for column in X.columns if X[column].isna().all()])
    if X.empty:
        raise HTTPException(status_code=422, detail="No usable feature columns remain after excluding the target and empty fields.")
    if len(X.columns) > 100:
        raise HTTPException(status_code=422, detail="Sandbox training is limited to 100 feature columns.")
    y = labels.map({classes[0]: 0, classes[1]: 1}).astype(int)
    try:
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=43, stratify=y_train_val)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="This dataset is too small or imbalanced for stratified splits.") from exc
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=1, random_state=42),
        "extra_trees": ExtraTreesClassifier(n_estimators=60, min_samples_leaf=2, class_weight="balanced", n_jobs=1, random_state=42),
    }
    selected_names = list(dict.fromkeys(request.models))
    if not selected_names or any(name not in candidates for name in selected_names):
        raise HTTPException(status_code=422, detail="Choose supported models: logistic_regression, random_forest, extra_trees.")
    comparison = []
    validation_probabilities = {}
    for name in selected_names:
        try:
            pipeline = build_sandbox_pipeline(X_train, candidates[name], request.numeric_imputation, request.categorical_imputation, request.scale_numeric)
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_val)[:, 1]
            result = evaluate(y_val, probabilities, threshold=0.5)
            comparison.append({"model": name, **result})
            validation_probabilities[name] = probabilities
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not train {name}; inspect missing values and column types.") from exc
    champion = max(comparison, key=lambda result: result["average_precision"])
    champion_name = champion["model"]
    threshold, validation_f1 = choose_f1_threshold(y_val, validation_probabilities[champion_name])
    final_pipeline = build_sandbox_pipeline(X_train_val, candidates[champion_name], request.numeric_imputation, request.categorical_imputation, request.scale_numeric)
    final_pipeline.fit(X_train_val, y_train_val)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    baseline_metrics = evaluate(y_test, test_probabilities, threshold=0.5)
    tuned_metrics = evaluate(y_test, test_probabilities, threshold=threshold)
    run_id = uuid4().hex
    if len(SANDBOX_MODELS) >= MAX_CACHED_DATASETS:
        SANDBOX_MODELS.pop(next(iter(SANDBOX_MODELS)))
    SANDBOX_MODELS[run_id] = {"pipeline": final_pipeline, "features": list(X.columns), "threshold": float(threshold), "selected_model": champion_name}
    SANDBOX_RUN_HISTORY.append({
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": request.target,
        "selected_model": champion_name,
        "rows": int(len(frame)),
        "feature_count": int(len(X.columns)),
        "validation_average_precision": float(champion["average_precision"]),
        "test_roc_auc": float(tuned_metrics["roc_auc"]),
        "test_f1": float(tuned_metrics["f1"]),
        "threshold": float(threshold),
        "preprocessing_config": {"numeric_imputation": request.numeric_imputation, "categorical_imputation": request.categorical_imputation, "scale_numeric": request.scale_numeric},
    })
    return {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "sandbox_only": True,
        "rows": int(len(frame)),
        "feature_count": int(len(X.columns)),
        "preprocessing_config": {"numeric_imputation": request.numeric_imputation, "categorical_imputation": request.categorical_imputation, "scale_numeric": request.scale_numeric},
        "target": request.target,
        "target_classes": classes,
        "selected_model": champion_name,
        "selection_metric": "validation average_precision",
        "validation_model_comparison": comparison,
        "threshold_selection": {"method": "maximize F1 on validation split", "threshold": float(threshold), "validation_f1_at_selected_threshold": float(validation_f1)},
        "final_test_metrics_at_0_5": baseline_metrics,
        "final_test_metrics": tuned_metrics,
        "excluded_features": ["duration"] if "duration" in frame.columns else [],
        "limitations": ["Sandbox artifacts are held in process memory only.", "This exploratory run uses a random split; temporal and subgroup validation are not included.", "The sandbox run never replaces the deployed production model."],
    }


@app.post("/workflow/{dataset_id}/predict")
def predict_sandbox_model(dataset_id: str, request: SandboxPredictionRequest) -> dict[str, Any]:
    _get_uploaded_dataset(dataset_id)
    artifact = SANDBOX_MODELS.get(request.run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Sandbox model run is missing or was evicted. Train the model again.")
    try:
        row = pd.DataFrame([request.customer]).reindex(columns=artifact["features"])
        probability = float(artifact["pipeline"].predict_proba(row)[0, 1])
        threshold = float(artifact["threshold"])
        return {"run_id": request.run_id, "selected_model": artifact["selected_model"], "subscription_probability": probability, "threshold": threshold, "predicted_positive": bool(probability >= threshold), "sandbox_only": True}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not score this sandbox profile: {exc}") from exc


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    global PREDICTION_COUNT, LAST_PREDICTION_AT, DRIFT_SAMPLE_COUNT, FEATURE_REFERENCE
    started = time.perf_counter()
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing. Download the dataset and run: python -m ml.train")
    try:
        artifact = joblib.load(MODEL_PATH)
        if not FEATURE_REFERENCE:
            FEATURE_REFERENCE = artifact.get("feature_reference", {})
        pipeline = artifact["pipeline"]
        expected = artifact["features"]
        row = pd.DataFrame([request.customer]).reindex(columns=expected)
        probability = float(pipeline.predict_proba(row)[0, 1])
        explanations = []
        baseline_customer = artifact.get("baseline_customer")
        if isinstance(baseline_customer, dict) and baseline_customer:
            try:
                baseline_row = pd.DataFrame([baseline_customer]).reindex(columns=expected)
                baseline_probability = float(pipeline.predict_proba(baseline_row)[0, 1])
                for feature in expected:
                    candidate = dict(baseline_customer)
                    candidate[feature] = request.customer.get(feature, baseline_customer.get(feature))
                    candidate_row = pd.DataFrame([candidate]).reindex(columns=expected)
                    candidate_probability = float(pipeline.predict_proba(candidate_row)[0, 1])
                    impact = candidate_probability - baseline_probability
                    explanations.append({
                        "feature": feature,
                        "impact_probability": impact,
                        "direction": "higher" if impact > 0 else "lower" if impact < 0 else "neutral",
                        "customer_value": request.customer.get(feature),
                        "baseline_value": baseline_customer.get(feature),
                    })
                explanations.sort(key=lambda item: abs(item["impact_probability"]), reverse=True)
                explanations = explanations[:5]
            except Exception:
                explanations = []
        threshold = float(request.threshold if request.threshold is not None else artifact.get("threshold", 0.5))
        drift_updated = False
        for feature, reference in FEATURE_REFERENCE.items():
            value = request.customer.get(feature)
            if value is None:
                continue
            try:
                numeric_value = float(value)
                if not np.isfinite(numeric_value):
                    continue
                edges = reference.get("edges", [])
                proportions = reference.get("proportions", [])
                if len(edges) < 2 or not proportions:
                    continue
                bucket = int(np.searchsorted(edges[1:-1], numeric_value, side="right"))
                bucket = max(0, min(bucket, len(proportions) - 1))
                counts = LIVE_FEATURE_COUNTS.setdefault(feature, [0] * len(proportions))
                counts[bucket] += 1
                drift_updated = True
            except (TypeError, ValueError):
                continue
        if drift_updated:
            DRIFT_SAMPLE_COUNT += 1
        PREDICTION_COUNT += 1
        LAST_PREDICTION_AT = datetime.now(timezone.utc).isoformat()
        PREDICTION_LATENCIES.append((time.perf_counter() - started) * 1000)
        return {
            "subscription_probability": probability,
            "predicted_subscription": bool(probability >= threshold),
            "threshold": threshold,
            "selected_model": artifact.get("selected_model", "unknown"),
            "top_feature_impacts": explanations,
            "explanation_method": "single-feature baseline sensitivity; not causal and not SHAP",
            "note": "Educational estimate, not a guarantee. Validate model quality and tune the decision threshold before operational use.",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process supplied features: {exc}") from exc
