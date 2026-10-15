"""DepositWise inference API and browser workbench."""
from __future__ import annotations
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
REPORT_PATH = ROOT / "reports/generated/model_comparison.json"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
app = FastAPI(title="DepositWise API", description="Term-deposit subscription propensity prediction for educational use.", version="0.3.0")


class PredictionRequest(BaseModel):
    customer: dict[str, Any] = Field(..., description="Dataset feature names and values, excluding target y and post-call duration.")


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
            frame = pd.read_csv(io.BytesIO(raw), low_memory=False)
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


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing. Download the dataset and run: python -m ml.train")
    try:
        artifact = joblib.load(MODEL_PATH)
        pipeline = artifact["pipeline"]
        expected = artifact["features"]
        row = pd.DataFrame([request.customer]).reindex(columns=expected)
        probability = float(pipeline.predict_proba(row)[0, 1])
        threshold = float(artifact.get("threshold", 0.5))
        return {
            "subscription_probability": probability,
            "predicted_subscription": bool(probability >= threshold),
            "threshold": threshold,
            "selected_model": artifact.get("selected_model", "unknown"),
            "note": "Educational estimate, not a guarantee. Validate model quality and tune the decision threshold before operational use.",
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process supplied features: {exc}") from exc
