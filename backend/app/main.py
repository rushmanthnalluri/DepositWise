"""DepositWise inference API and browser workbench."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend"
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
app = FastAPI(title="DepositWise API", description="Term-deposit subscription propensity prediction for educational use.", version="0.2.0")

class PredictionRequest(BaseModel):
    customer: dict[str, Any] = Field(..., description="Feature names and values from the dataset, excluding target y and post-call duration.")

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
    page = page.replace('<script src="/static/app.js" defer></script>', f"<script>{script}</script>")
    return HTMLResponse(content=page)

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
        return {"subscription_probability": probability, "predicted_subscription": bool(probability >= threshold), "threshold": threshold, "selected_model": artifact.get("selected_model", "unknown"), "note": "Educational estimate, not a guarantee. Validate model quality and tune the decision threshold before operational use."}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process supplied features: {exc}") from exc
