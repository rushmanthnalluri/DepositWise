"""DepositWise inference API."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
app = FastAPI(title="DepositWise API", description="Term-deposit subscription propensity prediction for educational use.", version="0.2.0")

class PredictionRequest(BaseModel):
    customer: dict[str, Any] = Field(..., description="Feature names and values from the dataset, excluding target y and post-call duration.")

@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/")
def root() -> dict[str, str]:
    return {"name": "DepositWise", "docs": "/docs", "health": "/health"}

@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model artifact is missing. Train it with: python -m ml.train")
    try:
        artifact = joblib.load(MODEL_PATH)
        pipeline = artifact["pipeline"]
        expected = artifact["features"]
        row = pd.DataFrame([request.customer]).reindex(columns=expected)
        probability = float(pipeline.predict_proba(row)[0, 1])
        return {"subscription_probability": probability, "predicted_subscription": bool(probability >= 0.5), "threshold": 0.5, "note": "Educational estimate, not a guarantee. Threshold is a baseline and should be tuned on validation data."}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not process supplied features: {exc}") from exc
