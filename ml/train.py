"""Train baseline DepositWise classifiers on a local Kaggle CSV.

Supports comma- or semicolon-separated CSV files. The target column is expected
to be named 'y' and contain yes/no labels. Generated artifacts are not committed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DEPOSITWISE_DATA_PATH", ROOT / "data/raw/bank-full.csv"))
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
REPORT_PATH = ROOT / "reports/generated/baseline_metrics.json"


def read_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle dataset and set "
            "DEPOSITWISE_DATA_PATH to the labeled CSV file."
        )
    # The common UCI/Kaggle banking release is semicolon-delimited.
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(c).strip().strip('"') for c in frame.columns]
    if "y" not in frame.columns:
        raise ValueError(
            f"Expected target column 'y'; found {list(frame.columns)}. "
            "Check the downloaded file and configure the correct labeled CSV."
        )
    frame = frame.dropna(subset=["y"]).copy()
    frame["y"] = frame["y"].astype(str).str.strip().str.lower()
    frame = frame[frame["y"].isin(["yes", "no"])]
    if frame.empty:
        raise ValueError("Target column 'y' contains no usable yes/no labels.")
    return frame


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Call duration is post-contact information. Exclude it from the pre-call
    # targeting model to avoid an operationally misleading feature set.
    X = X.drop(columns=["duration"], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    classifier = LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    )
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def main() -> None:
    frame = read_dataset(DATA_PATH)
    y = frame.pop("y").map({"no": 0, "yes": 1}).astype(int)
    X = frame.drop(columns=["duration"], errors="ignore")
    if y.nunique() != 2:
        raise ValueError("Training requires both target classes: yes and no.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_val)[:, 1]
    # Default 0.5 is reported as a baseline only; tune a threshold on validation
    # for the intended business objective before deployment.
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "model": "LogisticRegression(class_weight=balanced)",
        "target": "y (yes=1, no=0)",
        "excluded_features": ["duration"],
        "validation_rows": int(len(y_val)),
        "positive_rate_validation": float(y_val.mean()),
        "threshold": 0.5,
        "average_precision": float(average_precision_score(y_val, probabilities)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
        "precision": float(precision_score(y_val, predictions, zero_division=0)),
        "recall": float(recall_score(y_val, predictions, zero_division=0)),
        "f1": float(f1_score(y_val, predictions, zero_division=0)),
        "features": list(X.columns),
        "note": "Validation metrics only; not an independent final test result.",
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": list(X.columns)}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
