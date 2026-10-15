"""Compare leakage-aware classifiers for the DepositWise bank marketing dataset."""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
REPORT_PATH = ROOT / "reports/generated/model_comparison.json"
RANDOM_STATE = 42


def read_dataset(path: Path) -> pd.DataFrame:
    """Read the comma- or semicolon-delimited dataset and validate its target."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download the Kaggle dataset and set "
            "DEPOSITWISE_DATA_PATH to the labeled CSV file."
        )
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    if "y" not in frame.columns:
        raise ValueError(
            f"Expected target column 'y'; found {list(frame.columns)}. "
            "Use the labeled bank marketing dataset, not an unlabeled targets file."
        )
    frame = frame.dropna(subset=["y"]).copy()
    frame["y"] = frame["y"].astype(str).str.strip().str.lower()
    frame = frame[frame["y"].isin(["yes", "no"])]
    if frame.empty or frame["y"].nunique() != 2:
        raise ValueError("Target column 'y' must contain both yes and no labels.")
    return frame


def build_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    """Build preprocessing inside the pipeline to avoid validation leakage."""
    # Duration is only known after a call and is not appropriate for pre-call targeting.
    X = X.drop(columns=["duration"], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
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
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def main() -> None:
    frame = read_dataset(DATA_PATH)
    y = frame.pop("y").map({"no": 0, "yes": 1}).astype(int)
    X = frame.drop(columns=["duration"], errors="ignore")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    results = []
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        result = {
            "model": name,
            "threshold": 0.5,
            "average_precision": float(average_precision_score(y_val, probabilities)),
            "roc_auc": float(roc_auc_score(y_val, probabilities)),
            "precision": float(precision_score(y_val, predictions, zero_division=0)),
            "recall": float(recall_score(y_val, predictions, zero_division=0)),
            "f1": float(f1_score(y_val, predictions, zero_division=0)),
        }
        results.append(result)
        fitted[name] = pipeline
        print(f"Finished {name}: average_precision={result['average_precision']:.4f}")

    # Choose by average precision, which is informative for an imbalanced positive class.
    # This validation comparison is for model selection, not an independent test estimate.
    best = max(results, key=lambda result: result["average_precision"])
    selected_pipeline = fitted[best["model"]]
    report = {
        "project": "DepositWise",
        "target": "y (yes=1, no=0)",
        "selection_metric": "average_precision",
        "selected_model": best["model"],
        "excluded_features": ["duration"],
        "rows": int(len(frame)),
        "validation_rows": int(len(y_val)),
        "positive_rate_full_dataset": float(y.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "random_state": RANDOM_STATE,
        "test_set_used": False,
        "models": results,
        "caveat": (
            "Metrics are from one stratified validation split used for model selection. "
            "Use a separate untouched test set before making final performance claims. "
            "The default 0.5 threshold is not a business-optimal threshold."
        ),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": selected_pipeline, "features": list(X.columns),
         "selected_model": best["model"], "threshold": 0.5},
        MODEL_PATH,
    )
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Selected model saved to: {MODEL_PATH}")
    print(f"Comparison report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
