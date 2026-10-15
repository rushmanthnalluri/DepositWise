"""Compare leakage-aware classifiers on the public UCI/Kaggle bank marketing data.

If data/raw/bank-full.csv is absent, fetch the official UCI archive, which hosts
this same 45,211-row bank-full.csv dataset. Set DEPOSITWISE_DATA_PATH to override.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = Path(os.getenv("DEPOSITWISE_DATA_PATH", ROOT / "data/raw/bank-full.csv"))
MODEL_PATH = Path(os.getenv("DEPOSITWISE_MODEL_PATH", ROOT / "models/depositwise_model.joblib"))
REPORT_PATH = Path(os.getenv("DEPOSITWISE_REPORT_PATH", ROOT / "reports/generated/model_comparison.json"))
RANDOM_STATE = 42
UCI_ARCHIVE_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def ensure_dataset(path: Path) -> None:
    """Fetch the upstream archive only when no local CSV is available."""
    if path.exists():
        return
    if os.getenv("DEPOSITWISE_DISABLE_DATA_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        raise FileNotFoundError(f"Dataset not found at {path} and automatic download is disabled.")
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Dataset not found at {path}; downloading official UCI Bank Marketing archive.")
    try:
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "bank-marketing.zip"
            request = urllib.request.Request(UCI_ARCHIVE_URL, headers={"User-Agent": "DepositWise/1.0 educational project"})
            with urllib.request.urlopen(request, timeout=45) as response, archive_path.open("wb") as output:
                output.write(response.read())
            csv_bytes = None
            with zipfile.ZipFile(archive_path) as archive:
                # UCI's top-level archive currently contains nested bank.zip and
                # bank-additional.zip files rather than loose CSV files.
                direct_member = next((name for name in archive.namelist() if name.endswith("/bank-full.csv") or name == "bank-full.csv"), None)
                if direct_member is not None:
                    csv_bytes = archive.read(direct_member)
                else:
                    nested_archives = [name for name in archive.namelist() if name.lower().endswith(".zip")]
                    for nested_name in nested_archives:
                        with zipfile.ZipFile(io.BytesIO(archive.read(nested_name))) as nested:
                            member = next((name for name in nested.namelist() if name.endswith("/bank-full.csv") or name == "bank-full.csv"), None)
                            if member is not None:
                                csv_bytes = nested.read(member)
                                break
            if csv_bytes is None:
                raise ValueError("The UCI archive and its nested ZIP files did not contain bank-full.csv.")
            path.write_bytes(csv_bytes)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Could not obtain dataset from the official UCI archive: {exc}. Download bank-full.csv manually and set DEPOSITWISE_DATA_PATH.") from exc
    print(f"Downloaded dataset to {path}")


def read_dataset(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSV and validate the label column."""
    ensure_dataset(path)
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = [str(column).strip().strip('"') for column in frame.columns]
    if "y" not in frame.columns:
        raise ValueError(f"Expected target column 'y'; found {list(frame.columns)}. Use the labeled bank marketing dataset.")
    frame = frame.dropna(subset=["y"]).copy()
    frame["y"] = frame["y"].astype(str).str.strip().str.lower()
    frame = frame[frame["y"].isin(["yes", "no"])]
    if frame.empty or frame["y"].nunique() != 2:
        raise ValueError("Target column 'y' must contain both yes and no labels.")
    return frame


def build_pipeline(X: pd.DataFrame, classifier) -> Pipeline:
    """Fit imputation, scaling and encoding inside each model pipeline."""
    X = X.drop(columns=["duration"], errors="ignore")
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [column for column in X.columns if column not in numeric]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def evaluate(y_true, probabilities) -> dict[str, float]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "threshold": 0.5,
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def main() -> None:
    frame = read_dataset(DATA_PATH)
    y = frame.pop("y").map({"no": 0, "yes": 1}).astype(int)
    # Exclude call duration: it is only known after the contact and leaks future information.
    X = frame.drop(columns=["duration"], errors="ignore")
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_STATE + 1, stratify=y_train_val)

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE),
        "extra_trees": ExtraTreesClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE),
    }
    comparison = []
    for name, classifier in candidates.items():
        pipeline = build_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        metrics = evaluate(y_val, pipeline.predict_proba(X_val)[:, 1])
        comparison.append({"model": name, **metrics})
        print(f"Validation {name}: average_precision={metrics['average_precision']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    best = max(comparison, key=lambda result: result["average_precision"])
    # The test set did not influence model selection; score it only after refitting the winner.
    final_pipeline = build_pipeline(X_train_val, candidates[best["model"]])
    final_pipeline.fit(X_train_val, y_train_val)
    test_metrics = evaluate(y_test, final_pipeline.predict_proba(X_test)[:, 1])
    report = {
        "project": "DepositWise",
        "dataset_source": "UCI Machine Learning Repository, Bank Marketing (dataset 222), bank-full.csv; equivalent upstream dataset used by Kaggle copies",
        "dataset_source_url": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
        "rows": int(len(frame)),
        "feature_count_before_exclusions": int(frame.shape[1]),
        "target": "y (yes=1, no=0)",
        "excluded_features": ["duration"],
        "selection_metric": "validation average_precision",
        "selected_model": best["model"],
        "split_rows": {"train": int(len(y_train)), "validation": int(len(y_val)), "test": int(len(y_test))},
        "positive_rate_full_dataset": float(y.mean()),
        "positive_rate_validation": float(y_val.mean()),
        "positive_rate_test": float(y_test.mean()),
        "random_state": RANDOM_STATE,
        "validation_model_comparison": comparison,
        "final_test_metrics": test_metrics,
        "threshold_note": "0.5 is a baseline, not a threshold optimized for campaign costs or capacity.",
        "limitations": ["One stratified random split; temporal validation may be more realistic for future campaigns.", "Assess calibration, subgroup performance, and campaign utility before operational decisions."],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final_pipeline, "features": list(X.columns), "selected_model": best["model"], "threshold": 0.5}, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Selected model saved: {MODEL_PATH}")
    print(f"Evaluation report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
