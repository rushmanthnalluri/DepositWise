"""Compare leakage-aware classifiers and evaluate the selected model honestly."""
from __future__ import annotations

import io
import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
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
                direct_member = next((name for name in archive.namelist() if name.endswith("/bank-full.csv") or name == "bank-full.csv"), None)
                if direct_member is not None:
                    csv_bytes = archive.read(direct_member)
                else:
                    # The official UCI archive nests bank-full.csv inside bank.zip.
                    for nested_name in [name for name in archive.namelist() if name.lower().endswith(".zip")]:
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
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    prep = ColumnTransformer([("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)])
    return Pipeline([("preprocess", prep), ("classifier", classifier)])


def evaluate(y_true, probabilities, threshold: float = 0.5) -> dict:
    """Calculate ranking, classification, calibration and confusion-matrix metrics."""
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {"true_negative": int(tn), "false_positive": int(fp), "false_negative": int(fn), "true_positive": int(tp)},
    }


def choose_f1_threshold(y_true, probabilities) -> tuple[float, float]:
    """Choose a decision threshold on validation data only, never on the test set."""
    candidates = np.arange(0.10, 0.901, 0.01)
    scores = [(float(f1_score(y_true, probabilities >= threshold, zero_division=0)), float(threshold)) for threshold in candidates]
    best_f1, best_threshold = max(scores, key=lambda pair: (pair[0], pair[1]))
    return best_threshold, best_f1


def top_encoded_feature_importance(pipeline: Pipeline, limit: int = 15) -> list[dict[str, float | str]]:
    """Return model-based importance for the top encoded features, when supported."""
    preprocessor = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
        method = "tree impurity importance"
    elif hasattr(classifier, "coef_"):
        values = np.abs(np.asarray(classifier.coef_, dtype=float)).ravel()
        method = "absolute logistic coefficient"
    else:
        return []
    order = np.argsort(values)[::-1][:limit]
    return [{"feature": str(names[index]), "importance": float(values[index]), "method": method} for index in order]



def build_numeric_feature_reference(frame: pd.DataFrame) -> dict[str, dict[str, list[float]]]:
    """Create numeric reference histograms from training+validation features only."""
    reference = {}
    for column in frame.select_dtypes(include="number").columns:
        values = frame[column].dropna().to_numpy(dtype=float)
        if values.size < 2:
            continue
        low, high = float(values.min()), float(values.max())
        epsilon = max(abs(low), abs(high), 1.0) * 1e-9
        quantiles = [float(value) for value in np.quantile(values, [0.2, 0.4, 0.6, 0.8])]
        edges = sorted(set([low - epsilon, *quantiles, high + epsilon]))
        if len(edges) < 2:
            edges = [low - 0.5, high + 0.5]
        counts = np.histogram(values, bins=edges)[0]
        proportions = np.maximum(counts.astype(float) / float(values.size), 1e-6)
        proportions = proportions / proportions.sum()
        reference[str(column)] = {
            "edges": [float(edge) for edge in edges],
            "proportions": [float(value) for value in proportions],
        }
    return reference


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
        "gradient_boosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE),
    }
    comparison = []
    fitted = {}
    validation_probabilities = {}
    candidate_thresholds = {}
    for name, classifier in candidates.items():
        pipeline = build_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_val)[:, 1]
        metrics = evaluate(y_val, probabilities, threshold=0.5)
        comparison.append({"model": name, **metrics})
        fitted[name] = pipeline
        validation_probabilities[name] = probabilities
        candidate_thresholds[name] = choose_f1_threshold(y_val, probabilities)[0]
        print(f"Validation {name}: average_precision={metrics['average_precision']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")

    best = max(comparison, key=lambda result: result["average_precision"])
    validation_threshold, validation_best_f1 = choose_f1_threshold(y_val, validation_probabilities[best["model"]])
    # The test set did not influence model or threshold selection.
    final_pipeline = build_pipeline(X_train_val, candidates[best["model"]])
    final_pipeline.fit(X_train_val, y_train_val)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    test_metrics_baseline = evaluate(y_test, test_probabilities, threshold=0.5)
    test_metrics_tuned = evaluate(y_test, test_probabilities, threshold=validation_threshold)
    fraction_positive, mean_predicted = calibration_curve(y_test, test_probabilities, n_bins=10, strategy="quantile")
    report = {
        "project": "DepositWise",
        "dataset_source": "UCI Machine Learning Repository, Bank Marketing (dataset 222), bank-full.csv; upstream variant used by Kaggle copies",
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
        "threshold_selection": {"method": "maximize F1 on validation split", "threshold": validation_threshold, "validation_f1_at_selected_threshold": validation_best_f1},
        "candidate_thresholds": {name: float(value) for name, value in candidate_thresholds.items()},
        "final_test_metrics_at_0_5": test_metrics_baseline,
        "final_test_metrics": test_metrics_tuned,
        "test_calibration_curve": [{"mean_predicted_probability": float(predicted), "observed_positive_fraction": float(observed)} for predicted, observed in zip(mean_predicted, fraction_positive)],
        "top_encoded_feature_importance": top_encoded_feature_importance(final_pipeline),
        "limitations": [
            "The threshold maximizes validation F1 and is not optimized for campaign costs, customer contact capacity, or profit.",
            "The test set is a stratified random split; temporal validation may better reflect future campaigns.",
            "Feature importance is model-based association, not causal explanation. Review calibration, subgroup performance, privacy, and consent before operational use.",
        ],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline_customer = {}
    for column in X.columns:
        if column in X_train_val.select_dtypes(include="number").columns:
            baseline_customer[column] = float(X_train_val[column].median())
        else:
            modes = X_train_val[column].dropna().mode()
            baseline_customer[column] = modes.iloc[0] if not modes.empty else "unknown"
    feature_reference = build_numeric_feature_reference(X_train_val)
    candidate_pipelines = {}
    for candidate_name, candidate_classifier in candidates.items():
        candidate_pipeline = build_pipeline(X_train_val, candidate_classifier)
        candidate_pipeline.fit(X_train_val, y_train_val)
        candidate_pipelines[candidate_name] = candidate_pipeline
    joblib.dump({
        "pipeline": final_pipeline,
        "candidate_pipelines": candidate_pipelines,
        "candidate_thresholds": candidate_thresholds,
        "features": list(X.columns),
        "selected_model": best["model"],
        "threshold": validation_threshold,
        "baseline_customer": baseline_customer,
        "feature_reference": feature_reference,
    }, MODEL_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Selected model saved: {MODEL_PATH}")
    print(f"Evaluation report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
