from pathlib import Path

import pandas as pd
import pytest

from ml.train import build_pipeline, read_dataset
from sklearn.linear_model import LogisticRegression


def test_read_dataset_accepts_semicolon_delimited_csv(tmp_path: Path):
    path = tmp_path / "bank.csv"
    path.write_text("age;job;y\n30;admin.;no\n42;technician;yes\n", encoding="utf-8")
    frame = read_dataset(path)
    assert list(frame.columns) == ["age", "job", "y"]
    assert frame["y"].tolist() == ["no", "yes"]


def test_read_dataset_rejects_missing_target(tmp_path: Path):
    path = tmp_path / "wrong.csv"
    path.write_text("age,job\n30,admin.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected target column"):
        read_dataset(path)


def test_pipeline_fits_mixed_types_and_ignores_unknown_categories():
    X = pd.DataFrame({
        "age": [25, 41, 32, 50],
        "job": ["admin.", "technician", "admin.", "services"],
    })
    y = [0, 1, 0, 1]
    pipeline = build_pipeline(X, LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    predictions = pipeline.predict_proba(pd.DataFrame({"age": [35], "job": ["new-job"]}))
    assert predictions.shape == (1, 2)
