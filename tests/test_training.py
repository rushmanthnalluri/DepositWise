import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.train import build_pipeline, read_dataset


def test_read_dataset_accepts_semicolon_delimited_csv(tmp_path):
    path = tmp_path / "bank-full.csv"
    path.write_text(
        'age;job;y\n35;management;yes\n42;technician;no\n28;student;no\n51;retired;yes\n',
        encoding="utf-8",
    )
    frame = read_dataset(path)
    assert frame.columns.tolist() == ["age", "job", "y"]
    assert frame["y"].tolist() == ["yes", "no", "no", "yes"]


def test_read_dataset_rejects_missing_target(tmp_path):
    path = tmp_path / "unlabeled.csv"
    path.write_text("age,job\n35,management\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected target column 'y'"):
        read_dataset(path)


def test_pipeline_handles_unseen_categories_and_excludes_duration():
    X = pd.DataFrame({
        "age": [21, 32, 45, 53, 27, 38],
        "job": ["student", "admin.", "technician", "retired", "student", "admin."],
        "duration": [50, 80, 120, 95, 60, 90],
    })
    y = pd.Series([0, 0, 1, 1, 0, 1])
    pipeline = build_pipeline(X, LogisticRegression(max_iter=500))
    pipeline.fit(X, y)
    sample = pd.DataFrame({"age": [30], "job": ["never-seen-before"]})
    probability = pipeline.predict_proba(sample)[0, 1]
    assert 0 <= probability <= 1
    selected_columns = set()
    for _, _, columns in pipeline.named_steps["preprocess"].transformers:
        if isinstance(columns, list):
            selected_columns.update(columns)
    assert "duration" not in selected_columns
