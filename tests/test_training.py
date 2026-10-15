import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml.train import build_numeric_feature_reference, build_pipeline, read_dataset


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


def test_threshold_selection_uses_validation_probabilities():
    from ml.train import choose_f1_threshold, evaluate

    labels = pd.Series([0, 0, 0, 1, 1, 1])
    probabilities = pd.Series([0.08, 0.22, 0.41, 0.38, 0.67, 0.91])
    threshold, validation_f1 = choose_f1_threshold(labels, probabilities)
    assert 0.1 <= threshold <= 0.9
    assert 0 <= validation_f1 <= 1
    metrics = evaluate(labels, probabilities, threshold=threshold)
    assert "brier_score" in metrics
    assert "balanced_accuracy" in metrics
    assert metrics["confusion_matrix"]["true_positive"] >= 1


def test_numeric_feature_reference_uses_valid_histogram_proportions():
    frame = pd.DataFrame({"age": list(range(1, 11)), "constant": [5] * 10, "job": ["admin"] * 10})
    reference = build_numeric_feature_reference(frame)
    assert set(reference) == {"age", "constant"}
    for item in reference.values():
        assert len(item["edges"]) == len(item["proportions"]) + 1
        assert sum(item["proportions"]) == pytest.approx(1.0)
        assert all(value > 0 for value in item["proportions"])
