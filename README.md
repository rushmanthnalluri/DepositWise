# DepositWise

**DepositWise** is an end-to-end machine-learning project for predicting whether a customer will subscribe to a term deposit following a bank marketing campaign.

It adapts engineering ideas from [PropPulse](https://github.com/rushmanthnalluri/prop-pulse)—reproducible pipelines, leakage-aware evaluation, model comparison, explainability, API serving, monitoring and a guided workbench—to a binary-classification banking problem.

## Dataset

[Kaggle: Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)

Download the dataset from Kaggle and place the supplied CSV files under `data/raw/`. Do not commit credentials or private customer data. Check the dataset's terms before redistribution.

The common dataset release uses `y` as the target (whether the client subscribed to a term deposit). Confirm the actual column names and train/test format from the downloaded files before running the pipeline.

## Project goals

- Explore customer and campaign attributes and document class imbalance.
- Build leakage-safe preprocessing for numeric and categorical fields.
- Establish a Logistic Regression baseline and compare tree-based classifiers.
- Evaluate using stratified validation and metrics suited to imbalanced classification: PR-AUC, ROC-AUC, precision, recall, F1, confusion matrix and probability calibration.
- Select a decision threshold using validation data and an explicit business objective.
- Explain predictions with feature importance and, where appropriate, SHAP.
- Expose prediction and model-health endpoints through FastAPI.
- Add a responsive dashboard, automated tests, CI, Docker support and monitoring as the project matures.

## Important modeling safeguards

- Treat `duration` carefully: call duration is only known after a call. Exclude it from a pre-call targeting model, or label a duration-inclusive model explicitly as post-call analysis.
- Fit imputation, encoding, scaling, feature selection and resampling only within training folds.
- Keep the final test set untouched until model and threshold selection are complete.
- Do not report fabricated metrics. Results belong in reports only after the corresponding experiment has run.
- Predictions are decision support, not guarantees of customer behavior. Review fairness, privacy, consent and applicable banking/marketing requirements before real-world use.

## Intended architecture

```text
Kaggle CSVs
    -> validation and EDA
    -> leakage-aware feature pipeline
    -> baseline and candidate model training
    -> validation-based model/threshold selection
    -> versioned model artifact and evaluation report
    -> FastAPI prediction service
    -> React dashboard / ML workbench
    -> tests, CI, Docker and drift monitoring
```

## Repository layout

```text
DepositWise/
├── backend/             # FastAPI application
├── frontend/            # Web dashboard (planned)
├── ml/
│   ├── data/             # Dataset validation and split helpers
│   ├── features/         # Preprocessing pipeline
│   ├── training/         # Model training and comparison
│   ├── evaluation/       # Metrics and threshold selection
│   └── explainability/   # Feature explanations
├── data/raw/             # Downloaded data; ignored by Git
├── models/               # Generated model artifacts; ignored by Git
├── notebooks/            # EDA notebooks
├── reports/              # Reproducible experiment reports
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## Quick start

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
source .venv/bin/activate

python -m pip install -r requirements.txt
```

Dataset-specific commands and API run instructions will be added alongside the implementation. Never place API keys in source control.

## References

- [Banking Dataset — Marketing Targets (Kaggle)](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [PropPulse](https://github.com/rushmanthnalluri/prop-pulse)
- [PlacementPredict](https://github.com/rushmanthnalluri/PlacementPredict) (verify the repository URL/access before using it as a source)

## Status

Initial project scaffold. Model metrics, deployment status and production readiness are not claimed until verified by running the implementation and CI.
