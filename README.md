# DepositWise

**DepositWise** is an end-to-end machine-learning project that estimates whether a customer may subscribe to a term deposit after a bank marketing campaign. It is designed as a portfolio project with reproducible experiments, an inference API, and transparent model evaluation.

It borrows engineering ideas from [PropPulse](https://github.com/rushmanthnalluri/prop-pulse), adapting them to binary classification and marketing-campaign data.

## Dataset

Primary reference: [Kaggle Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets). If `data/raw/bank-full.csv` is not present, the training and EDA scripts automatically download the corresponding upstream `bank-full.csv` from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing). You can still provide the Kaggle CSV locally; set `DEPOSITWISE_DATA_PATH` to override the default path. The loader detects comma- and semicolon-delimited files and validates the `y` target (`yes`/`no`).

Do not commit downloaded data, customer records, credentials, model artifacts, or generated reports. Check the dataset's terms before redistributing it.

## Live demo

- **Dashboard:** https://depositwise.onrender.com/dashboard
- **API health:** https://depositwise.onrender.com/health
- **Model evaluation report:** https://depositwise.onrender.com/model-info
- **Interactive API docs:** https://depositwise.onrender.com/docs

The free Render service may sleep when idle. The Docker build downloads the public upstream dataset, generates the EDA profile, compares candidate models, and packages the selected model artifact into the image. The first deploy after a code change may take longer while model training runs.

## Current capabilities

- CSV loading, delimiter detection, and target validation.
- EDA profile output with class balance, duplicates, missing-value counts, and a target-distribution chart.
- Leakage-aware preprocessing for numeric and categorical features.
- Comparison of Logistic Regression, Random Forest, and Extra Trees.
- Model selection by validation average precision, with ROC-AUC, precision, recall, F1, balanced accuracy, Brier score, and confusion matrix reporting.
- Decision-threshold tuning by validation F1, with the untouched test set evaluated at both the baseline and tuned thresholds.
- Test-set calibration curve and model-based encoded-feature importance, exposed through `/model-info`.
- FastAPI health and prediction endpoints.
- Responsive browser workbench served at `/dashboard`, connected to the prediction API.
- Automated tests and a GitHub Actions workflow for Python 3.11 and 3.12.
- Dockerfile for API and dashboard containerization.

## Important modeling safeguards

- `duration` is excluded because call duration is only known after contact. The model is intended for pre-call targeting.
- Preprocessing is fitted within each model pipeline, after the train/validation split.
- The dataset is split into training, validation, and a separate test set. Candidate selection uses validation average precision; the selected model is then refit on train+validation and evaluated once on the held-out test set.
- The model selects a decision threshold by validation F1; it is not yet optimized for campaign costs, contact capacity, or profit. The test set is not used to select the threshold.
- Model outputs are estimates, not guarantees. Review privacy, consent, fairness, and applicable marketing/banking requirements before real-world use.

## Setup

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/rushmanthnalluri/DepositWise.git
cd DepositWise
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run EDA and training

If you have the Kaggle CSV, you can place it at `data/raw/bank-full.csv`. Otherwise the script downloads the upstream UCI dataset automatically. Then run:

```bash
python -m ml.eda
python -m ml.train
```

EDA outputs are written to `reports/generated/data_profile.json` and `reports/figures/target_distribution.png`. Model comparison, validation results, and held-out test metrics are written to `reports/generated/model_comparison.json`. The selected pipeline is saved to `models/depositwise_model.joblib`. These generated files are intentionally ignored by Git.

To use another dataset path:

```bash
# macOS/Linux
export DEPOSITWISE_DATA_PATH=/absolute/path/to/bank-full.csv
python -m ml.train
```

```powershell
# Windows PowerShell
$env:DEPOSITWISE_DATA_PATH = "C:\path\to\bank-full.csv"
python -m ml.train
```

## Run the API

Train the model first, then run:

```bash
uvicorn backend.app.main:app --reload
```

- Health: http://127.0.0.1:8000/health
- Model comparison and held-out test metrics: http://127.0.0.1:8000/model-info
- Interactive API docs: http://127.0.0.1:8000/docs

Example request body for `POST /predict` (use feature names from the dataset):

```json
{
  "customer": {
    "age": 35,
    "job": "management",
    "marital": "married",
    "education": "tertiary",
    "default": "no",
    "balance": 1200,
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "day": 15,
    "month": "may",
    "campaign": 1,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown"
  }
}
```

The example is illustrative; use the exact feature columns and valid values from the downloaded dataset. The API returns a probability and a baseline threshold classification.

## Tests and CI

Run tests locally:

```bash
pytest -q
```

GitHub Actions runs the test suite on pushes to `main`, pull requests, and manual dispatches. Check the [Actions page](https://github.com/rushmanthnalluri/DepositWise/actions) for actual run results; a workflow being configured does not mean it has passed.

## Docker

Build the image (the Docker build downloads the dataset, generates EDA outputs, and trains the model):

```bash
docker build -t depositwise .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" depositwise
```

On Windows PowerShell, replace the volume argument with `-v "${PWD}/models:/app/models:ro"`. The selected model artifact is baked into the image at build time, so the container can serve predictions immediately after a successful build. Rebuild the image to retrain on an updated dataset or pipeline.

## Repository structure

```text
backend/app/       FastAPI service
ml/train.py        model comparison and artifact creation
ml/eda.py          dataset profiling and target plot
data/raw/          local input CSVs (ignored)
models/            generated model artifacts (ignored)
reports/generated/ generated JSON reports (ignored)
reports/figures/   generated plots (ignored)
tests/             API and ML tests
.github/workflows/ GitHub Actions CI
```

## Current workbench capabilities

- Dark, responsive campaign-intelligence shell with navigation for Overview, Customer Scoring, Data Explorer, Model Lab, Evaluation, and Model Health.
- Nine-stage guided ML workbench navigation, linked to the relevant workspace views.
- Live dashboard metrics and feature importance are loaded from the generated model report rather than hardcoded in the interface.
- CSV/XLSX upload profiling runs in memory and reports dimensions, column types, missing cells, duplicate rows, unique counts, and a sample preview.
- Uploaded files are kept in a bounded in-memory cache only; they do not replace the bundled dataset or production model. Temporary profiles can be evicted when the cache is full or the service restarts.
- Customer scoring calls the deployed model API and displays the estimated probability, selected threshold, classification, and up to five single-feature baseline-sensitivity signals. These signals are explicitly not presented as causal effects or SHAP explanations.
- Model evaluation shows held-out metrics, validation-threshold context, candidate comparison where present, top encoded feature importance, a calibration curve, and a confusion matrix.
- Uploaded datasets can run feature inventory, descriptive statistics, missing-value analysis, numeric correlation summaries, and target-distribution analysis.
- Binary-classification sandbox training can compare Logistic Regression, Random Forest, and Extra Trees; configure numeric mean/median imputation, categorical most-frequent/constant imputation, and numeric scaling; select a champion using validation average precision; tune its threshold on validation data; report held-out test metrics; and score an edited JSON customer profile. Sandbox models stay in memory and never replace the deployed champion.
- Model-health page distinguishes actual API/model availability from monitoring capabilities that are not yet wired up.
- Keyboard focus states, responsive layouts, reduced-motion support, upload-size/type checks, and API error states are included.

## Feature parity status

The interface and core interaction patterns are being adapted from [PropPulse](https://github.com/rushmanthnalluri/prop-pulse) and [PlacementPredict](https://github.com/rushmanthnalluri/placement-predict), with banking-specific equivalents instead of property valuation or student salary functionality.

**Not yet implemented:** persistent experiment tracking/model registry, SHAP-based local explanations, campaign lift and cost optimization, feature-value prediction logging, PSI drift monitoring, subgroup fairness reports, and temporal validation. The current metrics endpoint deliberately retains only aggregate counts and latency, not raw customer features. The guided workbench now connects upload, profiling, EDA, sandbox training, evaluation, and sandbox scoring; preprocessing configuration and persistent run management remain future work.

## Roadmap

1. Complete the upload-to-evaluation sandbox pipeline with schema validation, configurable preprocessing, training jobs, and comparison reports.
2. Add per-prediction explanations and campaign lift/cost scenarios.
3. Implement privacy-aware prediction logging, feature-drift monitoring, and human-approved retraining recommendations.
4. Add subgroup/fairness evaluation and temporal validation for future campaigns.

## References

- [Kaggle Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [PropPulse](https://github.com/rushmanthnalluri/prop-pulse)
- [PlacementPredict](https://github.com/rushmanthnalluri/placement-predict)

## Status

The current branch contains the dashboard redesign, report-backed overview/evaluation views, customer scoring, in-memory CSV/XLSX profiling and EDA endpoints, sandbox binary-classifier training/scoring, and corresponding tests. This does **not** mean full feature parity is complete. No uploaded dataset is persisted or used to retrain the deployed model. Verify the latest CI run and Render deployment before claiming the new revision is live.
