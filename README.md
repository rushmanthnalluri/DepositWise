# DepositWise

**DepositWise** is an end-to-end machine-learning project that estimates whether a customer may subscribe to a term deposit after a bank marketing campaign. It is designed as a portfolio project with reproducible experiments, an inference API, and transparent model evaluation.

It borrows engineering ideas from [PropPulse](https://github.com/rushmanthnalluri/prop-pulse), adapting them to binary classification and marketing-campaign data.

## Dataset

Use the [Kaggle Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets). Download the labeled `bank-full.csv` file and put it at `data/raw/bank-full.csv`. The loader auto-detects common comma- and semicolon-delimited CSV files and expects the target column `y` with `yes`/`no` labels.

Do not commit downloaded data, customer records, credentials, model artifacts, or generated reports. Check the dataset's terms before redistributing it.

## Current capabilities

- CSV loading, delimiter detection, and target validation.
- EDA profile output with class balance, duplicates, missing-value counts, and a target-distribution chart.
- Leakage-aware preprocessing for numeric and categorical features.
- Comparison of Logistic Regression, Random Forest, and Extra Trees.
- Model selection by validation average precision, with ROC-AUC, precision, recall, and F1 reported for each candidate.
- FastAPI health and prediction endpoints.
- Responsive browser workbench served at `/dashboard`, connected to the prediction API.
- Automated tests and a GitHub Actions workflow for Python 3.11 and 3.12.
- Dockerfile for API and dashboard containerization.

## Important modeling safeguards

- `duration` is excluded because call duration is only known after contact. The model is intended for pre-call targeting.
- Preprocessing is fitted within each model pipeline, after the train/validation split.
- The current comparison uses one stratified validation split. It is for model selection, **not** an independent final test result.
- The default probability threshold of 0.5 is a baseline. Choose a threshold against an explicit campaign cost/benefit objective before deployment.
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

Place the labeled CSV at `data/raw/bank-full.csv`, then run:

```bash
python -m ml.eda
python -m ml.train
```

EDA outputs are written to `reports/generated/data_profile.json` and `reports/figures/target_distribution.png`. Model comparison results are written to `reports/generated/model_comparison.json`. The selected pipeline is saved to `models/depositwise_model.joblib`. These generated files are intentionally ignored by Git.

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

Build the API image after training the model locally:

```bash
docker build -t depositwise .
docker run --rm -p 8000:8000 -v "$(pwd)/models:/app/models:ro" depositwise
```

On Windows PowerShell, replace the volume argument with `-v "${PWD}/models:/app/models:ro"`. The model artifact is not baked into the image. The mounted `models/depositwise_model.joblib` must exist, or the API will report that prediction is unavailable.

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

## Roadmap

- Add a separate untouched test set and threshold selection based on campaign economics.
- Add confusion matrix, calibration checks, feature explanations, and drift monitoring.
- Add an untouched test set and threshold selection based on campaign economics.
- Add confusion matrix, calibration checks, feature explanations, and drift monitoring.
- Persist a trained model artifact in deployment so live predictions can run.

## References

- [Kaggle Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [PropPulse](https://github.com/rushmanthnalluri/prop-pulse)
- [PlacementPredict](https://github.com/rushmanthnalluri/PlacementPredict) (repository access was not verified in this session)

## Status

The API, browser workbench, model-training scaffold, Docker configuration, and CI are committed. The test suite has passed in GitHub Actions for Python 3.11 and 3.12. The Kaggle CSV has not been added and no trained model artifact is committed; prediction requests return a clear unavailable response until a model is trained and supplied. A Render service has been configured for deployment.
