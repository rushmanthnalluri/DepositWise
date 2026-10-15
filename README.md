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
- Model selection by validation average precision, with ROC-AUC, precision, recall, and F1 reported for each candidate.
- FastAPI health and prediction endpoints.
- Responsive browser workbench served at `/dashboard`, connected to the prediction API.
- Automated tests and a GitHub Actions workflow for Python 3.11 and 3.12.
- Dockerfile for API and dashboard containerization.

## Important modeling safeguards

- `duration` is excluded because call duration is only known after contact. The model is intended for pre-call targeting.
- Preprocessing is fitted within each model pipeline, after the train/validation split.
- The dataset is split into training, validation, and a separate test set. Candidate selection uses validation average precision; the selected model is then refit on train+validation and evaluated once on the held-out test set.
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

## Roadmap

- Add threshold selection based on campaign economics.
- Add confusion matrix, calibration checks, feature explanations, and drift monitoring.
- Add temporal validation to test generalization to future campaigns.

## References

- [Kaggle Banking Dataset — Marketing Targets](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
- [PropPulse](https://github.com/rushmanthnalluri/prop-pulse)
- [PlacementPredict](https://github.com/rushmanthnalluri/PlacementPredict) (repository access was not verified in this session)

## Status

The API, browser workbench, auto-downloadable dataset pipeline, EDA script, three-model comparison, held-out test evaluation, Docker build-time training, and CI are committed. No CSV or model artifact is stored in Git. CI results and the latest Render deploy should be checked before claiming the current deployment is healthy; model metrics should be read from the generated evaluation report rather than assumed.
