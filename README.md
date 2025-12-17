# Credit Risk Model

This repository is a proof-of-concept BNPL / credit-risk pipeline that demonstrates:

- Transaction-level feature engineering and aggregation to customer-level features.
- Proxy target creation using RFM (Recency, Frequency, Monetary) and K-Means clustering.
- Weight-of-Evidence (WoE) transformation and Information Value (IV)-based feature selection.
- A scikit-learn `Pipeline`-based preprocessing + model training workflow (Logistic Regression and LightGBM), with hyperparameter tuning via GridSearchCV.
- Regression models to recommend loan amount and duration per customer.
- Experiment tracking and artifact logging using MLflow (local `mlruns/`).
- A FastAPI serving layer (predict endpoint) and optional Docker deployment.

This README documents how to run preprocessing, training, and serving locally, and includes troubleshooting tips.

**Repository layout (actual)**

```
.dockerignore
.env (local environment file - do not commit secrets)
.git/
.github/
.gitignore
.pytest_cache/ (generated cache - ignore/remove)
api/
data/
docker-compose.yml
Dockerfile
mlflow.db (local mlflow sqlite - exclude from repo if desired)
mlruns/
models/
notebooks/
README.md
requirements.txt
scripts/
src/
tests/
```

**Important note about versions**

This project pins `scikit-learn==1.7.2` in `requirements.txt` to avoid known model unpickle/compatibility issues between sklearn versions (especially `SimpleImputer`). If you get unpickling errors, match this version in your environment.

**Quick start (development)**

Run these commands from the repository root in PowerShell (adjust paths if your virtualenv location differs):

1) Create and activate venv (once):

```powershell
python -m venv .venv
& .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Preprocess raw transactions (creates `data/processed/processed_data.csv` and saves the preprocessor):

```powershell
& '.\.venv\Scripts\python.exe' src\preprocessing.py
```

3) Train models and log MLflow runs (classifier + regressors):

```powershell
& '.\.venv\Scripts\python.exe' src\train.py
```

4) (Optional) Train loan regressors if separate:

```powershell
& '.\.venv\Scripts\python.exe' src\train_loan_model.py
```

5) Start the API locally:

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn api.main:app --reload
```

6) Example POST to `/predict` (curl):

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @- <<'JSON'
{
  "CustomerId": "CustomerId_4406",
  "Amount": 100.0,
  "Value": 100.0,
  "CurrencyCode": "UGX",
  "CountryCode": "256",
  "ProductCategory": "airtime",
  "PricingStrategy": "2",
  "ChannelId": "ChannelId_3",
  "ProviderId": "ProviderId_6",
  "ProductId": "ProductId_10",
  "TransactionStartTime": "2018-11-15T02:18:49Z"
}
JSON
```

**MLflow**

Run the MLflow UI locally to inspect experiments and downloaded artifacts:

```powershell
& '.\.venv\Scripts\python.exe' -m mlflow ui --port 5000
# open http://127.0.0.1:5000
```

The project logs model artifacts (pipelines) and metrics to local `mlruns/` by default.

**Docker**

The repository includes a `Dockerfile` and `docker-compose.yml` to run the API in a container. Two options:

- Build an image that contains trained artifacts (copy `models/` into image) and run.
- Or mount `models/` and `data/` as volumes to the container at runtime (recommended for development).

Example (local Docker Desktop):

```powershell
docker compose up --build
```

Production note: Prefer model registry or object storage (S3) instead of baking models into images.

**Testing**

Run unit and integration tests with pytest:

```powershell
& '.\.venv\Scripts\python.exe' -m pytest -q
```

Current test coverage focuses on pipeline components and API endpoints.

**Design & Implementation Notes**

- Feature engineering:
  - `src/pipeline.py` provides `DateTimeExtractor`, `PerCustomerAggregates`, and a robust `build_preprocessor()` that returns an sklearn `Pipeline` with imputation, scaling and one-hot encoding.
  - `src/feature_engineering.py` implements a `WoEIVTransformer` and `IVFeatureSelector` for IV-based selection.

- Target engineering:
  - `src/preprocessing.py` computes RFM metrics per `CustomerId`, scales them, and applies K-Means (3 clusters by default) to segment customers; the least engaged cluster is labeled `is_high_risk`.

- Modeling:
  - Classifiers: Logistic Regression (interpretable) and LightGBM (performance). Both are trained via `Pipeline` that includes feature engineering and preprocessor to avoid train/serving mismatch.
  - Regressors: LightGBM regressors for `loan_amount` and `loan_duration` trained in `src/train_loan_model.py`.

**Serving behavior**

- The `api/predict.py` code loads preprocessor and models (joblib). For single-row inference the API uses customer-level historical aggregates (loaded from `data/processed/processed_data.csv`) to augment the request.
- For development, the API contains a small fallback predictor to allow the `/predict` endpoint to respond even if artifacts are missing. Remove or replace this fallback for production before deployment.

**Troubleshooting**

- ModuleNotFoundError when unpickling: ensure `src/` is importable at runtime (the server adds `src` to `sys.path`) and install `scikit-learn==1.7.2`.
- `SimpleImputer` attribute errors: caused by sklearn version mismatch; reinstall the pinned version.
- If `/predict` returns 404 for a customer: the `CustomerId` must exist in `data/processed/processed_data.csv`. Run `src/preprocessing.py` to generate processed data first.

**CI suggestions**

Add a GitHub Actions workflow that:

1. Installs Python and dependencies (cache pip).
2. Runs `pytest`.
3. Optionally builds the Docker image (and runs containerized smoke tests).

I can add that workflow for you — tell me if you want the workflow to also build and push the Docker image.

**Cleaning generated files**

To remove temporary caches and trained artifacts locally:

```powershell
Get-ChildItem -Path . -Include '__pycache__' -Recurse | Remove-Item -Recurse -Force
Remove-Item -Path models\trained\* -Force -ErrorAction SilentlyContinue
Remove-Item -Path mlruns\* -Recurse -Force -ErrorAction SilentlyContinue
```

---

If you want, I can now:

- (A) Add a GitHub Actions CI workflow that runs tests and optionally builds the Docker image.
- (B) Remove generated artifacts (`models/trained/`, `mlruns/`, `__pycache__`) from the repository.
- (C) Add a protected `/admin/retrain` endpoint to the API to trigger model retraining and artifact reload.

Tell me which of (A)/(B)/(C) you prefer and I'll implement it next.
# Credit Risk Model

An End-to-End Implementation for Building, Deploying, and Automating a Credit Risk Mode for Bati bank.

## Features

- **EDA Notebook**: 



## Project Structure
```
credit-risk-model/
├── .github/workflows/ci.yml   
├── data/                       
│   ├── raw/                   
│   └── processed/             
├── notebooks/
│   └── eda.ipynb         
├── src/
│   ├── __init__.py
│   ├── data_processing.py     
│   ├── train.py               
│   ├── predict.py             
│   └── api/
│       ├── main.py            
│       └── pydantic_models.py 
├── tests/
│   └── test_data_processing.py  
├── Dockerfile
├── .dockerignore
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zerubabelalpha/credit-risk-model
   ```

   go to the dir
   ```
   cd credit-risk-model
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Example

Quick start (development):

1. Activate virtual environment (Windows PowerShell):

```powershell
Set-Location 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\credit-risk-model'
& 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\risk_and_predictive_analytics\.venv\Scripts\Activate.ps1'
```

2. Run preprocessing (creates `data/processed/processed_data.csv` and `models/preprocessor.joblib`):

```powershell
& 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\risk_and_predictive_analytics\.venv\Scripts\python.exe' src\preprocessing.py
```

3. Train models (classifier + regressors) and log runs to MLflow:

```powershell
& 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\risk_and_predictive_analytics\.venv\Scripts\python.exe' src\train.py
```

4. Start the API (from project root):

```powershell
& 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\risk_and_predictive_analytics\.venv\Scripts\python.exe' -m uvicorn api.main:app --reload
```

5. Example `curl` request to `/predict`:

```powershell
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @- <<'JSON'
{
   "CustomerId": "CustomerId_4406",
   "Amount": 100.0,
   "Value": 100.0,
   "CurrencyCode": "UGX",
   "CountryCode": "256",
   "ProductCategory": "airtime",
   "PricingStrategy": "2",
   "ChannelId": "ChannelId_3",
   "ProviderId": "ProviderId_6",
   "ProductId": "ProductId_10",
   "TransactionStartTime": "2018-11-15T02:18:49Z"
}
JSON
```

MLflow UI:

```powershell
& 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\risk_and_predictive_analytics\.venv\Scripts\python.exe' -m mlflow ui --port 5000
```

Open http://127.0.0.1:5000 to inspect experiment runs and logged artifacts.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Credit Scoring Business Understanding

1) Basel II (and subsequent supervisory guidance) makes banks responsible for credible internal risk estimates and requires robust model governance: conceptual soundness, independent validation, ongoing monitoring, and clear documentation of model design, data, assumptions and use. In practice this means regulators and supervisors expect models used for credit risk or regulatory capital to be auditable and explainable — not black boxes — so that errors, biases or changes in performance can be detected and addressed. 


2) Why a proxy is needed: when a true, observed “default” outcome is absent or incomplete (e.g., only approved loans are visible, or time horizon is too short), you must create a proxy (delinquency >X days, charge-off, treatment flag, or behavioral trigger) so you can train a supervised model. The World Bank and empirical studies emphasize that many practical scoring systems rely on such operational proxies when full default history is unavailable. 

Business risks of using a proxy:

Label error / misclassification: the proxy may systematically mislabel good/bad outcomes (e.g., short-term delinquency ≠ eventual default), which biases PD estimates. 

Targeting & economic risk: decisions (approve/price/limit) based on a noisy proxy can increase credit losses or shrink revenue (wrong accept/reject/pricing decisions).

Regulatory & governance risk: supervisors may challenge models whose proxies are weak, opaque, or insufficiently conservative.


3) Key trade-offs: simple interpretable model (Logistic + WoE) vs complex high-performance model (Gradient Boosting) in a regulated context

Interpretability & governance

Simple (Logistic + WoE): Highly interpretable, easier to document/explain, simpler to validate and monitor; aligns well with regulatory expectations for transparency and auditability. Faster to implement governance, scorecard conversion, and manual overrides. 

Complex (GBM): Often higher raw predictive power (better separation/AUC) but harder to explain at feature-level; needs additional interpretability tools (SHAP, PDP, surrogate models) and stronger validation to satisfy supervisors.

###  Project for KAIM 

**Troubleshooting & Notes**

- If the API raises import/unpickling errors for sklearn objects, ensure your venv uses the pinned `scikit-learn==1.7.2` (see `requirements.txt`).
- The API contains a development fallback that will return sample predictions when model artifacts are missing; run `src/preprocessing.py` and `src/train.py` to create production artifacts.
- Docker: `docker-compose.yml` is provided. To build and run the container locally:

```powershell
Set-Location 'C:\Users\Acer\Documents\KAIM_PROJECT\TEST\credit-risk-model'
docker compose up --build
```

- To remove temporary `__pycache__` folders and other generated artifacts:

```powershell
Get-ChildItem -Path . -Include '__pycache__' -Recurse | Remove-Item -Recurse -Force
Remove-Item -Path models\trained\* -Force -ErrorAction SilentlyContinue
```

If you'd like, I can add a GitHub Actions workflow to run tests and build the Docker image on push/PR.