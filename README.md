# Credit Risk Model

An outcome-oriented, end-to-end credit risk pipeline designed to demonstrate modern Machine Learning Engineering (MLE) practices in a regulated application domain.

## Overview

This repository implements a BNPL (Buy Now, Pay Later) credit risk modeling system. It demonstrates how to translate transaction-level data into customer-level credit scores using both traditional interpretable models (Logistic Regression + WoE) and high-performance ensemble methods (LightGBM).

Key capabilities include:
- **Feature Engineering**: Transaction aggregation, RFM (Recency, Frequency, Monetary) analysis, and Weight of Evidence (WoE) transformation.
- **Proxy Target Creation**: Utilizing K-Means clustering on RFM metrics to define "good" vs. "bad" credit proxies in the absence of explicit default labels.
- **Modeling**: 
  - **Classification**: Predicting default probability (PD).
  - **Regression**: Recommending optimal loan amounts and durations.
- **MLOps**: Automated pipelines (`scikit-learn`), experiment tracking (`MLflow`), and containerized serving (`FastAPI` + `Docker`).

## Business Understanding & Methodology

### Regulatory Context (Basel II)
In credit risk, models must be "conceptually sound" and explainable. While black-box models often yield higher performance, regulatory frameworks (like Basel II) favor interpretable models where the impact of each feature can be audited. This project balances these needs by implementing both:
- **Logistic Regression with WoE**: Highly interpretable, standard scorecards.
- **Gradient Boosting (LightGBM)**: Higher predictive power, used for benchmarking and challenger models.

### The "Proxy" Problem
In many emerging markets or new product launches, historical default labels are unavailable. This project demonstrates a standard industry approach:
1. **RFM Analysis**: Segment customers based on behavior.
2. **K-Means Clustering**: Group customers into behavioral clusters.
3. **Proxy Labeling**: The least engaged/riskiest cluster is treated as the logical proxy for "high risk" (Target = 1), enabling supervised training.

## Project Structure

```
credit-risk-model/
├── .github/workflows/   # CI/CD workflows
├── api/                 # FastAPI serving layer
├── data/                # Data storage (raw & processed)
├── notebooks/           # Jupyter notebooks for EDA and prototyping
├── src/                 # Source code
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── train.py
│   └── train_loan_model.py
├── tests/               # Unit and integration tests
├── Dockerfile           # Docker build instructions
├── docker-compose.yml   # Container orchestration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Docker (optional, for containerized execution)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/zerubabelalpha/credit-risk-model.git
   cd credit-risk-model
   ```

2. **Create and activate a virtual environment**
   ```bash
   # Linux/Mac
   python3 -m venv .venv
   source .venv/bin/activate

   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: This project pins `scikit-learn==1.7.2` to ensure model reproducibility and prevent pickling errors.

## Usage Guide

### 1. Data Preprocessing
Run the preprocessing pipeline to transform raw transaction data into a customer-level analytic dataset. This step performs cleaning, feature extraction, and saves the `processed_data.csv` and the fitted `preprocessor.joblib`.

```bash
python src/preprocessing.py
```

### 2. Model Training
Train the credit scoring models (Classifier) and loan recommendation models (Regressors). This script logs all experiments and artifacts to the local MLflow registry.

```bash
python src/train.py
# Optional: Train specific loan parameter models
python src/train_loan_model.py
```

### 3. API Serving
Start the FastAPI server to serve predictions in real-time.

```bash
# Run locally
uvicorn api.main:app --reload
```

**Example Request:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
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
         }'
```

### 4. Experiment Tracking (MLflow)
Inspect model performance, hyperparameters, and artifacts.

```bash
mlflow ui --port 5000
# Access at http://127.0.0.1:5000
```

### 5. Docker Deployment
Build and run the entire application stack using Docker.

```bash
docker compose up --build
```
This mounts the `models/` and `data/` directories, allowing for seamless development and persistence.

## Testing

Run the test suite to ensure pipeline integrity.

```bash
pytest -q
```

## Troubleshooting

- **ModuleNotFoundError / Unpickling Errors**: Ensure you are running commands from the project root so `src` is in the python path. Verify `scikit-learn` version matches `requirements.txt`.
- **404 on Prediction**: The model relies on historical features. If a `CustomerId` is not found in `data/processed/processed_data.csv`, the API cannot fetch aggregate features. Ensure preprocessing is complete.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/NewFeature`).
3. Commit your changes (`git commit -m 'Add NewFeature'`).
4. Push to the branch (`git push origin feature/NewFeature`).
5. Open a Pull Request.