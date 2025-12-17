# predict risk

import mlflow
import pandas as pd
import joblib
import os
import sys
from typing import Dict
from api.credit_scoring import generate_credit_score
# Ensure our `src/` package is importable (required when running via uvicorn)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_path = os.path.join(repo_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS

# -------------------------------
# MLflow config (same as training)
# -------------------------------

mlflow.set_tracking_uri("file:./mlruns")

# MODEL_NAME = "credit_risk_lightgbm"   # or credit_risk_logistic
# MODEL_STAGE = "Production"            # use "None" if not staged

PREPROCESSOR_PATH = "models/preprocessor.joblib"
RISK_MODEL_PATHS = [
    "models/trained/risk_lgbm_pipeline.joblib",
    "models/trained/risk_lgbm_classifier.joblib",
    "models/trained/risk_logistic_pipeline.joblib",
    "models/trained/logistic_regression_pipeline.joblib"
]
LOAN_AMOUNT_MODEL_PATH = "models/trained/loan_amount_lgbm_regressor.joblib"
LOAN_DURATION_MODEL_PATH = "models/trained/loan_duration_lgbm_regressor.joblib"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"


def load_customer_features():
    try:
        df = pd.read_csv(PROCESSED_DATA_PATH)
        return df.groupby("CustomerId").last().reset_index()
    except Exception as e:
        print(f"[predict] Warning: processed data not found ({e}). Using empty history.")
        # Minimal empty dataframe with expected columns
        cols = [
            "CustomerId", "total_txn_amount", "avg_txn_amount",
            "count_txn", "std_txn_amount"
        ]
        return pd.DataFrame(columns=cols)


# -------------------------------
# Load model (Pipeline)
# -------------------------------

def load_model():
    """
    Loads the full sklearn Pipeline:
    preprocessing + model
    """
    # Ensure local `src/` modules (e.g. `preprocessing`) are importable for
    # objects that were pickled with module-level imports.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_path = os.path.join(repo_root, "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    # model = mlflow.sklearn.load_model(model_uri)

    # Attempt to load artifacts; if missing, trigger training and reload
    def _load_or_none(path):
        try:
            return joblib.load(path)
        except Exception:
            return None

    preprocessor = _load_or_none(PREPROCESSOR_PATH)

    risk_clf = None
    for p in RISK_MODEL_PATHS:
        if os.path.exists(p):
            risk_clf = _load_or_none(p)
            if risk_clf is not None:
                break

    amt_reg = _load_or_none(LOAN_AMOUNT_MODEL_PATH)
    dur_reg = _load_or_none(LOAN_DURATION_MODEL_PATH)

    # If any critical artifact missing, try to run full training (this will create artifacts)
    if preprocessor is None or risk_clf is None or amt_reg is None or dur_reg is None:
        try:
            print("Artifacts missing or failed to load. Running training pipeline...")
            # Import local training script and run it (this will create artifacts)
            import src.train as train_script
            train_script.main()
        except Exception as e:
            print(f"Training failed: {e}")

        preprocessor = _load_or_none(PREPROCESSOR_PATH)
        for p in RISK_MODEL_PATHS:
            if os.path.exists(p):
                risk_clf = _load_or_none(p)
                if risk_clf is not None:
                    break

        amt_reg = _load_or_none(LOAN_AMOUNT_MODEL_PATH)
        dur_reg = _load_or_none(LOAN_DURATION_MODEL_PATH)

    # Final sanity check: if models missing, provide lightweight fallback predictors
    if preprocessor is None or risk_clf is None or amt_reg is None or dur_reg is None:
        import numpy as _np

        class DummyRisk:
            def predict_proba(self, X):
                # Accept either a preprocessed array or DataFrame
                try:
                    if hasattr(X, "columns") and "Amount" in X.columns:
                        amt = X["Amount"].astype(float).values
                    else:
                        # try to handle numpy arrays: assume single value
                        amt = _np.atleast_1d(_np.array(X)).astype(float)
                    p = _np.clip(0.01 + (amt.mean() / 10000.0), 0.0001, 0.99)
                except Exception:
                    p = 0.01
                return _np.array([[1 - p, p]])

        class DummyRegressor:
            def predict(self, X):
                try:
                    if hasattr(X, "columns") and "Amount" in X.columns:
                        amt = X["Amount"].astype(float).values
                        return _np.array([amt.mean() * 0.8])
                    else:
                        return _np.array([100.0])
                except Exception:
                    return _np.array([100.0])

        print("[predict] Warning: returning fallback models for prediction")
        return {
            "preprocessor": preprocessor,
            "risk": DummyRisk(),
            "amount": DummyRegressor(),
            "duration": DummyRegressor(),
            "fallback": True
        }

    return {
        "preprocessor": preprocessor,
        "risk": risk_clf,
        "amount": amt_reg,
        "duration": dur_reg,
        "fallback": False
    }


# -------------------------------
# Prediction function
# -------------------------------


customer_features = load_customer_features()

def predict_risk(models: Dict, input_data: dict):

    df = pd.DataFrame([input_data])

    # Lookup historical customer features
    cust_id = input_data["CustomerId"]
    history = customer_features[customer_features["CustomerId"] == cust_id]

    if history.empty:
        raise ValueError("Customer not found in historical data")

    # Attach historical aggregates
    for col in [
        "total_txn_amount",
        "avg_txn_amount",
        "count_txn",
        "std_txn_amount"
    ]:
        df[col] = history.iloc[0][col]

    # Extract datetime features
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df["txn_hour"] = df["TransactionStartTime"].dt.hour
    df["txn_day"] = df["TransactionStartTime"].dt.day
    df["txn_month"] = df["TransactionStartTime"].dt.month
    df["txn_year"] = df["TransactionStartTime"].dt.year

    
    
    # Risk Prediction
    preprocessor = models["preprocessor"]
    risk_clf = models["risk"]

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols]
    X_trans_risk = preprocessor.transform(X)
    risk_prob = risk_clf.predict_proba(X_trans_risk)[0][1]

    score_output = generate_credit_score(risk_prob)

    is_high_risk = 1 if risk_prob > 0.5 else 0

    score_output["is_high_risk"] = is_high_risk

    # -------------------------------
    # Regression Predictions
    # -------------------------------
    # We bypass the 'feature_eng' step (index 0) of the regression pipelines
    # because it relies on batch aggregations which we did manually above.
    
    # Function to extract step from pipeline or use raw model
    def get_step(model, step_name):
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            return model.named_steps[step_name]
        return model

    # Loan Amount Prediction
    amt_pipeline = models["amount"]
    # If it's our trained pipeline, it has "preprocessor" and "regressor" steps
    # If fallback, it's just a dummy object
    try:
        amt_pre = get_step(amt_pipeline, "preprocessor")
        amt_reg = get_step(amt_pipeline, "regressor")
        
        # Transform using the model's OWN preprocessor (not the risk one)
        X_trans_amt = amt_pre.transform(X)
        loan_amount_pred = amt_reg.predict(X_trans_amt)[0]
    except Exception as e:
        print(f"Warning: Loan amount prediction fallback due to: {e}")
        loan_amount_pred = amt_pipeline.predict(X)[0] # Fallback for dummy

    # Loan Duration Prediction
    dur_pipeline = models["duration"]
    try:
        dur_pre = get_step(dur_pipeline, "preprocessor")
        dur_reg = get_step(dur_pipeline, "regressor")
        
        X_trans_dur = dur_pre.transform(X)
        loan_duration_pred = dur_reg.predict(X_trans_dur)[0]
    except Exception as e:
        print(f"Warning: Loan duration prediction fallback due to: {e}")
        loan_duration_pred = dur_pipeline.predict(X)[0]

    score_output["loan_amount"] = round(float(loan_amount_pred), 2)
    score_output["loan_duration"] = int(round(loan_duration_pred))

    return score_output

