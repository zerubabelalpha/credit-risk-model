import mlflow
import pandas as pd
import joblib
from typing import Dict

# -------------------------------
# MLflow config (same as training)
# -------------------------------

mlflow.set_tracking_uri("file:./mlruns")

# MODEL_NAME = "credit_risk_lightgbm"   # or credit_risk_logistic
# MODEL_STAGE = "Production"            # use "None" if not staged

MODEL_LOCAL_PATH = "models/trained/lightgbm_pipeline.joblib"   # since its the best performed model
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"


def load_customer_features():
    df = pd.read_csv(PROCESSED_DATA_PATH)
    return df.groupby("CustomerId").last().reset_index()


# -------------------------------
# Load model (Pipeline)
# -------------------------------

def load_model():
    """
    Loads the full sklearn Pipeline:
    preprocessing + model
    """
    # model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    # model = mlflow.sklearn.load_model(model_uri)

    model = joblib.load(MODEL_LOCAL_PATH)
    return model


# -------------------------------
# Prediction function
# -------------------------------


customer_features = load_customer_features()

def predict_risk(model, input_data: dict):

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

    
    risk_prob = model.predict_proba(df)[0][1]

    return {
        "risk_probability": float(risk_prob),
        "is_high_risk": int(risk_prob >= 0.5)
    }

