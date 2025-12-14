# src/inference.py

import joblib
import mlflow
import pandas as pd
import numpy as np
from typing import Dict

PIPELINE_PATH = "../models/transformer_pipeline.joblib"
MODEL_NAME = "credit_risk_lightgbm"   # or logistic if that was best
MODEL_STAGE = "Production"            # or "None" if not staged


# -------------------------------
# Load artifacts once (singleton)
# -------------------------------

def load_preprocessor():
    artifacts = joblib.load(PIPELINE_PATH)
    return artifacts["preprocessor"]


def load_model():
    model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    model = mlflow.pyfunc.load_model(model_uri)
    return model


# -------------------------------
# Prediction function
# -------------------------------

def predict_risk(
    model,
    preprocessor,
    input_data: Dict
) -> Dict:
    """
    input_data: dictionary of raw features
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Apply preprocessing
    X = preprocessor.transform(df)

    # Predict
    risk_prob = model.predict_proba(X)[0][1]
    risk_label = int(risk_prob >= 0.5)

    return {
        "risk_probability": float(risk_prob),
        "is_high_risk": risk_label
    }
