import mlflow
import pandas as pd
import joblib
from typing import Dict

# -------------------------------
# MLflow config (same as training)
# -------------------------------

mlflow.set_tracking_uri("file:./mlruns")

MODEL_NAME = "credit_risk_lightgbm"   # or credit_risk_logistic
MODEL_STAGE = "Production"            # use "None" if not staged
MODEL_LOCAL_PATH = "models/trained/lightgbm_pipeline.joblib"

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

def predict_risk(
    model,
    input_data: Dict
) -> Dict:
    """
    input_data: dictionary of RAW features
    (same format as training data before preprocessing)
    """

    # Convert input to DataFrame
    df = pd.DataFrame([input_data])

    # Pipeline handles preprocessing + prediction
    risk_prob = model.predict_proba(df)[0][1]
    risk_label = int(risk_prob >= 0.5)

    return {
        "risk_probability": float(risk_prob),
        "is_high_risk": risk_label
    }
