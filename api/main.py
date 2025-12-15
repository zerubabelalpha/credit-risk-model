from fastapi import FastAPI
from api.predict import load_model, predict_risk
from api.pydantic_models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse
)

app = FastAPI(
    title="Credit Risk Scoring API",
    description="API for predicting customer credit risk probability",
    version="1.0.0"
)

# --------------------------------
# Load model on startup
# --------------------------------

@app.on_event("startup")
def load_artifacts():
    global model
    model = load_model()


# --------------------------------
# Health check
# --------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


# --------------------------------
# Prediction endpoint
# --------------------------------

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    result = predict_risk(
        model=model,
        input_data=request.dict()
    )

    return result
