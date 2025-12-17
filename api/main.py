from fastapi import FastAPI, HTTPException
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

# model placeholder to be initialized at startup
model = None

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
    try:
        # Ensure model is loaded (in case startup didn't complete successfully)
        global model
        if model is None:
            try:
                model = load_model()
            except Exception:
                # let predict_risk handle missing artifacts via its own fallbacks
                pass

        result = predict_risk(models=model, input_data=request.dict())
        return result
    except ValueError as e:
        # Known client errors (e.g., customer not found)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        # Surface unexpected errors to help debugging while developing
        raise HTTPException(status_code=500, detail=str(e))
