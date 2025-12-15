from pydantic import BaseModel, Field
from typing import Optional


# -------------------------------
# Request model
# -------------------------------

class PredictionRequest(BaseModel):
    CustomerId: str
    Amount: float = Field(..., gt=0, description="Transaction amount")
    Value: Optional[float]
    CurrencyCode: Optional[str]
    CountryCode: Optional[str]
    ProductCategory: Optional[str]
    PricingStrategy: Optional[str]
    ChannelId: Optional[str]
    ProviderId: Optional[str]
    ProductId: Optional[str]
    TransactionStartTime: str = Field(
        ..., description="ISO datetime string (YYYY-MM-DD HH:MM:SS)"
    )


# -------------------------------
# Response models
# -------------------------------

class PredictionResponse(BaseModel):
    risk_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of high credit risk"
    )
    is_high_risk: int = Field(
        ..., description="1 = High Risk, 0 = Low Risk"
    )


class HealthResponse(BaseModel):
    status: str
