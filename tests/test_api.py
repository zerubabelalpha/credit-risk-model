from fastapi.testclient import TestClient
from api.main import app


client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_sample():
    payload = {
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
        "TransactionStartTime": "2018-11-15 02:18:49+00:00"
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "risk_probability" in data
    assert "loan_amount" in data
