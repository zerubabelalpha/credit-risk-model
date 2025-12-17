import pandas as pd
from src.pipeline import get_feature_engineering_pipeline, build_preprocessor
from preprocessing import NUMERIC_COLS, CATEGORICAL_COLS


def make_row():
    return {
        "TransactionId": "t1",
        "CustomerId": "CustomerId_1",
        "Amount": 100.0,
        "Value": 100.0,
        "TransactionStartTime": "2018-11-15 02:18:49+00:00",
        "CurrencyCode": "UGX",
        "CountryCode": "256",
        "ProductCategory": "airtime",
        "PricingStrategy": "2",
        "ChannelId": "ChannelId_3",
        "ProviderId": "ProviderId_6",
        "ProductId": "ProductId_10",
    }


def test_feature_engineering():
    df = pd.DataFrame([make_row()])
    fe = get_feature_engineering_pipeline()
    out = fe.fit_transform(df)
    assert "txn_hour" in out.columns
    assert "total_txn_amount" in out.columns


def test_preprocessor_fit_transform():
    df = pd.DataFrame([make_row()])
    pre = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)
    # Should be able to fit on at least one row without exception
    pre.fit(df)
    out = pre.transform(df)
    assert out.shape[0] == 1
