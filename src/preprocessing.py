import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans

RAW_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
RANDOM_STATE = 42

# --------------------- Custom Transformers ---------------------

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_col="TransactionStartTime"):
        self.datetime_col = datetime_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X[self.datetime_col] = pd.to_datetime(X[self.datetime_col], errors="coerce")
        X["txn_hour"] = X[self.datetime_col].dt.hour
        X["txn_day"] = X[self.datetime_col].dt.day
        X["txn_month"] = X[self.datetime_col].dt.month
        X["txn_year"] = X[self.datetime_col].dt.year
        return X

class PerCustomerAggregates(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col="CustomerId", amount_col="Amount"):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        agg = X.groupby(self.customer_id_col)[self.amount_col].agg(
            total_txn_amount="sum",
            avg_txn_amount="mean",
            count_txn="count",
            std_txn_amount="std"
        ).reset_index()
        agg["std_txn_amount"] = agg["std_txn_amount"].fillna(0)
        return X.merge(agg, on=self.customer_id_col, how="left")

# --------------------- RFM Target Engineering ---------------------

def create_target(df):
    df = df.copy()
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"], errors="coerce")
    snapshot = df["TransactionStartTime"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerId").agg(
        recency=("TransactionStartTime", lambda x: (snapshot - x.max()).days),
        frequency=("TransactionStartTime", "count"),
        monetary=("Amount", "sum")
    ).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    rfm["risk_score"] = (
        rfm["recency"].rank(ascending=False) +
        rfm["frequency"].rank(ascending=True) +
        rfm["monetary"].rank(ascending=True)
    )

    high_risk_cluster = rfm.groupby("cluster")["risk_score"].mean().idxmax()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]], scaler, kmeans, snapshot

# --------------------- Build Preprocessing Pipeline ---------------------

def build_preprocessor(numeric_cols, categorical_cols):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

# --------------------- Run Preprocessing ---------------------

def run_preprocessing():
    df = pd.read_csv(RAW_PATH)
    df = DateTimeExtractor().transform(df)
    df = PerCustomerAggregates().transform(df)

    target_df, scaler, kmeans, snapshot = create_target(df)
    df = df.merge(target_df, on="CustomerId", how="left")
    df["is_high_risk"] = df["is_high_risk"].fillna(0)

    numeric_cols = [
        "Amount", "Value", "total_txn_amount", "avg_txn_amount",
        "count_txn", "std_txn_amount", "txn_hour", "txn_day", "txn_month", "txn_year"
    ]

    categorical_cols = [
        "CurrencyCode", "CountryCode", "ProductCategory",
        "PricingStrategy", "ChannelId", "ProviderId", "ProductId"
    ]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(df[numeric_cols + categorical_cols])

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(preprocessor, PIPELINE_PATH)
    df.to_csv(PROCESSED_PATH, index=False)

    print("✅ Preprocessing pipeline saved")


def main():
    print("🚀 Starting preprocessing pipeline...")
    run_preprocessing()
    print("🎯 Preprocessing completed successfully")


if __name__ == "__main__":
    main()
