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


# =====================================================
# CONFIG
# =====================================================

RAW_PATH = "../data/raw/data.csv"
PROCESSED_PATH = "../data/processed/processed_data.csv"
PIPELINE_PATH = "../models/transformer_pipeline.joblib"
RANDOM_STATE = 42


# =====================================================
# CUSTOM TRANSFORMERS
# =====================================================

class DateTimeExtractor(BaseEstimator, TransformerMixin):
    """Extracts time-based features from TransactionStartTime."""
    
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
    """Creates aggregate transaction features per CustomerId."""

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


# =====================================================
# RFM + TARGET ENGINEERING
# =====================================================

def calculate_rfm(df, customer_id="CustomerId", date_col="TransactionStartTime", amount_col="Amount"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_id).agg(
        recency=(date_col, lambda x: (snapshot_date - x.max()).days),
        frequency=(date_col, "count"),
        monetary=(amount_col, "sum")
    ).reset_index()

    return rfm, snapshot_date


def assign_high_risk_cluster(rfm, n_clusters=3):
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    cluster_summary = rfm.groupby("cluster").agg(
        recency_mean=("recency", "mean"),
        frequency_mean=("frequency", "mean"),
        monetary_mean=("monetary", "mean")
    )

    cluster_summary["risk_score"] = (
        cluster_summary["recency_mean"].rank(ascending=False) +
        cluster_summary["frequency_mean"].rank(ascending=True) +
        cluster_summary["monetary_mean"].rank(ascending=True)
    )

    high_risk_cluster = cluster_summary["risk_score"].idxmax()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm, kmeans, scaler, high_risk_cluster


# =====================================================
# PREPROCESSING PIPELINE
# =====================================================

def build_preprocessor(numeric_cols, categorical_cols):

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ],
        remainder="drop"
    )


# =====================================================
# MAIN ORCHESTRATION FUNCTION
# =====================================================

def run_preprocessing(
    raw_path=RAW_PATH,
    processed_path=PROCESSED_PATH,
    pipeline_path=PIPELINE_PATH
):
    print("📥 Loading raw data...")
    df = pd.read_csv(raw_path)

    # Feature engineering
    df = DateTimeExtractor().transform(df)
    df = PerCustomerAggregates().transform(df)

    # Target engineering
    rfm, snapshot = calculate_rfm(df)
    rfm_labeled, kmeans, scaler, high_risk_cluster = assign_high_risk_cluster(rfm)

    df = df.merge(
        rfm_labeled[["CustomerId", "is_high_risk"]],
        on="CustomerId",
        how="left"
    )

    df["is_high_risk"] = df["is_high_risk"].fillna(0).astype(int)

    # Build preprocessing pipeline
    numeric_cols = [
        "Amount", "Value",
        "total_txn_amount", "avg_txn_amount",
        "count_txn", "std_txn_amount",
        "txn_hour", "txn_day", "txn_month", "txn_year"
    ]

    categorical_cols = [
        "CurrencyCode", "CountryCode",
        "ProductCategory", "PricingStrategy",
        "ChannelId", "ProviderId", "ProductId"
    ]

    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    preprocessor.fit(df[numeric_cols + categorical_cols])

    # Save artifacts
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)

    df.to_csv(processed_path, index=False)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "rfm_scaler": scaler,
            "kmeans": kmeans,
            "high_risk_cluster": high_risk_cluster,
            "snapshot_date": snapshot
        },
        pipeline_path
    )

    print("✅ Preprocessing completed successfully")
    print(f"📁 Processed data saved to: {processed_path}")
    print(f"📦 Pipeline saved to: {pipeline_path}")

    return df


# =====================================================
# SCRIPT ENTRY POINT
# =====================================================

if __name__ == "__main__":
    run_preprocessing()
