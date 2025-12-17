import os
import joblib
import pandas as pd
import numpy as np

from feature_engineering import WoEIVTransformer, IVFeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from pipeline import build_preprocessor as build_pipeline_preprocessor

RAW_PATH = "data/raw/data.csv"
PROCESSED_PATH = "data/processed/processed_data.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
IV_PATH = "models/iv_table.csv"

RANDOM_STATE = 42

NUMERIC_COLS = [
    "Amount", "Value", "total_txn_amount", "avg_txn_amount",
    "count_txn", "std_txn_amount", "txn_hour", "txn_day",
    "txn_month", "txn_year"
]

CATEGORICAL_COLS = [
    "CurrencyCode", "CountryCode", "ProductCategory",
    "PricingStrategy", "ChannelId", "ProviderId", "ProductId"
]

WOE_COLS = [
    "Amount", "Value", "total_txn_amount", "avg_txn_amount",
    "count_txn", "std_txn_amount", "txn_hour", "txn_day", "txn_month"
]


# ============================================================
# Custom Feature Transformers
# ============================================================

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


# ============================================================
# RFM-Based Target Engineering
# ============================================================

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

    # Ensure n_clusters <= n_samples
    n_clusters = min(3, max(1, rfm.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    rfm["cluster"] = kmeans.fit_predict(rfm_scaled)

    rfm["risk_score"] = (
        rfm["recency"].rank(ascending=False) +
        rfm["frequency"].rank(ascending=True) +
        rfm["monetary"].rank(ascending=True)
    )

    high_risk_cluster = rfm.groupby("cluster")["risk_score"].mean().idxmax()
    rfm["is_high_risk"] = (rfm["cluster"] == high_risk_cluster).astype(int)

    return rfm[["CustomerId", "is_high_risk"]]

# ============================================================
# Build Preprocessing Pipeline
# ============================================================

def build_preprocessor(numeric_cols, categorical_cols, woe_cols):

    woe_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("woe", WoEIVTransformer(bins=10)),
        ("iv_select", IVFeatureSelector(iv_threshold=0.02))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    return ColumnTransformer([
        ("woe", woe_pipeline, woe_cols),
        ("num", numeric_pipeline, [c for c in numeric_cols if c not in woe_cols]),
        ("cat", categorical_pipeline, categorical_cols)
    ])

def get_feature_engineering_pipeline():
    """Returns the feature engineering steps as a Pipeline"""
    return Pipeline([
        ("datetime_extractor", DateTimeExtractor()),
        ("customer_aggregates", PerCustomerAggregates())
    ])

# ============================================================
# Run Preprocessing
# ============================================================

def run_preprocessing():
    df = pd.read_csv(RAW_PATH)

    df = get_feature_engineering_pipeline().fit_transform(df)

    target_df = create_target(df)
    df = df.merge(target_df, on="CustomerId", how="left")
    df["is_high_risk"] = df["is_high_risk"].fillna(0)



    # Use pipeline's robust preprocessor builder (wraps ensure_cols)
    preprocessor = build_pipeline_preprocessor(
        NUMERIC_COLS, CATEGORICAL_COLS
    )

    preprocessor.fit(
        df[NUMERIC_COLS + CATEGORICAL_COLS],
        df["is_high_risk"]
    )

    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    joblib.dump(preprocessor, PIPELINE_PATH)
    df.to_csv(PROCESSED_PATH, index=False)

    iv_table = (
        preprocessor.named_transformers_["woe"]
        .named_steps["woe"]
        .get_iv()
    )

    iv_table.to_csv(IV_PATH)

    print("✅ Preprocessing pipeline saved")
    print("📊 IV-based feature selection applied")

# ============================================================
# Main
# ============================================================

def main():
    print("🚀 Running preprocessing with WoE + IV feature selection...")
    run_preprocessing()
    print("🎯 Pipeline completed successfully")

if __name__ == "__main__":
    main()
