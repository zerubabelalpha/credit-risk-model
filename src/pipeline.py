import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np


class EnsureColumns(BaseEstimator, TransformerMixin):
    """Ensure required columns exist on the DataFrame (fill with NaN/defaults)."""

    def __init__(self, required_columns=None):
        self.required_columns = required_columns or []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.required_columns:
            if c not in X.columns:
                X[c] = np.nan
        return X


# Simple feature transformers
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


def get_feature_engineering_pipeline():
    """Return a small pipeline that applies datetime extraction and per-customer aggregates."""
    return Pipeline([
        ("datetime", DateTimeExtractor()),
        ("cust_agg", PerCustomerAggregates())
    ])


def build_preprocessor(numeric_cols, categorical_cols):
    """Build a robust ColumnTransformer for numeric and categorical features."""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    ct = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ], remainder="drop")

    # Wrap in a Pipeline that ensures missing columns are created before ColumnTransformer
    return Pipeline([
        ("ensure_cols", EnsureColumns(required_columns=numeric_cols + categorical_cols)),
        ("ct", ct)
    ])
