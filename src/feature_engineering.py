import pandas as pd
import numpy as np

from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================
# WoE + IV Transformer
# ============================================================

class WoEIVTransformer(BaseEstimator, TransformerMixin):
    """
    Weight of Evidence transformer with:
    - Stable binning learned in fit()
    - IV computation
    - Safe numeric output
    """

    def __init__(self, bins=10, min_pct=1e-6):
        self.bins = bins
        self.min_pct = min_pct

        self.bin_edges_ = {}
        self.woe_maps_ = {}
        self.iv_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)

        df = X.copy()
        df["target"] = y

        total_good = (df["target"] == 0).sum()
        total_bad = (df["target"] == 1).sum()

        for col in X.columns:
            # -----------------------------
            # Create bins (store edges)
            # -----------------------------
            try:
                bins = pd.qcut(
                    df[col],
                    q=self.bins,
                    retbins=True,
                    duplicates="drop"
                )[1]
                df["bin"] = pd.cut(df[col], bins=bins, include_lowest=True)
                self.bin_edges_[col] = bins
            except Exception:
                # Fallback for low-cardinality columns
                df["bin"] = df[col]
                self.bin_edges_[col] = None

            # -----------------------------
            # WoE & IV calculation
            # -----------------------------
            grouped = df.groupby("bin", observed=False)["target"].agg(
                total="count",
                bad="sum"
            )
            grouped["good"] = grouped["total"] - grouped["bad"]

            grouped["bad_pct"] = grouped["bad"] / total_bad
            grouped["good_pct"] = grouped["good"] / total_good

            # Prevent divide-by-zero
            grouped["bad_pct"] = grouped["bad_pct"].clip(lower=self.min_pct)
            grouped["good_pct"] = grouped["good_pct"].clip(lower=self.min_pct)

            grouped["woe"] = np.log(grouped["good_pct"] / grouped["bad_pct"])
            grouped["iv"] = (grouped["good_pct"] - grouped["bad_pct"]) * grouped["woe"]

            self.woe_maps_[col] = grouped["woe"].to_dict()
            self.iv_[col] = grouped["iv"].sum()

        return self

    def transform(self, X):
        check_is_fitted(self, "woe_maps_")
        X = pd.DataFrame(X).copy()

        for col in X.columns:
            bins = self.bin_edges_.get(col)

            if bins is not None:
                binned = pd.cut(X[col], bins=bins, include_lowest=True)
                mapped = binned.map(self.woe_maps_[col])
            else:
                mapped = X[col].map(self.woe_maps_[col])

            # 🔑 Ensure numeric output
            X[col] = pd.to_numeric(mapped, errors="coerce").fillna(0.0)

        # Attach IV metadata
        X.attrs["iv"] = pd.Series(self.iv_)
        return X

    def get_iv(self):
        return (
            pd.DataFrame.from_dict(self.iv_, orient="index", columns=["IV"])
            .sort_values("IV", ascending=False)
        )


# ============================================================
# IV-Based Feature Selector
# ============================================================

class IVFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, iv_threshold=0.02):
        self.iv_threshold = iv_threshold
        self.selected_features_ = []

    def fit(self, X, y=None):
        iv_series = X.attrs.get("iv")
        if iv_series is None:
            raise ValueError("IV metadata missing from WoE transformer")

        self.selected_features_ = iv_series[
            iv_series >= self.iv_threshold
        ].index.tolist()

        return self

    def transform(self, X):
        return X[self.selected_features_]

