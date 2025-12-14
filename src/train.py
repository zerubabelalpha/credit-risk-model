# src/

import os
import joblib
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from lightgbm import LGBMClassifier


# =====================================================
# CONFIG
# =====================================================

PROCESSED_DATA_PATH = "../data/processed/processed_data.csv"
PIPELINE_PATH = "../models/transformer_pipeline.joblib"
MODEL_OUTPUT_DIR = "../models/trained"

RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "credit_risk_model_training"


# =====================================================
# UTILITY FUNCTIONS
# =====================================================

def load_data():
    df = pd.read_csv(PROCESSED_DATA_PATH)

    if "is_high_risk" not in df.columns:
        raise ValueError("Target column 'is_high_risk' not found.")

    y = df["is_high_risk"]
    X = df.drop(columns=["is_high_risk"])

    return X, y


def load_preprocessor():
    artifacts = joblib.load(PIPELINE_PATH)
    return artifacts["preprocessor"]


def evaluate_model(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob)
    }


# =====================================================
# MODEL TRAINING FUNCTION
# =====================================================

def train_and_log_model(
    model_name,
    model,
    param_grid,
    X_train,
    y_train,
    X_test,
    y_test
):
    with mlflow.start_run(run_name=model_name):

        grid = GridSearchCV(
            model,
            param_grid,
            scoring="roc_auc",
            cv=5,
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_prob)

        # -------------------------
        # MLflow logging
        # -------------------------
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)

        if model_name == "LightGBM":
            mlflow.lightgbm.log_model(best_model, "model")
        else:
            mlflow.sklearn.log_model(best_model, "model")

        return best_model, metrics


# =====================================================
# MAIN TRAINING PIPELINE
# =====================================================

def main():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("📥 Loading data...")
    X, y = load_data()

    print("📦 Loading preprocessing pipeline...")
    preprocessor = load_preprocessor()

    # Apply preprocessing
    X_transformed = preprocessor.transform(X)

    print("✂ Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # =================================================
    # Logistic Regression
    # =================================================

    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE
    )

    log_reg_params = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "solver": ["lbfgs"]
    }

    log_model, log_metrics = train_and_log_model(
        "LogisticRegression",
        log_reg,
        log_reg_params,
        X_train, y_train,
        X_test, y_test
    )

    joblib.dump(
        log_model,
        f"{MODEL_OUTPUT_DIR}/logistic_regression_model.joblib"
    )

    # =================================================
    # LightGBM
    # =================================================

    lgbm = LGBMClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    lgbm_params = {
        "n_estimators": [100, 300],
        "max_depth": [-1, 5, 10],
        "learning_rate": [0.01, 0.1],
        "num_leaves": [31, 50]
    }

    lgbm_model, lgbm_metrics = train_and_log_model(
        "LightGBM",
        lgbm,
        lgbm_params,
        X_train, y_train,
        X_test, y_test
    )

    joblib.dump(
        lgbm_model,
        f"{MODEL_OUTPUT_DIR}/lightgbm_model.joblib"
    )

    # =================================================
    # Select & Register Best Model
    # =================================================

    print("\n🏆 Selecting best model based on ROC-AUC...")

    if lgbm_metrics["roc_auc"] > log_metrics["roc_auc"]:
        best_model = lgbm_model
        best_model_name = "credit_risk_lightgbm"
        best_metrics = lgbm_metrics
    else:
        best_model = log_model
        best_model_name = "credit_risk_logistic"
        best_metrics = log_metrics

    print(f"🏆 Best model: {best_model_name}")
    print("📊 Metrics:", best_metrics)

    # Register best model
    with mlflow.start_run(run_name="BestModelRegistration"):
        mlflow.log_metrics(best_metrics)
        mlflow.sklearn.log_model(
            best_model,
            "model",
            registered_model_name=best_model_name
        )

    print("✅ Training & registration complete.")


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":
    main()
