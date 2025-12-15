import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from lightgbm import LGBMClassifier

# --------------------- CONFIG ---------------------

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
MODEL_DIR = "models/trained"

TEST_SIZE = 0.2
RANDOM_STATE = 42

# MLflow local tracking (file-based)
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-risk-demo")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------- Utility ---------------------

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred), # Sensitivity
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
        }

# --------------------- Load data ---------------------

print("📥 Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)
y = df.pop("is_high_risk")
X = df

print("📦 Loading preprocessing pipeline...")
preprocessor = joblib.load(PIPELINE_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

# =====================================================
# Logistic Regression Experiment
# =====================================================

log_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

log_param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

with mlflow.start_run():

    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    grid = GridSearchCV(
        log_pipeline,
        log_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    metrics = evaluate_model(best_pipeline, X_test, y_test)
# new added
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])

    mlflow.log_params(grid.best_params_)

    mlflow.sklearn.log_model(best_pipeline, "model")

    joblib.dump(
        best_pipeline,
        f"{MODEL_DIR}/logistic_regression_pipeline.joblib"
    )

    print("✅ Logistic Regression tracked in MLflow")

# =====================================================
# LightGBM Experiment
# =====================================================

lgbm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LGBMClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    ))
])

lgbm_param_grid = {
    "model__n_estimators": [200],
    "model__learning_rate": [0.05, 0.1],
    "model__num_leaves": [31, 50]
}

with mlflow.start_run():

    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)

    grid = GridSearchCV(
        lgbm_pipeline,
        lgbm_param_grid,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)
    best_pipeline = grid.best_estimator_

    metrics = evaluate_model(best_pipeline, X_test, y_test)

        # new added
    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])

    mlflow.log_params(grid.best_params_)

    mlflow.lightgbm.log_model(best_pipeline, "model")

    joblib.dump(
        best_pipeline,
        f"{MODEL_DIR}/lightgbm_pipeline.joblib"
    )

    print("✅ LightGBM tracked in MLflow")

print("🚀 Training complete. Run 'mlflow ui' to inspect experiments.")
