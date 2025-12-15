import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
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

PROCESSED_DATA_PATH = "data/processed/processed_data.csv"
PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
MODEL_DIR = "models/trained"

TEST_SIZE = 0.3
RANDOM_STATE = 42

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("credit-risk-demo")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Utility Functions
# =====================================================

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),     
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }


def display_best_model(results):
    best = max(results, key=lambda x: x["roc_auc"])

    print("\n📊 Model Performance (ROC-AUC)\n")
    for r in results:
        print(f"{r['model_name']:<20} : {r['roc_auc']:.4f}")

    print("\n🏆 Best Performing Model\n")
    print(f"{best['model_name']}  (ROC-AUC = {best['roc_auc']:.4f})")

    return best


# =====================================================
# Load Data
# =====================================================

print("📥 Loading processed data...")
df = pd.read_csv(PROCESSED_DATA_PATH)

y = df.pop("is_high_risk")
X = df

print("📦 Loading preprocessing pipeline...")
preprocessor = joblib.load(PIPELINE_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

results = []

# =====================================================
# Logistic Regression
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

with mlflow.start_run(run_name="Logistic_Regression"):

    mlflow.log_param("model_type", "LogisticRegression")

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

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_params(grid.best_params_)
    mlflow.sklearn.log_model(best_pipeline, "model")

    joblib.dump(
        best_pipeline,
        f"{MODEL_DIR}/logistic_regression_pipeline.joblib"
    )

    results.append({
        "model_name": "Logistic Regression",
        "roc_auc": metrics["roc_auc"]
    })

    print("✅ Logistic Regression completed")

# =====================================================
# LightGBM
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

with mlflow.start_run(run_name="LightGBM"):

    mlflow.log_param("model_type", "LightGBM")

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

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    mlflow.log_params(grid.best_params_)
    mlflow.lightgbm.log_model(best_pipeline, "model")

    joblib.dump(
        best_pipeline,
        f"{MODEL_DIR}/lightgbm_pipeline.joblib"
    )

    results.append({
        "model_name": "LightGBM",
        "roc_auc": metrics["roc_auc"]
    })

    print("✅ LightGBM completed")

# =====================================================
# Compare & Display Best Model
# =====================================================

best_model = display_best_model(results)

print("\n Training complete.   Run: mlflow ui")

