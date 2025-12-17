import os
import joblib
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from preprocessing import (
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    build_preprocessor,
    run_preprocessing,
    PROCESSED_PATH
)
from pipeline import get_feature_engineering_pipeline
from train_loan_model import main as train_loan_models

MODEL_DIR = "models/trained"
os.makedirs(MODEL_DIR, exist_ok=True)


def evaluate_classifier(model, X_test, y_test):
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs)
    }


def train_classifier(df):
    # Prepare per-customer dataset (one row per customer)
    cust = df.groupby("CustomerId").last().reset_index()

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = cust[feature_cols]
    y = cust["is_high_risk"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build pipeline: feature engineering -> preprocessor -> model
    feature_eng = get_feature_engineering_pipeline()
    preprocessor = build_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS, NUMERIC_COLS)  # woe cols are subset of numeric; build_preprocessor in preprocessing handles that

    # --- Logistic Regression
    lr_pipeline = Pipeline([
        ("feature_eng", feature_eng),
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])

    lr_grid = {
        "model__C": [0.01, 0.1, 1, 10]
    }

    mlflow.set_tracking_uri("file:./mlruns")

    with mlflow.start_run(run_name="Logistic_Regression"):
        print("🔄 Training Logistic Regression with GridSearchCV...")
        grid = GridSearchCV(lr_pipeline, lr_grid, scoring="roc_auc", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)

        best = grid.best_estimator_
        metrics = evaluate_classifier(best, X_test, y_test)

        # Log metrics and params
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))

        mlflow.log_params(grid.best_params_)
        mlflow.sklearn.log_model(best, "logistic_model")

        joblib.dump(best, f"{MODEL_DIR}/risk_logistic_pipeline.joblib")
        print("✅ Logistic Regression saved")

    # --- LightGBM Classifier
    try:
        from lightgbm import LGBMClassifier

        lgbm_pipeline = Pipeline([
            ("feature_eng", feature_eng),
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1))
        ])

        lgbm_grid = {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1]
        }

        with mlflow.start_run(run_name="LightGBM_Classifier"):
            print("🔄 Training LightGBM classifier with GridSearchCV...")
            grid = GridSearchCV(lgbm_pipeline, lgbm_grid, scoring="roc_auc", cv=5, n_jobs=-1)
            grid.fit(X_train, y_train)

            best = grid.best_estimator_
            metrics = evaluate_classifier(best, X_test, y_test)

            for k, v in metrics.items():
                mlflow.log_metric(k, float(v))

            mlflow.log_params(grid.best_params_)
            mlflow.lightgbm.log_model(best, "lightgbm_model")

            joblib.dump(best, f"{MODEL_DIR}/risk_lgbm_pipeline.joblib")
            print("✅ LightGBM classifier saved")

    except Exception as e:
        print(f"LightGBM training skipped: {e}")


def main():
    # Ensure processed data exists; if not, run preprocessing
    if not os.path.exists(PROCESSED_PATH):
        print("Processed data not found — running preprocessing...")
        run_preprocessing()

    df = pd.read_csv(PROCESSED_PATH)

    print("Starting classifier training...")
    train_classifier(df)

    print("Starting regression model training (loan amount & duration)...")
    train_loan_models()

    print("\n🎯 All training complete. Use `mlflow ui` to inspect runs.")


if __name__ == "__main__":
    main()