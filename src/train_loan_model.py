import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import mlflow

# Import from existing modules
from preprocessing import (
    get_feature_engineering_pipeline,
    NUMERIC_COLS,
    CATEGORICAL_COLS
)

# ======== Paths & Config ========
RAW_DATA_PATH = "data/raw/data.csv"
MODEL_DIR = "models/trained"
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42

# ======== Step 1: Engineer Loan Targets ========

def engineer_loan_targets(df):
    """
    Engineers target variables: LoanAmount and LoanDuration.
    """
    # Working on a copy to avoid side effects
    loan_txns = df[df["Amount"] > 0].copy()
    loan_txns["TransactionStartTime"] = pd.to_datetime(loan_txns["TransactionStartTime"])

    # Aggregate by CustomerId
    loan_amount = loan_txns.groupby("CustomerId")["Amount"].sum().rename("LoanAmount")
    
    # Calculate duration
    loan_duration = loan_txns.groupby("CustomerId").agg(
        LoanStart=("TransactionStartTime", "min"),
        LoanEnd=("TransactionStartTime", "max")
    )
    loan_duration["LoanDuration"] = (loan_duration["LoanEnd"] - loan_duration["LoanStart"]).dt.days

    # Merge into single target DataFrame
    loan_targets = pd.concat([loan_amount, loan_duration["LoanDuration"]], axis=1).reset_index()

    return loan_targets

# ======== Step 2: Custom Regression Preprocessor ========

def build_regression_preprocessor(numeric_cols, categorical_cols):
    """
    Builds a preprocessor suitable for regression (no WoE, just scaling/encoding).
    """
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

# ======== Step 3: Evaluation Function ========

def evaluate_regression_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"mse": mse, "r2": r2}

# ======== Step 4: Train Model Function ========

def train_lgbm_regressor(X_train, y_train, X_test, y_test, model_name, feature_pipeline, preprocessor):
    
    # Manual debugging steps to isolate error
    print("Debugging: Manual step execution...")
    
    # 1. Feature Engineering
    print(f"Step 1: Feature Engineering ({type(feature_pipeline).__name__})...")
    try:
        X_train_eng = feature_pipeline.fit_transform(X_train, y_train)
        X_test_eng = feature_pipeline.transform(X_test)
        print("Feature Engineering done.")
    except Exception as e:
        print(f"FAILED at Feature Engineering: {e}")
        import traceback; traceback.print_exc()
        raise e

    # 2. Preprocessing
    print(f"Step 2: Preprocessing ({type(preprocessor).__name__})...")
    try:
        X_train_prep = preprocessor.fit_transform(X_train_eng, y_train)
        X_test_prep = preprocessor.transform(X_test_eng)
        print("Preprocessing done.")
    except Exception as e:
        print(f"FAILED at Preprocessing: {e}")
        import traceback; traceback.print_exc()
        raise e

    # 3. Regressor
    print("Step 3: Regressor...")
    reg = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    try:
        reg.fit(X_train_prep, y_train)
        print("Regressor fitted.")
    except Exception as e:
        print(f"FAILED at Regressor: {e}")
        import traceback; traceback.print_exc()
        raise e
    
    # Construct final pipeline with FITTED steps
    best_model = Pipeline([
        ("feature_eng", feature_pipeline), # This is now fitted (in Step 1 reuse)
        ("preprocessor", preprocessor),     # This is now fitted
        ("regressor", reg)                  # This is now fitted
    ])

    print("Evaluating model...")
    metrics = evaluate_regression_model(best_model, X_test, y_test)

    # Log and save model with MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    with mlflow.start_run(run_name=f"LGBM_{model_name}"):
        mlflow.log_param("model_type", "lgbm_regressor")
        mlflow.log_metric("mse", float(metrics["mse"]))
        mlflow.log_metric("r2", float(metrics["r2"]))
        try:
            mlflow.lightgbm.log_model(reg, artifact_path="model")
        except Exception:
            # Fallback to sklearn logging if lightgbm flavor isn't available
            mlflow.sklearn.log_model(best_model, artifact_path="model")

    # Save the model locally
    save_path = os.path.join(MODEL_DIR, f"{model_name}_lgbm_regressor.joblib")
    joblib.dump(best_model, save_path)

    print(f"Trained {model_name}. MSE: {metrics['mse']:.2f}, R2: {metrics['r2']:.4f}")
    print(f"Model saved to {save_path}")
    return best_model

# ======== Step 5: Main Execution ========

def main():
    print("Loading raw data...")
    if not os.path.exists(RAW_DATA_PATH):
        print(f"Error: {RAW_DATA_PATH} not found.")
        return

    df = pd.read_csv(RAW_DATA_PATH)
    
    print("Engineering loan targets...")
    loan_targets = engineer_loan_targets(df)
    
    # Merge targets back to main dataframe
    df = df.merge(loan_targets, on="CustomerId", how="left")
    
    # Filter rows that have targets (implied: only customers who had valid loan transactions)
    # The user draft did dropna, which means we only train on customers with history.
    df = df.dropna(subset=["LoanAmount", "LoanDuration"])
    
    print(f"Data shape after adding loan targets: {df.shape}")

    # Prepare X and y
    # Note: X must contain columns required by feature_engineering pipeline (Amount, TransactionStartTime, etc.)
    X = df.drop(columns=["LoanAmount", "LoanDuration"])
    y_amount = df["LoanAmount"]
    y_duration = df["LoanDuration"]

    # Split
    X_train, X_test, y_train_amount, y_test_amount = train_test_split(
        X, y_amount, test_size=0.2, random_state=RANDOM_STATE
    )
    # We use the same split for duration to keep it simple/consistent or split again? 
    # User draft split again. I'll stick to variable names but ensures same random state yields same split if X is same.
    _, _, y_train_duration, y_test_duration = train_test_split(
        X, y_duration, test_size=0.2, random_state=RANDOM_STATE
    )

    # Build Pipeline Components
    feature_eng_pipeline = get_feature_engineering_pipeline()
    
    # We need to ensure we don't try to use WoE cols that might be generated or required.
    # The NUMERIC_COLS and CATEGORICAL_COLS from preprocessing are likely fine.
    # However, 'woe_cols' in preprocessing included derived aggregates.
    # Our regression preprocessor just treats them as numeric. 
    # NOTE: 'total_txn_amount' etc are generated by feature_eng_pipeline.
    # So we should include them in checking or just trust the columns list.
    
    # NUMERIC_COLS in src/preprocessing.py includes:
    # "Amount", "Value", "total_txn_amount", "avg_txn_amount", "count_txn", "std_txn_amount", ...
    # These match what feature_eng_pipeline produces.
    
    preprocessor = build_regression_preprocessor(NUMERIC_COLS, CATEGORICAL_COLS)

    # Train Models
    train_lgbm_regressor(
        X_train, y_train_amount, X_test, y_test_amount, 
        "loan_amount", feature_eng_pipeline, preprocessor
    )
    
    train_lgbm_regressor(
        X_train, y_train_duration, X_test, y_test_duration, 
        "loan_duration", feature_eng_pipeline, preprocessor
    )
    
    print("\nAll training tasks completed.")

if __name__ == "__main__":
    main()
