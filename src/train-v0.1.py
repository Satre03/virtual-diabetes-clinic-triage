import argparse
import json
import joblib
import os
import numpy as np
from datetime import datetime, timezone
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

def get_model():
    """Returns a linear regression pipeline with standard scaling."""
    print("Using LinearRegression (v0.1)")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

def main(version: str):
    print(f"Training model version: {version}")

    # Load dataset
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Initialize and train model
    model = get_model()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")

    # Create artifacts directory
    os.makedirs("artifacts", exist_ok=True)

    # Save model
    model_path = f"artifacts/model_v{version}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save feature list
    with open("artifacts/feature_list.json", "w") as f:
        json.dump(list(X.columns), f)
    print("Feature list saved.")

    # Save metadata
    metadata = {
        "version": version,
        "rmse": rmse,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open("artifacts/meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("Training metadata saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Model version (e.g. 0.1)")
    args = parser.parse_args()
    main(args.version)
