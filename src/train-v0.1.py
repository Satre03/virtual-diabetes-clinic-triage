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

def get_model(version: str):
    if version == "0.1":
        print("Using LinearRegression (v0.1)")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    else:
        raise ValueError(f"Unknown model version: {version}")

def main(version: str):
    print(f"Training model version {version}")
    data = load_diabetes(as_frame=True)
    X = data.frame.drop(columns=["target"])
    y = data.frame["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

    model = get_model(version)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"Test RMSE: {rmse:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    model_path = f"artifacts/model_v{version}.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    feature_path = "artifacts/feature_list.json"
    with open(feature_path, "w") as f:
        json.dump(list(X.columns), f)
    print(f"Features saved to {feature_path}")

    meta = {
        "version": version,
        "rmse": rmse,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    with open("artifacts/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Meta saved to artifacts/meta.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Model version (e.g., 0.1)")
    args = parser.parse_args()
    main(args.version)
