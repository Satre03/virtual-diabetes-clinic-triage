import argparse
import json
import joblib
import os
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
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
    elif version == "0.2":
        print("Using RandomForestRegressor (v0.2)")
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=10,
                max_features='sqrt',
                random_state=RANDOM_SEED
            ))
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

    metrics = {"version": version, "rmse": rmse}
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("Metrics saved to artifacts/metrics.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Model version (e.g., 0.1 or 0.2)")
    args = parser.parse_args()
    main(args.version)
