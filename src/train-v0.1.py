import argparse
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def main(version: str):
    print(f"Starting training for version {version}")

    # Load data
    data = load_diabetes(as_frame=False)
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define pipeline (currently only LinearRegression for all versions, 
    # but can add branching logic here if you want different models for different versions)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    print(f"RMSE: {rmse:.4f}")

    # Save artifacts
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / f"model_v{version}.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    meta = {
        "pipeline": "linear_regression",
        "version": version,
        "rmse": rmse,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    meta_path = artifacts_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")

    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear regression model.")
    parser.add_argument(
        "--version",
        type=str,
        required=True,
        help="Model version string, e.g. 0.1"
    )
    args = parser.parse_args()
    main(args.version)
