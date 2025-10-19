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
import os

def main():
    # Load data
    data = load_diabetes(as_frame=False)
    X, y = data.data, data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define pipeline: scaler + linear regression
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Evaluate model
    preds = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))
    print(f"RMSE: {rmse:.2f}")

    # Define root directory as parent of current working dir (assuming script runs from src)
    ROOT_DIR = Path(os.getcwd()).parent.resolve()

    # Artifacts directory at repo root (already exists)
    artifacts_dir = ROOT_DIR / "artifacts"
    print(f"Artifacts directory: {artifacts_dir}")

    # Test writing a test file to check permissions
    test_file = artifacts_dir / "test_write.txt"
    try:
        test_file.write_text("This is a test file to check write permissions.")
        print(f"Test file written successfully: {test_file}")
        test_file.unlink()  # Clean up test file after checking
    except Exception as e:
        print(f"Failed to write test file: {e}")

    # Save the trained model
    try:
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(pipeline, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"Failed to save model: {e}")

    # Save metadata
    try:
        meta = {
            "pipeline": "LinearRegression + StandardScaler",
            "version": "0.1.0",
            "rmse": rmse,
            "trained_at": datetime.now(timezone.utc).isoformat()
        }
        meta_path = artifacts_dir / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Metadata saved to {meta_path}")
    except Exception as e:
        print(f"Failed to save metadata: {e}")

if __name__ == "__main__":
    main()
