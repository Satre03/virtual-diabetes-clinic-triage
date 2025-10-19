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

    # Define root directory (two levels up from this script)
    ROOT_DIR = Path(__file__).parent.parent.resolve()

    # Prepare artifacts directory at repo root
    artifacts_dir = ROOT_DIR / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save the trained model
    model_path = artifacts_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Model saved to {model_path}")

    # Save metadata
    meta = {
        "pipeline": "LinearRegression + StandardScaler",
        "version": "0.1.0",
        "rmse": rmse,
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    meta_path = artifacts_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {meta_path}")

if __name__ == "__main__":
    main()
