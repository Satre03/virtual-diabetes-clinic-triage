from pathlib import Path
from datetime import datetime, timezone
import json

import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
META_PATH = ARTIFACT_DIR / "model_meta.json"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

def main() -> None:
    data = load_diabetes(as_frame=False)  # no pandas dependency
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, preds, squared=False))

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    meta = {
        "pipeline": "baseline",
        "version": "0.1.0",
        "seed": 42,
        "n_features": int(X.shape[1]),
        "trained_at": datetime.now(timezone.utc).isoformat()
    }
    metrics = {"rmse": rmse}

    META_PATH.write_text(json.dumps(meta, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(json.dumps({"meta": meta, "metrics": metrics}))

if __name__ == "__main__":
    main()
