from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import argparse
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(description="Train v0.2 diabetes triage model")
parser.add_argument(
    "--pipeline", choices=["ridge", "rf"], default="ridge", help="Which model to train"
)
parser.add_argument("--calibrate", action="store_true", help="Compute high-risk flag metrics")
parser.add_argument(
    "--poly", action="store_true", help="Add polynomial features (degree=2)"
)
args = parser.parse_args()

# ------------------------------
# Load data
# ------------------------------
data = load_diabetes(as_frame=False)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# Choose model with tuned hyperparameters
# ------------------------------
if args.pipeline == "ridge":
    model = Ridge(alpha=0.1, random_state=42)  # tuned alpha
else:
    model = RandomForestRegressor(
        n_estimators=500, max_depth=5, min_samples_leaf=5, random_state=42
    )

# ------------------------------
# Build pipeline
# ------------------------------
steps = []
if args.poly:
    steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
steps.append(("scaler", StandardScaler()))
steps.append(("model", model))
pipeline = Pipeline(steps)

# ------------------------------
# Train
# ------------------------------
pipeline.fit(X_train, y_train)

# ------------------------------
# Evaluate RMSE
# ------------------------------
preds = pipeline.predict(X_test)
rmse = float(mean_squared_error(y_test, preds, squared=False))
print(f"RMSE: {rmse:.2f}")

# ------------------------------
# Optional calibration (high-risk flag)
# ------------------------------
precision = recall = None
if args.calibrate:
    threshold = np.percentile(y_train, 75)
    y_train_binary = (y_train > threshold).astype(int)
    y_pred_binary = (pipeline.predict(X_train) > threshold).astype(int)

    precision = precision_score(y_train_binary, y_pred_binary)
    recall = recall_score(y_train_binary, y_pred_binary)

# ------------------------------
# Artifacts directory
# ------------------------------
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

# Compute delta RMSE vs previous
prev_rmse = None
meta_path = artifacts_dir / "meta.json"
if meta_path.exists():
    try:
        prev_meta = json.loads(meta_path.read_text())
        prev_rmse = prev_meta.get("rmse")
    except Exception:
        pass
delta_rmse = None if prev_rmse is None else rmse - prev_rmse

# ------------------------------
# Save model & metadata
# ------------------------------
joblib.dump(pipeline, artifacts_dir / "model.joblib")

meta = {
    "pipeline": args.pipeline,
    "version": "0.2.0",
    "rmse": rmse,
    "trained_at": datetime.now(timezone.utc).isoformat()
}
(artifacts_dir / "meta-v0.2.json").write_text(json.dumps(meta, indent=2))

metrics = {
    "rmse": rmse,
    "precision": precision,
    "recall": recall,
    "delta_vs_prev": {"rmse": delta_rmse}
}
(artifacts_dir / "metrics-v0.2.json").write_text(json.dumps(metrics, indent=2))

# ------------------------------
# Print summary
# ------------------------------
print(json.dumps(meta, indent=2))
if precision is not None:
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
if delta_rmse is not None:
    print(f"Δ RMSE vs previous: {delta_rmse:+.3f}")

# Load old model
pipeline = joblib.load("artifacts/model.joblib")

# Example input
x_example = np.array([[0.02, -0.044, 0.06, -0.03, -0.02, 0.03, -0.02, 0.02, 0.02, -0.001]])
pred = pipeline.predict(x_example)
print(pred)
