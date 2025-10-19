from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

# -----------------------------
# Load data
# -----------------------------
data = load_diabetes(as_frame=False)
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Define Ridge pipeline
# -----------------------------
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0, random_state=42))
])

# -----------------------------
# Train Ridge model
# -----------------------------
pipeline.fit(X_train, y_train)

# -----------------------------
# Evaluate RMSE
# -----------------------------
preds_train = pipeline.predict(X_train)
preds_test = pipeline.predict(X_test)
rmse = float(mean_squared_error(y_test, preds_test, squared=False))
print(f"RMSE (Ridge): {rmse:.2f}")

# -----------------------------
# Create “high-risk” flag (top 25% of targets)
# -----------------------------
thr_value = float(np.percentile(y_train, 75))
y_train_high = (y_train >= thr_value).astype(int)
y_test_high = (y_test >= thr_value).astype(int)

# -----------------------------
# Calibrate using Isotonic Regression
# -----------------------------
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(preds_train, y_train_high)
prob_test = iso.predict(preds_test)

y_pred_flag = (prob_test >= 0.5).astype(int)
precision = float(precision_score(y_test_high, y_pred_flag, zero_division=0))
recall = float(recall_score(y_test_high, y_pred_flag, zero_division=0))

print(f"Precision@0.5: {precision:.2f}")
print(f"Recall@0.5: {recall:.2f}")

# -----------------------------
# Save artifacts
# -----------------------------
artifacts_dir = Path("artifacts/v0_2")
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, artifacts_dir / "model.joblib")
joblib.dump(iso, artifacts_dir / "isotonic_calibrator.joblib")

(artifacts_dir / "risk_threshold.json").write_text(json.dumps({
    "threshold_label": "75th percentile",
    "threshold_value_on_y_train": thr_value,
    "classification_threshold_on_calibrated_prob": 0.5
}, indent=2))

meta = {
    "pipeline": "v0.2",
    "selected_model": "Ridge",
    "version": "0.2.0",
    "dataset": "sklearn.diabetes",
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "rmse": rmse,
    "metrics": {
        "precision_at_0.5": precision,
        "recall_at_0.5": recall,
        "high_risk_threshold_on_y_train": "75th percentile"
    }
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

# -----------------------------
# Update CHANGELOG.md
# -----------------------------
changelog_path = Path("CHANGELOG.md")
fmt = lambda x: f"{x:.4f}"

lines = []
lines.append(f"## v0.2 — {datetime.now(timezone.utc).date().isoformat()}")
lines.append("**Iteration 2:** Switched to Ridge Regression, improved preprocessing with StandardScaler, and added calibrated high-risk flag (Isotonic Regression).")
lines.append("")
lines.append("### Metrics (test set)")
lines.append(f"- Ridge RMSE={fmt(rmse)}")
lines.append(f"- precision@0.5={fmt(precision)}, recall@0.5={fmt(recall)}")
lines.append("")

if changelog_path.exists():
    existing = changelog_path.read_text(encoding="utf-8")
    new_content = existing + ("\n" if not existing.endswith("\n") else "") + "\n".join(lines) + "\n"
else:
    new_content = "# CHANGELOG\n\n" + "\n".join(lines) + "\n"

changelog_path.write_text(new_content, encoding="utf-8")

print(json.dumps(meta, indent=2))
