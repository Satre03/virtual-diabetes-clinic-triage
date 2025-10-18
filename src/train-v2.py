from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, roc_auc_score

# Load data
data = load_diabetes(as_frame=False)
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define models and parameter grids
models = [
    ("Ridge", Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ]), {
        "model__alpha": [0.1, 1.0, 10.0, 100.0]
    }),
    ("RandomForest", Pipeline([
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
    ]), {
        "model__n_estimators": [200, 500],
        "model__max_depth": [None, 8, 16],
        "model__min_samples_leaf": [1, 3, 5]
    })
]

# Train and select best model
best_model = None
best_name = None
best_rmse = float("inf")
best_params = None

for name, pipeline, params in models:
    grid = GridSearchCV(
        pipeline,
        params,
        scoring="neg_root_mean_squared_error",
        cv=5,
        n_jobs=-1,
        refit=True
    )
    grid.fit(X_train, y_train)
    rmse_cv = -grid.best_score_
    if rmse_cv < best_rmse:
        best_rmse = rmse_cv
        best_model = grid.best_estimator_
        best_name = name
        best_params = grid.best_params_

# Evaluate
preds = best_model.predict(X_test)
rmse = float(mean_squared_error(y_test, preds, squared=False))
print(f"Best model: {best_name}")
print(f"RMSE: {rmse:.2f}")

# Optional classification metrics (high-risk threshold)
threshold = np.percentile(y_train, 75)
y_true_flag = (y_test >= threshold).astype(int)
y_pred_flag = (preds >= threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true_flag, y_pred_flag, average="binary", zero_division=0
)
try:
    auroc = float(roc_auc_score(y_true_flag, preds))
except ValueError:
    auroc = float("nan")

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, AUROC: {auroc:.3f}")

# Save artifacts
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(best_model, artifacts_dir / "model.joblib")

meta = {
    "pipeline": best_name,
    "version": "0.2.0",
    "cv_rmse": float(best_rmse),
    "rmse": rmse,
    "trained_at": datetime.now(timezone.utc).isoformat(),
    "params": best_params,
    "classification": {
        "threshold_percentile": 75,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc
    }
}
(artifacts_dir / "meta.json").write_text(json.dumps(meta, indent=2))

print(json.dumps(meta, indent=2))
