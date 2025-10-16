from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pathlib import Path
import joblib
import json
import numpy as np

ART_DIR = Path("artifacts")
MODEL_PATH = ART_DIR / "model.joblib"
META_PATH = ART_DIR / "meta.json"

class PredictRequest(BaseModel):
    features: list[float] = Field(..., description="Feature vector of length 10 (sklearn diabetes)")

class PredictResponse(BaseModel):
    prediction: float
    model_version: str

app = FastAPI(title="Virtual Diabetes Clinic Triage", version="0.1.0")
_model = None
_meta = {"version": "unknown"}

@app.on_event("startup")
def _startup():
    global _model, _meta
    try:
        if MODEL_PATH.exists():
            _model = joblib.load(MODEL_PATH)
        if META_PATH.exists():
            _meta = json.loads(META_PATH.read_text())
    except Exception as e:
        _model = None
        _meta = {"version": "unknown"}
        # still boot; observability via JSON errors

@app.get("/health")
def health():
    return {"status": "ok", "model_version": _meta.get("version", "unknown")}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(req.features) != 10:
        raise HTTPException(status_code=422, detail=f"Expected 10 features, got {len(req.features)}")
    X = np.array(req.features, dtype=float).reshape(1, -1)
    y = float(_model.predict(X)[0])
    return PredictResponse(prediction=y, model_version=_meta.get("version", "unknown"))

@app.exception_handler(Exception)
async def _json_error(_, exc: Exception):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
