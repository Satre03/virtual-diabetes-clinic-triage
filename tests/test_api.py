from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"

def test_prediction():
    payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    assert "prediction" in res.json()
