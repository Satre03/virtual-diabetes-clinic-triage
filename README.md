# Virtual-Diabetes-Clinic-Triage-

# Diabetes Risk Service

Detta projekt är en liten ML-tjänst som förutspår kortsiktig diabetesprogression baserat på det öppna *scikit-learn Diabetes*-datasetet.  
Projektet är byggt som en reproducerbar **MLOps-pipeline** med **GitHub Actions** och **Docker**.

---

## Kommandon för att köra projektet

### Klona projektet
```bash
git clone [https://github.com/MelissaWestberg/diabetes_risk_service.git](https://github.com/Satre03/virtual-diabetes-clinic-triage.git)
cd diabetes_risk_service
```

Skapa och aktivera virtuell miljö
MAC:
```bash
python3 -m venv venv
source venv/bin/activate 
```

WINDOWS:
``` bash
py -m venv venv
venv\Scripts\activate
```
Installerar beroenden

```bash
pip install -r requirements.txt
```
Träna modellen manuellt (valfritt)
```bash
python src/train.py
```
Bygg Docker-image
```bash
docker build -t ghcr.io/melissawestberg/diabetes_risk_service:v0.3 .
```
Kör containern
```bash
docker run -p 8000:8000 ghcr.io/melissawestberg/diabetes_risk_service:v0.3
```
Öppna sedan i webbläsaren:
```bash
http://localhost:8000/health
```

Förväntat svar:
```bash
{"status": "ok", "model_version": "v0.3"}
```


Exempel på payload för /predict
Skicka med curl:
```bash
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}'
```
Förväntat svar:
```bash
{"prediction": 153.2}
```
Kör containern
```bash
docker run -p 8002:8000 ghcr.io/melissawestberg/diabetes_risk_service:v0.3
```
Då nås API:t på:
```bash
http://localhost:8002/health
```

