FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=$MODEL_VERSION

COPY models/model_v${MODEL_VERSION}.joblib ./models/model_v${MODEL_VERSION}.joblib
COPY models/feature_list.json ./models/feature_list.json

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
