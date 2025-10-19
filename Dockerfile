FROM python:3.11-slim

ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=$MODEL_VERSION

WORKDIR /app

COPY requirements.txt requirements.txt
COPY src/ artifacts/
COPY src/ src/

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
