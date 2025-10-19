FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# This ARG will be set by GitHub Actions
ARG MODEL_VERSION
ENV MODEL_VERSION=${MODEL_VERSION}

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
