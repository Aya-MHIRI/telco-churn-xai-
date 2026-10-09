# API FastAPI (Hugging Face Spaces « Docker » ou tout hébergeur de conteneurs)
FROM python:3.11-slim
WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src ./src
COPY api ./api
EXPOSE 7860
# Un seul worker : les sessions d'entraînement sont gardées en mémoire
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
