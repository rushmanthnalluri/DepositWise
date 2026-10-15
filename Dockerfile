FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    DEPOSITWISE_MODEL_PATH=/app/models/depositwise_model.joblib

WORKDIR /app
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY backend ./backend
COPY frontend ./frontend
COPY ml ./ml
# Download the public upstream dataset, generate the EDA profile, and train at image build time.
# This keeps the dataset and trained model out of Git while making the deployed API usable.
RUN python -m ml.eda && python -m ml.train
RUN useradd --create-home --uid 10001 appuser && mkdir -p /app/models && chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT"]
