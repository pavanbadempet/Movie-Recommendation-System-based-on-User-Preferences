FROM python:3.11-slim

WORKDIR /app

# Install system dependencies — no-install-recommends keeps the layer lean
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
# Adding torch explicitly for the Quantum ML algorithms
# hadolint ignore=DL3013
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# hadolint ignore=DL3059
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source and the serving artifacts used by local compose builds.
COPY backend ./backend
COPY data/processed ./data/processed
COPY data/evaluation ./data/evaluation
COPY models ./models

# Create a non-root user and hand over ownership before switching
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
