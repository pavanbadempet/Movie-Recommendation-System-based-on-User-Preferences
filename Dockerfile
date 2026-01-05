# Multi-stage Dockerfile for Movie Recommendation System
# Stage 1: Build ETL artifacts
# Stage 2: Lightweight runtime

FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY etl/ ./etl/
COPY backend/ ./backend/
COPY streamlit_app.py .

# Copy Pre-computed Models and Data
COPY models/ ./models/
COPY data/processed/ ./data/processed/

# Create other directories
RUN mkdir -p data/raw logs

# -------------------------------------------
# Stage 2: Runtime image
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Default command: run backend API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
