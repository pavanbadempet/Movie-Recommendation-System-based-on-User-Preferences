# Multi-stage Dockerfile for Movie Recommendation System
# Stage 1: Build React frontend
# Stage 2: Build ETL artifacts
# Stage 3: Lightweight runtime

FROM node:24-slim AS frontend_builder

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
ENV VITE_BASE_PATH=/ui/
RUN npm run build

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary for lightning-fast dependency installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# Copy source code
COPY etl/ ./etl/
COPY backend/ ./backend/
COPY frontend/streamlit_app.py ./frontend/streamlit_app.py
COPY REVISION* ./

# Build and install the Rust core extension module
RUN uv pip install --system maturin && \
    maturin build --manifest-path backend/rust_core/Cargo.toml --release --interpreter python3.11 --out dist && \
    uv pip install --system --no-cache dist/*.whl

# Fail image builds early if synced Python source has a syntax error.
RUN python -m compileall backend etl frontend/streamlit_app.py

# Copy Pre-computed Models and Data (present in production builds; skipped in CI)
# Use COPY with a wildcard so the layer is a no-op when the directories are absent
# rather than failing with "no source files were specified".
# hadolint ignore=DL3059
RUN mkdir -p models data/processed data/evaluation
COPY models/ ./models/
COPY data/processed/ ./data/processed/
COPY data/evaluation/ ./data/evaluation/

# Create other directories
# hadolint ignore=DL3059
RUN mkdir -p data/raw logs

# Pre-download serving model files to make container startup instant
# hadolint ignore=DL3059
RUN python -c "from backend.models.model_loader import ensure_model_files, default_artifacts_for_serving_profile; from pathlib import Path; ensure_model_files(Path('backend/models'), default_artifacts_for_serving_profile())"

# -------------------------------------------
# Stage 2: Runtime image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Install runtime dependencies only
# hadolint ignore=DL3008
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /app /app
COPY --from=frontend_builder /frontend/dist /app/frontend/dist

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports (7860 is required for Hugging Face Spaces, 8000 for Render, 8501 for Streamlit)
EXPOSE 7860 8000 8501

# Set default port to 7860 for Hugging Face Spaces
# Render will override this environment variable at runtime
ENV PORT=7860
ENV NOVA_LOW_MEMORY=false
ENV NOVA_FORCE_VECTOR_ARTIFACTS=true
ENV NOVA_EAGER_MODEL_DOWNLOAD=true
ENV NOVA_REFRESH_PIPELINE_MANIFEST=true
ENV NOVA_HEALTH_LOAD_RECOMMENDER=false
ENV NOVA_BACKGROUND_RECOMMENDER_WARMUP=true
ENV NOVA_ASYNC_EVALUATION_CACHE=true
ENV NOVA_PRECOMPUTE_SEMANTIC_BENCHMARK=false
ENV NOVA_PRECOMPUTE_RECOMMENDATION_BENCHMARK=false
ENV NOVA_RECOMMENDER_CIRCUIT_FAILURE_THRESHOLD=3
ENV NOVA_RECOMMENDER_CIRCUIT_OPEN_SECONDS=60
ENV NOVA_RECOMMENDER_CACHE_ENABLED=true
ENV NOVA_RECOMMENDER_CACHE_READS=true
ENV NOVA_RECOMMENDER_CACHE_MAX_ENTRIES=512
ENV NOVA_RECOMMENDER_CACHE_TTL_SECONDS=300
ENV NOVA_RECOMMENDER_STALE_CACHE_TTL_SECONDS=21600
ENV NOVA_RECOMMENDER_DISTRIBUTED_CACHE_ENABLED=true
ENV NOVA_RECOMMENDER_DISTRIBUTED_CACHE_TIMEOUT_SECONDS=1.5
ENV NOVA_SLO_WINDOW_SECONDS=3600
ENV NOVA_SLO_MIN_REQUESTS=5
ENV NOVA_SLO_MIN_ROUTE_REQUESTS=20
ENV NOVA_SLO_LATENCY_P95_MS=2500
ENV NOVA_SLO_ERROR_RATE=0.03
ENV NOVA_SLO_MAX_EVENTS=5000
ENV NOVA_SLO_EXCLUDED_ROUTE_PREFIXES=/docs,/redoc,/openapi.json,/favicon.ico,/v1/artifacts,/v1/diagnostics,/v1/evaluation,/v1/platform/readiness
ENV NOVA_SLO_ROUTE_LATENCY_BUDGETS=/:1000,/health:1000,/v1/frontends/status:3000,/v1/platform/slo:1000,/v1/search:2500,/v1/recommendations/id/{movie_id}:25000
ENV NOVA_FRONTEND_PRIORITY=github_pages,react,streamlit
ENV NOVA_FRONTEND_STREAMLIT_URL=https://a-movie-recommendation-system.streamlit.app
ENV NOVA_FRONTEND_REACT_URL=/ui/
ENV NOVA_FRONTEND_PAGES_URL=https://movie-recommendation-system-6bm.pages.dev
ENV NOVA_FRONTEND_HEALTH_TIMEOUT_SECONDS=2.5
ENV NOVA_FRONTEND_HEALTH_CACHE_SECONDS=30

# Default command: run backend API.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]
ENV NOVA_APP_COMMIT=f4baf923f7fb1a50850f7eedab23421bcb6df487
ENV NOVA_APP_COMMIT=9a1db81cd9d34eff7b8e9c551e53d0ce51e472aa
ENV NOVA_APP_COMMIT=7f9da00330b42d38ced8e27e1b42718c7d9e8ea1
