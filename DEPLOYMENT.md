# Deployment Guide

Complete guide for deploying APEX across all supported environments.

---

## Table of Contents

- [Environment Variables Reference](#environment-variables-reference)
- [Local Development (Docker Compose)](#local-development-docker-compose)
- [Production: Render (Backend)](#production-render-backend)
- [Production: Cloudflare Pages (Frontend)](#production-cloudflare-pages-frontend)
- [Production: GitHub Pages (Frontend)](#production-github-pages-frontend)
- [Backup Frontend: Streamlit Cloud](#backup-frontend-streamlit-cloud)
- [Serving Tier Selection](#serving-tier-selection)
- [Health Check & Verification](#health-check--verification)
- [Upgrading](#upgrading)

---

## Environment Variables Reference

| Variable | Required | Description | Example |
|---|---|---|---|
| `TMDB_API_KEY` | Yes | TMDB API key for movie metadata enrichment | `abc123...` |
| `JWT_SECRET_KEY` | Yes | Secret for JWT token signing — generate with `openssl rand -hex 32` | `a1b2c3...` |
| `OPENROUTER_API_KEY` | No | OpenRouter key for LLM explanation generation (GPT-4o / Llama 3) | `sk-or-...` |
| `DATABASE_URL` | No | PostgreSQL connection string. Defaults to SQLite if unset | `postgresql://user:pass@host/db` |
| `REDIS_URL` | No | Redis connection string for feature store. In-memory fallback if unset | `redis://localhost:6379/0` |
| `NOVA_SERVING_PROFILE` | No | `full` forces Tier1 behavior; `lite` forces Tier3. Auto-detected if unset | `lite` |
| `NOVA_SERVING_TIER` | No | Explicit tier override: `tier1`, `tier2`, or `tier3` | `tier2` |
| `NOVA_ADMIN_TOKEN` | No | Bearer token for admin endpoints (`/v1/admin/*`) | `secret-token` |
| `SENTRY_DSN` | No | Sentry DSN for error monitoring | `https://...@sentry.io/...` |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins. Defaults to known frontend URLs | `https://myapp.pages.dev` |
| `NOVA_HEALTH_LOAD_RECOMMENDER` | No | Set `false` to skip recommender load on `/health` (faster cold start) | `false` |
| `NOVA_BACKGROUND_RECOMMENDER_WARMUP` | No | Set `true` to warm recommender in background after startup | `true` |
| `KAFKA_BROKER_URL` | No | Kafka broker for streaming events | `kafka:9092` |

Copy `.env.example` to `.env` and fill in the required values before running locally.

---

## Local Development (Docker Compose)

The full stack — backend, frontend, Kafka, Spark, Redis, PostgreSQL, Prometheus, Grafana — runs with a single command.

```bash
# Start all services
docker compose up --build

# Start only the backend + Redis (faster for API development)
docker compose up nova-backend redis --build

# Stop everything
docker compose down

# Stop and remove volumes (full reset)
docker compose down -v
```

### Service URLs

| Service | URL | Notes |
|---|---|---|
| Backend API | http://localhost:8000 | FastAPI |
| API Docs (Swagger) | http://localhost:8000/docs | Auto-generated |
| React Frontend | http://localhost:5173 | Vite dev server |
| Prometheus | http://localhost:9090 | Metrics scraping |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |
| Spark Master UI | http://localhost:8080 | PySpark cluster |
| PostgreSQL | localhost:5432 | nova_user/nova_password |
| Redis | localhost:6379 | Feature store |

### First-time data setup

```bash
# Build FAISS index and serving artifacts from scratch
docker compose run --rm nova-backend python scripts/rebuild_serving_artifacts.py

# Or run the full medallion ETL pipeline
docker compose run --rm nova-backend python scripts/etl_pipeline.py
```

---

## Production: Render (Backend)

### Deploy via Blueprint (recommended)

1. Push this repository to GitHub.
2. Go to [Render Dashboard](https://dashboard.render.com) → **New** → **Blueprint**.
3. Connect your repository. Render reads `render.yaml` automatically.
4. The blueprint deploys the backend on the free plan with `NOVA_SERVING_PROFILE=lite` (Tier 3).
5. Add secret environment variables in the Render dashboard (never commit these):
   - `JWT_SECRET_KEY`
   - `NOVA_ADMIN_TOKEN`
   - `DATABASE_URL` (optional — SQLite used if unset)
   - `REDIS_URL` (optional)
   - `TMDB_API_KEY`
   - `OPENROUTER_API_KEY` (optional)

### Upgrade to Tier 2 (CPU ONNX — Standard plan)

Edit `render.yaml`:
```yaml
plan: standard
envVars:
  - key: NOVA_SERVING_PROFILE
    value: full
  - key: NOVA_SERVING_TIER
    value: tier2
```

### Upgrade to Tier 1 (GPU — Pro plan)

```yaml
plan: pro
envVars:
  - key: NOVA_SERVING_PROFILE
    value: full
  - key: NOVA_SERVING_TIER
    value: tier1
```

### Manual deploy (without Blueprint)

1. **New** → **Web Service** → Connect repository.
2. Environment: **Docker**.
3. Docker context: `.` (root).
4. Dockerfile path: `./Dockerfile`.
5. Health check path: `/health`.
6. Add environment variables as above.

---

## Production: Cloudflare Pages (Frontend)

Recommended for the primary frontend — global CDN, zero cost, automatic HTTPS.

1. Go to [Cloudflare Pages](https://pages.cloudflare.com) → **Create a project** → **Connect to Git**.
2. Select your repository.
3. Build settings:
   - **Root directory:** `frontend`
   - **Build command:** `npm ci && npm run build`
   - **Build output directory:** `dist`
   - **Node version:** `22` (Cloudflare Pages max supported version)
4. Environment variables (optional):
   - `VITE_API_URL` — primary backend URL (e.g., `https://your-api.onrender.com`)
   - `VITE_BACKUP_API_URL` — backup backend URL
5. Deploy.

The React frontend has built-in request-level API failover — if the primary backend is sleeping (free tier cold start), it automatically retries the backup URL.

---

## Production: GitHub Pages (Frontend)

Zero-cost static hosting via GitHub Actions.

1. In your repository settings, go to **Pages** → set source to **GitHub Actions**.
2. The workflow `.github/workflows/frontend-pages.yml` runs automatically on pushes to `main` that touch `frontend/**`.
3. To trigger manually: **Actions** → **Frontend GitHub Pages** → **Run workflow**.

---

## Backup Frontend: Streamlit Cloud

The Streamlit app (`frontend/streamlit_app.py`) provides a lightweight fallback UI.

1. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
2. Repository: your repo. Branch: `main`. Main file: `frontend/streamlit_app.py`.
3. In **Advanced settings** → **Secrets**, add:
   ```toml
   API_URL = "https://your-api.onrender.com"
   TMDB_API_KEY = "your_tmdb_key"
   ```
4. Deploy.

---

## Serving Tier Selection

APEX auto-detects hardware at startup and selects the appropriate tier. You can override this with environment variables.

| Tier | Hardware Condition | Active Models | Typical Latency |
|---|---|---|---|
| **Tier 1** | GPU present + RAM ≥ 16 GB | Full 6-model ensemble + RL + Active Inference | 50–200 ms |
| **Tier 2** | No GPU + RAM ≥ 8 GB | ONNX-quantized ensemble | 200–800 ms |
| **Tier 3** | RAM < 8 GB | FAISS + TF-IDF only | 800–2000 ms |

Check the active tier at runtime:
```bash
curl https://your-api.onrender.com/health
# Response includes: "serving_tier": "tier3", "tier_selection_reason": "legacy_profile_mapping"
```

---

## Health Check & Verification

After deploying, verify the service is healthy:

```bash
# Basic health check
curl https://your-api.onrender.com/health

# Platform readiness (detailed component status)
curl https://your-api.onrender.com/v1/platform/ready

# Semantic benchmark (17 curated intent cases)
curl https://your-api.onrender.com/v1/evaluation/semantic-benchmark

# Test a recommendation
curl "https://your-api.onrender.com/v1/recommendations/id/155"
# Should return recommendations for The Dark Knight
```

Expected `/health` response:
```json
{
  "status": "ok",
  "movie_count": 10000,
  "serving_tier": "tier3",
  "app_version": "2.0.0"
}
```

Expected `/v1/platform/ready` response includes component statuses for: `catalog`, `artifact_health`, `vector_serving`, `search_smoke`, `recommendation_smoke`.

---

## Upgrading

### Updating model weights

```bash
# Rebuild all serving artifacts (FAISS index, embeddings, ONNX models)
python scripts/rebuild_serving_artifacts.py

# Or just retrain the ensemble with IPS debiasing
python scripts/causal_debias_training.py

# Validate artifacts before deploying
python scripts/validate_serving_artifacts.py
```

### Updating the catalog (new movies)

```bash
# Download latest TMDB data and rebuild
python scripts/download_real_datasets.py
python scripts/rebuild_serving_artifacts.py
```

### Rolling back

Render keeps the last 5 deploys. Go to **Deploys** in the Render dashboard and click **Rollback** on any previous deploy.
