# Architecture

This document is the definitive technical reference for the APEX Movie Recommendation System.
It describes the actual running system — not a prototype or aspirational design.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Technology Stack](#2-technology-stack)
3. [Adaptive Serving Tiers](#3-adaptive-serving-tiers)
4. [10-Stage Recommendation Pipeline](#4-10-stage-recommendation-pipeline)
5. [6-Model Ensemble Engine](#5-6-model-ensemble-engine)
6. [Data Pipeline — Medallion Architecture](#6-data-pipeline--medallion-architecture)
7. [Real-Time Learning](#7-real-time-learning)
8. [Infrastructure Services](#8-infrastructure-services)
9. [Frontend](#9-frontend)
10. [Deployment](#10-deployment)
11. [CI/CD & Quality Gates](#11-cicd--quality-gates)
12. [Observability](#12-observability)

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENTS                                          │
│   React 19/Vite (Cloudflare Pages)  │  Streamlit (backup)              │
└────────────────────────┬────────────────────────────────────────────────┘
                         │  HTTPS / JWT
┌────────────────────────▼────────────────────────────────────────────────┐
│                   GATEWAY LAYER                                         │
│   Render.com (gateway)  ←→  HuggingFace Spaces (primary API)           │
│   SlowAPI rate limiter (30 req/min)  +  Redis token bucket (B2B SaaS)  │
└────────────────────────┬────────────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────────────┐
│                   FASTAPI BACKEND  (Python 3.11)                        │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │              10-STAGE RECOMMENDATION PIPELINE                    │   │
│  │  FAISS ANN → TF-IDF → MultiModal → KG → Ensemble → RL → LGB →  │   │
│  │  MMR → SafetyFilter → LLM Explanation                           │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────┐   ┌──────────────────────────────────────┐    │
│  │  ApexEnsembleEngine │   │  OnlineLearner (BPR background thread)│    │
│  │  6 models in        │   │  ActiveInferenceEngine (free-energy)  │    │
│  │  ThreadPoolExecutor │   └──────────────────────────────────────┘    │
│  └─────────────────────┘                                                │
└──────┬──────────┬──────────┬──────────┬──────────────────────────────┘
       │          │          │          │
  ┌────▼───┐ ┌───▼────┐ ┌───▼────┐ ┌───▼──────────────────────────────┐
  │ Redis  │ │Postgres│ │ Kafka  │ │  PySpark + Delta Lake (Medallion) │
  │feature │ │ JWT /  │ │ event  │ │  Bronze → Silver → Gold           │
  │ store  │ │ multi- │ │ stream │ │  ALS rank=16, RMSE=0.8754         │
  │ TTL 60s│ │ tenant │ │        │ │  SBERT 768-dim, FAISS 251MB       │
  └────────┘ └────────┘ └────────┘ └──────────────────────────────────┘
```

---

## 2. Technology Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI (Python 3.11), Uvicorn, orjson |
| ML / inference | PyTorch, ONNX Runtime, scikit-learn, LightGBM |
| Vector search | FAISS IVF (251 MB index, 768-dim SBERT + 512-dim CLIP) |
| Embeddings | `all-mpnet-base-v2` (SBERT, 768-dim), CLIP ViT (512-dim) |
| Graph reasoning | NetworkX (knowledge graph multi-hop) |
| Data processing | PySpark 3.4, Delta Lake, Apache Airflow |
| Message broker | Apache Kafka + Zookeeper (Confluent 7.3) |
| Feature store / cache | Redis 7 (TTL 60 s, token bucket rate limiting) |
| Relational DB | PostgreSQL 15 (multi-tenancy, JWT auth, user events) |
| Frontend | React 19, TypeScript, Vite 7 |
| Observability | Prometheus, Grafana, Sentry |
| Containerisation | Docker (multi-stage), docker-compose (9 services) |
| CI/CD | GitHub Actions (10 workflows) |

---

## 3. Adaptive Serving Tiers

At startup, `backend/serving_tier.py` auto-detects hardware via `HardwareProfile`
(GPU availability via `torch.cuda.is_available()`, RAM via `psutil`, CPU cores via
`os.cpu_count()`) and resolves one of three tiers. The tier can be overridden with
the `NOVA_SERVING_TIER` environment variable.

```
startup
   │
   ├─ NOVA_SERVING_TIER set? ──yes──► use that tier (tier1 / tier2 / tier3)
   │
   └─ auto-detect hardware
         │
         ├─ RAM < 8 GB ──────────────────────────────► Tier 3 (Starter)
         ├─ GPU detected AND RAM >= 16 GB ───────────► Tier 1 (Enterprise)
         └─ otherwise ───────────────────────────────► Tier 2 (Professional)
```

### Tier 1 — Enterprise

Condition: GPU detected + RAM ≥ 16 GB (or `NOVA_SERVING_TIER=tier1`)

- Full 6-model `ApexEnsembleEngine` loaded on CUDA
- `torch.compile` applied to all six sub-models
- `OnlineLearner` background thread started (BPR gradient updates to LightGCN)
- Redis feature store active
- Dynamic INT8 quantization skipped (GPU path)

### Tier 2 — Professional

Condition: CPU-only machine with RAM ≥ 8 GB (or `NOVA_SERVING_TIER=tier2`)

- ONNX Runtime inference path (`backend/onnx_engine.py`)
- All 6 models exported to ONNX; Python GIL bypassed for 2–5× CPU speedup
- Dynamic INT8 quantization applied to KAN and Diffusion linear layers
- PostgreSQL for persistence; no OnlineLearner
- Falls back to Tier 3 behaviour if no ONNX models are found

### Tier 3 — Starter

Condition: RAM < 8 GB (or `NOVA_SERVING_TIER=tier3`)

- FAISS + TF-IDF only; neural models not loaded
- TF-IDF vocabulary capped at 12,000 features
- SQLite for lightweight persistence
- Sparse retrieval index built lazily on first AI search request
- FAISS/SBERT artifacts optional (skipped if `NOVA_LOW_MEMORY=1`)


---

## 4. 10-Stage Recommendation Pipeline

Every recommendation request passes through up to ten sequential stages.
Later stages operate on the candidate set produced by earlier ones.

```
User request (movie_id or query text)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1 — FAISS ANN Retrieval                              │
│  SBERT all-mpnet-base-v2, 768-dim embeddings                │
│  IVF index (251 MB), top-100 nearest neighbours             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 2 — TF-IDF Sparse Retrieval (hybrid)                 │
│  Up to 50,000 features (12,000 in Tier 3), bigrams          │
│  Merges with FAISS candidates for cold-start resilience     │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 3 — Multi-Modal FAISS                                │
│  CLIP ViT 512-dim (poster/visual) fused with SBERT 768-dim  │
│  MultiModalFusionIndex loaded from backend/multimodal_fusion │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 4 — Knowledge Graph Multi-Hop Reasoning              │
│  NetworkX graph (backend/knowledge_graph.py)                │
│  Cross-domain enrichment (backend/cross_domain_kg.py)       │
│  Traverses director / genre / franchise / actor edges       │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 5 — 6-Model Ensemble Reranking                       │
│  ApexEnsembleEngine (see Section 5)                         │
│  All 6 models run in parallel via ThreadPoolExecutor        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 6 — RL Score Shift                                   │
│  A2C ActorCriticPolicy (backend/rl_policy.py)               │
│  State vector: 20-dim (log-scaled ratings, clicks, views,   │
│  ALS 16-dim user embedding)                                 │
│  Loaded from models/rl_policy.pth                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 7 — LightGBM Ranker                                  │
│  backend/ranker.py, trained offline on interaction signals  │
│  Loaded from models/nova_ranker.joblib                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 8 — MMR Diversity Reranking                          │
│  backend/diversity_reranker.py                              │
│  λ = 0.7 relevance / 0.3 max-similarity-to-selected        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 9 — RLSafetyFilter (hard constraints)                │
│  Never recommends items the user has explicitly disliked    │
│  Enforced before any result leaves the pipeline             │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Stage 10 — LLM Explanation Generation                      │
│  backend/llm_explanations.py via OpenRouter                 │
│  Models: GPT-4o (primary), Llama 3 (fallback)              │
│  Per-recommendation natural-language explanation            │
└─────────────────────────────────────────────────────────────┘
```

Stages 3–10 are skipped or degraded gracefully in Tier 3 (low-memory) mode.
Each stage annotates the result with `retrieval_stage` and `retrieval_signals`
fields so the frontend can show provenance.


---

## 5. 6-Model Ensemble Engine

`backend/ensemble_engine.py` — class `ApexEnsembleEngine(nn.Module)`

### Model Registry

| # | Model | Paradigm | Default Weight | File |
|---|-------|----------|---------------|------|
| 1 | LightGCN | Graph Neural Network (collaborative filtering) | **0.65** | `models/lightgcn.pth` |
| 2 | Quantum Fluid | Neural ODE + wave interference (continuous-time) | **0.25** | `models/quantum_fluid.pth` |
| 3 | SASRec | Transformer sequential (self-attention) | **0.10** | `models/sasrec.pth` |
| 4 | KAN | Kolmogorov-Arnold B-spline activations | 0.00 | `models/kan_ranker.pth` |
| 5 | Hyperbolic | Poincaré manifold (hierarchical geometry) | 0.00 | `models/hyperbolic.pth` |
| 6 | Diffusion | Latent denoising (score-based generative) | 0.00 | *(in-memory)* |

Weights are loaded from `models/ensemble_weights.json` at startup. If the file is
missing or malformed, the hard-coded defaults above are used. Weights are
re-normalised automatically if they do not sum to 1.0. Hot-reload is available
via `engine.reload_weights()` without restarting the process.

Weights can be tuned offline with a Dirichlet grid-search:
`scripts/optimize_ensemble_weights.py`

### Inference Flow

```
predict_ensemble(user_id, candidate_item_ids)
        │
        ├─ ONNX available? ──yes──► _predict_ensemble_onnx()
        │                           (bypasses Python GIL, 2-5× speedup)
        │
        └─ no ──────────────────► _predict_ensemble_pytorch()
                                   │
                                   ├─ Pre-compute shared embeddings (cached)
                                   ├─ Build session sequence for SASRec
                                   │   (real-time index → event store → zero fallback)
                                   ├─ Attention-weighted user embedding blend
                                   │   (0.7 × base + 0.3 × attended)
                                   │
                                   ├─ Submit all 6 scoring functions to
                                   │   module-level ThreadPoolExecutor
                                   │   (avoids per-request thread creation overhead)
                                   │
                                   └─ Weighted sum → final score per candidate
```

### PySpark Prior Injection

On initialisation, `_inject_pyspark_priors()` loads real ALS embeddings from
`data/datalake/gold/model_user_embeddings` and `model_item_embeddings` (Parquet)
and injects them into LightGCN, Quantum Fluid, Hyperbolic, and SASRec embedding
tables. This anchors all neural models in the collaborative-filtering signal
learned from the full MovieLens dataset before any fine-tuning.

### Session Cache

An LRU `OrderedDict` (cap: 10,000 entries, TTL: 60 s) caches per-user session
sequences. On miss, the real-time in-memory index (`backend/realtime_feature_updater`)
is tried first (~1 ms), then the background event-store index (rebuilt every 5 min),
then a zero-padded cold-start fallback.

### CPU Optimisations (Tier 2)

Dynamic INT8 quantization is applied to KAN and Diffusion linear layers on CPU,
reducing model size ~4× and improving inference throughput 2–3× with ~1% accuracy
loss. Quantization is skipped on CUDA.


---

## 6. Data Pipeline — Medallion Architecture

```
Raw Source (MovieLens 100K)
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  BRONZE LAYER  (raw ingest)                                   │
│  100,836 ratings · 610 users · 9,724 rated movies            │
│  Stored as Delta Lake tables (append-only, immutable)         │
└───────────────────────┬───────────────────────────────────────┘
                        │  PySpark transformations
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  SILVER LAYER  (cleaned, validated)                           │
│  Deduplication, schema enforcement, SCD tracking             │
│  Slowly Changing Dimensions for historical user preferences   │
└───────────────────────┬───────────────────────────────────────┘
                        │  PySpark ALS + SBERT encoding
                        ▼
┌───────────────────────────────────────────────────────────────┐
│  GOLD LAYER  (serving-ready features)                         │
│  PySpark ALS: rank=16, RMSE=0.8754                           │
│  User embeddings → data/datalake/gold/model_user_embeddings   │
│  Item embeddings → data/datalake/gold/model_item_embeddings   │
│  SBERT all-mpnet-base-v2 (768-dim) → models/sbert_embeddings.npy │
│  FAISS IVF index → models/faiss.index (251 MB)               │
│  Movie IDs map → models/movie_ids.npy                        │
│  Pipeline manifest → models/pipeline_manifest.json           │
└───────────────────────────────────────────────────────────────┘
```

### Orchestration

- **Apache Airflow** DAGs schedule the Bronze → Silver → Gold pipeline
- **Apache Kafka** streams real-time user events into the Bronze layer
- **Daily refresh** triggered by GitHub Actions (`data-refresh.yml`) using a
  Kaggle GPU runner; artifacts are pushed to HuggingFace Hub via `sync-hf.yml`

### Artifact Integrity

At startup, `Recommender._validate_vector_artifacts()` enforces strict row-alignment
contracts between the FAISS index, SBERT embedding matrix, `movie_ids.npy`, and the
serving catalog. A SHA-256 hash of the movie-ID vector is compared against the
pipeline manifest. Any mismatch disables vector serving and falls back to TF-IDF.

### SBERT Embeddings

Loaded with `np.load(..., mmap_mode='r')` — memory-mapped from disk so the full
768-dim × N matrix never occupies RAM on constrained deployments.


---

## 7. Real-Time Learning

### OnlineLearner (`backend/online_learner.py`)

A daemon thread that incrementally updates LightGCN embeddings from live events
without a full retraining cycle. Only active in Tier 1.

```
FastAPI request path
        │  enqueue(event)
        ▼
  bounded queue (max 10,000 events)
        │  background daemon thread
        ▼
  accumulate batch (default: 32 events)
        │
        ▼
  _apply_gradient_step(batch)
  ┌─────────────────────────────────────────────────────────┐
  │  BPR-style loss per (user, pos_item, neg_item) triple   │
  │  rating ≥ 4.0  → weight +1.0  (strong positive)        │
  │  rating < 2.5  → weight -0.5  (negative, roles swapped)│
  │  click         → weight +0.3  (weak positive)           │
  │  Adam optimiser (persistent, momentum accumulates)      │
  │  Gradient clip: max L2 norm = 1.0                       │
  └─────────────────────────────────────────────────────────┘
        │  every 1,000 events
        ▼
  checkpoint → models/lightgcn_online.pth
```

If the queue fills, the oldest event is dropped (FIFO eviction) and a WARNING
is logged. The thread is a daemon so it never blocks process shutdown.

### Active Inference Engine (`backend/active_inference_engine.py`)

Implements Karl Friston's Free Energy Principle for real-time self-healing on
negative user feedback (thumbs-down). Triggered as a FastAPI `BackgroundTask`
within ~50 ms of the feedback event.

```python
free_energy = ||movie_embedding - dynamic_prior||₂ × (-reward)
# reward = +1.0 (thumbs up) → low surprise → no update
# reward = -1.0 (thumbs down) → high surprise → SGD step on dynamic_prior
```

The `dynamic_prior` parameter (shape `[1, emb_dim]`) shifts to reduce future
surprise from similar items. Gradients are clipped to max norm 1.0.

### Real-Time Feature Updater (`backend/realtime_feature_updater.py`)

Maintains an in-memory user→session-sequence index pre-loaded from the event
store at startup (up to 10,000 most recent users). Provides sub-millisecond
session lookups for SASRec without hitting the JSONL event store on every
request. The background event-store index (rebuilt every 5 min) serves as a
fallback for users not in the real-time index.


---

## 8. Infrastructure Services

### Redis

- **Role:** Feature store, session cache, B2B SaaS rate-limiting token bucket
- **TTL:** 60 seconds for session sequences
- **Image:** `redis:7-alpine`
- **Persistence:** RDB snapshot every 60 s (at least 1 write)
- **Port:** 6379

### PostgreSQL

- **Role:** Multi-tenancy, JWT authentication, user event storage, SCD tables
- **Image:** `postgres:15-alpine`
- **Schema:** Managed via SQLAlchemy models (`backend/database.py`)
- **Init scripts:** `sql/` directory mounted at container start
- **Port:** 5432

### Apache Kafka + Zookeeper

- **Role:** Real-time event streaming from frontend interactions into the data pipeline
- **Image:** `confluentinc/cp-kafka:7.3.0` / `confluentinc/cp-zookeeper:7.3.0`
- **Port:** 9092 (internal), 29092 (host)

### PySpark Cluster

- **Role:** Medallion ETL, ALS training, feature engineering
- **Image:** `bitnami/spark:3.4.1`
- **Topology:** 1 master + 1 worker (4 GB RAM, 2 cores per worker in compose)
- **Ports:** 8080 (Spark UI), 7077 (master)
- **Data mount:** `./data` shared with backend for Delta Lake access

### Rate Limiting

Two layers of rate limiting protect the API:

| Layer | Mechanism | Limit |
|-------|-----------|-------|
| Public API | SlowAPI (`slowapi`) per-IP | 30 req/min |
| B2B SaaS | Redis token bucket (`backend/middleware/rate_limiter.py`) | Per-tenant SLA quota |


---

## 9. Frontend

**Stack:** React 19 + TypeScript + Vite 7

### Multi-Backend Failover

The frontend uses `Promise.any()` to race requests across multiple backend
endpoints simultaneously. The first successful response wins:

```
Promise.any([
  fetch(primary),    // HuggingFace Spaces
  fetch(backup),     // Render.com gateway
  fetch(localhost),  // local dev
])
```

This gives sub-second failover with no user-visible error when any single
backend is cold-starting or unavailable.

### Key Features

- **JWT authentication** — login/register flow, token stored in memory (not localStorage)
- **Semantic search** — free-text query routed to the TF-IDF + FAISS hybrid pipeline
- **Visual recommendations** — poster-first card layout with similarity scores and
  per-recommendation LLM explanations
- **Knowledge graph recommendations** — dedicated view showing multi-hop graph paths
  that explain why a movie was recommended
- **Retrieval signal transparency** — each result card shows `retrieval_stage` and
  `retrieval_signals` from the pipeline

### Build

```bash
cd frontend
npm ci
npm run build   # outputs to frontend/dist/
```

The built `dist/` is served by FastAPI as a static mount at `/ui/` when present.
It can also be deployed independently to GitHub Pages via
`.github/workflows/frontend-pages.yml`.

---

## 10. Deployment

### Docker

Multi-stage `backend/Dockerfile`:
1. **Stage 1 (Node 24):** builds the React frontend (`npm run build`)
2. **Stage 2 (Python 3.11-slim):** installs Python dependencies, copies built
   frontend, runs as a non-root user

### docker-compose Services (9 total)

| # | Service | Image / Build | Port(s) |
|---|---------|--------------|---------|
| 1 | `apex-backend` | `backend/Dockerfile` | 8000 |
| 2 | `apex-frontend` | `frontend/Dockerfile` | 5173 |
| 3 | `zookeeper` | `confluentinc/cp-zookeeper:7.3.0` | 2181 |
| 4 | `kafka` | `confluentinc/cp-kafka:7.3.0` | 9092, 29092 |
| 5 | `redis` | `redis:7-alpine` | 6379 |
| 6 | `spark-master` | `bitnami/spark:3.4.1` | 8080, 7077 |
| 7 | `spark-worker` | `bitnami/spark:3.4.1` | — |
| 8 | `postgres` | `postgres:15-alpine` | 5432 |
| 9 | `prometheus` | `prom/prometheus:latest` | 9090 |
| 10 | `grafana` | `grafana/grafana:latest` | 3000 |

*(Grafana is the 10th container but the compose file groups it with Prometheus
under the observability section.)*

All services share the `apex-net` bridge network.

### Live Environments

| Environment | URL | Purpose |
|-------------|-----|---------|
| Primary API | `https://pavanbadempet-movie-rec-api.hf.space` | HuggingFace Spaces |
| Gateway | `https://movie-recs-api-5qvy.onrender.com` | Render.com |
| Frontend | Cloudflare Pages | React SPA |
| Backup UI | Streamlit Cloud | `frontend/streamlit_app.py` |


---

## 11. CI/CD & Quality Gates

10 GitHub Actions workflows run on every push to `main`/`develop`.

### Workflow Summary

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | push / PR | Unit, API, PySpark, ML, frontend, Docker validation |
| `serving-quality.yml` | post-deploy / daily 08:30 UTC | Live quality gate against HF Spaces + Render |
| `data-refresh.yml` | daily | Kaggle GPU runner re-trains ALS, rebuilds FAISS, pushes artifacts |
| `sync-hf.yml` | post-data-refresh | Syncs artifacts to HuggingFace Hub |
| `frontend-pages.yml` | push to main | Tests, builds, and deploys React to GitHub Pages |
| `load-test.yml` | scheduled | k6 load tests against live endpoints |
| `synthetic-monitoring.yml` | scheduled | Lightweight uptime probes |
| `secrets-scan.yml` | push / PR | Detects accidentally committed secrets |
| `keep-alive.yml` | scheduled | Prevents HF Spaces / Render cold-start eviction |
| `backfill-serving-artifacts.yml` | manual | Backfills missing serving artifacts |

### CI Test Matrix (`ci.yml`)

| Job | Tests |
|-----|-------|
| Unit & Property-Based | `test_session_sequence`, `test_ensemble_weights`, `test_online_learner`, `test_rl_wiring`, `test_events`, `test_slo`, `test_frontend_failover`, ensemble math, RL policy, fairness, explanations, … |
| API Integration | `test_api`, `test_api_endpoints`, `test_security_api`, `test_artifact_health`, `test_catalogs`, `test_database` |
| Data Pipeline | `test_pyspark_scd`, `test_etl`, `test_delta_lakehouse`, `test_semantic_artifacts` |
| ML Models | `test_semantic_benchmark`, `test_recommendation_benchmark`, model fuzzing, replay, integration pipeline |
| Frontend | ESLint + `npm run build` (Node 24) |
| Docker | `docker compose config` + Hadolint Dockerfile lint |

### Live Quality Gates (`serving-quality.yml`)

The serving quality workflow evaluates the live HuggingFace Spaces deployment
after every data refresh and on a daily schedule. It fails the build if any
threshold is breached:

| Metric | Threshold |
|--------|-----------|
| Search top-1 hit rate | ≥ 0.98 |
| Search hit rate | = 1.00 |
| Recommendation hit rate | ≥ 0.95 |
| MRR | ≥ 0.35 |
| NDCG | ≥ 0.25 |
| Recommendation benchmark hit rate | ≥ 0.90 |
| Recommendation benchmark pass rate | ≥ 0.80 |
| Bad match rate | ≤ 0.05 |
| Explanation coverage | ≥ 0.90 |


---

## 12. Observability

### Prometheus Metrics

`backend/main.py` exposes a `/metrics` endpoint (Prometheus ASGI app) with two
custom metrics instrumented via middleware on every HTTP request:

| Metric | Type | Labels |
|--------|------|--------|
| `apex_http_requests_total` | Counter | `method`, `endpoint`, `http_status` |
| `apex_http_request_duration_seconds` | Histogram | `method`, `endpoint` |

These implement the **RED method** (Rate, Errors, Duration) for service-level
monitoring.

### Grafana

Grafana (`grafana/grafana:latest`, port 3000) is pre-configured to scrape
Prometheus and visualise RED dashboards and SLO burn-rate charts.

### SLO Tracker

`backend/slo.py` — `RequestSloTracker` records per-route latency and error
samples in-process. The `/v1/slo` endpoint returns a live SLO report without
requiring Prometheus to be running.

### Sentry

Error monitoring via `sentry_sdk` is initialised at startup when `SENTRY_DSN`
is set. Both `traces_sample_rate` and `profiles_sample_rate` are set to 1.0
for full distributed tracing in production.

### Experiment Tracking

`backend/experiments.py` provides A/B experiment assignment and metric
aggregation. Experiments are attached to recommendation requests via
`attach_experiment()` and summarised at `/v1/experiments/{id}/metrics`.

---

## Key File Map

```
backend/
├── main.py                    FastAPI app, lifespan, middleware, routes
├── serving_tier.py            Hardware detection, tier resolution
├── recommender.py             Recommender class, 10-stage pipeline
├── ensemble_engine.py         ApexEnsembleEngine, 6-model parallel inference
├── online_learner.py          BPR background gradient updates to LightGCN
├── active_inference_engine.py Free-energy self-healing on negative feedback
├── realtime_feature_updater.py In-memory session sequence index
├── onnx_engine.py             ONNX Runtime inference (Tier 2)
├── lightgcn.py                LightGCN graph neural network
├── sasrec.py                  SASRec transformer sequential model
├── neural_ode_recommender.py Neural ODE + wave interference
├── hyperbolic_recommender.py  Poincaré manifold recommender
├── kan_ranker.py              Kolmogorov-Arnold B-spline ranker
├── diffusion_recommender.py   Latent denoising diffusion model
├── knowledge_graph.py         NetworkX KG engine
├── multimodal_fusion.py       CLIP + SBERT fusion index
├── rl_policy.py               A2C ActorCriticPolicy (stage 6)
├── ranker.py                  LightGBM ranker (stage 7)
├── diversity_reranker.py      MMR diversity (stage 8)
├── llm_explanations.py        OpenRouter GPT-4o / Llama 3 (stage 10)
├── feature_store.py           Redis feature store client
├── database.py                SQLAlchemy models (PostgreSQL / SQLite)
├── auth.py                    JWT authentication, multi-tenancy
└── slo.py                     In-process SLO tracker

models/
├── faiss.index                FAISS IVF index (251 MB)
├── sbert_embeddings.npy       768-dim SBERT vectors (memory-mapped)
├── movie_ids.npy              FAISS row → movie_id map
├── ensemble_weights.json      Blend weights for 6-model ensemble
├── lightgcn.pth               Trained LightGCN weights
├── sasrec.pth                 Trained SASRec weights
├── quantum_fluid.pth          Trained Quantum Fluid weights
├── hyperbolic.pth             Trained Hyperbolic weights
├── kan_ranker.pth             Trained KAN weights
├── nova_ranker.joblib         LightGBM ranker
└── rl_policy.pth              A2C policy weights

data/datalake/
├── bronze/                    Raw MovieLens ingest (Delta Lake)
├── silver/                    Cleaned, SCD-tracked tables
└── gold/
    ├── model_user_embeddings/ ALS user embeddings (Parquet)
    └── model_item_embeddings/ ALS item embeddings (Parquet)
```
