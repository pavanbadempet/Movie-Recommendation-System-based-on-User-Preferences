# APEX — Recommendation API

> Netflix-quality recommendations for your platform. No ML team required.

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-1A2B3C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)

APEX is an API that gives any streaming or media platform personalized, explainable recommendations powered by a production-grade 6-model ensemble — the same architecture used by Netflix, YouTube, and Amazon, without requiring a dedicated ML team.

**HR@10 = 0.785 · NDCG@10 = 0.542 · Semantic HR@10 = 1.0**

```bash
# Get your first recommendation in one call
curl "https://your-apex-api.onrender.com/v1/recommendations/id/550?n=10&explain=true" \
  -H "X-Nova-API-Key: YOUR_KEY"
```

**[View Live Demo](https://your-apex-api.onrender.com/docs) · [Quick Start](docs/QUICKSTART.md) · [API Reference](docs/API_REFERENCE.md) · [Pricing](#-deployment-tiers)**

---

<details>
<summary>🧠 How it works — 4-Layer Intelligence Stack</summary>

APEX implements a **4-Layer Intelligence Stack** across 18 systematic phases:

---

## 🧠 Architecture Overview

APEX is structured into 18 systematic phases across 4 intelligence layers:

### Layer 1: Data Platform & Streaming
* **High-Concurrency ETL Pipeline**: Processes massive datasets using Parquet and memory-mapped NumPy arrays.
* **In-Memory Feature Store**: Redis-backed cache for real-time user state (clicks, ratings, session velocity) with sub-millisecond retrieval.
* **Streaming Telemetry**: Kafka-style event appending and aggregation for real-time behavioral updates.

### Layer 2: Machine Learning Engine
* **Two-Tower Neural Retrieval**: SBERT-based dual encoders map users and items into a shared 768-dimensional latent space.
* **Vector Search (FAISS)**: Sub-millisecond ANN (Approximate Nearest Neighbor) retrieval across millions of dense vectors.
* **Multi-Task Learning (MMoE)**: A Multi-gate Mixture-of-Experts (MMoE) ranker that simultaneously predicts click-through rate (CTR) and user rating, dynamically weighting the loss.
* **LightGBM Ranker**: High-speed gradient boosting tree used as a fast fallback ranker.

### Layer 3: Advanced Aesthetics & Multi-Modal Understanding
* **Visual Encoders (CLIP)**: Uses OpenAI's CLIP model to extract 512-dimensional aesthetic embeddings from movie posters.
* **Multi-Modal FAISS Fusion**: Mathematically fuses SBERT text vectors (60%) and CLIP visual vectors (40%) into a unified 1280-dimensional search space for aesthetic + thematic matching.
* **Latent Diffusion Similarities**: Recommends items based on visual generative latent structures.

### Layer 4: Cognitive Intelligence & Compliance
* **Reinforcement Learning (A2C)**: An Actor-Critic neural network optimizing for long-term retention (7-day return probability) rather than cheap clickbait, trained via Conservative Q-Learning (CQL).
* **Deep Content Understanding (NLP)**: Uses HuggingFace Zero-Shot classification (`nli-distilroberta-base`) to extract abstract human concepts (Moral Dilemmas, Moods) and NER for entities.
* **Semantic Knowledge Graphs**: NetworkX-powered multi-hop reasoning (`User -> Liked Theme -> New Movie`).
* **LLM Personalization**: OpenRouter integration (GPT-4o / Llama 3) to dynamically generate personalized 1-sentence explanations ("Because you loved X, you'll enjoy Y").
* **Differential Privacy**: Mathematical bounding (Laplace/Gaussian noise) on user embeddings to guarantee GDPR / EU AI Act compliance.
* **Counterfactual Evaluation**: Inverse Propensity Scoring (IPS) to mathematically simulate model deployments offline before exposing them to users.

</details>

---

## 🗺️ Architecture Diagram

```mermaid
flowchart TD
    subgraph Serving["Serving Path"]
        U[UserRequest] --> API[FastAPI]
        API --> TD[TierDetector\nbackend.serving]
        TD -->|GPU + ≥16GB RAM| T1["Tier1: GPU / Full Ensemble\nLightGCN · Quantum · SASRec\nKAN · Hyperbolic · Diffusion"]
        TD -->|No GPU + ≥8GB RAM| T2["Tier2: ONNX CPU\nQuantized Inference"]
        TD -->|< 8GB RAM| T3["Tier3: FAISS + TF-IDF Only\nLow-Memory Mode"]
        T1 --> RP[RetrievalPipeline\nbackend.pipeline]
        T2 --> RP
        T3 --> RP
        RP --> RK[RankingPipeline\nbackend.pipeline]
        RK --> RR[RerankingPipeline\nbackend.pipeline]
        RR --> Resp[Response]
    end

    subgraph Retrieval["Retrieval Sources"]
        FAISS[FAISS ANN Index] --> RP
        TFIDF[TF-IDF Sparse Index] --> RP
        KG[Knowledge Graph] --> RP
    end

    subgraph Ranking["Ranking Components — 6 Ensemble Models (DR-Optimized Weights)\nbackend.models"]
        RK --> LGC[LightGCN\nweight 0.005]
        RK --> QNN[Quantum-Fluid NeuralODE\nweight 0.010]
        RK --> SAS[SASRec\nweight 0.659]
        RK --> KAN2[KAN\nweight 0.298]
        RK --> HYP[Hyperbolic\nweight 0.004]
        RK --> DIF[Diffusion\nweight 0.024]
    end

    subgraph DataPipeline["Data Pipeline"]
        TMDB[TMDB API] --> ETL[ETL Jobs]
        Kaggle[Kaggle Dataset] --> ETL
        ETL --> Bronze[Delta Lake Bronze\nRaw Ingestion]
        Bronze --> Silver[Delta Lake Silver\nCleaned + Joined]
        Silver --> Gold[Delta Lake Gold\nFeature Vectors]
        Gold --> MT[Model Training\nPySpark + PyTorch]
        MT --> Artifacts[Serving Artifacts\nFAISS + ONNX + Weights]
    end

    subgraph Compliance["Compliance & Fairness\nbackend.privacy · backend.metrics"]
        DP[Differential Privacy\nLaplace/Gaussian ε-DP]
        IPS[IPS Debiasing\nDoubly Robust weights]
        FA[Fairness Auditor\nGini + KL divergence]
    end
```

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```ini
TMDB_API_KEY=your_tmdb_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
JWT_SECRET_KEY=generate_a_strong_random_secret
REDIS_URL=redis://localhost:6379/0  # Optional for Layer 1 features
```

### 3. Data Pipeline & Model Generation
Run the core pipeline to generate all artifacts, neural embeddings, and FAISS indices:
```bash
python scripts/rebuild_serving_artifacts.py
```

### 4. Launch the API
Start the high-concurrency FastAPI backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚀 Deployment Tiers

APEX auto-detects hardware at startup and selects the appropriate serving tier. The live demo runs in **Tier 3** (free Render plan) — the full ensemble requires a paid plan.

| Tier | Plan | Profile | Active Models | Latency |
|------|------|---------|---------------|---------|
| **Tier 1** | Paid (GPU) | `full` | 6-model ensemble + RL + Active Inference | 50–200 ms |
| **Tier 2** | Paid (CPU) | `full` | ONNX-accelerated ensemble | 200–800 ms |
| **Tier 3** | Free | `lite` | FAISS + TF-IDF only | 800–2000 ms |

### Live Demo (Current: Tier 3)

The Render deployment uses `plan: free` with `NOVA_SERVING_PROFILE=lite`, which activates Tier 3 (degraded mode). This is intentional for cost reasons — the architecture fully supports all three tiers.

### Upgrading to Tier 1 or Tier 2

To enable the full ensemble on a paid Render plan, update `render.yaml`:

```yaml
# Tier 2 (CPU ONNX — Standard plan)
envVars:
  - key: NOVA_SERVING_PROFILE
    value: full
  - key: NOVA_SERVING_TIER
    value: tier2

# Tier 1 (GPU — Pro plan with GPU instance)
envVars:
  - key: NOVA_SERVING_PROFILE
    value: full
  - key: NOVA_SERVING_TIER
    value: tier1
```

---

## 📡 Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/recommendations/id/{movie_id}` | `GET` | Core Deep Neural Recommendation (MMoE Ranked) |
| `/v1/recommendations/visually-similar/{movie_id}` | `GET` | Multi-Modal (Text + Vision) Fusion Search |
| `/v1/recommendations/knowledge-graph/{movie_id}` | `GET` | Multi-Hop Semantic Reasoning Search |
| `/v1/search/semantic` | `GET` | Vector-based semantic search (handles misspellings & abstract concepts) |

*Append `?explain=true` to any recommendation endpoint to trigger the OpenRouter LLM for personalized natural-language justifications.*

---

## 📊 Ensemble Evaluation & Model Cards

### Offline Evaluation Results

Evaluation protocol: leave-one-out, 200 users, 100 candidates per user.  
Ensemble weights determined by **Doubly Robust IPS grid search** (200 Dirichlet-sampled candidates) — corrects for popularity bias in the logging policy.

| Model | HR@10 | NDCG@10 | DR Weight | Paradigm |
|---|---|---|---|---|
| **Ensemble** | **0.785** | **0.542** | — | Weighted blend |
| SASRec | 0.761 | 0.520 | 0.659 | Sequential Transformer |
| KAN | 0.694 | 0.439 | 0.298 | Kolmogorov-Arnold Network |
| LightGCN | 0.672 | 0.411 | 0.005 | Graph Collaborative Filtering |
| Quantum-Fluid | 0.583 | 0.354 | 0.010 | Neural ODE + Complex Embeddings |
| Diffusion | 0.521 | 0.309 | 0.024 | Generative Latent Diffusion |
| Hyperbolic | 0.498 | 0.287 | 0.004 | Poincaré Ball Manifold |

**Ensemble lift over best individual model (SASRec): +4.3% NDCG@10**

Semantic benchmark (17 curated intent cases, `reports/semantic_benchmark_report.json`): **HR@10 = 1.0, bad-hit rate = 0.0**

### Cross-Architecture Design Rationale

Each model addresses a distinct failure mode of the others:

| Failure Mode | Model That Addresses It |
|---|---|
| Ignores interaction order | SASRec (causal Transformer) |
| Misses multi-hop graph patterns | LightGCN (graph propagation) |
| Linear scoring bottleneck | KAN (learnable edge functions) |
| Static preference assumption | Quantum-Fluid (continuous-time ODE) |
| Euclidean hierarchy distortion | Hyperbolic (Poincaré manifold) |
| Candidate ranking bottleneck | Diffusion (generative retrieval) |
| Popularity bias in training | IPS-weighted BPR + DR weight selection |

### Causal Debiasing

All models are trained with **Inverse Propensity Scoring (IPS)** following Schnabel et al. "Recommendations as Treatments" (ICML 2016):
- Propensity estimation from empirical impression frequency (Laplace-smoothed)
- IPS-weighted BPR loss: each sample weighted by 1/propensity (clipped at 10.0)
- Doubly Robust weight selection: combines direct reward imputation with IPS correction
- Script: `scripts/causal_debias_training.py`

### Ablation Study

Run leave-one-out ablation to measure each model's marginal contribution:
```bash
python scripts/ablation_study.py --sample-size 1000 --output reports/ablation_report.json
```

Full model cards (training data, architecture details, known limitations): [`docs/MODEL_CARDS.md`](docs/MODEL_CARDS.md)

---

## 🛡️ Enterprise Fairness & Compliance

APEX includes a rigorous `FairnessAuditor` (`scripts/fairness_audit.py`) that mathematically verifies:
1. **Popularity Bias**: Enforces a Gini Coefficient `< 0.70` to prevent the model from blindly surfacing blockbuster content and starving niche creators.
2. **Calibration (KL Divergence)**: Ensures the recommended item distributions perfectly mirror the user's organic taste distribution without forcing them into a filter bubble.
3. **Safety Filters**: The Reinforcement Learning architecture utilizes an absolute hard-boundary to guarantee the AI will never recommend content a user explicitly dislikes.

---

## 📊 Observability

APEX ships a complete production observability stack that starts automatically with `docker compose up`.

### Prometheus + Grafana

| Service | URL | Credentials |
|---|---|---|
| Prometheus | http://localhost:9090 | — |
| Grafana | http://localhost:3000 | admin / admin |

The **APEX Overview** dashboard is provisioned automatically and shows:
- Request rate, error rate, and p50/p95/p99 latency per endpoint
- SLO burn rate for recommendations (<25s p95) and search (<2.5s p95)
- Real-time error rate gauge with 1%/3% threshold coloring

### Alerting Rules (`prometheus.rules.yml`)

| Alert | Condition | Severity |
|---|---|---|
| `HighErrorRate` | 5xx rate > 3% for 2 min | critical |
| `RecommendationLatencyHigh` | rec p95 > 25s for 3 min | warning |
| `SearchLatencyHigh` | search p95 > 2.5s for 3 min | warning |
| `NoTraffic` | 0 requests for 10 min | warning |
| `RecommendationEndpointErrors` | rec 5xx > 0.01/s for 2 min | critical |

### In-Process SLO Tracking

Every request is tracked by `RequestSloTracker` in `backend/slo.py`. The `/v1/platform/slo` endpoint returns the current SLO window (error rate, p95 latency, request counts per route) without requiring Prometheus.

### Sentry Error Monitoring

Set `SENTRY_DSN` in `.env` to enable full error tracking with stack traces, performance profiling, and release tracking. Gracefully disabled when not configured.

---

## 🧪 Testing

APEX maintains a rigorous testing suite covering neural network bounds, safety constraints, mathematical normalization, and offline replay evaluation.

```bash
python -m pytest backend/tests/ -v
```

---

## 🧬 Mutation Testing

APEX uses [mutmut](https://mutmut.readthedocs.io/) to verify that property-based tests actually detect logic errors in the serving tier and ONNX engine modules. A weekly GitHub Actions workflow runs this automatically.

To run locally:

```bash
pip install mutmut
mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py
mutmut results
```

The weekly CI workflow (`.github/workflows/mutation-tests.yml`) runs every Monday at 10:00 UTC and prints the mutation score automatically.

---
*Built as a state-of-the-art reference architecture for large-scale applied AI engineering.*

---

## 📚 Documentation Index

| Document | Description |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | Local setup, code standards, PR process, adding new ML models |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Full deployment guide — Render, Cloudflare Pages, Docker Compose, env vars |
| [CHANGELOG.md](CHANGELOG.md) | Version history and breaking changes |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Complete API endpoint reference with request/response schemas |
| [docs/API_CHANGELOG.md](docs/API_CHANGELOG.md) | API version history and deprecation notices |
| [docs/openapi.json](docs/openapi.json) | Machine-readable OpenAPI 3.1 spec (52 endpoints, 19 schemas) |
| [docs/swagger-ui.html](docs/swagger-ui.html) | Static Swagger UI — browse the full API without running the server |
| [docs/MODEL_CARDS.md](docs/MODEL_CARDS.md) | Model cards for all 6 ensemble models (architecture, metrics, limitations) |
| [docs/ARCHITECTURE_DECISIONS.md](docs/ARCHITECTURE_DECISIONS.md) | Architecture Decision Records (ADRs) — why the system is built this way |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | High-level system architecture overview |
| [docs/ENTERPRISE_GUIDE.md](docs/ENTERPRISE_GUIDE.md) | Multi-tenancy, B2B SaaS features, and enterprise deployment |
| [docs/APEX_WHITEPAPER.md](docs/APEX_WHITEPAPER.md) | Technical whitepaper — full system design and research context |
| [frontend/README.md](frontend/README.md) | Frontend setup, test coverage, deployment |
