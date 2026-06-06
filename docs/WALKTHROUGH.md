# APEX System Walkthrough

A guided tour of the full APEX recommendation pipeline — from raw data to a served recommendation — with concrete examples at each stage.

---

## Overview

A single recommendation request travels through four stages:

```
User Request
    │
    ▼
[1] Retrieval Pipeline      ← FAISS ANN + TF-IDF + Knowledge Graph
    │  ~500 candidates
    ▼
[2] Ranking Pipeline        ← 6-model ensemble (SASRec, KAN, LightGCN, ...)
    │  scored + sorted
    ▼
[3] Reranking Pipeline      ← MMR diversity + RL safety + LLM explanation
    │  final top-N
    ▼
Response
```

---

## Stage 1: Data Pipeline

Before any recommendation can be served, the data pipeline builds the serving artifacts.

### 1.1 Data Sources

| Source | Content | Format |
|---|---|---|
| TMDB API | Movie metadata (title, genres, overview, cast, director, poster) | JSON → Parquet |
| Kaggle MovieLens | User ratings (userId, movieId, rating, timestamp) | CSV → Parquet |
| TMDB Poster Images | Movie poster images for CLIP visual encoding | JPEG |

### 1.2 Medallion ETL (PySpark)

Raw data flows through three Delta Lake layers:

```
Bronze (raw ingestion)
    ↓  clean, deduplicate, join
Silver (cleaned + joined)
    ↓  feature engineering, embeddings
Gold (feature vectors)
    ↓  model training
Serving Artifacts (FAISS + ONNX + weights)
```

Run the full pipeline:
```bash
python scripts/rebuild_serving_artifacts.py
```

This produces:
- `models/faiss.index` — 768-dim SBERT embeddings, ~10k movies
- `models/two_tower_faiss.index` — 128-dim Two-Tower embeddings
- `models/sbert_embeddings.npy` — raw embedding matrix
- `models/tfidf_vectorizer.joblib` — TF-IDF vocabulary (12k features)
- `models/lightgcn.pth` / `models/lightgcn_ips.pth` — LightGCN weights (IPS-debiased)
- `models/sasrec.pth` — SASRec transformer weights
- `models/kan_ranker.pth` — KAN ranker weights
- `models/ensemble_weights.json` — DR-optimized blend weights

---

## Stage 2: Serving Tier Detection

At startup, `TierDetector` inspects the hardware and selects the appropriate serving mode:

```python
from backend.serving.serving_tier import resolve_serving_tier
tier, reason = resolve_serving_tier()
# e.g., ("tier3", "legacy_profile_mapping")
```

| Tier | Condition | What loads |
|---|---|---|
| Tier 1 | GPU + RAM ≥ 16 GB | Full 6-model PyTorch ensemble + `torch.compile` |
| Tier 2 | No GPU + RAM ≥ 8 GB | ONNX Runtime quantized models (2–5× faster CPU inference) |
| Tier 3 | RAM < 8 GB | FAISS + TF-IDF only (no neural ensemble) |

Check the active tier:
```bash
curl http://localhost:8000/health | python -m json.tool
# "serving_tier": "tier1"
```

---

## Stage 3: Retrieval Pipeline

Given a query movie (e.g., *The Dark Knight*, id=155), the retrieval pipeline fetches ~500 candidates from three sources in parallel:

### 3.1 FAISS ANN Search

The query movie's SBERT embedding is looked up and used to find the 200 nearest neighbors in the FAISS index:

```python
query_vector = sbert_embeddings[movie_idx]  # shape: (768,)
distances, indices = faiss_index.search(query_vector.reshape(1, -1), 200)
```

This returns movies with similar plot text and genre descriptions.

### 3.2 TF-IDF Sparse Search

The query movie's title + genres + overview are tokenized and scored against the TF-IDF vocabulary. This catches keyword matches that dense embeddings sometimes miss (e.g., exact franchise names).

### 3.3 Knowledge Graph Traversal

The NetworkX knowledge graph is traversed from the query movie:
```
The Dark Knight → [Crime, Action, Drama] → [The Godfather, Heat, Se7en, ...]
The Dark Knight → [Christopher Nolan] → [Inception, Interstellar, Tenet, ...]
```

Multi-hop reasoning surfaces thematically related movies that may not be textually similar.

### 3.4 Candidate Fusion

The three candidate sets are merged and deduplicated. Each candidate carries metadata about which retrieval stage produced it (`faiss`, `tfidf`, `knowledge_graph`, `content_sparse_fallback`).

---

## Stage 4: Ranking Pipeline

The ~500 candidates are scored by the 6-model ensemble and blended using DR-optimized weights.

### 4.1 The 6 Models

Each model scores every candidate independently:

| Model | What it captures | Implementation |
|---|---|---|
| **SASRec** (weight: 0.659) | Sequential intent — what the user wants *next* based on their session | Causal Transformer, 50-item sequence window |
| **KAN** (weight: 0.298) | Non-linear feature interactions between user and item embeddings | Fourier basis functions on edges |
| **LightGCN** (weight: 0.005) | Multi-hop collaborative filtering — "users who liked A also liked B" | 3-layer graph convolution, BPR loss |
| **Diffusion** (weight: 0.024) | Generative diversity — denoises toward the ideal item embedding | DDPM conditioned on user embedding |
| **Quantum-Fluid** (weight: 0.010) | Temporal preference drift — how taste evolves over time | Neural ODE with complex-valued embeddings |
| **Hyperbolic** (weight: 0.004) | Hierarchical genre structure — franchise and sub-genre relationships | Poincaré ball manifold |

### 4.2 Ensemble Blending

```python
final_score = (
    lightgcn_score * 0.005 +
    sasrec_score   * 0.659 +
    kan_score      * 0.298 +
    diffusion_score * 0.024 +
    quantum_score  * 0.010 +
    hyperbolic_score * 0.004
)
```

Weights are loaded from `models/ensemble_weights.json` and can be hot-reloaded without restart:
```bash
curl -X POST http://localhost:8000/v1/admin/reload-ensemble-weights \
  -H "Authorization: Bearer $NOVA_ADMIN_TOKEN"
```

### 4.3 Learned Ranker (optional)

If `models/nova_ranker.joblib` is present, a LightGBM gradient boosting ranker re-scores the top-50 candidates using additional features (vote_count, popularity, metadata completeness).

---

## Stage 5: Reranking Pipeline

The ranked list goes through three final passes:

### 5.1 MMR Diversity Reranking

Maximal Marginal Relevance (MMR) balances relevance and diversity:
```
score = λ × relevance - (1-λ) × max_similarity_to_already_selected
```

This prevents the top-10 from being dominated by sequels of the same franchise.

### 5.2 RL Safety Filter

The `RLSafetyFilter` removes any candidates the user has explicitly disliked (rating < 2.5):
```python
safe_candidates = [c for c in candidates if c["id"] not in user_dislikes]
```

This is a hard constraint — the RL policy cannot override it.

### 5.3 LLM Explanation (optional)

When `?explain=true` is appended to the request, OpenRouter (GPT-4o or Llama 3) generates a personalized one-sentence explanation for each recommendation:

> "Because you loved *The Dark Knight*'s psychological tension and moral complexity, you'll enjoy *Se7en*'s equally dark exploration of human nature."

---

## Stage 6: A Complete Example

Request:
```bash
curl "http://localhost:8000/v1/recommendations/id/155?n=5&explain=true"
```

Response (abbreviated):
```json
{
  "query_movie": { "id": 155, "title": "The Dark Knight" },
  "recommendations": [
    {
      "id": 49026,
      "title": "The Dark Knight Rises",
      "similarity_score": 0.967,
      "retrieval_stage": "content_sparse_fallback",
      "explanation": [
        "Shared genres: Action, Crime",
        "Same director (Christopher Nolan)",
        "Semantic twin concepts: batman, crime, gotham"
      ],
      "explanation_text": "Because you loved The Dark Knight's..."
    },
    {
      "id": 272,
      "title": "Batman Begins",
      "similarity_score": 0.621,
      "retrieval_stage": "content_sparse_fallback"
    }
  ]
}
```

The `retrieval_stage` field tells you which pipeline stage produced each candidate — useful for debugging recommendation quality.

---

## Stage 7: Online Learning

After a recommendation is served, user interactions feed back into the system:

1. **Click / Rating event** → `POST /v1/events` → `OnlineLearner.enqueue()`
2. **OnlineLearner** runs a mini-batch BPR update on LightGCN every 30 seconds
3. **Active Inference Engine** triggers a self-heal gradient step for extreme ratings (≥ 4.0 or ≤ 2.0)
4. **Real-time feature updater** updates the in-memory session sequence index for SASRec

This closes the feedback loop: the system improves continuously from live user behavior without requiring a full retraining cycle.

---

## Observability

Every request is tracked:

- **Prometheus metrics** at `/metrics` — request count, latency histograms per endpoint
- **SLO tracker** — per-route p50/p95 latency and error rate, available at `/v1/platform/slo`
- **Sentry** — exception tracking (configure `SENTRY_DSN` env var)
- **Event store** — all recommendation requests and impressions logged to `data/events/`

```bash
# Check SLO report
curl http://localhost:8000/v1/platform/slo | python -m json.tool

# Check Prometheus metrics
curl http://localhost:8000/metrics | grep nova_http
```

---

## Further Reading

- [docs/ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) — why each architectural choice was made
- [docs/MODEL_CARDS.md](MODEL_CARDS.md) — detailed model cards for all 6 ensemble models
- [docs/API_REFERENCE.md](API_REFERENCE.md) — complete API endpoint reference
- [CONTRIBUTING.md](../CONTRIBUTING.md) — how to add a new model or feature
