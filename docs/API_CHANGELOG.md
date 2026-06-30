# API Changelog

All breaking and non-breaking changes to the APEX recommendation API are documented here.
Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [v2.0.0] — 2026-05-26

### Added
- `GET /health` now returns `serving_tier`, `hardware_profile`, and `tier_selection_reason` fields (Adaptive Serving Tiers)
- `POST /v1/admin/reload-ensemble-weights` — hot-reload ensemble blend weights without restart
- `GET /v1/platform/slo` — SLO report with per-route p95 latency and error rate
- `POST /v1/events` now triggers `OnlineLearner.enqueue` for `click` and `rating` events
- `POST /v1/events` now dispatches `ActiveInferenceEngine.self_heal` as a background task for ratings ≥ 4.0 or ≤ 2.0
- `GET /v1/recommendations/id/{movie_id}` now applies RL policy score shift when `models/rl_policy.pth` is present
- `GET /v1/recommendations/user/{user_id}` — personalized recommendations from user behavior profile
- `GET /v1/browse` — catalog browsing with genre/year filters
- `GET /v1/chat` — LLM-powered conversational movie discovery (requires `OPENROUTER_API_KEY`)
- Experiment metadata (`experiment`, `variant`) attached to recommendation `retrieval_signals` when A/B test is active

### Changed
- `GET /v1/recommendations/id/{movie_id}` — SASRec now uses real session sequences from the event store instead of zero-padded dummy input
- Ensemble weights loaded from `models/ensemble_weights.json` (Dirichlet-optimized) instead of hard-coded constants
- `/health` response shape extended — existing fields (`status`, `movie_count`, `app_version`, `app_commit`) unchanged

### Deprecated
- `NOVA_SERVING_PROFILE` env var — superseded by `NOVA_SERVING_TIER`. Remains functional; mapped to nearest tier.

---

## [v1.5.0] — 2026-03-10

### Added
- `GET /v1/recommendations/visually-similar/{movie_id}` — multimodal CLIP+SBERT fusion search
- `GET /v1/recommendations/knowledge-graph/{movie_id}` — multi-hop NetworkX semantic reasoning
- `GET /v1/search/ai` — dense vector semantic search with cross-encoder reranking
- `GET /v1/evaluation/semantic-benchmark` — live serving quality metrics (HR@10, NDCG@10, MRR)
- `GET /v1/platform/readiness` — component-level readiness probe
- `GET /v1/artifacts/health` — artifact contract validation
- `?explain=true` query parameter on recommendation endpoints — triggers LLM personalized explanation

### Changed
- `GET /v1/search` — now supports hybrid retrieval (dense + sparse) with MMR diversity reranking

---

## [v1.0.0] — 2025-11-01

### Added
- `GET /v1/recommendations/id/{movie_id}` — content-based FAISS recommendation
- `GET /v1/search` — TF-IDF sparse search
- `POST /v1/events` — behavior event ingestion
- `GET /health` — basic health check
- `GET /v1/auth/token` — JWT authentication
- `POST /v1/auth/register` — user registration
- Multi-tenant API key authentication via `X-Nova-API-Key` header

---

## Versioning Policy

- **Major version** (`v2` → `v3`): breaking changes to existing response schemas or endpoint removal. Announced 30 days in advance with a deprecation notice in `/health`.
- **Minor version** (`v1.5` → `v1.6`): new endpoints or non-breaking additions to existing responses.
- **Patch**: bug fixes, performance improvements, no schema changes.
- All endpoints remain under `/v1/` until a breaking change requires `/v2/`.
- Deprecated fields are kept for **2 major versions** before removal.
- Breaking changes are announced via the `X-Nova-Deprecation` response header on affected endpoints.
