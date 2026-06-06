# Changelog — APEX Recommendation Engine

All notable changes follow [Semantic Versioning](https://semver.org/) and
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions.

---

## [Unreleased]

### Added — 2026-06-10
- `backend/serving/serving_tier.py`: GPU VRAM check added to `TierDetector.detect()`
  via `torch.cuda.get_device_properties`. `_auto_select()` now requires `gpu_vram_gb >= 8.0`
  for Tier1; unknown VRAM (`gpu_vram_gb == 0.0`) falls back to Tier2 safely.
- `backend/serving/app_startup.py`: startup/shutdown orchestration extracted from `main.py`
  lifespan into dedicated module with `startup()` / `shutdown()` public API.
- `backend/models/ensemble_engine.py`: `predict_ensemble()` and `_predict_ensemble_pytorch()`
  accept `user_emb_override: torch.Tensor | None` — DP-noised embedding is used directly
  without mutating the shared LightGCN embedding table, fixing the thread-safety issue.
- `backend/privacy/privacy_preserving_ml.py`: `privatize_user_embedding` now accepts
  `delta` keyword argument, correctly threading it to `add_gaussian_noise`.
- CI: `type-check` job added to `ci.yml` running `mypy backend/` with `pyproject.toml`
  config, gating all downstream test jobs.
- Tests: `test_dp_thread_safety.py`, `test_serving_tier_vram.py`, `test_cold_start_properties.py`
  added to cover DP thread-safety, VRAM-aware tier selection, and cold-start invariants.

### Planned
- GraphQL API support
- Multi-language catalog support
- DDIM sampling for diffusion model (10-step vs. 100-step inference)
- Lazy initialization of zero-weight ensemble models on Tier3

---

## [2.1.0] — 2026-06-05

### Added — Online Learning Coordinator
- `backend/learning/online_learning_coordinator.py` — `OnlineLearningCoordinator` fans out
  every live click/rating event to all three online-capable models simultaneously
  via independent daemon threads. Single `enqueue(event)` call reaches LightGCN,
  SASRec, and KAN.
- `backend/learning/sasrec_online_learner.py` — `SASRecOnlineLearner`: incremental gradient
  updates to SASRec item embeddings and last attention block from live events.
  Uses live session cache for accurate sequence context. Checkpoint interval: 500 events.
- `backend/learning/kan_online_learner.py` — `KANOnlineLearner`: incremental updates to KAN
  Fourier sin/cos coefficients using LightGCN embeddings as detached input.
  Decoupled gradient flows prevent conflicting updates between learners.
- `backend/serving/__init__.py` extended with all three online learner exports.

### Changed — Ensemble Engine
- `_predict_ensemble_pytorch`: contextual weights (`neural_weight_optimizer`) now
  applied on both ONNX and PyTorch paths (previously ONNX-only).
- `_predict_ensemble_pytorch`: uncertainty-gated blending added — per-item weighted
  variance across 6 model scores computes a confidence gate. Items where models
  strongly disagree receive up to 50% score reduction.
- `_predict_ensemble_onnx`: contextual weights unified with PyTorch path.

### Added — IPS at Inference
- `RankingConfig` extended with `apply_ips_reranking` (default `True`),
  `item_popularity`, `ips_clip_val` fields.
- `RankingPipeline.rank()`: Step 3b applies Inverse Propensity Scoring re-weighting
  using item popularity lazy-loaded from the event store. Niche items with equal
  model scores receive a soft boost; blockbusters receive a soft penalty.
- `RankingPipeline._get_popularity()`: lazy-loads and caches item popularity from
  `debiased_metrics.compute_item_popularity`.

### Added — Differential Privacy at Inference
- `apply_learned_ranker` in `recommender_core.py` now calls
  `privatize_user_embedding` (Gaussian ε-DP) on the user's LightGCN embedding
  at every recommendation request. ε configurable via `APEX_DP_EPSILON` env var
  (default 1.0). Guarantees GDPR / EU AI Act compliance at serving time.

### Added — Long-Horizon RL at Inference
- `apply_learned_ranker` now calls `estimate_churn_risk` and
  `compute_preference_stability` per request (cached from user event log).
- `long_horizon_score_adjustment` applied per candidate — at-risk users receive
  quality-boosted recommendations; preference-shifting users receive genre-diverse
  candidates.
- Metrics exposed in `candidate["metrics"]` response field:
  `churn_risk`, `preference_stability`, `cold_start`.

### Added — Cold-Start Boost
- `cold_start_boost` from `uncertainty_estimator` applied in `apply_learned_ranker`
  for users with < 5 interactions. Boosts content-quality and popularity signals to
  compensate for sparse collaborative filtering signal.

### Fixed — Privacy Package
- `backend/privacy/__init__.py`: `DifferentialPrivacyEngine` and `anonymize_telemetry`
  were not exported from the `backend.privacy.privacy` package (pre-existing import shadow
  between `backend/privacy.py` and `backend/privacy/` directory). Fixed by inlining
  the class definition into the package `__init__.py`.

### Added — Code Organization
- `backend/intelligence/__init__.py` — new subpackage grouping all Layer 4 cognitive
  modules: knowledge graph, semantic twin, NLP, LLM, multimodal, long-horizon RL,
  temporal preference, bandit, exploration, vision encoder.
- `backend/data/__init__.py` — new subpackage grouping the Layer 1 data platform:
  event streaming, feature store, real-time session index, experiments, usage, cache, SLO.
- All existing subpackages (`models`, `pipeline`, `serving`, `metrics`, `privacy`)
  expanded to export all relevant symbols (96 total across 7 subpackages, 60/84 flat
  modules covered, 24 intentional entry points left uncovered).
- `docs/PACKAGE_STRUCTURE.md` rewritten with complete module map, import graph,
  and contribution guide.

---

## [2.0.0] — 2025-06-01

### Added — 6-Model Ensemble
- `backend/sasrec.py` — Self-Attentive Sequential Recommendation (causal Transformer,
  2 blocks, 2 heads, hidden_dim=64, seq_len=50).
- `backend/kan_ranker.py` — Kolmogorov-Arnold Network ranker using Fourier basis
  functions (3-layer, grid_size=5).
- `backend/hyperbolic_recommender.py` — Poincaré ball manifold recommender
  (curvature=1.0, Möbius addition, Fermi-Dirac loss).
- `backend/diffusion_recommender.py` — Latent Diffusion generative recommender
  (DDPM, 100 timesteps, linear β schedule 1e-4→0.02, conditioned on user embedding).
- `backend/neural_ode_recommender.py` — Quantum-Fluid Neural ODE with complex
  embeddings (Euler approximation, 4 steps, wave interference scoring).
- `backend/ensemble_engine.py` — `ApexEnsembleEngine` unifying all 6 models with
  DR-optimized blend weights, parallel thread pool execution, dynamic INT8
  quantization on CPU, `torch.compile` on GPU.

### Added — DR-Optimized Ensemble Weights (ADR-007)
- Doubly Robust IPS grid search over 200 Dirichlet-sampled weight candidates.
- DR-optimized weights: SASRec 0.659, KAN 0.298, Diffusion 0.024, Quantum 0.010,
  LightGCN 0.005, Hyperbolic 0.004.
- Weights stored in `models/ensemble_weights.json`, hot-reloadable without restart
  via `ApexEnsembleEngine.reload_weights()`.
- `backend/neural_weight_optimizer.py` — `ContextualWeightNetwork`: small neural
  network producing context-dependent ensemble weights based on user behavior profile.

### Added — Pipeline Decomposition (ADR-006)
- `backend/pipeline_types.py` — `CandidateItem`, `RankedItem`, `FinalItem` dataclasses.
- `backend/retrieval_pipeline.py` — `RetrievalPipeline`: FAISS + TF-IDF + KG retrieval.
- `backend/ranking_pipeline.py` — `RankingPipeline`: ensemble + learned ranker scoring.
- `backend/reranking_pipeline.py` — `RerankingPipeline`: RL safety + quality gate +
  MMR diversity + LLM reranking.

### Added — 3-Tier Serving (ADR-005)
- `backend/serving/serving_tier.py` — `TierDetector`: hardware auto-detection at startup,
  including GPU VRAM check (`torch.cuda.get_device_properties`) — Tier1 requires ≥ 8 GB VRAM.
- `backend/serving/onnx_engine.py` — ONNX Runtime quantized inference (Tier2, 2–5× speedup).
- `backend/serving/app_startup.py` — startup/shutdown orchestration extracted from `main.py`.
- `backend/active_inference_engine.py` — Active Inference self-healing via
  free-energy minimization.
- `backend/online_learner.py` — `OnlineLearner`: real-time LightGCN BPR mini-batch
  updates from live events.
- `backend/realtime_feature_updater.py` — In-memory user session sequence index.

### Added — Compliance & Fairness
- `backend/privacy/privacy_preserving_ml.py` — `privatize_user_embedding` (Laplace/Gaussian
  ε-DP, thread-safe per-request copy — shared embedding table never mutated),
  `k_anonymize_profile`, `federated_average_gradients`.
- `backend/metrics/debiased_metrics.py` — IPS-corrected NDCG, calibration score, beyond-
  accuracy metrics.
- `backend/intelligence/uncertainty_estimator.py` — ensemble disagreement, coverage uncertainty,
  cold-start boost.
- `scripts/fairness_audit.py` — `FairnessAuditor` enforcing Gini < 0.70 and
  KL divergence calibration.

### Added — Cognitive Intelligence Layer
- `backend/knowledge_graph.py` — `KnowledgeGraphEngine`: NetworkX multi-hop
  thematic reasoning.
- `backend/cross_domain_kg.py` — Cross-domain KG enrichment (books, music, games).
- `backend/semantic_twin.py` — Deterministic semantic item twin construction.
- `backend/content_understanding.py` — `ContentUnderstandingEngine`: HuggingFace
  Zero-Shot NLP + NER.
- `backend/multimodal_fusion.py` — CLIP visual + SBERT text fusion (60/40 blend).
- `backend/vision_encoder.py` — `VisionEncoder`: OpenAI CLIP poster embedding.
- `backend/llm_explanations.py` — LLM-generated 1-sentence personalized explanations
  (OpenRouter GPT-4o / Llama 3).
- `backend/long_horizon_rl.py` — 30/90-day churn risk + preference stability.
- `backend/temporal_preference.py` — Exponential decay temporal user profile.
- `backend/contextual_bandit.py` — Thompson Sampling / UCB exploration.
- `backend/exploration_engine.py` — `ThompsonSamplingBandit` for discovery.
- `backend/attention_user_model.py` — Attention-weighted session user model.
- `backend/query_understanding.py` — Intent parsing (mood, genre, era, abstract concept).

### Added — Data Platform
- `backend/events.py` — JSONL + Postgres dual-mode event store with DLQ quarantine.
- `backend/feature_store.py` — Redis-backed real-time user feature cache.
- `backend/experiments.py` — A/B experiment assignment and metric tracking.
- `backend/benchmark_cache.py` — Background benchmark computation with caching.

### Added — Observability
- `backend/slo.py` — `RequestSloTracker`: bounded in-process p50/p95/p99 latency
  tracking with per-route budgets.
- Prometheus + Grafana stack with auto-provisioned APEX Overview dashboard.
- Alerting rules: `HighErrorRate`, `RecommendationLatencyHigh`, `SearchLatencyHigh`.

### Changed — API
- `/v1/recommendations/id/{movie_id}` now returns `metrics` field with per-model
  scores, churn risk, preference stability, and cold-start flag.
- Rate limiting: 30 requests/minute per IP via `slowapi`.
- B2B SaaS plan enforcement middleware (`backend/middleware/plan_enforcer.py`).

### Infrastructure
- Docker Compose with Kafka cluster configuration.
- Kubernetes Helm chart (`k8s/helm/apex/`).
- Render deployment with `render.yaml` (3-tier auto-detection).
- GitHub Actions CI: 11 workflows covering lint → unit → API integration →
  data pipeline → ML → frontend → Docker → Helm → OpenAPI export →
  mutation testing → load tests.

---

## [1.0.0] — 2025-01-27

### Added
- Hybrid AI search (sparse TF-IDF + dense SBERT + FAISS ANN)
- Learned ranker (LightGBM) with promotion gates
- Behavior-aware personalization from implicit feedback
- Delta Lake medallion model (Bronze/Silver/Gold) with PySpark ETL
- SCD Type 2 dimension management
- Kafka streaming for real-time behavioral events
- Serving artifact versioning and SHA-256 validation
- Frontend failover (Cloudflare Pages / React / Streamlit)
- System readiness probes and SLO monitoring
- Per-seed recommendation diagnostic reports
- CSV catalog onboarding with schema profiling
- Customer API-key authentication mode
- Offline benchmark evaluation gates (HR@10, NDCG@10)
- Hugging Face model hub integration
- Render / free-tier deployment support
