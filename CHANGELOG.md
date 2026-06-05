# Changelog

All notable changes to APEX are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [Unreleased]

### Added
- **Model Cards** (`docs/MODEL_CARDS.md`): Full model cards for all 6 ensemble models following Mitchell et al. (2019) standard — architecture, training data, evaluation metrics (HR@10, NDCG@10), DR-optimized weights, intended use, and known limitations
- **`pyproject.toml`**: Single source of truth for ruff (lint + format), mypy, pytest, and coverage config — replaces `pytest.ini`
- **Ruff lint gate in CI**: New `lint` job runs `ruff check` + `ruff format --check` before all test jobs; lint failure blocks the entire pipeline
- **Ensemble evaluation section in README**: Surfaces real offline metrics (HR@10=0.785, NDCG@10=0.542), DR-optimized weights, cross-architecture design rationale, and causal debiasing methodology
- **IPS-debiased LightGCN weights**: `ensemble_engine.py` now prefers `lightgcn_ips.pth` (trained with IPS-weighted BPR) over `lightgcn.pth` when available
- **`docs/LOCAL_DEVELOPMENT.md`** → merged into `CONTRIBUTING.md`

### Changed
- **Architecture diagram weights updated**: Mermaid diagram in README now shows DR-optimized weights (SASRec=0.659, KAN=0.298) instead of stale placeholder values (LightGCN=0.65, KAN=0.00)
- **`CONTRIBUTING.md`** rewritten: covers ruff setup, test commands, PR format, and step-by-step guide for adding new ML models
- **`DEPLOYMENT.md`** rewritten: full environment variable reference table, Docker Compose service URLs, Tier 1/2/3 upgrade instructions, health check verification commands
- **`docs/ARCHITECTURE_DECISIONS.md`** ADR weights updated to reflect DR-optimized values

### Fixed (bugs)
- `backend/recommender.py`: `cand_text` used in LLM reranking f-string but never defined — would crash at runtime
- `backend/search_benchmark.py`: `_or_jloads` called instead of `_orjson.loads` — JSON parsing silently broken
- `backend/serving_tier.py`: `threading` used as type annotation but never imported
- `backend/ensemble_engine.py`: `ThreadPoolExecutor` used as string annotation with import inside function body
- `backend/recommendation_routes.py`: `remote_recommender_status()` called in dead `if False` branch
- `backend/auth.py`: `== False` SQLAlchemy comparison → `~APIKey.is_revoked`
- `backend/active_inference_engine.py`: 3 f-strings with no placeholders
- `scripts/benchmark_ensemble.py`, `scripts/generate_synthetic_interactions.py`: B023 loop-variable capture bugs in closures

### Fixed (code quality)
- 93 unused imports auto-removed across `backend/`
- 925 style issues auto-fixed (whitespace, import sorting, type annotation modernization, comprehension simplification)
- 165 files reformatted to consistent style (double quotes, 120-char line length, sorted imports)
- `backend/recommender_helpers.py`: E731 lambda assignments replaced with proper `def` functions
- `backend/tests/test_replay.py`: E701 multiple statements on one line
- `tests/test_finetune_two_tower.py`: 3 nested `with` blocks collapsed to single `with` statements
- `scripts/causal_debias_training.py`: Added academic references (Schnabel ICML 2016, Saito WSDM 2020, Dudík ICML 2011)

---

## [2.0.0] — 2025-05

### Added
- **4-Layer Intelligence Stack**: Two-Tower neural retrieval, MMoE multi-task ranker,
  CLIP multi-modal fusion, A2C reinforcement learning, knowledge graph reasoning
- **Adaptive Serving Tiers**: Auto-detects hardware at startup; degrades gracefully
  from GPU ensemble (Tier 1) → ONNX CPU (Tier 2) → FAISS+TF-IDF lite (Tier 3)
- **ONNX Export & Runtime**: CPU-accelerated inference for Tier 2 deployments
- **Differential Privacy**: Laplace/Gaussian noise on user embeddings for GDPR compliance
- **Counterfactual Evaluation**: Inverse Propensity Scoring (IPS) for offline model simulation
- **Fairness Auditor**: Gini coefficient and KL divergence checks to prevent popularity bias
- **LLM Personalization**: OpenRouter integration (GPT-4o / Llama 3) for natural-language
  recommendation explanations
- **Multi-Tenancy & Auth**: JWT-based auth, PostgreSQL-backed tenant/user management
- **Observability**: Prometheus metrics, Grafana dashboards, Sentry error monitoring, SLO tracking
- **Property-Based Testing**: Hypothesis + mutation testing (mutmut) with 80% coverage gate
- **Full CI/CD**: 6-job GitHub Actions pipeline (unit, API, data pipeline, ML, frontend, Docker)
- **React + Vite Frontend**: TypeScript UI with Vitest coverage
- **PySpark Medallion Pipeline**: Delta Lake bronze/silver/gold ETL with SCD Type 2

### Architecture
- FastAPI backend with async lifespan management
- Redis feature store for sub-millisecond session state
- Kafka event streaming for real-time behavioral updates
- Distributed PySpark cluster for large-scale ETL
- FAISS ANN vector search across 768-dim SBERT and 1280-dim multi-modal spaces

---

## [1.0.0] — Initial Release

- Content-based movie recommendation using TF-IDF and cosine similarity
- Basic FastAPI REST API
- Streamlit frontend
- TMDB API integration for movie metadata enrichment
