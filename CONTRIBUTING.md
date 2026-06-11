# Contributing to APEX

Thanks for your interest. This document covers everything you need to get from zero to a working dev environment and submit a quality PR.

---

## Table of Contents

- [Philosophy](#philosophy)
- [Local Development Setup](#local-development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Running Tests](#running-tests)
- [Submitting a PR](#submitting-a-pr)
- [Adding a New ML Model](#adding-a-new-ml-model)

---

## Philosophy

- **Correctness over cleverness.** A clear, well-commented implementation beats a clever one-liner.
- **Comments explain *why*, not *what*.** The code shows what happens; comments explain the reasoning behind the decision.
- **Keep the request path fast.** Heavy computation belongs in the ETL pipeline or background tasks, not in a request handler.
- **Every change needs a test.** Property-based tests (Hypothesis) are preferred for ML components; unit tests for everything else.

---

## Local Development Setup

### Prerequisites

- Python 3.11+
- Node.js 24+ (for frontend)
- Docker + Docker Compose (for the full stack)

### Backend only (fastest)

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment variables
cp .env.example .env
# Edit .env and add your TMDB_API_KEY and JWT_SECRET_KEY

# 4. Build serving artifacts (FAISS index, embeddings)
python scripts/rebuild_serving_artifacts.py

# 5. Start the API
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`.

### Full stack (Docker Compose)

```bash
# Starts backend + frontend + Kafka + Spark + Redis + PostgreSQL + Prometheus + Grafana
docker compose up --build
```

Services:
| Service | URL |
|---|---|
| Backend API | http://localhost:8000 |
| React Frontend | http://localhost:5173 |
| API Docs | http://localhost:8000/docs |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (password from `GF_ADMIN_PASSWORD` in `.env`, default `change-me-in-env`) |
| Spark UI | http://localhost:8080 |

### Frontend only

```bash
cd frontend
npm install
npm run dev
```

---

## Project Structure

```
backend/                  # FastAPI application — all Python serving code
  __init__.py             # Public API surface (CandidateItem, TierDetector, __version__)
  main.py                 # App entry point, middleware, lifespan
  recommender.py          # Thin orchestrator — delegates to pipeline modules
  #
  # Sub-packages (logical groupings with __init__.py exports)
  pipeline/               # 3-stage pipeline types and implementations
    __init__.py           # Exports: CandidateItem, RankedItem, FinalItem, all 3 pipelines
  models/                 # ML model implementations
    __init__.py           # Exports: LightGCN, SASRec, KAN, Quantum, Hyperbolic, Diffusion, ...
  serving/                # Tier detection and serving infrastructure
    __init__.py           # Exports: TierDetector, HardwareProfile, resolve_serving_tier
  metrics/                # Evaluation and debiased metrics
    __init__.py           # Exports: ips_ndcg_at_k, calibration_score, beyond_accuracy_metrics
  privacy/                # Differential privacy and federated learning
    __init__.py           # Exports: add_laplace_noise, privatize_user_embedding, ...
  #
  # Core pipeline modules (flat, in backend/ for backward compatibility)
  pipeline_types.py       # CandidateItem, RankedItem, FinalItem dataclasses
  retrieval_pipeline.py   # Stage 1: FAISS + TF-IDF + KG → candidates
  ranking_pipeline.py     # Stage 2: Ensemble + Learned Ranker → scores
  reranking_pipeline.py   # Stage 3: MMR + RL Safety + LLM → final list
  ensemble_engine.py      # 6-model ensemble (LightGCN, SASRec, KAN, ...)
  serving_tier.py         # Hardware auto-detection (Tier 1/2/3)
  tests/                  # Backend unit + property-based tests

etl/                      # Data pipeline (Pandas + PySpark)
scripts/                  # Training, evaluation, and utility scripts
tests/                    # Integration and system tests
frontend/                 # React + TypeScript UI (Vite 7, React 19)
docs/                     # Architecture docs, ADRs, model cards, OpenAPI spec
models/                   # Trained model weights (.pth, .joblib, .json)
data/                     # Delta Lake (bronze/silver/gold) + evaluation sets
```

---

## Code Standards

All Python code is linted and formatted with **ruff**. Configuration is in `pyproject.toml`.

```bash
# Check for lint errors
python -m ruff check backend/ tests/ scripts/ etl/

# Auto-fix fixable issues
python -m ruff check backend/ tests/ scripts/ etl/ --fix

# Format code
python -m ruff format backend/ tests/ scripts/ etl/
```

The CI pipeline runs `ruff check` and `ruff format --check` as a gate before any tests. A PR with lint errors will not be merged.

### Pre-commit hooks (recommended)

Install once and all quality gates run automatically on every `git commit`:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

This runs ruff, secret detection (gitleaks), large-file checks, and Dockerfile linting locally — the same checks CI runs. You will never push a commit that fails CI lint.

To run all hooks against all files manually:

```bash
pre-commit run --all-files
```

Key rules:
- Line length: 120 characters
- Imports: sorted (isort-compatible), one import per line
- Type annotations: required on all public functions
- f-strings: use them; avoid `%` formatting and `.format()`
- No `lambda` assignments — use `def` instead

For TypeScript (frontend):
```bash
cd frontend
npm run lint        # ESLint
npm run type-check  # TypeScript strict check
```

---

## Running Tests

### Backend

```bash
# All unit + property-based tests (fast, no ML model loading)
python -m pytest tests/test_serving_tier_properties.py tests/test_ensemble_weights.py tests/test_slo.py -v

# Full test suite with coverage
python -m pytest tests/ backend/tests/ --cov=backend --cov-report=term-missing --cov-fail-under=80

# Property-based tests only
python -m pytest tests/ -k "property" -v

# Specific test file
python -m pytest tests/test_ranker.py -v
```

### Frontend

```bash
cd frontend
npm run test              # Run all tests once
npm run test:coverage     # With coverage report (80% threshold)
```

### Mutation testing (weekly CI, or run locally)

```bash
pip install mutmut
mutmut run --paths-to-mutate backend/serving/serving_tier.py,backend/serving/onnx_engine.py
mutmut results
```

---

## Submitting a PR

1. **Branch from `develop`**, not `main`. Use a descriptive name: `feat/add-cross-encoder-reranker`, `fix/sasrec-cold-start-padding`, `docs/update-model-cards`.

2. **Keep PRs focused.** One logical change per PR. If you're fixing a bug and refactoring, split them.

3. **Write tests.** For ML components, prefer property-based tests with Hypothesis. For API changes, add an integration test in `tests/test_api.py`.

4. **Update docs.** If you change a public API endpoint, update `docs/API_REFERENCE.md`. If you make an architectural decision, add an ADR to `docs/ARCHITECTURE_DECISIONS.md`.

5. **PR description format:**
   ```
   ## What
   Brief description of the change.

   ## Why
   The problem this solves or the improvement it makes.

   ## How
   Key implementation decisions. Link to relevant ADR if applicable.

   ## Testing
   What tests were added or modified.
   ```

6. **CI must pass.** All 6 CI jobs (lint, unit-tests, api-tests, data-pipeline-tests, ml-tests, frontend-tests) must be green before review.

---

## Adding a New ML Model

1. Create `backend/your_model.py` with a `nn.Module` subclass. Follow the pattern in `backend/lightgcn.py` or `backend/sasrec.py`.

2. Add a weight key to `ApexEnsembleEngine._REQUIRED_WEIGHT_KEYS` and `_DEFAULT_WEIGHTS` in `backend/ensemble_engine.py`.

3. Add a scoring function in `ApexEnsembleEngine._predict_ensemble_pytorch()` following the existing pattern.

4. Add a model card section to `docs/MODEL_CARDS.md` documenting architecture, training, evaluation metrics, and known limitations.

5. Add property-based tests in `backend/tests/` that verify:
   - Output shape invariants (scores are always the right length)
   - Score range invariants (scores are finite floats)
   - Graceful degradation (model handles cold-start / zero inputs without crashing)

6. Run the ablation study to measure marginal contribution:
   ```bash
   python scripts/ablation_study.py --sample-size 1000
   ```

7. If the model contributes positively, run DR weight optimization:
   ```bash
   python scripts/causal_debias_training.py --skip-lgcn
   ```
