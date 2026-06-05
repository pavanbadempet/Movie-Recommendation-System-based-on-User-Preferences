# Design Document: APEX Final Polish

## Overview

This spec closes the four remaining gaps to bring the APEX Movie Recommendation System to a 10/10 production-readiness rating. All frontend pages, hooks, and component updates are already implemented. The work is purely backend, infrastructure, and documentation.

The four tracks are independent and can be executed in parallel:

- **Track 1 — Feature Completeness**: Offline evaluation pipeline + API endpoint + CI registration of 3 missing PBT files + mutation testing workflow
- **Track 2 — Repo Hygiene**: `git rm --cached` for tracked artifacts + `.gitignore` additions
- **Track 3 — Production Deployment Docs**: README Deployment Tiers section + whitepaper placeholder replacement
- **Track 4 — main.py Decomposition**: Extract admin routes, benchmark cache, and recommender helpers into dedicated modules

---

## Track 1: Feature Completeness

### 1.1 Offline Evaluation Script (`scripts/run_offline_evaluation.py`)

The script implements leave-one-out evaluation against MovieLens 100K.

**Algorithm:**
1. Load ratings from `data/raw/u.data` (tab-separated: user_id, item_id, rating, timestamp). Download from `https://files.grouplens.org/datasets/movielens/ml-100k.zip` if absent.
2. For each user, sort interactions by timestamp ascending. Hold out the last interaction as the test item. The remaining interactions form the training set.
3. For each user, call `recommender.recommend_by_id(last_training_item_id, n=50)` to get ranked candidates.
4. Compute metrics:
   - **NDCG@10**: `1.0 / log2(rank + 2)` if test item in top-10, else 0. Average across users.
   - **Recall@50**: 1 if test item in top-50, else 0. Average across users.
   - **ILD**: Load `models/sbert_embeddings.npy`. For each user's top-10, compute mean pairwise cosine distance. Average across users. If embeddings unavailable, set to `null`.
   - **Cold-Start NDCG@10**: Same as NDCG@10 but only for users with ≤5 training interactions (count computed dynamically from the split, not a pre-computed flag).
5. Write `reports/offline_eval_report.json`:
   ```json
   {
     "generated_at": "2025-01-01T00:00:00Z",
     "num_users": 943,
     "ndcg_at_10": 0.142,
     "recall_at_50": 0.387,
     "ild": 0.621,
     "cold_start_ndcg_at_10": 0.089,
     "evaluation_protocol": "leave_one_out",
     "model_version": "2.0.0"
   }
   ```
6. Update `docs/APEX_WHITEPAPER.md` Section 6.1: replace `| Pending offline eval run |` with computed values formatted to 3 decimal places using regex substitution.

**CLI interface:**
```
python scripts/run_offline_evaluation.py [--output reports/offline_eval_report.json]
```

**Determinism**: Use `numpy.random.seed(42)` before any sampling. Sort all interactions by (user_id, timestamp) before splitting to ensure stable ordering.

### 1.2 Offline Metrics API Endpoint

Add to `backend/evaluation_routes.py` inside `create_evaluation_router`:

```python
@router.get("/v1/evaluation/offline-metrics")
async def offline_metrics():
    report_path = Path("reports/offline_eval_report.json")
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first."
        )
    try:
        content = report_path.read_text(encoding="utf-8")
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Offline eval report contains malformed JSON: {exc}")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Could not read offline eval report: {exc}")
```

No authentication required — metrics are public read-only data.

### 1.3 CI Registration of Missing PBTs

Three test files exist but are absent from the `unit-tests` job in `.github/workflows/ci.yml`:
- `tests/test_serving_tier_properties.py`
- `tests/test_onnx_thread_count.py`
- `tests/test_orjson_roundtrip.py`

The `test_orjson_roundtrip.py` file does not yet exist and must be created. It imports `_json_dumps` and `_json_loads` from `backend.main` and uses `@given` with `st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(), st.none(), st.lists(st.integers())))` to assert round-trip consistency.

### 1.4 Mutation Testing Workflow

New file `.github/workflows/mutation-tests.yml`:

```yaml
name: Mutation Testing
on:
  workflow_dispatch:
  schedule:
    - cron: '0 10 * * 1'  # Every Monday at 10:00 UTC
jobs:
  mutation-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: 'pip'
      - name: Install dependencies
        run: |
          pip install mutmut pytest hypothesis
          pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt
      - name: Run mutation tests
        env:
          JWT_SECRET_KEY: test-jwt-secret-key-for-ci-only
          NOVA_DISABLE_MODEL_DOWNLOADS: "1"
        run: |
          mutmut run \
            --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py \
            --runner "python -m pytest tests/test_serving_tier_properties.py tests/test_onnx_thread_count.py -x -q"
      - name: Print mutation score
        run: mutmut results
        if: always()
```

---

## Track 2: Repo Hygiene

### 2.1 Files to Untrack

The following files are currently tracked by Git and must be removed from tracking with `git rm --cached`:

| File | Status |
|---|---|
| `docker-compose.yml.bak` | Tracked — confirmed by `git ls-files` |
| `test_llm.py` | Tracked — confirmed by `git ls-files` |

The following files appear in the analysis but are already covered by `.gitignore` patterns and may not be tracked (verify with `git ls-files` before running `git rm --cached`):
- `.env` — already in `.gitignore` as `.env`
- `nova_db.sqlite3` — already in `.gitignore` as `*.sqlite3`
- `benchmark_temp.json` — already in `.gitignore` as `benchmark_temp.json`
- `movies_temp.parquet` — already in `.gitignore` as `movies_temp.parquet`
- `frontend-vite.err.log`, `frontend-vite.log` — already covered by `*.log`

### 2.2 .gitignore Additions

The current `.gitignore` is missing these patterns:

```gitignore
# Backup files
*.bak

# Root-level debug/verification scripts
test_llm.py
test_recommendations.py
test_delta_implementation.py
final_verification.py
verify_implementation.py
```

The `*.log` pattern already covers log files. The `*.sqlite3` pattern already covers database files. The `benchmark_temp.json` and `movies_temp.parquet` entries already exist.

---

## Track 3: Production Deployment Documentation

### 3.1 README Deployment Tiers Section

Add a new "🚀 Deployment Tiers" section to `README.md` after the "Quick Start" section:

```markdown
## 🚀 Deployment Tiers

APEX auto-detects hardware at startup and selects the appropriate serving tier. The live demo runs in **Tier 3** (free Render plan) — the full ensemble requires a paid plan.

| Tier | Plan | Profile | Active Models | Latency |
|------|------|---------|---------------|---------|
| **Tier 1** | Paid (GPU) | `full` | 6-model ensemble + RL + Active Inference | 50–200 ms |
| **Tier 2** | Paid (CPU) | `full` | ONNX-accelerated ensemble | 200–800 ms |
| **Tier 3** | Free | `lite` | FAISS + TF-IDF only | 800–2000 ms |

### Live Demo (Current: Tier 3)
The Render deployment uses `plan: free` with `NOVA_SERVING_PROFILE=lite`, which activates Tier 3 (degraded mode). This is intentional for cost reasons — the architecture supports all three tiers.

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
```

### 3.2 Whitepaper Placeholder Replacement

Replace all occurrences of `| Pending offline eval run |` in `docs/APEX_WHITEPAPER.md` with `| Requires local execution — run scripts/run_offline_evaluation.py |`.

This is a one-time text substitution. When `run_offline_evaluation.py` is executed, it will overwrite these with real values.

---

## Track 4: main.py Decomposition

### 4.1 `backend/admin_routes.py`

Extract all admin-only endpoints from `backend/main.py` into a new `APIRouter`. The router uses the existing `resolve_admin_token` dependency for authentication.

**Endpoints to extract:**
- `POST /v1/admin/reload-ensemble-weights`
- `POST /v1/artifacts/reload` (if currently in main.py — check first)
- Any other endpoints guarded by `resolve_admin_token`

**Module structure:**
```python
from fastapi import APIRouter, Depends
from backend.auth import resolve_admin_token

def create_admin_router(
    *,
    get_apex_engine,
    reload_local_recommender,
    refresh_artifact_files,
    ...
) -> APIRouter:
    router = APIRouter(tags=["Admin"])
    # ... endpoints
    return router
```

The router uses the factory pattern (matching `create_evaluation_router`) to avoid circular imports — dependencies are injected at registration time in `main.py`.

**Registration in main.py:**
```python
from backend.admin_routes import create_admin_router
admin_router = create_admin_router(
    get_apex_engine=get_apex_engine,
    reload_local_recommender=_reload_local_recommender,
    ...
)
app.include_router(admin_router)
```

### 4.2 `backend/benchmark_cache.py`

Extract benchmark caching state and helpers into a standalone module with a clean public API.

**Public API:**
```python
def get_cached_semantic_benchmark(k: int) -> dict | None: ...
def compute_semantic_benchmark_cached(rec, k: int) -> dict: ...
def start_background_semantic_benchmark(k: int) -> None: ...
def warming_semantic_benchmark_report(k: int) -> dict: ...
def get_cached_recommendation_benchmark(k: int) -> dict | None: ...
def compute_recommendation_benchmark_cached(rec, k: int) -> dict: ...
def start_background_recommendation_benchmark(k: int) -> None: ...
def warming_recommendation_benchmark_report(k: int) -> dict: ...
def semantic_benchmark_ttl_seconds() -> int: ...
def recommendation_benchmark_ttl_seconds() -> int: ...
```

Module-level state (cache dicts, locks) is private (`_semantic_benchmark_cache`, etc.). All public functions access state through the module's own functions, not directly.

### 4.3 `backend/recommender_helpers.py`

Extract the three large helper functions:

```python
def reload_local_recommender(force_download: bool) -> Recommender: ...
def refresh_artifact_files(force_download: bool) -> dict[str, bool]: ...
def background_recommender_warmup() -> None: ...
```

These functions need access to the module-level `_recommender` singleton in `main.py`. Use a getter/setter pattern:

```python
# In recommender_helpers.py
_get_recommender: Callable[[], Recommender | None] = lambda: None
_set_recommender: Callable[[Recommender | None], None] = lambda r: None

def configure(get_rec: Callable, set_rec: Callable) -> None:
    """Called once from main.py lifespan to wire up the singleton accessors."""
    global _get_recommender, _set_recommender
    _get_recommender = get_rec
    _set_recommender = set_rec
```

### 4.4 Line Count Target

After all three extractions, `backend/main.py` must be fewer than 1500 lines. The current ~2600 lines break down approximately as:

| Section | Estimated Lines |
|---|---|
| Imports + setup | ~120 |
| Lifespan + app init | ~80 |
| Middleware + CORS | ~60 |
| Response models | ~100 |
| Benchmark cache helpers | ~120 → moves to `benchmark_cache.py` |
| Recommender helpers | ~80 → moves to `recommender_helpers.py` |
| Admin endpoints | ~150 → moves to `admin_routes.py` |
| Core API endpoints | ~1800 |
| Utility functions | ~90 |

Target after extraction: ~2250 - 350 = ~1900 lines. To reach <1500, also extract the `_event_logging_enabled`, `_safe_float`, `_temporary_env`, `_artifact_refresh_env` utility functions into `recommender_helpers.py` or a new `backend/utils.py`.

---

## File Change Summary

| File | Action |
|---|---|
| `scripts/run_offline_evaluation.py` | **Create** |
| `backend/evaluation_routes.py` | **Modify** — add `GET /v1/evaluation/offline-metrics` |
| `tests/test_orjson_roundtrip.py` | **Create** |
| `.github/workflows/ci.yml` | **Modify** — add 3 test files to unit-tests job |
| `.github/workflows/mutation-tests.yml` | **Create** |
| `.gitignore` | **Modify** — add `*.bak` and debug script patterns |
| `README.md` | **Modify** — add Deployment Tiers section + Mutation Testing section |
| `docs/APEX_WHITEPAPER.md` | **Modify** — replace placeholder text |
| `backend/admin_routes.py` | **Create** |
| `backend/benchmark_cache.py` | **Create** |
| `backend/recommender_helpers.py` | **Create** |
| `backend/main.py` | **Modify** — remove extracted code, import new modules |
| `docker-compose.yml.bak` | **Delete** (untrack + delete) |
| `test_llm.py` | **Delete** (untrack + delete) |
