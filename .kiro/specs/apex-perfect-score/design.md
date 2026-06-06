# Design Document: APEX Perfect Score

## Overview

This document describes the technical design for closing the four remaining gaps in the APEX Movie Recommendation System. The work is organized into four parallel tracks that can be executed concurrently where dependencies allow.

---

## Track 1: ML Completeness

### 1.1 Ensemble Weight Optimizer Execution

**File:** `scripts/optimize_ensemble_weights.py` (already exists — verify and run)

The script already implements the Dirichlet grid-search. The gap is that it has never been executed against real data. The design is:

1. Load the Event Store validation split (20% of historical interactions held out by timestamp)
2. For each of 500+ Dirichlet-sampled weight vectors, call `ApexEnsembleEngine.predict_ensemble` on validation users and compute NDCG@10 and Hit_Rate@10
3. Track the best vector; log top-5 to stdout
4. Write `models/ensemble_weights.json` with schema:
   ```json
   {
     "lightgcn": 0.35,
     "quantum": 0.20,
     "sasrec": 0.15,
     "kan": 0.12,
     "hyperbolic": 0.10,
     "diffusion": 0.08,
     "evaluated_at": "2026-05-01T06:00:00Z",
     "ndcg_at_10": 0.312,
     "hit_rate_at_10": 0.847,
     "num_candidates_evaluated": 500
   }
   ```

The `Ensemble_Engine._load_weights()` already reads this file at startup — no backend changes needed.

### 1.2 Offline Evaluation Pipeline

**New file:** `scripts/run_offline_evaluation.py`

**Algorithm:**
```
1. Load MovieLens 100K ratings (data/raw/ml-latest-small/ratings.csv or HF Hub)
2. For each user, sort interactions by timestamp ascending
3. Hold out the last interaction as the test item
4. For each user, call recommender.recommend_by_id(training_items[-1], n=50)
5. Compute NDCG@10, Recall@50 against the held-out test item
6. Compute ILD: for each user's top-10, compute mean pairwise cosine distance
   between SBERT embeddings (loaded from models/sbert_embeddings.npy)
7. Cold-start subset: users with ≤5 training interactions → compute NDCG@10
8. Write reports/offline_eval_report.json
9. Update docs/APEX_WHITEPAPER.md Section 6.1 via regex replacement
```

**NDCG@k formula:**
```python
def ndcg_at_k(ranked_ids: list[int], relevant_id: int, k: int) -> float:
    for i, mid in enumerate(ranked_ids[:k]):
        if mid == relevant_id:
            return 1.0 / math.log2(i + 2)
    return 0.0
```

**ILD formula:**
```python
def ild(embeddings: np.ndarray) -> float:
    # embeddings: [k, dim]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / (norms + 1e-8)
    sim_matrix = normed @ normed.T
    n = len(embeddings)
    if n < 2:
        return 0.0
    total = sum(1 - sim_matrix[i, j] for i in range(n) for j in range(i+1, n))
    return total / (n * (n - 1) / 2)
```

**Output schema** (`reports/offline_eval_report.json`):
```json
{
  "ndcg_at_10": 0.312,
  "recall_at_50": 0.521,
  "ild": 0.387,
  "cold_start_ndcg_at_10": 0.143,
  "evaluated_at": "2026-05-01T06:00:00Z",
  "num_users": 610,
  "num_cold_start_users": 87,
  "candidate_pool_size": 9724
}
```

### 1.3 Offline Metrics API Endpoint

**File:** `backend/evaluation_routes.py` (add to existing router)

```python
@router.get("/v1/evaluation/offline-metrics")
async def offline_metrics():
    path = Path("reports/offline_eval_report.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first.")
    return _json_loads(path.read_text(encoding="utf-8"))
```

### 1.4 Whitepaper Update

The script patches `docs/APEX_WHITEPAPER.md` Section 6.1 using regex:
```python
import re
content = Path("docs/APEX_WHITEPAPER.md").read_text()
content = re.sub(r'(NDCG@10.*?)\| Pending offline eval run \|',
                 rf'\1| {ndcg:.3f} |', content)
# ... similar for Recall@50, ILD, Cold-Start
Path("docs/APEX_WHITEPAPER.md").write_text(content)
```

---

## Track 2: Testing Completeness

### 2.1 Coverage Gate

**File:** `.github/workflows/ci.yml`

Add `--cov-fail-under=80` to the pytest command:
```yaml
python -m pytest ... --cov=backend --cov-report=term-missing --cov-report=xml --cov-fail-under=80
```

Add Vitest coverage to `frontend/vite.config.ts`:
```typescript
test: {
  coverage: {
    provider: 'v8',
    reporter: ['text', 'lcov'],
    thresholds: { lines: 80 }
  }
}
```

Update `ci.yml` frontend job:
```yaml
- name: Run Vitest with coverage
  working-directory: ./frontend
  run: npm run test -- --coverage
```

Add coverage badge to `README.md`:
```markdown
![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)
```

### 2.2 Missing Property Tests

**New file:** `tests/test_serving_tier_properties.py`

Contains Properties 1, 2, 3 using `hypothesis` with `@settings(max_examples=100)`:

- **Property 1** — `HardwareProfile` type invariants: mock `torch.cuda.is_available`, `psutil.virtual_memory`, `os.cpu_count` with `@given` strategies including error-raising callables. Assert `gpu_available: bool`, `ram_gb: float > 0`, `cpu_cores: int >= 1`.

- **Property 2** — Tier resolution totality: `@given(st.floats(min_value=0, max_value=1000), st.booleans())` → assert result always in `{"tier1", "tier2", "tier3"}`, never raises.

- **Property 3** — Auto-selection boundaries:
  - `ram_gb < 8.0` → `"tier3"` (regardless of GPU)
  - `gpu_available=True AND ram_gb >= 16.0` → `"tier1"`
  - else → `"tier2"`

**New file:** `tests/test_onnx_thread_count.py`

Contains Property 4: mock `ort.InferenceSession`, `@given(st.integers(min_value=1, max_value=256))`, assert `intra_op_num_threads == cpu_cores`.

**New file:** `tests/test_orjson_roundtrip.py`

Contains Property 5: `@given` with nested dicts of string keys and int/float/str/None/list values, assert `_json_loads(_json_dumps(payload)) == payload`.

### 2.3 Mutation Testing

**New file:** `.github/workflows/mutation-tests.yml`

```yaml
name: Mutation Tests
on:
  workflow_dispatch:
  schedule:
    - cron: '0 10 * * 1'  # Weekly on Monday

jobs:
  mutmut:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install mutmut pytest -r requirements.txt
      - run: |
          mutmut run \
            --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py \
            --runner "python -m pytest tests/test_serving_tier_properties.py tests/test_onnx_thread_count.py -x -q"
          mutmut results
```

Add to `README.md`:
```markdown
## Mutation Testing
Run locally: `mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py`
View results: `mutmut results`
```

---

## Track 3: Frontend Completeness

### 3.1 New Pages Architecture

The React SPA gains five new pages and one updated component:

```
frontend/src/
├── pages/
│   ├── Dashboard.tsx          (NEW)
│   ├── KnowledgeGraph.tsx     (NEW)
│   ├── Evaluation.tsx         (NEW)
│   ├── UserProfile.tsx        (NEW)
│   └── AdminPanel.tsx         (NEW)
├── components/
│   └── RecommendationCard.tsx (UPDATE — add provenance + explanation)
├── hooks/
│   ├── useSlo.ts              (NEW)
│   ├── useHealth.ts           (NEW)
│   └── useKnowledgeGraph.ts   (NEW)
└── test/
    └── accessibility.test.tsx (NEW)
```

### 3.2 Dashboard Page (`Dashboard.tsx`)

```typescript
// Fetches /health and /v1/platform/slo in parallel
// Renders:
//   - TierBadge: colored chip (tier1=green, tier2=blue, tier3=orange)
//   - HardwareCard: gpu_available, ram_gb, cpu_cores
//   - SloMetrics: p95_latency_ms, error_rate, request_rate from /v1/platform/slo
//   - Degraded state: grey banner when /v1/platform/slo returns 5xx or network error
```

**TierBadge color map:**
- `tier1` → `bg-green-500` "Enterprise (GPU)"
- `tier2` → `bg-blue-500` "Professional (CPU)"
- `tier3` → `bg-orange-500` "Starter (Lite)"

### 3.3 RecommendationCard Update

Add to the existing card component:
```typescript
// Below movie title:
{movie.explanation_text && (
  <p className="text-sm italic text-gray-600 mt-1">{movie.explanation_text}</p>
)}

// Below poster:
{movie.retrieval_stage && (
  <span className="badge">{movie.retrieval_stage}</span>
)}
{movie.retrieval_signals && (
  <dl className="text-xs">
    {Object.entries(movie.retrieval_signals).map(([k, v]) => (
      <div key={k}><dt>{k}</dt><dd>{String(v)}</dd></div>
    ))}
  </dl>
)}
```

### 3.4 Knowledge Graph Page (`KnowledgeGraph.tsx`)

Uses `d3-force` (already a common dependency) or `cytoscape`:

```typescript
// State: seedMovieId, graphData { nodes: Node[], edges: Edge[] }
// On seed selection: fetch /v1/recommendations/knowledge-graph/{id}
// Transform response into D3 nodes/edges:
//   - seed node: { id: movie_id, label: title, type: 'seed' }
//   - rec nodes: { id: movie_id, label: title, type: 'rec' }
//   - edges: { source: seed_id, target: rec_id, label: retrieval_stage }
// Render SVG force simulation
// Click handler: show side panel with title, poster, overview
// Empty state: "No knowledge graph connections found for this movie."
```

**D3 dependency:** Add `d3` to `frontend/package.json` (pinned version).

### 3.5 Evaluation Page (`Evaluation.tsx`)

```typescript
// Parallel fetch: Promise.allSettled([
//   fetch('/v1/evaluation/semantic-benchmark?sync=false'),
//   fetch('/v1/evaluation/recommendation-benchmark?sync=false'),
//   fetch('/v1/evaluation/offline-metrics'),
// ])
// Show each section independently — partial results if one fails
// MetricsTable: columns [Metric, Value, Threshold, Status]
// LoadingSpinner per section while fetching
```

### 3.6 User Profile Page (`UserProfile.tsx`)

```typescript
// Guard: if !isAuthenticated → show LoginPrompt
// Fetch /v1/events/features?limit=20 for behavior features
// Fetch /v1/recommendations/user/{userId}?n=10 for personalized recs
// BehaviorCard: total_ratings, avg_rating, click_count, view_count
//   - Validate: value >= 0 ? value : "—"
// PersonalizedRecs: reuse RecommendationCard grid
```

### 3.7 Admin Panel Page (`AdminPanel.tsx`)

```typescript
// Guard: if !isAdmin → show "Admin access required"
// Reload Weights button → POST /v1/admin/reload-ensemble-weights
// On success: display WeightsTable { model, weight } for all 6 models
// On error (network, 401, 403, timeout): display ErrorBanner with message
// No unhandled promise rejections — all errors caught in try/catch
```

### 3.8 Accessibility Audit (`accessibility.test.tsx`)

```typescript
import { render } from '@testing-library/react'
import { axe, toHaveNoViolations } from 'jest-axe'
expect.extend(toHaveNoViolations)

const pages = [Dashboard, RecommendationPage, KnowledgeGraph, Evaluation, UserProfile]

pages.forEach(Page => {
  it(`${Page.name} has no critical accessibility violations`, async () => {
    const { container } = render(<Page />)
    const results = await axe(container, {
      runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }
    })
    // Critical and serious violations fail; moderate violations are reported
    const criticalOrSerious = results.violations.filter(
      v => v.impact === 'critical' || v.impact === 'serious'
    )
    expect(criticalOrSerious).toHaveLength(0)
  })
})
```

**Dependencies to add:**
```json
"jest-axe": "8.0.0",
"@axe-core/react": "4.10.0"
```

### 3.9 Router Updates (`main.tsx` or `App.tsx`)

Add routes:
```typescript
<Route path="/dashboard" element={<Dashboard />} />
<Route path="/knowledge-graph" element={<KnowledgeGraph />} />
<Route path="/evaluation" element={<Evaluation />} />
<Route path="/profile" element={<UserProfile />} />
<Route path="/admin" element={<AdminPanel />} />
```

Add navigation links to the existing nav bar.

---

## Track 4: Spec Completeness

### 4.1 `backend/onnx_engine.py` — cpu_cores Wiring

Current state: `intra_op_num_threads` is hardcoded to 0. Changes:

```python
class ONNXEngine:
    def __init__(self, cpu_cores: int = 0):
        self._cpu_cores = cpu_cores
        self.sessions: dict[str, ort.InferenceSession] = {}
        # ... existing init

    def load_model(self, name: str, path: str) -> None:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = self._cpu_cores  # 0 = auto-detect
        session = ort.InferenceSession(path, sess_options=opts)
        self.sessions[name] = session

    def has_any_onnx_models(self) -> bool:
        return len(self.sessions) > 0

# Singleton factory update:
_onnx_engine: ONNXEngine | None = None

def get_onnx_engine(cpu_cores: int = 0) -> ONNXEngine:
    global _onnx_engine
    if _onnx_engine is None:
        _onnx_engine = ONNXEngine(cpu_cores=cpu_cores)
        # ... load models
    return _onnx_engine
```

### 4.2 `backend/ensemble_engine.py` — Device Placement Verification

The code already has `device` param, `_move_to_device()`, `_try_compile()`, `_try_compile_all()`, and `get_apex_engine(device=...)`. Verify:
- `__init__` calls `_move_to_device()` when `device != "cpu"` ✓
- `__init__` calls `_try_compile_all()` when `device == "cuda"` ✓
- `get_apex_engine` passes `device` to constructor ✓

No code changes needed — just verification and test coverage.

### 4.3 `backend/recommender.py` — Tier 3 Constraints

Add at the start of `Recommender.load()`:

```python
from backend.serving.serving_tier import resolve_serving_tier

def load(self) -> "Recommender":
    active_tier = resolve_serving_tier()

    if active_tier == "tier3":
        self._low_memory = True
        # Cap TF-IDF vocabulary
        current_max = int(os.getenv("NOVA_TFIDF_MAX_FEATURES", "50000"))
        if current_max > 12000:
            os.environ["NOVA_TFIDF_MAX_FEATURES"] = "12000"
        logger.info("Tier 3: low_memory=True, TF-IDF capped at 12000 features")

    # ... rest of existing load() logic

    # Skip diffusion model on tier3
    if active_tier == "tier3":
        self._diffusion_model = None
        logger.info("Tier 3: skipping Diffusion model load")

    # Defer sparse index on tier3
    if active_tier != "tier3":
        self._build_sparse_retrieval_index()
    else:
        logger.info("Tier 3: sparse retrieval index deferred to first request")

    return self
```

### 4.4 `backend/main.py` — Lifespan and /health Verification

The lifespan already has tier detection and branching. Verify:
- `get_tier_detector().resolve()` called before model loading ✓
- Tier 1: `get_apex_engine(device=...)` + `OnlineLearner` ✓
- Tier 2: `get_onnx_engine(cpu_cores=N)` + `has_any_onnx_models()` fallback ✓
- Tier 3: lazy load ✓
- `/health` returns `serving_tier`, `hardware_profile`, `tier_selection_reason` ✓

No code changes needed — just verification and test coverage.

---

## Dependency Graph

```
Track 1 (ML):
  1.1 (run optimizer) → 1.2 (offline eval) → 1.3 (API endpoint) → 1.4 (whitepaper)

Track 2 (Testing):
  2.1 (coverage gate) — independent
  2.2 (property tests) — independent
  2.3 (mutation testing) — depends on 2.2

Track 3 (Frontend):
  3.1 (new pages) → 3.2 (RecommendationCard) → 3.3 (KG page) → 3.4 (Eval page)
  3.5 (UserProfile) — independent of 3.3/3.4
  3.6 (AdminPanel) — independent
  3.7 (accessibility) — depends on all pages existing (3.1–3.6)
  3.8 (router) — depends on all pages existing

Track 4 (Spec):
  4.1 (onnx_engine) — independent
  4.2 (ensemble_engine verify) — independent
  4.3 (recommender tier3) — depends on serving_tier.py existing
  4.4 (main.py verify) — depends on 4.1, 4.3

Cross-track:
  Track 3 Evaluation page (3.4) depends on Track 1 API endpoint (1.3)
  Track 2 property tests (2.2) depend on Track 4 onnx_engine (4.1)
```

---

## File Change Summary

| File | Change Type | Track |
|------|-------------|-------|
| `scripts/optimize_ensemble_weights.py` | Verify + run | 1 |
| `scripts/run_offline_evaluation.py` | New | 1 |
| `backend/evaluation_routes.py` | Add endpoint | 1 |
| `docs/APEX_WHITEPAPER.md` | Update Section 6.1 | 1 |
| `reports/offline_eval_report.json` | New (generated) | 1 |
| `models/ensemble_weights.json` | Update (generated) | 1 |
| `.github/workflows/ci.yml` | Add coverage gates | 2 |
| `.github/workflows/mutation-tests.yml` | New | 2 |
| `frontend/vite.config.ts` | Add coverage config | 2 |
| `tests/test_serving_tier_properties.py` | New | 2 |
| `tests/test_onnx_thread_count.py` | New | 2 |
| `tests/test_orjson_roundtrip.py` | New | 2 |
| `README.md` | Add badges + mutation docs | 2 |
| `frontend/src/pages/Dashboard.tsx` | New | 3 |
| `frontend/src/pages/KnowledgeGraph.tsx` | New | 3 |
| `frontend/src/pages/Evaluation.tsx` | New | 3 |
| `frontend/src/pages/UserProfile.tsx` | New | 3 |
| `frontend/src/pages/AdminPanel.tsx` | New | 3 |
| `frontend/src/components/RecommendationCard.tsx` | Update | 3 |
| `frontend/src/hooks/useSlo.ts` | New | 3 |
| `frontend/src/hooks/useHealth.ts` | New | 3 |
| `frontend/src/hooks/useKnowledgeGraph.ts` | New | 3 |
| `frontend/src/test/accessibility.test.tsx` | New | 3 |
| `frontend/src/App.tsx` | Add routes + nav | 3 |
| `frontend/package.json` | Add d3, jest-axe, @axe-core/react | 3 |
| `backend/onnx_engine.py` | Add cpu_cores + has_any_onnx_models | 4 |
| `backend/recommender.py` | Add tier3 constraints | 4 |
