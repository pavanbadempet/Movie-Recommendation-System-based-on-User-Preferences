# Implementation Plan: APEX Perfect Score

## Overview

Close the four remaining gaps (ML completeness, testing completeness, frontend completeness, spec completeness) in parallel across four tracks. Each track is independently executable. Cross-track dependencies are noted per task.

---

## Track 1 — ML Completeness

- [ ] 1. Run ensemble weight optimizer against real interaction data
  - [x] 1.1 Verify `scripts/optimize_ensemble_weights.py` is complete and runnable
    - Confirm `run_dirichlet_grid_search` samples ≥500 Dirichlet candidates
    - Confirm it writes `models/ensemble_weights.json` with all 6 keys + metadata fields (`evaluated_at`, `ndcg_at_10`, `hit_rate_at_10`, `num_candidates_evaluated`)
    - Confirm all weights are non-negative and sum to 1.0 before writing
    - If any of the above are missing, implement them now
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [-] 1.2 Execute the optimizer and commit the resulting weights file
    - Run: `python scripts/optimize_ensemble_weights.py --num-candidates 500 --k 10`
    - Verify `models/ensemble_weights.json` is written with at least one of `kan`, `hyperbolic`, `diffusion` > 0.001
    - Commit `models/ensemble_weights.json` to the repository
    - _Requirements: 1.2, 1.5_

- [ ] 2. Build and run the offline evaluation pipeline
  - [-] 2.1 Create `scripts/run_offline_evaluation.py`
    - Implement leave-one-out split: for each user, hold out the most recent interaction as the test item
    - Load MovieLens 100K ratings from `data/raw/` (or download if absent)
    - For each user, call `recommender.recommend_by_id(last_training_item_id, n=50)` to get ranked candidates
    - Compute `ndcg_at_10` using `1.0 / log2(rank + 2)` if test item appears in top-10, else 0
    - Compute `recall_at_50`: 1 if test item in top-50, else 0; average across users
    - Compute `ild`: for each user's top-10, load SBERT embeddings from `models/sbert_embeddings.npy`, compute mean pairwise cosine distance; average across users
    - Compute `cold_start_ndcg_at_10`: same as NDCG@10 but only for users with ≤5 training interactions
    - Write `reports/offline_eval_report.json` with all required fields
    - Accept `--output` CLI argument; default to `reports/offline_eval_report.json`
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7_

  - [x] 2.2 Add `/v1/evaluation/offline-metrics` endpoint to `backend/evaluation_routes.py`
    - Add `GET /v1/evaluation/offline-metrics` handler that reads `reports/offline_eval_report.json`
    - Return 404 with message `"Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first."` if file absent
    - Return file contents as JSON on success
    - _Requirements: 2.8, 2.9_

  - [x] 2.3 Update `docs/APEX_WHITEPAPER.md` Section 6.1 with computed metrics
    - Add whitepaper-update logic to `scripts/run_offline_evaluation.py` (called after writing the report)
    - Use regex to replace `| Pending offline eval run |` placeholders in the NDCG@10, Recall@50, ILD, and Cold-Start rows with the computed values formatted to 3 decimal places
    - Run the script and commit the updated whitepaper
    - _Requirements: 2.10_

  - [x] 2.4 Run the offline evaluation pipeline
    - Execute: `python scripts/run_offline_evaluation.py`
    - Verify `reports/offline_eval_report.json` is written with all 8 required fields
    - Verify `docs/APEX_WHITEPAPER.md` Section 6.1 no longer contains "Pending offline eval run"
    - Commit both files
    - _Requirements: 2.7, 2.10_

---

## Track 2 — Testing Completeness

- [x] 3. Add code coverage enforcement gate
  - [x] 3.1 Add `--cov-fail-under=80` to the pytest command in `ci.yml`
    - In the `unit-tests` job, append `--cov-fail-under=80` to the existing `python -m pytest` command
    - Ensure `--cov-report=xml` is present so `coverage.xml` is always produced
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x] 3.2 Add Vitest coverage configuration to `frontend/vite.config.ts`
    - Add `test.coverage` block with `provider: 'v8'`, `reporter: ['text', 'lcov']`, `thresholds: { lines: 80 }`
    - Install `@vitest/coverage-v8` as a dev dependency in `frontend/package.json`
    - _Requirements: 3.5_

  - [x] 3.3 Update `ci.yml` frontend job to run Vitest with coverage
    - Replace `npm run test` with `npm run test -- --coverage` in the `frontend-tests` job
    - Add `--coverage-threshold=80` or rely on `vite.config.ts` threshold to fail the job
    - _Requirements: 3.5, 3.6_

  - [x] 3.4 Add coverage badge to `README.md`
    - Add `![Coverage](https://img.shields.io/badge/coverage-80%25-brightgreen)` badge below the existing badges
    - _Requirements: 3.7_

- [ ] 4. Write missing property tests from adaptive-serving-tiers spec
  - [x] 4.1 Create `tests/test_serving_tier_properties.py` — Properties 1, 2, 3
    - **Property 1 — HardwareProfile type invariants:**
      - Use `@given` with `st.one_of(st.booleans(), st.just(Exception("mock")))` for each of the three mocked callables
      - Mock `torch.cuda.is_available`, `psutil.virtual_memory`, `os.cpu_count` via `unittest.mock.patch`
      - Assert `gpu_available` is `bool`, `ram_gb` is `float` and `> 0`, `cpu_cores` is `int` and `>= 1`
      - Use `@settings(max_examples=100)`
    - **Property 2 — Tier resolution totality:**
      - Use `@given(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False), st.booleans())`
      - Construct `HardwareProfile(gpu_available=gpu, ram_gb=ram, cpu_cores=4)`
      - Assert `TierDetector._auto_select(profile)` returns value in `{"tier1", "tier2", "tier3"}` and never raises
      - Use `@settings(max_examples=100)`
    - **Property 3 — Auto-selection boundary conditions:**
      - `ram_gb < 8.0` → assert result is `"tier3"`
      - `gpu_available=True AND ram_gb >= 16.0` → assert result is `"tier1"`
      - `gpu_available=False AND ram_gb >= 8.0` → assert result is `"tier2"`
      - Use `@settings(max_examples=100)` per property
    - _Requirements: 4.1, 4.2, 4.3, 4.6, 4.7_

  - [x] 4.2 Create `tests/test_onnx_thread_count.py` — Property 4
    - **Property 4 — ONNX thread count binding:**
      - Use `@given(st.integers(min_value=1, max_value=256))`
      - Mock `ort.InferenceSession` to capture `sess_options.intra_op_num_threads`
      - Instantiate `ONNXEngine(cpu_cores=n)` and call `load_model("test", "fake.onnx")`
      - Assert captured `intra_op_num_threads == n`
      - Use `@settings(max_examples=100)`
    - _Requirements: 4.4, 4.6, 4.7_

  - [-] 4.3 Create `tests/test_orjson_roundtrip.py` — Property 5
    - **Property 5 — orjson round-trip consistency:**
      - Use `@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(), st.none(), st.lists(st.integers()))))`
      - Import `_json_dumps` and `_json_loads` from `backend.main`
      - Assert `_json_loads(_json_dumps(payload)) == payload`
      - Use `@settings(max_examples=100)`
    - _Requirements: 4.5, 4.6, 4.7_

  - [x] 4.4 Add new test files to `ci.yml` unit-tests job
    - Add `tests/test_serving_tier_properties.py`, `tests/test_onnx_thread_count.py`, `tests/test_orjson_roundtrip.py` to the pytest command in the `unit-tests` job
    - _Requirements: 4.1–4.7_

- [x] 5. Add mutation testing
  - [x] 5.1 Create `.github/workflows/mutation-tests.yml`
    - Trigger: `workflow_dispatch` + weekly schedule (`cron: '0 10 * * 1'`)
    - Install `mutmut` and project dependencies
    - Run: `mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py --runner "python -m pytest tests/test_serving_tier_properties.py tests/test_onnx_thread_count.py -x -q"`
    - Run: `mutmut results` to print the mutation score
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 5.2 Document mutation testing in `README.md`
    - Add a "Mutation Testing" section with local run instructions:
      ```bash
      pip install mutmut
      mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py
      mutmut results
      ```
    - _Requirements: 5.4_

---

## Track 3 — Frontend Completeness

- [x] 6. Add frontend dependencies
  - [x] 6.1 Add `d3`, `jest-axe`, `@axe-core/react` to `frontend/package.json`
    - `d3`: pinned to `7.9.0`
    - `jest-axe`: pinned to `8.0.0`
    - `@axe-core/react`: pinned to `4.10.0`
    - Run `npm install` to update `package-lock.json`
    - _Requirements: 8.1, 12.1_

- [ ] 7. Create custom hooks
  - [-] 7.1 Create `frontend/src/hooks/useHealth.ts`
    - Fetch `/health` on mount; return `{ data, loading, error }`
    - Expose `servingTier`, `hardwareProfile`, `tierSelectionReason` from response
    - _Requirements: 6.2, 6.3_

  - [-] 7.2 Create `frontend/src/hooks/useSlo.ts`
    - Fetch `/v1/platform/slo` on mount; return `{ data, loading, error }`
    - On network error or 5xx: set `degraded=true`, return null data without throwing
    - _Requirements: 6.1, 6.4_

  - [x] 7.3 Create `frontend/src/hooks/useKnowledgeGraph.ts`
    - Accept `movieId: number | null`; fetch `/v1/recommendations/knowledge-graph/{movieId}` when non-null
    - Transform response into `{ nodes: GraphNode[], edges: GraphEdge[] }`
    - Return `{ graphData, loading, error }`
    - _Requirements: 8.1_

- [x] 8. Create Dashboard page
  - [x] 8.1 Create `frontend/src/pages/Dashboard.tsx`
    - Use `useHealth()` and `useSlo()` hooks
    - Render `TierBadge` component: green for tier1, blue for tier2, orange for tier3
    - Render `HardwareCard`: `gpu_available` (boolean chip), `ram_gb` (formatted to 1 decimal), `cpu_cores`
    - Render `SloMetrics`: p95 latency, error rate, request rate from SLO response
    - Render degraded banner when `useSlo` returns `degraded=true`
    - All values have accessible labels (`aria-label` or `<label>` associations)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

- [x] 9. Update RecommendationCard component
  - [x] 9.1 Update `frontend/src/components/RecommendationCard.tsx` (or equivalent)
    - Add `retrieval_stage` badge below the poster image when non-null
    - Add `retrieval_signals` key-value list when non-null (use `<dl>` for accessibility)
    - Add `explanation_text` paragraph below the movie title when non-null and non-empty
    - Omit explanation section entirely when `explanation_text` is null or empty string
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 10. Create Knowledge Graph page
  - [x] 10.1 Create `frontend/src/pages/KnowledgeGraph.tsx`
    - Movie search input to select seed movie (reuse existing search component)
    - Use `useKnowledgeGraph(selectedMovieId)` hook
    - Render D3 force-directed SVG graph: seed node (larger), rec nodes, edges labeled by `retrieval_stage`
    - Click handler on any node: show side panel with `title`, `poster_path` (as `<img>`), `overview`
    - Empty state: `<p role="status">No knowledge graph connections found for this movie.</p>`
    - Loading state: spinner while fetching
    - All interactive elements keyboard-accessible (`tabIndex`, `onKeyDown` handlers)
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

- [x] 11. Create Evaluation page
  - [x] 11.1 Create `frontend/src/pages/Evaluation.tsx`
    - Use `Promise.allSettled` to fetch semantic benchmark, recommendation benchmark, and offline metrics in parallel
    - Render each section independently — show partial results if one call fails
    - `MetricsTable` component: columns [Metric, Value, Threshold, Status (pass/fail chip)]
    - Show NDCG@k, MRR@k, Hit-Rate@k, Bad-Match-Rate@k from benchmark responses
    - Show offline metrics section (NDCG@10, Recall@50, ILD, Cold-Start NDCG@10) when available
    - Loading spinner per section while fetching
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 12. Create User Profile page
  - [x] 12.1 Create `frontend/src/pages/UserProfile.tsx`
    - Guard: if not authenticated, render `<LoginPrompt />` and return early
    - Fetch `/v1/events/features` for behavior features
    - Fetch `/v1/recommendations/user/{userId}?n=10` for personalized recommendations
    - `BehaviorCard`: display `total_ratings`, `avg_rating`, `click_count`, `view_count`
      - Validate each value: `value >= 0 ? value.toLocaleString() : "—"`
    - Render personalized recommendations using existing `RecommendationCard` grid
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

- [x] 13. Create Admin Panel page
  - [x] 13.1 Create `frontend/src/pages/AdminPanel.tsx`
    - Guard: if not admin, render `<p>Admin access required.</p>` and return early
    - "Reload Ensemble Weights" button → `POST /v1/admin/reload-ensemble-weights`
    - On success: render `WeightsTable` with model name and weight value for all 6 models
    - On any error (network, 401, 403, timeout, unexpected): render `ErrorBanner` with the error message
    - All errors caught in try/catch — no unhandled promise rejections
    - _Requirements: 11.1, 11.2, 11.3, 11.4_

- [x] 14. Add new routes and navigation
  - [x] 14.1 Update `frontend/src/App.tsx` (or router file) to add new routes
    - Add routes: `/dashboard`, `/knowledge-graph`, `/evaluation`, `/profile`, `/admin`
    - Add navigation links to the existing nav bar for all new pages
    - _Requirements: 6.1, 8.1, 9.1, 10.1, 11.1_

- [ ] 15. Add accessibility audit tests
  - [x] 15.1 Create `frontend/src/test/accessibility.test.tsx`
    - Import `axe` from `jest-axe` and `toHaveNoViolations` matcher
    - Test each of: `Dashboard`, `RecommendationPage` (or equivalent), `KnowledgeGraph`, `Evaluation`, `UserProfile`
    - For each page: render with `@testing-library/react`, run `axe(container)`, assert no critical or serious violations
    - Use `runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }`
    - _Requirements: 12.1, 12.2, 12.3, 12.4_

---

## Track 4 — Spec Completeness

- [x] 16. Complete `backend/onnx_engine.py` — cpu_cores wiring
  - [x] 16.1 Add `cpu_cores` parameter to `ONNXEngine.__init__`
    - Change signature to `def __init__(self, cpu_cores: int = 0):`
    - Store as `self._cpu_cores = cpu_cores`
    - _Requirements: 13.1_

  - [x] 16.2 Apply `cpu_cores` in `load_model` session options
    - In `load_model`, set `opts.intra_op_num_threads = self._cpu_cores` (0 = ONNX auto-detect)
    - _Requirements: 13.2_

  - [x] 16.3 Add `has_any_onnx_models` method
    - Add `def has_any_onnx_models(self) -> bool: return len(self.sessions) > 0`
    - _Requirements: 13.3_

  - [x] 16.4 Update `get_onnx_engine` singleton factory
    - Change signature to `def get_onnx_engine(cpu_cores: int = 0) -> ONNXEngine:`
    - Pass `cpu_cores` to `ONNXEngine(cpu_cores=cpu_cores)`
    - _Requirements: 13.4_

- [ ] 17. Verify `backend/ensemble_engine.py` — device placement completeness
  - [x] 17.1 Verify `__init__` calls `_move_to_device()` when `device != "cpu"`
    - Read the current `__init__` and confirm the conditional call exists
    - If missing, add: `if self._device != "cpu": self._move_to_device()`
    - _Requirements: 14.1, 14.2_

  - [x] 17.2 Verify `__init__` calls `_try_compile_all()` when `device == "cuda"`
    - Confirm: `if self._device == "cuda": self._try_compile_all()`
    - If missing, add it after `_move_to_device()`
    - _Requirements: 14.3_

  - [x] 17.3 Verify `get_apex_engine` passes `device` to constructor
    - Confirm `get_apex_engine(device=...)` passes `device` to `ApexEnsembleEngine(..., device=device)`
    - If missing, add the parameter
    - _Requirements: 14.7_

- [x] 18. Add Tier 3 constraints to `backend/recommender.py`
  - [x] 18.1 Import `resolve_serving_tier` and read active tier at start of `load()`
    - Add `from backend.serving_tier import resolve_serving_tier` import
    - At the top of `load()`, call `active_tier = resolve_serving_tier()`
    - _Requirements: 15.1_

  - [x] 18.2 Apply Tier 3 low_memory override
    - After reading `active_tier`, add: `if active_tier == "tier3": self._low_memory = True`
    - _Requirements: 15.2_

  - [x] 18.3 Cap TF-IDF vocabulary on Tier 3
    - After setting `_low_memory`, add:
      ```python
      if active_tier == "tier3":
          current_max = int(os.getenv("NOVA_TFIDF_MAX_FEATURES", "50000"))
          if current_max > 12000:
              os.environ["NOVA_TFIDF_MAX_FEATURES"] = "12000"
      ```
    - _Requirements: 15.3_

  - [x] 18.4 Defer sparse retrieval index on Tier 3
    - Wrap the `_build_sparse_retrieval_index()` call: `if active_tier != "tier3": self._build_sparse_retrieval_index()`
    - _Requirements: 15.4_

  - [x] 18.5 Skip Diffusion model loading on Tier 3
    - Wrap the Diffusion model load block: `if active_tier != "tier3": # load diffusion`
    - _Requirements: 15.5_

- [x] 19. Verify `backend/main.py` — lifespan and /health completeness
  - [x] 19.1 Verify lifespan calls `get_tier_detector().resolve()` before model loading
    - Confirm the existing lifespan code calls `_tier_detector.resolve()` at the top
    - If the call is missing or after model loading, move it to the top of the lifespan function
    - _Requirements: 16.1_

  - [x] 19.2 Verify tier-branched engine startup
    - Confirm tier1 branch calls `get_apex_engine(device=...)` and starts `OnlineLearner`
    - Confirm tier2 branch calls `get_onnx_engine(cpu_cores=N)` and checks `has_any_onnx_models()`
    - Confirm tier3 branch does no pre-loading
    - _Requirements: 16.2, 16.3, 16.4_

  - [x] 19.3 Verify `/health` returns tier fields
    - Confirm `HealthResponse` model has `serving_tier`, `hardware_profile`, `tier_selection_reason`
    - Confirm the `/health` handler populates these from `_tier_detector`
    - Confirm `detection_pending` fallback when `_tier_detector` is None
    - _Requirements: 16.5, 16.6_

---

## Final Checkpoint

- [x] 20. Run full test suite and verify all gates pass
  - Run: `pytest tests/ backend/tests/ -v --cov=backend --cov-fail-under=80`
  - Run: `cd frontend && npm run test -- --coverage`
  - Verify no test failures and both coverage gates pass
  - Run: `python scripts/run_offline_evaluation.py` and verify report is written
  - Verify `docs/APEX_WHITEPAPER.md` Section 6.1 contains actual metric values

---

## Notes

- Tasks 1.2 and 2.4 require executing scripts locally (or in CI) — they produce committed artifacts
- Track 3 tasks can be executed in parallel with Tracks 1, 2, and 4
- Task 4.2 (Property 4 — ONNX thread count) depends on Track 4 Task 16 being complete first
- The `d3` library for the Knowledge Graph page should be imported as an ES module (`import * as d3 from 'd3'`)
- All new React components must use semantic HTML elements for accessibility compliance
- The offline evaluation script may take 10–30 minutes depending on hardware; it should log progress per user batch

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "3.1", "3.2", "3.3", "3.4", "6.1", "16.1", "17.1", "18.1", "19.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "4.1", "4.2", "4.3", "7.1", "7.2", "7.3", "16.2", "16.3", "16.4", "17.2", "17.3", "18.2"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.4", "5.1", "5.2", "8.1", "9.1", "10.1", "11.1", "12.1", "13.1", "18.3", "18.4", "18.5", "19.2"] },
    { "id": 3, "tasks": ["2.4", "14.1", "15.1", "19.3"] },
    { "id": 4, "tasks": ["20"] }
  ]
}
```
