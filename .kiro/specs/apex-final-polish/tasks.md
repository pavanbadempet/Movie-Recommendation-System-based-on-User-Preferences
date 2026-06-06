# Implementation Plan: APEX Final Polish

## Overview

Close the four remaining gaps to bring the APEX Movie Recommendation System to a 10/10 production-readiness rating. All frontend pages, hooks, and component updates are already implemented. The four tracks are independent and can be executed in parallel.

- **Track 1** — Feature Completeness: offline eval pipeline, offline metrics endpoint, missing PBT CI registration, mutation testing workflow
- **Track 2** — Repo Hygiene: untrack committed artifacts, update `.gitignore`
- **Track 3** — Production Deployment Docs: README Deployment Tiers section, whitepaper placeholder replacement
- **Track 4** — main.py Decomposition: extract admin routes, benchmark cache, recommender helpers

---

## Tasks

### Track 1 — Feature Completeness

- [x] 1. Create `scripts/run_offline_evaluation.py`
  - [x] 1.1 Implement dataset loading with auto-download
    - Load MovieLens 100K ratings from `data/raw/u.data` (tab-separated: user_id, item_id, rating, timestamp)
    - If `data/raw/u.data` is absent, download `https://files.grouplens.org/datasets/movielens/ml-100k.zip`, extract to `data/raw/`, and log `INFO`
    - Parse into a pandas DataFrame with columns `user_id`, `item_id`, `rating`, `timestamp`
    - Set `numpy.random.seed(42)` at the top of the script for determinism
    - _Requirements: 1.1, 1.2, 1.10_

  - [x] 1.2 Implement leave-one-out split
    - Sort all interactions by `(user_id, timestamp)` ascending
    - For each user, hold out the last interaction as the test item; the remaining interactions form the training set
    - Compute `cold_start_users`: set of user IDs whose training interaction count is ≤5 (computed dynamically from the split)
    - _Requirements: 1.1, 1.6_

  - [x] 1.3 Implement per-user recommendation and metric computation
    - For each user, call `recommender.recommend_by_id(last_training_item_id, n=50)` to get ranked candidates
    - Compute NDCG@10: `1.0 / log2(rank + 2)` if test item in top-10, else 0; average across all users
    - Compute Recall@50: 1 if test item in top-50, else 0; average across all users
    - Compute Cold-Start NDCG@10: same formula but restricted to `cold_start_users`; set to `null` if no cold-start users exist
    - Log progress every 100 users via `logger.info`
    - _Requirements: 1.3, 1.4, 1.6_

  - [x] 1.4 Implement ILD computation
    - Load `models/sbert_embeddings.npy`; if absent, set `ild = null` and log `WARNING`
    - For each user's top-10 results, compute mean pairwise cosine distance using `sklearn.metrics.pairwise.cosine_distances`
    - Average ILD across all users
    - _Requirements: 1.5_

  - [x] 1.5 Write Offline_Eval_Report and update whitepaper
    - Create `reports/` directory if absent
    - Write `reports/offline_eval_report.json` with fields: `generated_at` (ISO 8601 UTC), `num_users`, `ndcg_at_10`, `recall_at_50`, `ild`, `cold_start_ndcg_at_10`, `evaluation_protocol` (`"leave_one_out"`), `model_version` (from `APP_VERSION` or `"unknown"`)
    - After writing the report, open `docs/APEX_WHITEPAPER.md` and replace all occurrences of `| Pending offline eval run |` with the computed metric value formatted to 3 decimal places using `re.sub`
    - Accept `--output` CLI argument via `argparse`; default to `reports/offline_eval_report.json`
    - _Requirements: 1.7, 1.8, 1.9_

- [x] 2. Add `GET /v1/evaluation/offline-metrics` to `backend/evaluation_routes.py`
  - [x] 2.1 Add the endpoint inside `create_evaluation_router`
    - Add `from pathlib import Path` and `import json` imports at the top of the function (or module)
    - Implement `GET /v1/evaluation/offline-metrics` handler:
      - Check if `reports/offline_eval_report.json` exists; if not, raise `HTTPException(404, "Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first.")`
      - Read file contents; if `json.JSONDecodeError`, raise `HTTPException(500, f"Offline eval report contains malformed JSON: {exc}")`
      - If `OSError`, raise `HTTPException(500, f"Could not read offline eval report: {exc}")`
      - On success, return parsed JSON with HTTP 200
    - No authentication required — metrics are public read-only data
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 3. Create `tests/test_orjson_roundtrip.py`
  - [x] 3.1 Implement Property 5 — orjson round-trip consistency
    - Import `_json_dumps` and `_json_loads` from `backend.main`
    - Use `@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(), st.none(), st.lists(st.integers()))))` with `@settings(max_examples=100)`
    - Assert `_json_loads(_json_dumps(payload)) == payload` for all generated payloads
    - Include a unit test for an empty dict, a dict with nested list, and a dict with `None` values
    - _Requirements: 3.3, 3.5_

- [ ] 4. Register missing PBTs in `ci.yml` and create mutation testing workflow
  - [x] 4.1 Add three test files to the `unit-tests` job in `.github/workflows/ci.yml`
    - Add `tests/test_serving_tier_properties.py`, `tests/test_onnx_thread_count.py`, `tests/test_orjson_roundtrip.py` to the existing `python -m pytest` command in the `unit-tests` job
    - Preserve the existing `--cov-fail-under=80` and `-x` flags
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [-] 4.2 Create `.github/workflows/mutation-tests.yml`
    - Trigger: `workflow_dispatch` + `schedule: cron: '0 10 * * 1'`
    - Steps: checkout → setup Python 3.11 with pip cache → install `mutmut`, `pytest`, `hypothesis`, torch CPU, `requirements.txt` → run `mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py --runner "python -m pytest tests/test_serving_tier_properties.py tests/test_onnx_thread_count.py -x -q"` → run `mutmut results` with `if: always()`
    - Set `JWT_SECRET_KEY` and `NOVA_DISABLE_MODEL_DOWNLOADS` env vars on the run step
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

---

### Track 2 — Repo Hygiene

- [x] 5. Untrack committed artifacts and update `.gitignore`
  - [x] 5.1 Run `git rm --cached` for confirmed tracked files
    - Run `git rm --cached docker-compose.yml.bak` (confirmed tracked)
    - Run `git rm --cached test_llm.py` (confirmed tracked)
    - For each of `.env`, `nova_db.sqlite3`, `benchmark_temp.json`, `movies_temp.parquet`, `frontend-vite.err.log`, `frontend-vite.log`, `test_recommendations.py`, `test_delta_implementation.py`, `final_verification.py`, `verify_implementation.py`: run `git ls-files --error-unmatch <file>` first; only run `git rm --cached <file>` if the file is tracked
    - Do NOT delete the files from disk — only remove from Git tracking
    - _Requirements: 5.1–5.8_

  - [x] 5.2 Add missing patterns to `.gitignore`
    - Add `*.bak` under the "Local runtime state and temporary artifacts" section
    - Add the following under the existing "Temp debug scripts" section:
      ```
      test_llm.py
      test_recommendations.py
      test_delta_implementation.py
      final_verification.py
      verify_implementation.py
      ```
    - Verify `*.log` already covers log files (it does — present as `*.log` in the file)
    - Verify `*.sqlite3` already covers database files (it does — present as `*.sqlite3`)
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

---

### Track 3 — Production Deployment Documentation

- [x] 6. Add Deployment Tiers section to `README.md`
  - [x] 6.1 Insert the "🚀 Deployment Tiers" section after the "Quick Start" section
    - Add a table comparing Tier 1 / Tier 2 / Tier 3 by: plan type, serving profile, active models, expected latency range
    - Document that the current Render deployment uses `plan: free` with `NOVA_SERVING_PROFILE=lite` (Tier 3)
    - Provide the `render.yaml` environment variable changes needed to upgrade to Tier 2 (Standard plan) and Tier 1 (Pro GPU plan)
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [x] 6.2 Add "🧬 Mutation Testing" section to `README.md`
    - Add after the existing "🧪 Testing" section
    - Include local run instructions:
      ```bash
      pip install mutmut
      mutmut run --paths-to-mutate backend/serving_tier.py,backend/onnx_engine.py
      mutmut results
      ```
    - Note that the weekly CI workflow runs this automatically
    - _Requirements: 4.5_

- [x] 7. Replace whitepaper placeholders in `docs/APEX_WHITEPAPER.md`
  - [x] 7.1 Replace all `| Pending offline eval run |` occurrences
    - Open `docs/APEX_WHITEPAPER.md` and replace every occurrence of `| Pending offline eval run |` with `| Requires local execution — run scripts/run_offline_evaluation.py |`
    - Verify no occurrence of the literal string `"Pending offline eval run"` remains in the file after the replacement
    - _Requirements: 8.2, 8.3_

---

### Track 4 — main.py Decomposition

- [x] 8. Create `backend/benchmark_cache.py`
  - [x] 8.1 Extract benchmark cache state and all helper functions
    - Create `backend/benchmark_cache.py` with module-level private state: `_semantic_benchmark_cache`, `_semantic_benchmark_threads`, `_semantic_benchmark_cache_lock`, `_semantic_benchmark_compute_lock`, `_recommendation_benchmark_cache`, `_recommendation_benchmark_threads`, `_recommendation_benchmark_cache_lock`, `_recommendation_benchmark_compute_lock`
    - Move these functions from `main.py` into the new module (keeping the same logic): `_semantic_benchmark_ttl_seconds`, `_recommendation_benchmark_ttl_seconds`, `_warming_semantic_benchmark_report`, `_warming_recommendation_benchmark_report`, `_get_cached_semantic_benchmark`, `_compute_semantic_benchmark_cached`, `_background_semantic_benchmark`, `_start_background_semantic_benchmark`, `_get_cached_recommendation_benchmark`, `_compute_recommendation_benchmark_cached`, `_background_recommendation_benchmark`, `_start_background_recommendation_benchmark`
    - Export all functions as public (remove leading underscore from the public-facing ones: `get_cached_semantic_benchmark`, `compute_semantic_benchmark_cached`, `start_background_semantic_benchmark`, `warming_semantic_benchmark_report`, `get_cached_recommendation_benchmark`, `compute_recommendation_benchmark_cached`, `start_background_recommendation_benchmark`, `warming_recommendation_benchmark_report`)
    - _Requirements: 10.1_

  - [x] 8.2 Update `backend/main.py` to import from `benchmark_cache`
    - Add `from backend.metrics.benchmark_cache import (get_cached_semantic_benchmark, compute_semantic_benchmark_cached, start_background_semantic_benchmark, warming_semantic_benchmark_report, get_cached_recommendation_benchmark, compute_recommendation_benchmark_cached, start_background_recommendation_benchmark, warming_recommendation_benchmark_report)` to `main.py`
    - Remove the extracted functions and state from `main.py`
    - Update the `create_evaluation_router(...)` call in `main.py` to pass the imported functions
    - _Requirements: 10.2_

- [x] 9. Create `backend/recommender_helpers.py`
  - [x] 9.1 Extract recommender helper functions
    - Create `backend/recommender_helpers.py`
    - Move `_reload_local_recommender`, `_refresh_artifact_files`, `_background_recommender_warmup`, `_start_background_recommender_warmup`, `_temporary_env`, `_artifact_refresh_env`, `_event_logging_enabled`, `_safe_float` from `main.py` into the new module
    - For `_reload_local_recommender` and `_refresh_artifact_files`, use a `configure(get_rec, set_rec)` function to wire up the `_recommender` singleton accessors from `main.py` at startup
    - Export all functions as public (remove leading underscores from the public API)
    - _Requirements: 11.1_

  - [x] 9.2 Update `backend/main.py` to import from `recommender_helpers`
    - Add import of all extracted functions from `backend.pipeline.recommender_helpers`
    - Call `recommender_helpers.configure(lambda: _recommender, lambda r: globals().__setitem__('_recommender', r))` in the lifespan startup
    - Remove the extracted functions from `main.py`
    - _Requirements: 11.2_

- [x] 10. Create `backend/admin_routes.py`
  - [x] 10.1 Extract admin endpoints into a factory-pattern router
    - Create `backend/admin_routes.py` with a `create_admin_router(*, get_apex_engine, reload_local_recommender, refresh_artifact_files, ...) -> APIRouter` factory function
    - Move `POST /v1/admin/reload-ensemble-weights` from `main.py` into the router
    - Check `main.py` for any other endpoints guarded by `resolve_admin_token` and move them too
    - Preserve all existing authentication behavior (`resolve_admin_token` dependency)
    - The router must be self-contained: importing it without registering it must not cause import errors or side effects
    - _Requirements: 9.1, 9.2, 9.4_

  - [x] 10.2 Register the admin router in `backend/main.py`
    - Add `from backend.api.admin_routes import create_admin_router`
    - Instantiate and register: `app.include_router(create_admin_router(get_apex_engine=get_apex_engine, reload_local_recommender=reload_local_recommender, ...))`
    - Remove the extracted admin endpoints from `main.py`
    - _Requirements: 9.3_

- [ ] 11. Verify line count and run tests
  - [-] 11.1 Verify `backend/main.py` is under 1500 lines
    - Count lines: `python -c "print(sum(1 for _ in open('backend/main.py')))"`
    - If still over 1500, identify the next largest extractable block and move it to an appropriate module
    - _Requirements: 11.3_

  - [ ] 11.2 Run the full test suite to confirm no regressions
    - Run `pytest tests/test_api.py backend/tests/test_api_endpoints.py backend/tests/test_security_api.py -v --tb=short -x`
    - Run `pytest tests/ backend/tests/ -v --tb=short -q --cov=backend --cov-fail-under=80 -x`
    - All tests must pass; fix any import errors introduced by the refactoring before marking complete
    - _Requirements: 11.4_

---

## Notes

- Track 2 (repo hygiene) is the fastest track — complete it first to reduce noise in `git status`
- Track 3 (docs) is pure text editing — no code changes, no risk
- Track 1 task 1 (offline eval script) may take 10–30 minutes to run locally depending on hardware; it does not need to be executed as part of this spec — only the script needs to be created
- Track 4 tasks must be done in order: benchmark_cache → recommender_helpers → admin_routes → verify; each step depends on the previous to avoid broken imports
- The `create_evaluation_router` factory pattern in `evaluation_routes.py` means the offline-metrics endpoint (Task 2) does not need dependency injection — it reads a file directly
- `test_orjson_roundtrip.py` (Task 3) must be created before it can be added to `ci.yml` (Task 4.1)

---

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "5.1", "6.1", "7.1", "8.1"] },
    { "id": 1, "tasks": ["1.2", "5.2", "6.2", "8.2", "9.1"] },
    { "id": 2, "tasks": ["1.3", "3.1", "9.2", "10.1"] },
    { "id": 3, "tasks": ["1.4", "4.1", "10.2"] },
    { "id": 4, "tasks": ["1.5", "4.2", "11.1"] },
    { "id": 5, "tasks": ["2.1", "11.2"] }
  ]
}
```
