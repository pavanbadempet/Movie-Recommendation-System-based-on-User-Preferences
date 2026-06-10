# Requirements Document

## Introduction

The APEX Final Polish spec closes the four remaining gaps that prevent the APEX Movie Recommendation System from achieving a 10/10 production-readiness rating. The four gap areas are:

1. **Feature Completeness** — The offline evaluation pipeline is missing, three property-based tests are not registered in CI, and the mutation testing workflow does not exist. (The frontend pages, hooks, and RecommendationCard updates from the apex-perfect-score spec are already present in the filesystem and wired into routing.)
2. **Repo Hygiene** — Secrets, backup files, database files, temporary artifacts, log files, and loose debug scripts are committed to the repository and must be removed from tracking. The `.gitignore` must be updated to prevent re-commitment.
3. **Production Deployment Documentation** — The live Render deployment runs in degraded tier3 mode (free plan, `NOVA_SERVING_PROFILE: lite`) but the README and docs do not document this gap or provide an upgrade path. The whitepaper contains "Pending offline eval run" placeholders.
4. **main.py Decomposition** — `backend/main.py` is approximately 2600 lines. Admin endpoints, benchmark caching logic, and large helper functions must be extracted into dedicated modules to restore maintainability.

---

## Glossary

- **APEX**: The APEX Movie Recommendation System — the FastAPI backend and React frontend that form the subject of this spec.
- **Offline_Evaluation_Pipeline**: The `scripts/run_offline_evaluation.py` script and its associated `GET /v1/evaluation/offline-metrics` API endpoint that compute leave-one-out ranking metrics against the MovieLens dataset.
- **Offline_Eval_Report**: The JSON file written to `reports/offline_eval_report.json` by the Offline_Evaluation_Pipeline, containing NDCG@10, Recall@50, ILD, and Cold-Start NDCG@10 metrics.
- **CI**: The GitHub Actions continuous integration pipeline defined in `.github/workflows/ci.yml`.
- **PBT**: Property-Based Test — a test that uses the Hypothesis library to generate arbitrary inputs and assert universal invariants.
- **Mutation_Testing**: A software testing technique that introduces small code mutations and verifies that the existing test suite detects them, measured by a mutation score.
- **Repo_Hygiene**: The practice of ensuring no secrets, generated artifacts, temporary files, or debug scripts are committed to the version-controlled repository.
- **Gitignore**: The `.gitignore` file at the repository root that specifies which files Git should not track.
- **Render_Deployment**: The live production deployment of the APEX backend hosted on Render.com, currently using the free plan with `NOVA_SERVING_PROFILE: lite`.
- **Deployment_Tier**: One of three hardware-aware serving configurations — tier1 (full GPU ensemble), tier2 (ONNX CPU), or tier3 (degraded lightweight mode).
- **Whitepaper**: The `docs/APEX_WHITEPAPER.md` document describing the APEX architecture and evaluation results.
- **Admin_Router**: The `backend/admin_routes.py` module that will contain admin-only FastAPI endpoints extracted from `backend/main.py`.
- **Benchmark_Cache**: The `backend/benchmark_cache.py` module that will contain benchmark caching state and helper functions extracted from `backend/main.py`.
- **Recommender_Helpers**: The `backend/recommender_helpers.py` module that will contain the `_reload_local_recommender`, `_refresh_artifact_files`, and `_background_recommender_warmup` functions extracted from `backend/main.py`.
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10 — a ranking quality metric.
- **Recall@50**: The fraction of users for whom the held-out test item appears in the top-50 recommendations.
- **ILD**: Intra-List Diversity — the mean pairwise cosine distance among the top-10 recommendations, measuring recommendation variety.
- **Cold_Start_NDCG@10**: NDCG@10 computed only for users with 5 or fewer training interactions.
- **Leave_One_Out**: An evaluation protocol where the most recent interaction for each user is held out as the test item and the remaining interactions are used for training.
- **mutmut**: The Python mutation testing tool used to run the Mutation_Testing workflow.
- **orjson**: A fast JSON serialization library used in the APEX serving path; the `_json_dumps` / `_json_loads` helpers in `backend/main.py` wrap it with a stdlib fallback.

---

## Requirements

### Requirement 1: Offline Evaluation Script

**User Story:** As a machine learning engineer, I want a reproducible offline evaluation script, so that I can measure recommendation quality against a held-out test set and report standardized ranking metrics.

#### Acceptance Criteria

1. THE Offline_Evaluation_Pipeline SHALL implement a Leave_One_Out split by holding out the most recent interaction for each user as the test item.
2. WHEN the Offline_Evaluation_Pipeline is executed, THE Offline_Evaluation_Pipeline SHALL load MovieLens 100K ratings from `data/raw/` and download the dataset if it is absent.
3. WHEN the Leave_One_Out split is complete, THE Offline_Evaluation_Pipeline SHALL compute NDCG@10 for each user using the formula `1.0 / log2(rank + 2)` when the test item appears in the top-10 ranked results, and 0 otherwise.
4. WHEN the Leave_One_Out split is complete, THE Offline_Evaluation_Pipeline SHALL compute Recall@50 as 1 if the test item appears in the top-50 ranked results and 0 otherwise, averaged across all users.
5. WHEN the Leave_One_Out split is complete, THE Offline_Evaluation_Pipeline SHALL compute ILD for each user's top-10 results by loading SBERT embeddings from `models/sbert_embeddings.npy` and computing the mean pairwise cosine distance, then averaging across all users.
6. WHEN the Leave_One_Out split is complete, THE Offline_Evaluation_Pipeline SHALL compute Cold_Start_NDCG@10 using the same formula as NDCG@10 but restricted to users whose training interaction count is dynamically computed to be 5 or fewer (not a pre-computed flag).
7. WHEN all metrics are computed, THE Offline_Evaluation_Pipeline SHALL write the Offline_Eval_Report to `reports/offline_eval_report.json` containing the fields: `generated_at`, `num_users`, `ndcg_at_10`, `recall_at_50`, `ild`, `cold_start_ndcg_at_10`, `evaluation_protocol`, and `model_version`.
8. THE Offline_Evaluation_Pipeline SHALL accept an `--output` CLI argument that overrides the default output path of `reports/offline_eval_report.json`.
9. WHEN the Offline_Eval_Report is written, THE Offline_Evaluation_Pipeline SHALL update `docs/APEX_WHITEPAPER.md` Section 6.1 by replacing each `| Pending offline eval run |` placeholder with the computed metric value formatted to 3 decimal places.
10. FOR ALL valid MovieLens rating datasets, running the Offline_Evaluation_Pipeline twice with the same dataset SHALL produce Offline_Eval_Reports with identical metric values (determinism property).

---

### Requirement 2: Offline Metrics API Endpoint

**User Story:** As a frontend developer, I want a REST endpoint that serves the offline evaluation results, so that the Evaluation page can display offline metrics alongside online benchmark metrics.

#### Acceptance Criteria

1. THE System SHALL expose a `GET /v1/evaluation/offline-metrics` endpoint in `backend/evaluation_routes.py`.
2. WHEN `GET /v1/evaluation/offline-metrics` is called and `reports/offline_eval_report.json` exists, THE System SHALL return the file contents as a JSON response with HTTP status 200.
3. WHEN `GET /v1/evaluation/offline-metrics` is called and `reports/offline_eval_report.json` does not exist, THE System SHALL return HTTP status 404 with the message `"Offline evaluation has not been run yet. Execute scripts/run_offline_evaluation.py first."`.
4. IF `reports/offline_eval_report.json` contains malformed JSON, THEN THE System SHALL return HTTP status 500 with a descriptive error message.
5. WHEN `GET /v1/evaluation/offline-metrics` is called, THE System SHALL verify the file is readable before committing to a 200 response, and SHALL return HTTP status 500 with a descriptive error message if the file exists but cannot be read due to permission or I/O errors.

---

### Requirement 3: CI Registration of Property-Based Tests

**User Story:** As a developer, I want all property-based tests to run on every push, so that regressions in universal invariants are caught automatically before merging.

#### Acceptance Criteria

1. THE CI SHALL include `tests/test_serving_tier_properties.py` in the `unit-tests` job pytest command.
2. THE CI SHALL include `tests/test_onnx_thread_count.py` in the `unit-tests` job pytest command.
3. THE CI SHALL include `tests/test_orjson_roundtrip.py` in the `unit-tests` job pytest command.
4. WHEN any of the three property-based test files fail, THE CI SHALL fail the `unit-tests` job and block the pull request from merging.
5. FOR ALL arbitrary recommendation-shaped dictionaries with string keys and int, float, string, None, or list-of-int values, THE orjson round-trip property SHALL satisfy `_json_loads(_json_dumps(payload)) == payload`.

---

### Requirement 4: Mutation Testing Workflow

**User Story:** As a quality engineer, I want an automated mutation testing workflow, so that I can measure how effectively the property-based tests detect logic errors in the serving tier and ONNX engine modules.

#### Acceptance Criteria

1. THE System SHALL provide a `.github/workflows/mutation-tests.yml` workflow file.
2. THE Mutation_Testing workflow SHALL be triggerable via `workflow_dispatch` and SHALL run on a weekly schedule (`cron: '0 10 * * 1'`).
3. WHEN the Mutation_Testing workflow runs, THE Mutation_Testing workflow SHALL install `mutmut` and run it against `backend/serving_tier.py` and `backend/onnx_engine.py` using `tests/test_serving_tier_properties.py` and `tests/test_onnx_thread_count.py` as the test runner.
4. WHEN the Mutation_Testing workflow completes, THE Mutation_Testing workflow SHALL print the mutation score via `mutmut results`.
5. THE README SHALL include a "Mutation Testing" section with local run instructions for `mutmut`.

---

### Requirement 5: Repository Hygiene — Remove Committed Artifacts

**User Story:** As a security engineer, I want secrets and generated artifacts removed from the repository, so that credentials are not exposed and the repository size stays manageable.

#### Acceptance Criteria

1. THE System SHALL remove `.env` from Git tracking (the file contains real secrets and must not be version-controlled; `.env.example` SHALL remain).
2. THE System SHALL remove `docker-compose.yml.bak` from Git tracking.
3. THE System SHALL remove `nova_db.sqlite3` from Git tracking.
4. THE System SHALL remove `benchmark_temp.json` from Git tracking.
5. THE System SHALL remove `movies_temp.parquet` from Git tracking.
6. THE System SHALL remove `frontend-vite.err.log` from Git tracking.
7. THE System SHALL remove `frontend-vite.log` from Git tracking.
8. THE System SHALL remove the loose root-level debug scripts `test_llm.py`, `test_recommendations.py`, `test_delta_implementation.py`, `final_verification.py`, and `verify_implementation.py` from Git tracking.
9. WHEN any of the above file patterns are created in the future, THE Gitignore SHALL proactively prevent Git from tracking them by having the patterns in place before the files are created.

---

### Requirement 6: Repository Hygiene — .gitignore Updates

**User Story:** As a developer, I want the .gitignore to cover all artifact categories that should never be committed, so that accidental re-commits are prevented without manual intervention.

#### Acceptance Criteria

1. THE Gitignore SHALL include a pattern that matches `*.bak` files to prevent backup files from being tracked.
2. THE Gitignore SHALL include a pattern that matches `frontend-vite*.log` and `frontend-vite*.err.log` files (or a broader `frontend/*.log` pattern) to prevent Vite log files from being tracked.
3. THE Gitignore SHALL include patterns for the specific root-level debug script names (`test_llm.py`, `test_recommendations.py`, `test_delta_implementation.py`, `final_verification.py`, `verify_implementation.py`) or a broader pattern covering root-level `test_*.py` and `verify_*.py` files.
4. WHEN a developer runs `git status` after creating any of the artifact types listed in Requirement 5, THE Gitignore SHALL cause Git to report those files as untracked rather than staged.

---

### Requirement 7: Production Deployment Documentation — Deployment Tiers

**User Story:** As a developer evaluating the project, I want the README to clearly document the deployment tier gap, so that I understand why the live demo runs in degraded mode and how to upgrade to the full ensemble.

#### Acceptance Criteria

1. THE README SHALL include a "Deployment Tiers" section that describes all three Deployment_Tiers (tier1, tier2, tier3) with their hardware requirements and capability differences.
2. THE README SHALL document that the current Render_Deployment uses `plan: free` with `NOVA_SERVING_PROFILE: lite`, which activates tier3 (degraded mode) rather than the full ensemble.
3. THE README SHALL provide an upgrade path showing the `render.yaml` changes required to enable tier1 or tier2 on a paid Render plan, including the required environment variable changes (`NOVA_SERVING_PROFILE`, `NOVA_SERVING_TIER`).
4. THE README SHALL include a table or list comparing the three Deployment_Tiers by: plan type, serving profile, active models, and expected latency range.

---

### Requirement 8: Production Deployment Documentation — Whitepaper Placeholders

**User Story:** As a technical reviewer, I want the whitepaper to contain actual metric values or clearly marked placeholders, so that the document accurately represents the system's evaluated performance.

#### Acceptance Criteria

1. WHEN `scripts/run_offline_evaluation.py` has been executed, THE Whitepaper SHALL contain the computed NDCG@10, Recall@50, ILD, and Cold_Start_NDCG@10 values in Section 6.1 formatted to 3 decimal places.
2. IF `scripts/run_offline_evaluation.py` has not been executed, THEN THE Whitepaper SHALL replace each `| Pending offline eval run |` placeholder with `| Requires local execution — run scripts/run_offline_evaluation.py |` to make the pending state explicit.
3. THE Whitepaper SHALL not contain the literal string `"Pending offline eval run"` in any committed version of the document.

---

### Requirement 9: main.py Decomposition — Admin Router

**User Story:** As a backend developer, I want admin endpoints extracted into a dedicated module, so that `backend/main.py` is smaller and admin logic is independently testable.

#### Acceptance Criteria

1. THE System SHALL create `backend/admin_routes.py` containing a FastAPI `APIRouter` with all admin-only endpoints currently in `backend/main.py`, including the reload-ensemble-weights endpoint and the reload-artifacts endpoint.
2. WHEN `backend/admin_routes.py` is imported and its router is registered in `backend/main.py`, THE System SHALL respond to all previously working admin endpoint paths with the same HTTP status codes and response schemas as before the extraction. The router SHALL be self-contained such that importing it without registering it does not cause import errors or side effects.
3. THE `backend/main.py` SHALL import and register the Admin_Router using `app.include_router(admin_router)`, and the CI `api-tests` job SHALL validate that all admin endpoints remain reachable after registration.
4. IF an unauthenticated request is made to any admin endpoint, THEN THE Admin_Router SHALL return HTTP status 401 or 403, preserving the existing authentication behavior.

---

### Requirement 10: main.py Decomposition — Benchmark Cache Module

**User Story:** As a backend developer, I want benchmark caching state and helpers extracted into a dedicated module, so that the caching logic is reusable and `backend/main.py` is easier to navigate.

#### Acceptance Criteria

1. THE System SHALL create `backend/benchmark_cache.py` containing the benchmark cache dictionaries, cache lock objects, TTL helper functions, and the `_compute_semantic_benchmark_cached`, `_compute_recommendation_benchmark_cached`, `_get_cached_semantic_benchmark`, `_get_cached_recommendation_benchmark`, `_start_background_semantic_benchmark`, and `_start_background_recommendation_benchmark` functions currently in `backend/main.py`.
2. WHEN `backend/benchmark_cache.py` is imported in `backend/main.py`, THE System SHALL respond to `GET /v1/evaluation/semantic-benchmark` and `GET /v1/evaluation/recommendation-benchmark` with the same caching behavior as before the extraction. IF the import is missing from `backend/main.py`, THE CI `api-tests` job SHALL fail with a clear import error rather than silently serving stale or uncached responses.
3. THE Benchmark_Cache module SHALL expose a public API that allows `backend/main.py` and `backend/evaluation_routes.py` to call cache read and write operations without accessing module-private state directly. WHEN behavior changes are detected during testing after extraction, THE CI SHALL log a warning but SHALL allow deployment to proceed.

---

### Requirement 11: main.py Decomposition — Recommender Helpers Module

**User Story:** As a backend developer, I want the large recommender helper functions extracted into a dedicated module, so that `backend/main.py` drops below 1500 lines and each module has a single clear responsibility.

#### Acceptance Criteria

1. THE System SHALL create `backend/recommender_helpers.py` containing the `_reload_local_recommender`, `_refresh_artifact_files`, and `_background_recommender_warmup` functions currently in `backend/main.py`.
2. WHEN `backend/recommender_helpers.py` is imported in `backend/main.py`, THE System SHALL respond to artifact reload admin endpoints with the same behavior as before the extraction.
3. AFTER the extraction of Admin_Router, Benchmark_Cache, and Recommender_Helpers, THE `backend/main.py` SHALL be fewer than 1500 lines as measured by `wc -l` or equivalent.
4. WHEN the full test suite is run after the decomposition, THE CI `unit-tests` and `api-tests` jobs SHALL pass with no new failures introduced by the refactoring.

---

### Requirement 12: Frontend Completeness Verification

**User Story:** As a frontend developer, I want to confirm that all planned pages, hooks, and component updates are correctly wired into the application, so that users can navigate to every feature from the main navigation bar.

#### Acceptance Criteria

1. THE `frontend/src/main.tsx` (or equivalent router file) SHALL define routes for `/dashboard`, `/knowledge-graph`, `/evaluation`, `/profile`, and `/admin`.
2. THE navigation bar SHALL include links to Dashboard, Knowledge Graph, Evaluation, User Profile, and Admin Panel pages.
3. WHEN a user navigates to `/knowledge-graph`, THE System SHALL render the `KnowledgeGraphPage` component without a runtime error.
4. WHEN a user navigates to `/evaluation`, THE System SHALL render the `EvaluationPage` component and display at least one metrics section.
5. WHEN a user navigates to `/admin` without admin credentials, THE AdminPanel component SHALL not render at all — it SHALL render only an access-denied message and return early, preventing any admin controls from being visible in the DOM.
6. THE `RecommendationCard` component SHALL render a `retrieval_stage` badge when the `retrieval_stage` field is non-null.
7. THE `RecommendationCard` component SHALL render a `<dl>` element with retrieval signal key-value pairs when the `retrieval_signals` field is non-null and non-empty.
8. THE `RecommendationCard` component SHALL render an `explanation_text` paragraph when the `explanation_text` field is non-null and non-empty, and SHALL omit the paragraph entirely when the field is null or an empty string.

---

### Requirement 13: Accessibility Compliance for New Pages

**User Story:** As a user with assistive technology, I want all new frontend pages to meet WCAG 2.1 AA standards, so that the application is usable with screen readers and keyboard navigation.

#### Acceptance Criteria

1. THE `frontend/src/test/accessibility.test.tsx` file SHALL contain axe-core accessibility tests for Dashboard, KnowledgeGraph, Evaluation, UserProfile, and AdminPanel pages, and these tests SHALL be included in the CI `frontend-tests` job so that accessibility failures block deployment.
2. WHEN axe-core runs against each new page component with `runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }`, THE System SHALL report zero critical or serious violations. Any violations discovered during development MUST be fixed before the tests are committed — zero tolerance is enforced at the CI gate.
3. THE KnowledgeGraph D3 SVG graph SHALL have all interactive node elements keyboard-accessible via `tabIndex` and `onKeyDown` handlers.
4. THE `RecommendationCard` retrieval signals `<dl>` element SHALL include an `aria-label` attribute with the value `"Retrieval signals"`.
