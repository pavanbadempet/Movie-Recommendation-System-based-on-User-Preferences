# Requirements Document

## Introduction

The APEX Movie Recommendation System currently scores approximately 9.1/10 across all evaluation dimensions. Five concrete gaps prevent it from reaching a genuine 10/10:

1. **DevOps gap** — No Kubernetes/Helm deployment artifacts exist; the system stops at Docker Compose and has no production-grade orchestration layer.
2. **Code quality gap** — The `backend/` package has ~80 flat-level files alongside existing sub-packages (`models/`, `pipeline/`, `serving/`, `privacy/`) that are empty shells, creating a navigation and maintenance burden.
3. **ML depth gap** — `reports/offline_eval_report.json` carries a hard-coded timestamp of `2024-01-15T10:00:00Z` (over two years stale) and no CI job regenerates it from live data.
4. **Testing completeness gap** — Several Hypothesis property-based tests exist as files in `tests/` but their completeness and CI registration need formal verification: Properties 1–3 (adaptive-serving-tiers), Property 4 (ONNX thread count), Property 5 (orjson round-trip), Property 10 (artifact validator idempotence), and Property 11 (ablation report serialization round-trip).
5. **Frontend accessibility gap** — `frontend/src/test/accessibility.test.tsx` exists but Task 15 in the apex-perfect-score spec is unchecked, meaning the axe WCAG 2.0 AA tests are not confirmed complete and passing for all required page components.

This spec closes all five gaps to bring APEX to a verifiable 10/10.

---

## Glossary

- **Helm_Chart**: A Helm package consisting of a `Chart.yaml`, `values.yaml`, and template YAML files that render Kubernetes manifests.
- **HPA**: HorizontalPodAutoscaler — a Kubernetes resource that automatically scales a Deployment's replica count based on CPU and memory utilization metrics.
- **Backend_Package**: The Python package rooted at `backend/` containing the FastAPI application and all supporting modules.
- **Sub_Package**: One of the existing sub-directories under `backend/` that already has an `__init__.py`: `models/`, `pipeline/`, `serving/`, `privacy/`.
- **Offline_Eval_Report**: The JSON file at `reports/offline_eval_report.json` produced by `scripts/run_offline_evaluation.py` using leave-one-out evaluation on MovieLens 100K.
- **CI**: The GitHub Actions continuous integration pipeline defined in `.github/workflows/ci.yml`.
- **Hypothesis_Property**: A `@given`-decorated test function that uses the Hypothesis library to verify a universally quantified behavioral property across randomly generated inputs.
- **axe**: The `axe-core` accessibility engine surfaced through the `jest-axe` adapter and used to detect WCAG violations in rendered React components.
- **WCAG_AA**: Web Content Accessibility Guidelines 2.0 Level AA — the accessibility standard the frontend must satisfy.
- **kubeval**: A tool that validates Kubernetes manifest YAML files against the Kubernetes JSON Schema.
- **Serving_Tier**: One of `tier1` (GPU), `tier2` (ONNX/CPU), or `tier3` (FAISS/lightweight), selected at startup based on detected hardware.

---

## Requirements

### Requirement 1: Kubernetes Helm Chart Creation

**User Story:** As a DevOps engineer, I want a complete Helm chart for the APEX system, so that I can deploy all services to a Kubernetes cluster with a single `helm install` command.

#### Acceptance Criteria

1. THE Helm_Chart SHALL exist at `helm/apex-recommendation/` and contain `Chart.yaml`, `values.yaml`, and a `templates/` directory.
2. THE Helm_Chart SHALL include templates for the following Kubernetes resources for the backend service: `Deployment`, `Service`, `HorizontalPodAutoscaler`, `ConfigMap`, and `Ingress`.
3. THE Helm_Chart SHALL include templates for the following Kubernetes resources for the frontend service: `Deployment`, `Service`, and `Ingress`.
4. WHEN the backend `Deployment` template is rendered, THE Helm_Chart SHALL include a liveness probe and a readiness probe both targeting the `/health` HTTP endpoint on port 8000.
5. WHEN the backend `Deployment` template is rendered with `serving_tier=tier1` in values, THE Helm_Chart SHALL set memory requests to at least 4Gi and memory limits to at least 8Gi.
6. WHEN the backend `Deployment` template is rendered with `serving_tier=tier2` or `serving_tier=tier3` in values, THE Helm_Chart SHALL set memory requests to at least 1Gi and memory limits to at least 2Gi.
7. THE HPA SHALL be configured to scale the backend `Deployment` from a minimum of 1 replica to a maximum of 5 replicas.
8. WHEN backend CPU utilization exceeds 70%, THE HPA SHALL trigger a scale-up of the backend `Deployment`.
9. WHEN backend memory utilization exceeds 80%, THE HPA SHALL trigger a scale-up of the backend `Deployment`.
10. THE `values.yaml` file SHALL define default values for image repository, image tag, replica counts, resource requests/limits, and ingress host.

---

### Requirement 2: Helm Chart CI Validation

**User Story:** As a DevOps engineer, I want the CI pipeline to validate the Helm chart syntax on every push, so that broken chart templates are caught before they reach a real cluster.

#### Acceptance Criteria

1. WHEN a pull request is opened or a commit is pushed to `main` or `develop`, THE CI SHALL run a job named `helm-validate` that executes `helm lint helm/apex-recommendation/`.
2. WHEN the `helm lint` command reports any errors, THE CI `helm-validate` job SHALL fail and block merging.
3. WHEN the `helm lint` command reports only warnings or succeeds, THE CI `helm-validate` job SHALL pass.
4. THE CI `helm-validate` job SHALL also run `kubeval` against the rendered chart templates to validate them against the Kubernetes API schema.
5. WHEN `kubeval` reports any strict schema violations, THE CI `helm-validate` job SHALL fail.
6. THE `helm-validate` job SHALL run after the `lint` job and in parallel with other test jobs so it does not increase the critical path duration.

---

### Requirement 3: Backend Domain Package Migration

**User Story:** As a backend engineer, I want the backend's model, pipeline, serving, and privacy modules organized into their correct sub-packages, so that the codebase is navigable and imports reflect the logical architecture.

#### Acceptance Criteria

1. THE Backend_Package SHALL migrate the following files from `backend/` (flat level) into `backend/models/`: `lightgcn.py`, `sasrec.py`, `kan_ranker.py`, `mmoe_ranker.py`, `hyperbolic_recommender.py`, `diffusion_recommender.py`, `neural_ode_recommender.py`, `two_tower.py`, `rl_policy.py`, and `attention_user_model.py`.
2. THE Backend_Package SHALL migrate the following files from `backend/` (flat level) into `backend/pipeline/`: `retrieval_pipeline.py`, `ranking_pipeline.py`, `reranking_pipeline.py`, and `pipeline_types.py`.
3. THE Backend_Package SHALL migrate the following files from `backend/` (flat level) into `backend/serving/`: `serving_tier.py`, `onnx_engine.py`, `model_loader.py`, and `ensemble_engine.py`.
4. THE Backend_Package SHALL migrate the following files from `backend/` (flat level) into `backend/privacy/`: `privacy.py` and `privacy_preserving_ml.py`.
5. WHEN any module is migrated, THE Backend_Package SHALL update `backend/models/__init__.py`, `backend/pipeline/__init__.py`, `backend/serving/__init__.py`, and `backend/privacy/__init__.py` to re-export all public symbols from the migrated modules so that existing `from backend.X import Y` call sites continue to work without modification.
6. WHEN migration is complete, THE Backend_Package SHALL have zero import errors when all public symbols are imported from their original paths (e.g., `from backend.serving.serving_tier import TierDetector` resolves via `backend.serving.__init__` re-export).
7. WHEN migration is complete, THE CI unit-tests job SHALL pass without modification to any test file import paths.

---

### Requirement 4: Offline Evaluation Report Freshness

**User Story:** As an ML engineer, I want `reports/offline_eval_report.json` to reflect a current evaluation run rather than a stale hard-coded timestamp, so that the report accurately represents the system's measured performance.

#### Acceptance Criteria

1. THE Offline_Eval_Report SHALL have a `generated_at` field value that is not equal to `"2024-01-15T10:00:00Z"`.
2. WHEN `scripts/run_offline_evaluation.py` is executed, THE Offline_Eval_Report SHALL be overwritten with a report containing a `generated_at` timestamp equal to the UTC wall-clock time at the moment of execution, formatted as an ISO 8601 string ending in `Z`.
3. WHEN `scripts/run_offline_evaluation.py` is executed, THE Offline_Eval_Report SHALL contain all eight required fields: `generated_at`, `num_users`, `ndcg_at_10`, `recall_at_50`, `ild`, `cold_start_ndcg_at_10`, `evaluation_protocol`, and `model_version`.
4. WHEN `scripts/run_offline_evaluation.py` executes successfully, THE Offline_Eval_Report SHALL have `num_users` equal to the number of users in the MovieLens 100K test split (610 users for leave-one-out).
5. WHEN a CI job runs `scripts/run_offline_evaluation.py` and the script exits with code 0, THE CI SHALL upload `reports/offline_eval_report.json` as a named workflow artifact called `offline-eval-report`.
6. THE CI SHALL contain a job named `offline-eval` that is triggered on `workflow_dispatch` and on a weekly schedule (Monday 06:00 UTC) so the report can be refreshed on demand or automatically.
7. IF `scripts/run_offline_evaluation.py` exits with a non-zero code, THEN THE CI `offline-eval` job SHALL fail and mark the run as failed.

---

### Requirement 5: Property-Based Test Completeness — Adaptive Serving Tiers

**User Story:** As a quality engineer, I want Properties 1, 2, 3, and 4 from the adaptive-serving-tiers spec to be fully implemented and passing, so that the serving tier's behavioral guarantees are machine-verified.

#### Acceptance Criteria

1. THE file `tests/test_serving_tier_properties.py` SHALL exist and contain a `@given`-decorated test that verifies Property 1 (HardwareProfile type invariants): for any combination of mocked `torch.cuda.is_available`, `psutil.virtual_memory`, and `os.cpu_count` return values (including exception-raising callables), `TierDetector.detect()` SHALL return a `HardwareProfile` where `gpu_available` is `bool`, `ram_gb` is `float` greater than 0, and `cpu_cores` is `int` greater than or equal to 1.
2. THE file `tests/test_serving_tier_properties.py` SHALL contain a `@given`-decorated test that verifies Property 2 (tier resolution totality): for any `HardwareProfile` with arbitrary `ram_gb` and `gpu_available` values, `TierDetector._auto_select()` SHALL return a value in `{"tier1", "tier2", "tier3"}` and SHALL never raise an exception.
3. THE file `tests/test_serving_tier_properties.py` SHALL contain `@given`-decorated tests that verify Property 3 (auto-selection boundary conditions): when `ram_gb < 8.0` the result is `"tier3"`, when `gpu_available=True` and `ram_gb >= 16.0` the result is `"tier1"`, and when `gpu_available=False` and `ram_gb >= 8.0` the result is `"tier2"`.
4. WHEN `tests/test_serving_tier_properties.py` is run with Hypothesis, each property-decorated test SHALL use `@settings(max_examples=100)`.
5. THE file `tests/test_onnx_thread_count.py` SHALL exist and contain a `@given`-decorated test that verifies Property 4 (ONNX thread count binding): for any `cpu_cores` value in `[1, 256]`, instantiating `ONNXEngine(cpu_cores=n)` SHALL configure the ONNX Runtime `InferenceSession` with `intra_op_num_threads` equal to `n`.
6. WHEN `tests/test_serving_tier_properties.py` and `tests/test_onnx_thread_count.py` are executed in CI, THE CI unit-tests job SHALL include both files in the explicit pytest file list.

---

### Requirement 6: Property-Based Test Completeness — Serialization Round-Trips

**User Story:** As a quality engineer, I want Properties 5, 10, and 11 (serialization round-trips and artifact validator idempotence) to be fully implemented and passing, so that data fidelity guarantees are machine-verified.

#### Acceptance Criteria

1. THE file `tests/test_orjson_roundtrip.py` SHALL exist and contain a `@given`-decorated test that verifies Property 5 (orjson round-trip consistency): for any recommendation-shaped dictionary with integer, float, string, null, and list values within orjson's 64-bit signed integer range, `_json_loads(_json_dumps(payload))` SHALL equal `payload`.
2. WHEN `tests/test_orjson_roundtrip.py` is executed, the Property 5 test SHALL use `@settings(max_examples=100)`.
3. THE file `tests/test_ablation_serialization_property.py` SHALL exist and contain a `@given`-decorated test that verifies Property 11 (ablation report serialization round-trip): for any `AblationReport` with random `full_ensemble_ndcg`, `run_timestamp`, and list of `ModelAblationResult` objects, serializing to JSON via `json.dumps(dataclasses.asdict(report))` and then deserializing SHALL produce a dict whose values are equal to the original fields within a tolerance of `1e-9` for floats.
4. WHEN `tests/test_ablation_serialization_property.py` is executed, the Property 11 test SHALL use `@settings(max_examples=100)`.
5. THE file `tests/test_validate_serving_artifacts.py` SHALL exist and contain at least one test covering the idempotence property (Property 10) of `_validate_heavy_artifact_contract`: calling the function twice on the same manifest SHALL produce identical output dictionaries.
6. WHEN `tests/test_orjson_roundtrip.py` and `tests/test_ablation_serialization_property.py` are executed in CI, THE CI unit-tests job SHALL include both files in the explicit pytest file list.

---

### Requirement 7: Frontend Accessibility Test Completeness

**User Story:** As a frontend engineer, I want the accessibility audit tests in `frontend/src/test/accessibility.test.tsx` to cover all required page components and pass with zero WCAG 2.0 AA violations, so that the frontend meets accessibility standards.

#### Acceptance Criteria

1. THE file `frontend/src/test/accessibility.test.tsx` SHALL exist and contain axe-based accessibility tests for the following page components: `Dashboard`, `KnowledgeGraphPage`, `EvaluationPage`, `UserProfilePage`, and `AdminPanel`.
2. WHEN any of the above page components is rendered in a test environment and checked with `axe(container, { runOnly: { type: "tag", values: ["wcag2a", "wcag2aa"] } })`, THE accessibility test SHALL assert `toHaveNoViolations()`.
3. THE accessibility test file SHALL mock all network hooks (`useHealth`, `useSlo`, `useKnowledgeGraph`) and API calls (`apiGet`, `apiPost`) so that no real HTTP requests are made during test execution.
4. WHEN `UserProfilePage` is tested, THE accessibility test SHALL cover both the unauthenticated state (token null) and the authenticated state (token non-null).
5. WHEN `AdminPanel` is tested, THE accessibility test SHALL cover both the unauthenticated state and the authenticated state.
6. WHEN `KnowledgeGraphPage` is tested, THE accessibility test SHALL cover both the empty-titles state and a state with a non-empty titles list.
7. WHEN the frontend test suite is run with `npm run test`, THE accessibility test file SHALL be discovered and executed automatically by Vitest without additional configuration.
8. WHEN any of the tested components produces a WCAG 2.0 A or AA violation as reported by axe, THE Vitest test run SHALL fail with a descriptive violation message.
9. THE `jest-axe` package (version `^8.0.0`) and `@axe-core/react` (version `^4.10.0`) SHALL be listed as dev dependencies in `frontend/package.json`.
