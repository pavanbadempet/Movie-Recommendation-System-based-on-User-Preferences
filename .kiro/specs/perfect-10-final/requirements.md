# Requirements: Perfect 10 Final

## Overview

Close the remaining gaps preventing a 10/10 rating across all five evaluation categories. The work is organized into four tracks: Kubernetes/Helm infrastructure, remaining property-based tests, frontend accessibility tests, and offline evaluation refresh. All four tracks are independent and can execute in parallel.

---

## Requirements

### 1. Kubernetes & Helm (DevOps: 9 → 10)

**1.1** A Helm chart MUST exist at `k8s/helm/apex/` with a `Chart.yaml` declaring `name: apex`, `version: 1.0.0`, `appVersion: 2.0.0`.

**1.2** `k8s/helm/apex/values.yaml` MUST define configurable values for: backend image (`image.repository`, `image.tag`), replica count (`replicaCount`), resource requests/limits (`resources.requests.cpu`, `resources.requests.memory`, `resources.limits.cpu`, `resources.limits.memory`), ingress (`ingress.enabled`, `ingress.host`), serving tier (`servingTier`), and secrets references (`secretRefs.jwtSecretKey`, `secretRefs.tmdbApiKey`).

**1.3** A Kubernetes Deployment manifest MUST exist at `k8s/helm/apex/templates/deployment.yaml` that: uses `{{ .Values.image.repository }}:{{ .Values.image.tag }}` as the image, sets `NOVA_SERVING_TIER` from `values.yaml`, defines liveness and readiness probes on `/health`, and applies resource requests and limits from values.

**1.4** A Kubernetes Service manifest MUST exist at `k8s/helm/apex/templates/service.yaml` exposing port 8000 as `ClusterIP` by default.

**1.5** A Kubernetes HorizontalPodAutoscaler manifest MUST exist at `k8s/helm/apex/templates/hpa.yaml` targeting 70% CPU utilization with `minReplicas: 1` and `maxReplicas: 10`.

**1.6** A Kubernetes Ingress manifest MUST exist at `k8s/helm/apex/templates/ingress.yaml` with TLS support, conditionally rendered via `{{ if .Values.ingress.enabled }}`.

**1.7** A `k8s/helm/apex/templates/NOTES.txt` MUST exist with post-install usage instructions including how to get the service URL and run a health check.

**1.8** A `k8s/README.md` MUST document: prerequisites (kubectl, helm 3.x), how to install with `helm install apex ./k8s/helm/apex`, how to upgrade, how to configure serving tier, and how to set secrets via `--set` or a Kubernetes Secret.

**1.9** The CI workflow MUST add a `helm-lint` job that runs `helm lint k8s/helm/apex/` to validate chart syntax on every push.

---

### 2. Remaining Property-Based Tests (Testing: 9 → 10)

**2.1** `tests/test_serving_tier_properties.py` MUST exist with at minimum Property 1 (HardwareProfile type invariants), Property 2 (tier resolution totality), and Property 3 (auto-selection boundary conditions) using `@given` with `@settings(max_examples=100)`.

**2.2** `tests/test_onnx_thread_count.py` MUST exist with Property 4 (ONNX thread count binding): for any `cpu_cores` in [1, 256], the session `intra_op_num_threads` MUST equal the passed value.

**2.3** `tests/test_orjson_roundtrip.py` MUST exist with Property 5 (orjson round-trip consistency): `_json_loads(_json_dumps(payload)) == payload` for arbitrary recommendation-shaped dicts.

**2.4** All three test files MUST be registered in the `unit-tests` job in `.github/workflows/ci.yml`.

**2.5** `tests/test_serving_tier_properties.py`, `tests/test_onnx_thread_count.py`, `tests/test_orjson_roundtrip.py` MUST each run to completion without errors under `pytest -x`.

---

### 3. Frontend Accessibility Tests (Testing: 9 → 10)

**3.1** `frontend/src/test/accessibility.test.tsx` MUST exist and test each major page component (Dashboard, KnowledgeGraph page shell, Evaluation, UserProfile) for WCAG 2.0 AA violations using `jest-axe` with `runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }`.

**3.2** The accessibility test file MUST use `@testing-library/react` to render each component and `axe()` to check for violations, asserting `expect(results).toHaveNoViolations()`.

**3.3** All accessibility tests MUST pass in the `frontend-tests` CI job with zero critical or serious violations.

**3.4** `jest-axe` and `@axe-core/react` MUST be present in `frontend/package.json` devDependencies (already true — verify they are used in the test file).

---

### 4. Offline Evaluation Refresh (ML Depth: 9 → 10)

**4.1** `scripts/run_offline_evaluation.py` MUST exist and be fully implemented: leave-one-out split, per-user recommendation via `recommender.recommend_by_id`, NDCG@10, Recall@50, ILD, cold-start NDCG@10 computation, and JSON report writing.

**4.2** `reports/offline_eval_report.json` MUST have a `generated_at` value no older than 6 months from the current date (i.e., the report must be regenerated as part of this spec).

**4.3** `docs/APEX_WHITEPAPER.md` Section 6.1 MUST contain actual computed metric values — no occurrences of the literal string `"Pending offline eval run"`.

**4.4** The offline eval script MUST accept `--output` CLI argument and default to `reports/offline_eval_report.json`.

**4.5** The script MUST log progress every 100 users and complete without unhandled exceptions on a machine with the serving artifacts present.

---

### 5. Code Quality Polish (Code Quality: 8.5 → 10)

**5.1** The sub-package `__init__.py` files in `backend/models/`, `backend/pipeline/`, `backend/serving/`, `backend/privacy/` MUST exist and export the correct public symbols (already done — verify each exports match the modules documented in their docstrings).

**5.2** `backend/metrics/__init__.py` MUST export at minimum `compute_ndcg`, `compute_hit_rate` from `backend/debiased_metrics.py` with a module docstring explaining the metrics sub-package purpose.

**5.3** All sub-package `__init__.py` files MUST be importable without errors (verified by a simple `python -c "from backend.models import LightGCN"` style smoke test in CI).

**5.4** A `docs/PACKAGE_STRUCTURE.md` MUST document the backend sub-package layout with a table mapping each sub-package to the modules it contains, the public symbols it exports, and the design rationale for the grouping.
