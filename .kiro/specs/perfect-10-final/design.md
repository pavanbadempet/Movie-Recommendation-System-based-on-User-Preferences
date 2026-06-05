# Design: Perfect 10 Final

## Overview

Four independent tracks close the remaining gaps:

- **Track 1 — Kubernetes/Helm**: A standard Helm chart for `k8s/helm/apex/` with Deployment, Service, HPA, Ingress, and a `helm-lint` CI job.
- **Track 2 — Property-Based Tests**: Three test files covering serving tier hardware detection, ONNX thread count binding, and orjson round-trip consistency.
- **Track 3 — Frontend Accessibility Tests**: `accessibility.test.tsx` using `jest-axe` to validate WCAG 2.0 AA compliance across all major pages.
- **Track 4 — Offline Evaluation Refresh**: Verify/complete `run_offline_evaluation.py` and regenerate the stale `reports/offline_eval_report.json`.
- **Track 5 — Package Structure Docs**: `backend/metrics/__init__.py` and `docs/PACKAGE_STRUCTURE.md`.

All tracks are independent and can execute in parallel.

---

## Track 1 — Kubernetes / Helm Chart Design

### Chart Structure

```
k8s/
  helm/
    apex/
      Chart.yaml
      values.yaml
      templates/
        deployment.yaml
        service.yaml
        hpa.yaml
        ingress.yaml
        NOTES.txt
  README.md
```

### `Chart.yaml`

```yaml
apiVersion: v2
name: apex
description: APEX Recommendation API — production-grade 6-model ensemble
type: application
version: 1.0.0
appVersion: "2.0.0"
```

### `values.yaml` Key Sections

```yaml
replicaCount: 1

image:
  repository: ghcr.io/your-username/apex-backend
  pullPolicy: IfNotPresent
  tag: "latest"

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: false
  host: apex.example.com
  tls: []

resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

servingTier: "tier2"  # tier1 | tier2 | tier3
servingProfile: "full"

autoscaling:
  minReplicas: 1
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

secretRefs:
  jwtSecretKey: ""    # Name of a k8s Secret key
  tmdbApiKey: ""
  adminToken: ""
```

### Deployment Template

The deployment sets `NOVA_SERVING_TIER` from `{{ .Values.servingTier }}`, reads secrets from `secretKeyRef` when `secretRefs.*` are non-empty, and defines:

- **Liveness probe**: `httpGet /health` after `initialDelaySeconds: 60`, `periodSeconds: 30`
- **Readiness probe**: `httpGet /health` after `initialDelaySeconds: 30`, `periodSeconds: 10`

### HPA

Targets the Deployment, `minReplicas: 1`, `maxReplicas: 10`, `targetCPUUtilizationPercentage: 70`.

### CI helm-lint Job

Added to `.github/workflows/ci.yml` as a lightweight job:

```yaml
helm-lint:
  name: Helm Chart Lint
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - uses: azure/setup-helm@v4
      with:
        version: v3.16.0
    - run: helm lint k8s/helm/apex/
```

No dependencies — runs in parallel with all other CI jobs.

---

## Track 2 — Property-Based Tests Design

### `tests/test_serving_tier_properties.py`

Three `@given` properties:

**Property 1 — HardwareProfile type invariants**
- `@given(st.booleans(), st.floats(min_value=0.1, max_value=1000.0), st.integers(min_value=1, max_value=256))`
- Construct `HardwareProfile(gpu_available, ram_gb, cpu_cores)` directly
- Assert `isinstance(h.gpu_available, bool)`, `isinstance(h.ram_gb, float)`, `h.ram_gb > 0`, `isinstance(h.cpu_cores, int)`, `h.cpu_cores >= 1`

**Property 2 — Tier resolution totality**
- `@given(st.floats(min_value=0.0, max_value=1000.0, allow_nan=False), st.booleans())`
- Construct `HardwareProfile(gpu_available=gpu, ram_gb=ram, cpu_cores=4)`
- Call `TierDetector()._auto_select(profile)` and assert result in `{"tier1", "tier2", "tier3"}`

**Property 3 — Auto-selection boundary conditions**
- Parametrized unit tests asserting exact tier for boundary inputs:
  - `ram_gb=4.0, gpu=False` → `tier3`
  - `ram_gb=8.0, gpu=False` → `tier2`
  - `ram_gb=16.0, gpu=True` → `tier1`
  - `ram_gb=16.0, gpu=False` → `tier2`

### `tests/test_onnx_thread_count.py`

**Property 4 — ONNX thread count binding**
- `@given(st.integers(min_value=1, max_value=256))`
- Mock `onnxruntime.InferenceSession` using `unittest.mock.patch`
- Instantiate `ONNXEngine(cpu_cores=n)` and call a method that triggers session creation
- Assert `SessionOptions().intra_op_num_threads` was called with `n`

### `tests/test_orjson_roundtrip.py`

**Property 5 — orjson round-trip consistency**
- `@given(st.dictionaries(st.text(min_size=1), st.one_of(st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text(), st.none(), st.lists(st.integers()))))`
- Import `_json_dumps`, `_json_loads` from `backend.main`
- Assert `_json_loads(_json_dumps(payload)) == payload`
- `@settings(max_examples=100)`

---

## Track 3 — Frontend Accessibility Tests Design

### `frontend/src/test/accessibility.test.tsx`

Uses `jest-axe` + `@testing-library/react`. Pattern per component:

```tsx
import { render } from '@testing-library/react'
import { axe, toHaveNoViolations } from 'jest-axe'
expect.extend(toHaveNoViolations)

it('Dashboard has no WCAG 2.0 AA violations', async () => {
  const { container } = render(<Dashboard />)
  const results = await axe(container, {
    runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa'] }
  })
  expect(results).toHaveNoViolations()
})
```

Components to test: `Dashboard`, `Evaluation`, `UserProfile`, `AdminPanel`, and the `KnowledgeGraph` page shell (without D3 rendering — pass empty mock data).

Each test wraps the component in necessary context providers (Router, auth context mock).

---

## Track 4 — Offline Evaluation Refresh Design

### Verification Steps

1. Confirm `scripts/run_offline_evaluation.py` is complete (all 5 metric computations implemented).
2. If any computation is missing, implement the missing section.
3. The script writes `reports/offline_eval_report.json` with a fresh `generated_at` timestamp.
4. After the script runs, `docs/APEX_WHITEPAPER.md` Section 6.1 is updated via the regex replacement logic already in the script.

### Report Schema

```json
{
  "generated_at": "<ISO 8601 UTC>",
  "num_users": <int>,
  "ndcg_at_10": <float>,
  "recall_at_50": <float>,
  "ild": <float | null>,
  "cold_start_ndcg_at_10": <float | null>,
  "evaluation_protocol": "leave_one_out",
  "model_version": "2.0.0"
}
```

---

## Track 5 — Package Structure Documentation Design

### `backend/metrics/__init__.py`

Exports from `backend/debiased_metrics.py`:
- `compute_ndcg` (or equivalent function name — check actual module)
- `compute_hit_rate`
- Module docstring explaining the metrics sub-package purpose (debiased evaluation metrics using IPS)

### `docs/PACKAGE_STRUCTURE.md`

A markdown document with:

1. A table of all 5 sub-packages (models, pipeline, serving, privacy, metrics)
2. Per sub-package: modules contained, public exports, design rationale
3. A note on the "re-export pattern" — why source files remain flat in `backend/` while sub-packages provide logical namespacing
4. An import graph showing the dependency flow between sub-packages
