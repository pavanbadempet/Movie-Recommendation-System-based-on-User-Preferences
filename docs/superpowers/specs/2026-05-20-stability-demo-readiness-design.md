# Stability And Demo Readiness Design

## Context

The project is a full-stack movie recommendation system with a React/Vite frontend, FastAPI backend, ETL scripts, model artifacts, Docker/Render deployment files, and a broad test suite. The current worktree has many unrelated uncommitted changes, so implementation must stay tightly scoped and avoid reverting user work.

The audit found that the frontend builds, but the backend and deployment path are not reliable enough for a clean demo. The full test suite currently reports 240 passing tests, 4 skipped tests, and 7 failing tests. The failures cluster around auth/API-key behavior, PySpark SCD ordering, CLIP vision encoding, and SASRec ensemble dimensions.

## Goals

- Make the project easier to run, test, and deploy without changing the recommendation product direction.
- Fix the current high-confidence test failures that block trust in the codebase.
- Remove obvious duplicate backend registrations that create confusing behavior.
- Make the frontend configurable for local and hosted API URLs instead of hardcoding localhost.
- Improve first-run demo behavior when the API is unavailable or still warming.

## Non-Goals

- No broad rewrite of `backend/main.py` or `frontend/src/main.tsx`.
- No recommendation algorithm redesign.
- No large-scale artifact cleanup or deletion.
- No unrelated visual redesign.
- No reverting existing uncommitted changes.

## Backend Design

Use one active CORS setup and one active auth route implementation. Remove the later mock auth endpoints in `backend/main.py` so the database-backed auth flow is the only route registered for `/v1/auth/register` and `/v1/auth/token`.

Restore compatibility for `NOVA_API_KEYS` because existing tests and demo API-key behavior still expect it. The resolver should support configured static keys first, then fall back to database-backed API keys. When static API keys are configured, missing credentials should return 401 for protected context paths, and tenant/catalog mismatches should return 403.

Keep health and root endpoints light. `/health` should be able to report artifact health without forcing full recommender load when configured. Avoid adding new first-page frontend calls that trigger heavyweight model loading.

## Frontend Design

Replace hardcoded API base URLs with Vite environment-driven configuration:

- `VITE_API_URL` for the primary backend.
- `VITE_BACKUP_API_URL` for an optional backup backend.
- Same-origin API when the built frontend is served from the backend.
- Localhost fallback for local development.

Keep the existing UI structure. Improve first-run behavior by making unavailable/warming states actionable and by avoiding surprising login interception for the home-to-search demo path where possible. The auth flow can remain simple, but the UI should not imply real password security while using the demo password behavior.

Remove the NUL bytes from `frontend/src/styles.css` so source tools treat it as text.

## Test And Deployment Design

Make CI install the actual root `requirements.txt`, add ETL dependencies where needed, and run meaningful existing tests instead of placeholder echo commands. Keep the CI scope practical so it can complete on GitHub-hosted runners.

Fix the isolated failing tests:

- Vision encoder should handle the current Transformers CLIP output type and normalize a tensor.
- PySpark SCD ordering should use Spark column APIs rather than SQL text with unsupported `nulls last` syntax.
- SASRec ensemble dimensions should be internally consistent.
- Auth/API-key behavior should match tests and intended demo mode.

Align Docker/Render configuration so the backend build context has access to the correct requirements file and imports `backend.main:app` consistently.

## Verification

Run:

- `python -m pytest tests/test_api.py -q`
- targeted failing tests for vision, ensemble, and PySpark SCD
- `python -m pytest -q` if time and environment allow
- `npm run build` in `frontend`
- rendered frontend smoke test in the browser on a non-conflicting local port

## Risks

The repo includes many generated artifacts and a large dirty worktree. The implementation should avoid deleting artifacts or normalizing unrelated files. Full recommender warmup may still be slow after this pass; the target is to prevent basic health/demo paths from being blocked by that warmup, not to solve all model loading cost.
