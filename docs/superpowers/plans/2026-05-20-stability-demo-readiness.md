# Stability And Demo Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the movie recommendation app reliably testable, deployable, and demo-ready without redesigning the product.

**Architecture:** Keep the existing FastAPI and React/Vite structure. Make narrow fixes in the files already owning auth, serving configuration, ETL helpers, model helpers, and frontend API routing. Avoid broad module extraction because the repo has many unrelated dirty changes.

**Tech Stack:** Python, FastAPI, SQLAlchemy, PySpark, PyTorch, Transformers CLIP, React, TypeScript, Vite, Docker, GitHub Actions.

---

## File Structure

- Modify `backend/auth.py`: add static `NOVA_API_KEYS` parsing while preserving database-backed API keys.
- Modify `backend/main.py`: keep one CORS middleware and one auth route implementation, and keep light endpoints from doing unnecessary heavy work.
- Modify `backend/vision_encoder.py`: normalize the tensor returned by CLIP even when Transformers returns a structured output.
- Modify `backend/ensemble_engine.py` or `backend/sasrec.py`: make SASRec item embedding dimensions consistent with constructor inputs and tests.
- Modify `etl/pyspark_etl.py`: replace SQL text ordering with Spark column APIs for null-safe descending order.
- Modify `.github/workflows/ci.yml`: install correct dependency files and run real tests.
- Modify `render.yaml` and `backend/Dockerfile` or root `Dockerfile`: align backend Docker build context with requirements and module import paths.
- Modify `frontend/src/api.ts`: derive API backends from Vite env, same-origin, and localhost fallback.
- Modify `frontend/src/AuthPage.tsx` and/or `frontend/src/main.tsx`: adjust demo/auth copy and search flow so the first-run experience is not misleading.
- Modify `frontend/src/styles.css`: remove embedded NUL bytes.
- Optionally modify `README.md` and `DEPLOYMENT.md`: correct stale commands that directly contradict the code.

---

### Task 1: Fix API-Key Resolution And Duplicate Backend Routes

**Files:**
- Modify: `backend/auth.py`
- Modify: `backend/main.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Run failing API tests**

Run:

```bash
python -m pytest tests/test_api.py::TestPlatformEndpoint::test_platform_context_requires_key_when_configured tests/test_api.py::TestEventsEndpoint::test_event_rejects_cross_tenant_payload_when_api_key_configured -q
```

Expected before fix: failures around 200 vs 401 and 401 vs 403.

- [ ] **Step 2: Add static API-key parsing in `backend/auth.py`**

Implement helper functions equivalent to:

```python
def _configured_static_api_keys() -> dict[str, TenantContext]:
    entries: dict[str, TenantContext] = {}
    raw = os.getenv("NOVA_API_KEYS", "").strip()
    if not raw:
        return entries
    for item in raw.split(","):
        parts = [part.strip() for part in item.split(":")]
        if len(parts) < 4 or not parts[0]:
            continue
        key, tenant_id, catalog_id, plan = parts[:4]
        entries[key] = TenantContext(
            tenant_id=tenant_id,
            catalog_id=catalog_id,
            plan=plan,
            authenticated=True,
            api_key_label="static",
        )
    return entries
```

Then update `resolve_tenant_context` so:

```python
static_keys = _configured_static_api_keys()
if static_keys:
    if not x_nova_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-Nova-API-Key")
    for expected_key, static_context in static_keys.items():
        if hmac.compare_digest(expected_key, x_nova_api_key):
            return TenantContext(
                tenant_id=static_context.tenant_id,
                catalog_id=x_catalog_id or static_context.catalog_id,
                plan=static_context.plan,
                authenticated=True,
                api_key_label=static_context.api_key_label,
            )
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid X-Nova-API-Key")
```

Leave the existing database API-key path after this branch.

- [ ] **Step 3: Remove duplicate mock auth routes from `backend/main.py`**

Delete the later duplicate block near the recommendation endpoints:

```python
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm

class UserRegister(BaseModel):
    username: str
    password: str

@app.post("/v1/auth/register")
async def register_user(user: UserRegister):
    return {"message": "User registered successfully", "username": user.username}

@app.post("/v1/auth/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    return {"access_token": form_data.username, "token_type": "bearer"}
```

Keep the earlier database-backed auth routes.

- [ ] **Step 4: Remove duplicate broad CORS middleware from `backend/main.py`**

Keep the later environment-driven CORS block with `ALLOWED_ORIGINS` and `ALLOWED_ORIGIN_REGEX`. Remove the two earlier hardcoded CORS registrations:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

and:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 5: Verify duplicate route removal**

Run:

```bash
python -c "import backend.main as m; print([r.path for r in m.app.routes if getattr(r, 'path', '') in ['/v1/auth/register','/v1/auth/token']])"
```

Expected after fix:

```text
['/v1/auth/register', '/v1/auth/token']
```

- [ ] **Step 6: Re-run API tests**

Run:

```bash
python -m pytest tests/test_api.py -q
```

Expected after fix: all tests in `tests/test_api.py` pass.

---

### Task 2: Fix Isolated ML And ETL Test Failures

**Files:**
- Modify: `backend/vision_encoder.py`
- Modify: `backend/ensemble_engine.py` or `backend/sasrec.py`
- Modify: `etl/pyspark_etl.py`
- Test: `backend/tests/test_multimodal.py`
- Test: `backend/tests/test_ensemble_math.py`
- Test: `tests/test_pyspark_scd.py`

- [ ] **Step 1: Run targeted failing tests**

Run:

```bash
python -m pytest backend/tests/test_ensemble_math.py::test_apex_ensemble_initialization backend/tests/test_multimodal.py::test_vision_encoder tests/test_pyspark_scd.py -q
```

Expected before fix: SASRec dimension assertion, CLIP output normalization error, and Spark `nulls last` parse errors.

- [ ] **Step 2: Fix CLIP output handling in `backend/vision_encoder.py`**

After `self.model.get_image_features(**inputs)`, ensure `image_features` is a tensor:

```python
with torch.no_grad():
    image_features = self.model.get_image_features(**inputs)

if not isinstance(image_features, torch.Tensor):
    if hasattr(image_features, "image_embeds") and image_features.image_embeds is not None:
        image_features = image_features.image_embeds
    elif hasattr(image_features, "pooler_output") and image_features.pooler_output is not None:
        image_features = image_features.pooler_output
    elif hasattr(image_features, "last_hidden_state"):
        image_features = image_features.last_hidden_state[:, 0, :]
    else:
        raise TypeError(f"Unsupported CLIP image feature output: {type(image_features)!r}")

image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
```

- [ ] **Step 3: Fix SASRec dimension consistency**

Use one convention: `SASRec(num_items=N)` means there are `N` content item IDs plus one padding row. In `backend/ensemble_engine.py`, instantiate:

```python
self.sasrec = SASRec(num_items=num_items, hidden_dim=emb_dim)
```

Do not add an extra `+ 1` before passing `num_items`; `SASRec` already creates `nn.Embedding(num_items + 1, hidden_dim, padding_idx=0)`.

- [ ] **Step 4: Fix Spark null ordering in `etl/pyspark_etl.py`**

Replace SQL text expressions in `dedupe_latest_movies`:

```python
order_columns = [
    col(column).cast("double").desc_nulls_last()
    for column in ("vote_count", "popularity")
    if column in df.columns
]
```

Leave the existing fallback:

```python
if not order_columns:
    order_columns = [desc(key_columns[0])]
```

- [ ] **Step 5: Re-run targeted ML/ETL tests**

Run:

```bash
python -m pytest backend/tests/test_ensemble_math.py::test_apex_ensemble_initialization backend/tests/test_multimodal.py::test_vision_encoder tests/test_pyspark_scd.py -q
```

Expected after fix: targeted tests pass or optional model-heavy vision test skips if Transformers/Pillow are unavailable.

---

### Task 3: Fix Deployment And CI Configuration

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `backend/Dockerfile`
- Modify: `render.yaml`
- Modify: `README.md`
- Modify: `DEPLOYMENT.md`

- [ ] **Step 1: Fix CI dependency install**

In `.github/workflows/ci.yml`, replace:

```yaml
pip install -r backend/requirements.txt
pip install pyspark delta-spark
```

with:

```yaml
pip install -r requirements.txt
pip install -r requirements-etl.txt
```

Then replace placeholder backend test commands with a real but bounded test command:

```yaml
python -m pytest tests/test_api.py backend/tests/test_ensemble_math.py backend/tests/test_multimodal.py tests/test_pyspark_scd.py -q
```

- [ ] **Step 2: Align Render with root Dockerfile**

In `render.yaml`, use the repository root Dockerfile so the Docker build has access to root `requirements.txt`, `frontend`, `backend`, `models`, and `data/processed`:

```yaml
dockerContext: .
dockerfilePath: ./Dockerfile
```

Use a low-memory serving profile by default for the free plan:

```yaml
- key: NOVA_SERVING_PROFILE
  value: lite
- key: NOVA_HEALTH_LOAD_RECOMMENDER
  value: "false"
```

- [ ] **Step 3: Make `backend/Dockerfile` internally consistent**

If keeping it, change the backend-only image to copy from the parent context or document that Render uses the root Dockerfile. The safer narrow fix is to keep `backend/Dockerfile` for manual backend-only builds but make the command import the module available in that context:

```dockerfile
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Avoid `--workers 4` in the backend-only file because this recommender is memory-heavy.

- [ ] **Step 4: Correct stale docs**

In `README.md`, replace:

```bash
python scripts/run_pipeline.py
```

with an existing command:

```bash
python scripts/rebuild_serving_artifacts.py
```

In `DEPLOYMENT.md`, mention that Render should use `render.yaml` or the root Dockerfile, not `backend/requirements.txt`.

- [ ] **Step 5: Verify config syntax**

Run:

```bash
docker compose config
```

Expected: Compose config parses. If Docker is unavailable, record that local Docker verification could not run.

---

### Task 4: Fix Frontend API Configuration And Demo UX

**Files:**
- Modify: `frontend/src/api.ts`
- Modify: `frontend/src/AuthPage.tsx`
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/src/styles.css`
- Test: `frontend/package.json` build script

- [ ] **Step 1: Clean NUL bytes from CSS**

Remove NUL bytes from `frontend/src/styles.css` without changing CSS content.

Expected check:

```powershell
$bytes=[System.IO.File]::ReadAllBytes('frontend\src\styles.css'); ($bytes | Where-Object { $_ -eq 0 }).Count
```

Expected output:

```text
0
```

- [ ] **Step 2: Replace hardcoded backends in `frontend/src/api.ts`**

Implement helper functions equivalent to:

```ts
function normalizeBackend(url: string | undefined): string | undefined {
  const value = url?.trim().replace(/\/+$/, "");
  return value || undefined;
}

function sameOriginBackend(): string | undefined {
  if (typeof window === "undefined") return undefined;
  return normalizeBackend(window.location.origin);
}

const configuredBackends = [
  import.meta.env.VITE_API_URL,
  import.meta.env.VITE_BACKUP_API_URL,
  sameOriginBackend(),
  "http://localhost:8000",
]
  .map(normalizeBackend)
  .filter((url): url is string => Boolean(url));

export const API_BASES = Array.from(new Set(configuredBackends));
let activeBackend = API_BASES[0] || "http://localhost:8000";
```

- [ ] **Step 3: Make auth copy honest**

In `frontend/src/AuthPage.tsx`, change password placeholder/copy from implying real password auth to demo profile access. Example:

```tsx
placeholder="Demo password"
```

and supporting text:

```tsx
<p>{isLogin ? "Enter a demo username to personalize this browser session" : "Create a demo profile for this browser"}</p>
```

- [ ] **Step 4: Keep search demo reachable**

In `frontend/src/main.tsx`, do not force the search page behind auth. Move the `if (!token) return <AuthPage ... />` gate so it applies only when the user explicitly opens profile or auth-only actions. Search and home should remain usable in demo mode.

Minimal behavior:

```tsx
if (!token && page === "profile") {
  return <AuthPage onLogin={(t, u) => { setToken(t); setUsername(u); }} />;
}
```

Remove the global auth gate that blocks search.

- [ ] **Step 5: Rebuild frontend**

Run:

```bash
bun run build
```

from `frontend`.

Expected: TypeScript and Vite build pass.

---

### Task 5: Final Verification

**Files:**
- No new implementation files.
- Verify all modified files.

- [ ] **Step 1: Run targeted backend tests**

Run:

```bash
python -m pytest tests/test_api.py backend/tests/test_ensemble_math.py::test_apex_ensemble_initialization backend/tests/test_multimodal.py::test_vision_encoder tests/test_pyspark_scd.py -q
```

Expected: all targeted tests pass, with optional skips only for unavailable optional dependencies.

- [ ] **Step 2: Run full tests when environment allows**

Run:

```bash
python -m pytest -q
```

Expected: full suite passes or only known optional integration skips remain.

- [ ] **Step 3: Run frontend build**

Run:

```bash
bun run build
```

from `frontend`.

Expected: build succeeds.

- [ ] **Step 4: Rendered smoke test**

Start Vite on a non-conflicting port:

```bash
npm run dev -- --host 127.0.0.1 --port 5174
```

Open `http://127.0.0.1:5174/` in the in-app browser. Verify:

- Page title is `Nova Recommendation Console`.
- DOM is not blank.
- No Vite/React error overlay is visible.
- Console has no relevant app errors.
- Clicking `Search Movies` reaches `/search` and shows the search UI, not an unexpected login wall.

- [ ] **Step 5: Summarize residual risk**

Report any verification that could not run, especially Docker daemon availability, optional model downloads, or full recommender warmup cost.
