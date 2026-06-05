# APEX 10/10 Startup Product Roadmap

> **Goal:** Bring every product dimension from its current score to 10/10.
> This document covers the five gaps the existing technical specs do not address:
> Monetization, Product Identity, Onboarding, Demo Quality, and Code Maintainability.
> Technical specs (ML, CI, testing, architecture) are tracked separately in `.kiro/specs/`.

---

## Current Scorecard

| Dimension | Current | Target |
|---|---|---|
| Technical depth & ML architecture | 9.5 | 10 |
| Engineering depth & infra | 9.0 | 10 |
| DevOps / CI / Observability | 9.5 | 10 |
| Testing | 9.0 | 10 |
| Code organization & maintainability | 6.0 | 10 |
| Documentation | 9.0 | 10 |
| Frontend | 7.5 | 10 |
| **Monetization** | **2.0** | **10** |
| **Product definition & ICP clarity** | **4.0** | **10** |
| **Onboarding & self-serve** | **3.0** | **10** |
| **Demo & first impression** | **4.0** | **10** |

---

## Priority Order

Do these in this sequence. Earlier items unblock or amplify later ones.

| # | Item | Effort | Immediate Impact |
|---|---|---|---|
| 1 | Upgrade Render to Tier 2 | 30 min | Demo works properly today |
| 2 | Repo cleanup (artifacts, naming) | 2 hr | Professionalism signal |
| 3 | Self-serve signup + API key flow | 1 week | Customers can start using it |
| 4 | Stripe billing integration | 2–3 weeks | Revenue path exists |
| 5 | Landing page + value prop rewrite | 1 week | First impression |
| 6 | Quickstart / Getting Started docs | 3 days | Reduces onboarding friction |
| 7 | Pre-loaded demo data + demo video | 1 day | Pitch quality |
| 8 | RouterDeps refactor | 1 day | Long-term maintainability |
| 9 | Status page | 2 days | Customer trust signal |

---

## Gap 1 — Demo & First Impression (4/10 → 10/10)

### 1.1 Upgrade Render deployment to Tier 2

**File:** `render.yaml`

Change:
```yaml
plan: free
envVars:
  - key: NOVA_SERVING_PROFILE
    value: lite
```

To:
```yaml
plan: standard
envVars:
  - key: NOVA_SERVING_PROFILE
    value: full
  - key: NOVA_SERVING_TIER
    value: tier2
```

This activates ONNX inference (200–800 ms) instead of FAISS-only (800–2000 ms).
Cost: ~$25/month. Highest ROI action in this entire document.

---

### 1.2 Commit pre-built demo artifacts

**Files:** `models/`, `scripts/rebuild_serving_artifacts.py`

- Run `python scripts/rebuild_serving_artifacts.py` locally against the TMDB catalog
- Commit the resulting `models/faiss.index`, `models/sbert_embeddings.npy`,
  `models/tfidf_vectorizer.joblib`, `models/pipeline_manifest.json`
- The Render deployment must start with working recommendations without requiring a
  data pipeline run. Currently it returns empty results on first boot.
- Add `POST /v1/demo/reset` (admin-only) that reloads demo artifacts from disk

---

### 1.3 Record and embed demo video

**Files:** `README.md`, `frontend/src/pages/Landing.tsx`

Record a 90-second screen capture showing:
1. Dashboard page — serving tier badge, hardware profile card, SLO metrics
2. A recommendation request with `?explain=true` returning an LLM explanation
3. The Knowledge Graph visualization with a real movie
4. Offline eval metrics table

Embed the video on the landing page and link from the README hero section.

---

### 1.4 Add a public status page

**Files:** `frontend/src/pages/Status.tsx`, `frontend/src/App.tsx`

- New route `/status`
- Fetches `/health` and `/v1/platform/slo` — both endpoints already exist
- Renders: API status (green/red), current serving tier, p95 latency (last 24 h), error rate
- No auth required — this is a public trust signal
- Link from footer of landing page and README

---

## Gap 2 — Code Maintainability (6/10 → 10/10)

### 2.1 Delete committed junk files

Run these commands and commit the result:

```bash
git rm --cached docker-compose.yml.bak
git rm --cached frontend-vite.err.log
git rm --cached frontend-vite.log
git rm --cached nova_db.sqlite3
git rm --cached scripts/_slim_recommender.py
git rm --cached scripts/_slim_recommender2.py
git rm --cached scripts/_slim_recommender3.py
```

Then add to `.gitignore`:
```
*.bak
*.sqlite3
frontend-vite*.log
scripts/_slim_recommender*.py
```

---

### 2.2 Unify Nova/APEX naming

**Files:** `docker-compose.yml`, `render.yaml`, `backend/database.py`,
`sql/postgres_init.sql`, `.env.example`, `README.md`

The product is called **APEX**. The codebase uses "Nova" in:
- Container names (`nova-backend`, `nova-frontend`)
- Environment variables (`NOVA_SERVING_PROFILE`, `NOVA_API_KEYS`, `NOVA_TENANT_ID`)
- Database defaults (`nova_user`, `nova_password`, `nova_db`)
- Default tenant seed (`Demo Media Co`)

**Decision:** Keep `NOVA_` env var prefix (breaking change to rename it) but update all
human-facing strings — container names, DB names, seed data, docs — to say APEX.

Changes:
- `nova-backend` → `apex-backend` in `docker-compose.yml`
- `nova_db.sqlite3` → `apex.db` in `backend/database.py` default
- `"Demo Media Co"` seed → `"APEX Demo Tenant"` in `sql/postgres_init.sql`
- README repo description updated from "Movie Recommendation System" to "APEX API"

---

### 2.3 RouterDeps dataclass refactor

**Files:** `backend/main.py`, `backend/recommendation_routes.py`,
`backend/evaluation_routes.py`, `backend/browse_routes.py`

`create_recommendation_router` and related factories are called with ~40 keyword
arguments each. Replace with a shared dataclass:

```python
# backend/router_deps.py
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class RouterDeps:
    get_rec: Callable
    record_usage: Callable
    resolve_tenant_context: Callable
    remote_payload_or_raise: Callable
    record_recommendation_events: Callable
    build_user_behavior_profile: Callable
    assign_experiment: Callable
    attach_experiment: Callable
    aggregate_behavior_features: Callable
    append_event: Callable
    summarize_recommendation_events: Callable
    evaluate_artifact_health: Callable
    load_ranker: Callable
    enforce_payload_context: Callable
    get_db: Callable
    generate_chat_response: Callable
    summarize_usage: Callable
    event_storage_status: Callable
    get_events_path: Callable
    limiter: Any
```

Each router factory then takes `deps: RouterDeps` as a single argument.
`main.py` constructs one `RouterDeps` instance and passes it to all routers.

This is a purely mechanical refactor — no behavior changes.

---

## Gap 3 — Monetization (2/10 → 10/10)

### 3.1 Create `backend/billing.py` — Stripe integration

```python
# backend/billing.py

import stripe
import os

stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

PRICE_IDS = {
    "pro": os.getenv("STRIPE_PRICE_PRO"),        # e.g. price_xxx
    "enterprise": os.getenv("STRIPE_PRICE_ENT"),
}

def create_checkout_session(tenant_id: str, plan: str, success_url: str, cancel_url: str) -> str:
    """Returns a Stripe Checkout URL for the given plan."""
    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": PRICE_IDS[plan], "quantity": 1}],
        metadata={"tenant_id": tenant_id},
        success_url=success_url,
        cancel_url=cancel_url,
    )
    return session.url

def create_portal_session(stripe_customer_id: str, return_url: str) -> str:
    """Returns a Stripe Customer Portal URL for self-serve plan management."""
    session = stripe.billing_portal.Session.create(
        customer=stripe_customer_id,
        return_url=return_url,
    )
    return session.url

def handle_webhook(payload: bytes, sig_header: str) -> dict:
    """Process Stripe webhook. Returns event dict or raises."""
    event = stripe.Webhook.construct_event(
        payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
    )
    return event
```

**New API routes** (add to `backend/auth_routes.py` or new `backend/billing_routes.py`):
- `POST /v1/billing/checkout` — returns Stripe Checkout URL
- `POST /v1/billing/portal` — returns Stripe Customer Portal URL
- `POST /v1/billing/webhook` — Stripe webhook receiver (no auth, verified by signature)

**Webhook handler logic:**
- `invoice.paid` → update `dim_tenant.plan_tier` to the subscribed plan
- `customer.subscription.deleted` → downgrade `dim_tenant.plan_tier` to `'free'`
- `customer.subscription.updated` → sync plan tier

**New env vars to add to `.env.example`:**
```ini
STRIPE_SECRET_KEY=sk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PRO=price_...
STRIPE_PRICE_ENT=price_...
```

---

### 3.2 Add `stripe_customer_id` to `dim_tenant`

**Files:** `sql/migrations/`, `backend/database.py`

```sql
-- sql/migrations/V2__add_stripe_customer_id.sql
ALTER TABLE dim_tenant ADD COLUMN stripe_customer_id VARCHAR(255);
ALTER TABLE dim_tenant ADD COLUMN subscription_id VARCHAR(255);
```

Add corresponding columns to the `Tenant` ORM model in `backend/database.py`.

---

### 3.3 Create plan enforcement middleware

**File:** `backend/middleware/plan_enforcer.py`

```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

DAILY_LIMITS = {
    "free": 100,
    "pro": 10_000,
    "enterprise": None,  # unlimited
}

class PlanEnforcerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            plan = await get_tenant_plan(tenant_id)  # Redis cache → Postgres fallback
            limit = DAILY_LIMITS.get(plan)
            if limit is not None:
                usage = await get_daily_usage(tenant_id)
                if usage >= limit:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Daily request limit reached",
                            "plan": plan,
                            "limit": limit,
                            "upgrade_url": "/v1/billing/checkout?plan=pro",
                        }
                    )
        return await call_next(request)
```

Register in `main.py` after the existing `RedisRateLimiter` middleware.

---

### 3.4 Add pricing page to frontend

**File:** `frontend/src/pages/Pricing.tsx`

Three pricing cards side by side:

| Free | Pro | Enterprise |
|---|---|---|
| 100 req/day | 10,000 req/day | Unlimited |
| Tier 3 (FAISS only) | Tier 2 (ONNX ensemble) | Tier 1 (GPU ensemble) |
| Community support | Email support | SLA + dedicated support |
| $0 | $299/month | Contact us |

- "Get started free" → `/signup`
- "Start Pro trial" → `POST /v1/billing/checkout?plan=pro`
- "Contact sales" → mailto or Calendly link

Add route `/pricing` in `frontend/src/App.tsx`.

---

## Gap 4 — Product Identity & ICP Clarity (4/10 → 10/10)

### 4.1 Define and write the ICP

**Target customer:**
> Streaming and media platforms with 10,000–5,000,000 users that need
> Netflix-quality personalized recommendations without building and maintaining
> a dedicated ML infrastructure team.

**One-line value proposition:**
> "APEX gives your platform the same recommendation quality as Netflix —
> without the ML team. Plug in your catalog, get an API key, and go live in 30 minutes."

---

### 4.2 Rewrite `README.md` hero section

Replace the current opening (which leads with architecture) with:

```markdown
# APEX — Recommendation API

> Netflix-quality recommendations for your platform. No ML team required.

APEX is an API that gives any streaming or media product personalized,
explainable recommendations. Powered by a 6-model ensemble (HR@10 = 0.785),
differential privacy, and adaptive GPU/CPU/edge serving.

**Get started in 30 minutes:**
1. [Sign up](https://your-domain.com/signup) and get your API key
2. Upload your catalog via `/v1/catalog/upload`
3. Call `/v1/recommendations/id/{item_id}` from your backend

[View live demo](https://your-render-url.onrender.com/docs) ·
[Read the docs](docs/QUICKSTART.md) ·
[See pricing](https://your-domain.com/pricing)
```

Move the architecture diagram and model details to a collapsible section or to `docs/ARCHITECTURE.md`.

---

### 4.3 Create landing page

**File:** `frontend/src/pages/Landing.tsx`

Sections (in order):
1. **Hero** — value prop headline, sub-headline, "Get API key" CTA, demo video embed
2. **Social proof** — benchmark numbers: HR@10 = 0.785, NDCG@10 = 0.542, Semantic HR@10 = 1.0
3. **How it works** — 3 steps with icons: Upload catalog → Make API call → Improve over time
4. **Feature highlights** — 6 cards: Ensemble ML, Multi-modal search, Knowledge graph,
   Differential privacy, LLM explanations, Adaptive serving
5. **Pricing** — 3-tier pricing cards (links to `/pricing`)
6. **Footer** — links: Docs, API Reference, Status, GitHub

Add route `/` → `Landing` in `frontend/src/App.tsx` (currently `/` goes to the API root).

---

## Gap 5 — Onboarding & Self-Serve (3/10 → 10/10)

### 5.1 Self-serve signup flow

**File:** `frontend/src/pages/Signup.tsx`

Steps:
1. Email + password form → `POST /v1/auth/register`
2. On success: auto-create tenant (`plan_tier='free'`), auto-generate hashed API key
3. Show API key once with copy button + warning "Store this — you won't see it again"
4. Redirect to `/getting-started`

**Backend changes** (`backend/auth_routes.py`):
- After user registration, create a `Tenant` row and an `APIKey` row in the same transaction
- Return the plaintext API key in the registration response (only time it's returned unmasked)

---

### 5.2 Getting Started page

**File:** `frontend/src/pages/GettingStarted.tsx`

Four-step interactive wizard:

**Step 1 — Your API key**
- Show masked key with copy button
- Link to generate a new key if needed

**Step 2 — Upload your catalog**
- Drag-and-drop CSV uploader wired to `POST /v1/catalog/upload`
- Expected columns: `item_id`, `title`, `description` (optional: `genres`, `poster_url`)
- Show upload progress and row count on success

**Step 3 — Make your first call**
- Code snippet with the user's real API key pre-filled:
```bash
curl -H "X-Nova-API-Key: YOUR_KEY" \
  "https://your-api.onrender.com/v1/recommendations/id/550"
```
- Toggle between curl / Python / JavaScript
- "Try it now" button that fires the request and shows the live response inline

**Step 4 — Explore the dashboard**
- Link to `/dashboard`
- Show serving tier, SLO status, request count

---

### 5.3 Create `docs/QUICKSTART.md`

```markdown
# APEX Quickstart — Your first recommendation in 5 minutes

## 1. Get your API key
Sign up at https://your-domain.com/signup.
Your API key is shown once. Store it securely.

## 2. Upload your catalog
curl -X POST https://api.your-domain.com/v1/catalog/upload \
  -H "X-Nova-API-Key: YOUR_KEY" \
  -F "file=@catalog.csv"

## 3. Get recommendations
curl "https://api.your-domain.com/v1/recommendations/id/1?n=10" \
  -H "X-Nova-API-Key: YOUR_KEY"

## 4. Get an explanation
curl "https://api.your-domain.com/v1/recommendations/id/1?explain=true" \
  -H "X-Nova-API-Key: YOUR_KEY"

## Next steps
- [Full API reference](API_REFERENCE.md)
- [Manage your plan](https://your-domain.com/pricing)
- [View your dashboard](https://your-domain.com/dashboard)
```

---

## Implementation Checklist

Use this as your day-by-day execution tracker.

### Today (< 2 hours)
- [ ] Upgrade `render.yaml` to `plan: standard`, set `NOVA_SERVING_TIER=tier2`
- [ ] `git rm --cached` the committed junk files listed in 2.1
- [ ] Add missing patterns to `.gitignore`

### This week
- [ ] Build self-serve signup flow (5.1) — frontend + backend auth changes
- [ ] Build Getting Started page (5.2)
- [ ] Create `docs/QUICKSTART.md` (5.3)
- [ ] Rewrite README hero section (4.2)
- [ ] Commit pre-built demo artifacts (1.2)

### Next 2 weeks
- [ ] Stripe billing integration — `backend/billing.py` (3.1)
- [ ] Add `stripe_customer_id` migration (3.2)
- [ ] Plan enforcement middleware (3.3)
- [ ] Pricing page frontend (3.4)
- [ ] Landing page (4.3)
- [ ] Status page (1.4)

### This month
- [ ] RouterDeps refactor (2.3)
- [ ] APEX/Nova naming unification (2.2)
- [ ] Record and embed demo video (1.3)

---

## What the Existing Specs Already Cover

The following technical work is tracked in `.kiro/specs/` and does NOT need to be
re-implemented here. Execute those specs in parallel with this roadmap.

| Spec | Covers |
|---|---|
| `apex-peak-capability` | Session sequences, ensemble weight loading, online learner, RL wiring |
| `architecture-design-perfection` | Pipeline decomposition, ADRs, ablation study, PBT invariants |
| `adaptive-serving-tiers` | Hardware detection, tier routing, ONNX wiring, `serving_tier.py` |
| `apex-perfect-score` | Offline eval pipeline, frontend pages, accessibility tests, mutation testing |
| `apex-final-polish` | Repo hygiene, benchmark cache extraction, admin route factory |
| `perfect-10` | `recommender.py` < 600 lines, `main.py` < 800 lines, CI registration |

---

## Revised Target Score After All Work Complete

| Dimension | After Roadmap |
|---|---|
| Technical depth & ML architecture | 10 |
| Engineering depth & infra | 10 |
| DevOps / CI / Observability | 10 |
| Testing | 10 |
| Code organization & maintainability | 10 |
| Documentation | 10 |
| Frontend | 10 |
| Monetization | 10 |
| Product definition & ICP clarity | 10 |
| Onboarding & self-serve | 10 |
| Demo & first impression | 10 |
