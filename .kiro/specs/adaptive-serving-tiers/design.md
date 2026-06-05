# Design Document: Adaptive Serving Tiers

## Overview

This design introduces a hardware-aware serving tier system that auto-detects the deployment environment at startup and activates only the models and infrastructure components the hardware can sustain. The implementation is additive: a new `backend/serving_tier.py` module owns all detection and resolution logic, and the three existing serving modules (`ensemble_engine.py`, `recommender.py`, `onnx_engine.py`) are modified minimally to accept tier context. `main.py` wires everything together in the lifespan hook and exposes tier state through the `/health` endpoint.

---

## Architecture

```
startup (lifespan)
    │
    ▼
TierDetector.detect()          ← backend/serving_tier.py (new)
    │  reads: torch.cuda, psutil, os.cpu_count
    │  produces: HardwareProfile
    │
    ▼
resolve_serving_tier()         ← backend/serving_tier.py (new)
    │  reads: NOVA_SERVING_TIER, NOVA_SERVING_PROFILE
    │  produces: tier ("tier1" | "tier2" | "tier3"), reason
    │
    ├─ tier1 ──► ApexEnsembleEngine(device="cuda")   ← ensemble_engine.py (modified)
    │               torch.compile per model
    │               Redis session cache
    │
    ├─ tier2 ──► ONNXEngine(cpu_cores=N)             ← onnx_engine.py (modified)
    │               ORT_ENABLE_ALL, intra_op_num_threads=N
    │               ONNX models only (mmoe, lightgcn, hyperbolic)
    │
    └─ tier3 ──► Recommender.load() with tier3 flags ← recommender.py (modified)
                    NOVA_LOW_MEMORY=true equivalent
                    no neural models loaded
                    deferred sparse index

/health  ──► returns serving_tier, hardware_profile, tier_selection_reason
```

---

## Module Design

### 1. `backend/serving_tier.py` (new)

This module is the single source of truth for hardware detection and tier resolution. It has no imports from other backend modules to avoid circular dependencies.

#### `HardwareProfile` dataclass

```python
from dataclasses import dataclass

@dataclass
class HardwareProfile:
    gpu_available: bool
    ram_gb: float
    cpu_cores: int
```

#### `TierDetector` class

```python
class TierDetector:
    VALID_TIERS = frozenset({"tier1", "tier2", "tier3"})

    def __init__(self):
        self._profile: HardwareProfile | None = None
        self._tier: str | None = None
        self._reason: str | None = None
        self._detected: bool = False

    def detect(self) -> HardwareProfile:
        """Detect hardware metrics with safe defaults on any exception."""
        ...

    def resolve(self) -> tuple[str, str]:
        """Return (tier, reason). Calls detect() if not yet done."""
        ...
```

**`detect()` logic:**

Each metric is wrapped in its own try/except so a failure in one does not affect the others:

```python
def detect(self) -> HardwareProfile:
    import torch, psutil, os

    try:
        gpu_available = torch.cuda.is_available()
    except Exception as exc:
        logger.warning("GPU detection failed (%s); defaulting to False", exc)
        gpu_available = False

    try:
        ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    except Exception as exc:
        logger.warning("RAM detection failed (%s); defaulting to 4.0 GB", exc)
        ram_gb = 4.0

    try:
        cpu_cores = os.cpu_count() or 2
    except Exception as exc:
        logger.warning("CPU core detection failed (%s); defaulting to 2", exc)
        cpu_cores = 2

    self._profile = HardwareProfile(
        gpu_available=gpu_available,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
    )
    self._detected = True
    return self._profile
```

**`resolve()` logic:**

```python
def resolve(self) -> tuple[str, str]:
    if not self._detected:
        self.detect()

    profile = self._profile

    # 1. Explicit override
    explicit = os.getenv("NOVA_SERVING_TIER", "").strip().lower()
    if explicit:
        if explicit in self.VALID_TIERS:
            tier, reason = explicit, "explicit_override"
        else:
            logger.error(
                "NOVA_SERVING_TIER=%r is not valid; falling back to auto-detection", explicit
            )
            tier, reason = self._auto_select(profile)
    else:
        # 2. Legacy profile mapping
        legacy = os.getenv("NOVA_SERVING_PROFILE", "").strip().lower()
        if legacy == "full":
            tier, reason = "tier1", "legacy_profile_mapping"
        elif legacy in {"lite", "light", "low-memory", "metadata"}:
            tier, reason = "tier3", "legacy_profile_mapping"
        else:
            # 3. Hardware auto-detection
            tier, reason = self._auto_select(profile)

    self._tier = tier
    self._reason = reason
    logger.info(
        "Serving tier resolved: %s (reason=%s, gpu=%s, ram_gb=%.1f, cpu_cores=%d)",
        tier, reason, profile.gpu_available, profile.ram_gb, profile.cpu_cores,
    )
    return tier, reason

def _auto_select(self, profile: HardwareProfile) -> tuple[str, str]:
    if profile.ram_gb < 8.0:
        return "tier3", "hardware_auto_detection"
    if profile.gpu_available and profile.ram_gb >= 16.0:
        return "tier1", "hardware_auto_detection"
    return "tier2", "hardware_auto_detection"
```

**Module-level singleton and accessor:**

```python
_detector: TierDetector | None = None

def get_tier_detector() -> TierDetector:
    global _detector
    if _detector is None:
        _detector = TierDetector()
    return _detector

def resolve_serving_tier() -> tuple[str, str]:
    """Return (tier, reason). Safe to call before lifespan completes."""
    return get_tier_detector().resolve()
```

---

### 2. `backend/ensemble_engine.py` (modified)

**Change:** `ApexEnsembleEngine.__init__` accepts an optional `device` parameter. When `device="cuda"`, all sub-models are moved to GPU and `torch.compile` is attempted on each.

```python
def __init__(
    self,
    num_users: int = 1000,
    num_items: int = 10000,
    emb_dim: int = 16,
    device: str | None = None,          # NEW
):
    super().__init__()
    ...
    self._device = device or "cpu"
    self._compiled: dict[str, bool] = {}   # tracks per-model compile status

    # existing model construction unchanged ...

    # NEW: move to device and attempt compile
    if self._device != "cpu":
        self._move_to_device()
    if self._device == "cuda":
        self._try_compile_all()
```

**`_move_to_device()`:**

```python
def _move_to_device(self) -> None:
    for name in ("quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn"):
        try:
            model = getattr(self, name)
            model.to(self._device)
        except Exception as exc:
            logger.warning("Failed to move %s to %s: %s", name, self._device, exc)
```

**`_try_compile_all()`:**

```python
def _try_compile_all(self) -> None:
    for name in ("quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn"):
        if not self._compiled.get(name, False):
            self._try_compile(name)

def _try_compile(self, name: str) -> None:
    try:
        model = getattr(self, name)
        setattr(self, name, torch.compile(model))
        self._compiled[name] = True
        logger.info("torch.compile applied to %s", name)
    except Exception as exc:
        logger.warning("torch.compile failed for %s (%s); running uncompiled", name, exc)
        self._compiled[name] = False
```

**`predict_ensemble()` change:** Before each forward pass, if `_device == "cuda"` and a model is not yet compiled, call `_try_compile(name)`. Input tensors are moved to `self._device`.

**`get_apex_engine()` change:**

```python
def get_apex_engine(
    num_users: int = 1000,
    num_items: int = 10000,
    device: str | None = None,
) -> ApexEnsembleEngine:
    global _apex_engine
    if _apex_engine is None:
        _apex_engine = ApexEnsembleEngine(
            num_users=num_users,
            num_items=num_items,
            device=device,
        )
    return _apex_engine
```

---

### 3. `backend/onnx_engine.py` (modified)

**Change:** `ONNXEngine.__init__` accepts `cpu_cores: int` and passes it to `intra_op_num_threads`. The existing `0` (auto) is replaced with the detected core count.

```python
class ONNXEngine:
    def __init__(self, cpu_cores: int = 0):
        self.sessions = {}
        self.providers = ['CPUExecutionProvider']
        self._cpu_cores = cpu_cores

        self.load_model("mmoe_ranker", ONNX_DIR / "mmoe_ranker.onnx")
        self.load_model("lightgcn", ONNX_DIR / "lightgcn.onnx")
        self.load_model("hyperbolic", ONNX_DIR / "hyperbolic.onnx")

    def load_model(self, name: str, path: Path):
        if path.exists():
            try:
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sess_options.intra_op_num_threads = self._cpu_cores  # CHANGED
                self.sessions[name] = ort.InferenceSession(
                    str(path), sess_options, providers=self.providers
                )
                logger.info("Loaded ONNX Model: %s", name)
            except Exception as e:
                logger.error("Failed to load ONNX model %s: %s", name, e)
        else:
            logger.warning("ONNX Model not found: %s", path)
```

**Tier 2 fallback to Tier 3:** A helper `has_any_onnx_models()` checks whether any sessions loaded successfully. If none loaded, the caller (in `main.py` lifespan) falls back to tier3 behavior.

```python
def has_any_onnx_models(self) -> bool:
    return len(self.sessions) > 0
```

**`get_onnx_engine()` change:**

```python
def get_onnx_engine(cpu_cores: int = 0) -> ONNXEngine:
    global _onnx_engine
    if _onnx_engine is None:
        _onnx_engine = ONNXEngine(cpu_cores=cpu_cores)
    return _onnx_engine
```

---

### 4. `backend/recommender.py` (modified)

**Change:** `Recommender.load()` reads the active tier from `serving_tier.py` and skips neural model loading on tier3.

```python
def load(self) -> "Recommender":
    from backend.serving_tier import resolve_serving_tier
    active_tier, _ = resolve_serving_tier()
    is_tier3 = (active_tier == "tier3")

    # Existing low-memory flag is OR'd with tier3
    self._low_memory = self._low_memory or is_tier3

    # Skip diffusion model on tier3
    self.diffusion_model = None
    if not is_tier3:
        try:
            diffusion_path = MODELS_DIR / "diffusion_recommender.pth"
            if diffusion_path.exists() and not self._low_memory:
                ...  # existing load logic unchanged
        except Exception as e:
            logger.warning("Could not load Diffusion Recommender: %s", e)

    # Tier3: enforce deferred sparse index
    if is_tier3:
        os.environ.setdefault("NOVA_BUILD_SPARSE_ON_LOAD", "")
        # Ensure TF-IDF max features capped at 12000
        if int(os.getenv("NOVA_TFIDF_MAX_FEATURES", "50000")) > 12000:
            os.environ["NOVA_TFIDF_MAX_FEATURES"] = "12000"

    # Rest of load() unchanged ...
    return self
```

The existing `_low_memory_serving_enabled()` function and `_build_sparse_retrieval_index()` already handle the 12,000-feature cap and deferred build when `_low_memory=True`, so tier3 simply sets `_low_memory=True` and those paths activate automatically.

---

### 5. `backend/main.py` (modified)

#### orjson integration

At the top of `main.py`, replace the bare `import json` (used in several places) with a try/except shim:

```python
try:
    import orjson as _json_lib
    _ORJSON_AVAILABLE = True
    logger.info("orjson available; using fast JSON serialization")
except ImportError:
    import json as _json_lib
    _ORJSON_AVAILABLE = False
    logger.warning("orjson not installed; falling back to stdlib json")

def _json_dumps(obj) -> str:
    """Serialize obj to JSON string, falling back to stdlib json on error."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.dumps(obj).decode()
        except Exception as exc:
            logger.warning("orjson.dumps failed (%s); falling back to stdlib json", exc)
            import json
            return json.dumps(obj)
    return _json_lib.dumps(obj)

def _json_loads(s):
    """Deserialize JSON string, falling back to stdlib json on error."""
    if _ORJSON_AVAILABLE:
        try:
            return _json_lib.loads(s)
        except Exception as exc:
            logger.warning("orjson.loads failed (%s); falling back to stdlib json", exc)
            import json
            return json.loads(s)
    return _json_lib.loads(s)
```

All existing `json.dumps` / `json.loads` calls in the serving path are replaced with `_json_dumps` / `_json_loads`.

#### lifespan hook changes

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client, _online_learner, _tier_detector

    # --- NEW: resolve serving tier before any model loading ---
    from backend.serving_tier import get_tier_detector
    _tier_detector = get_tier_detector()
    active_tier, tier_reason = _tier_detector.resolve()

    http_client = httpx.AsyncClient(timeout=10.0)

    if active_tier == "tier1":
        device = "cuda" if _tier_detector._profile.gpu_available else "cpu"
        engine = get_apex_engine(device=device)
    elif active_tier == "tier2":
        from backend.onnx_engine import get_onnx_engine
        onnx_engine = get_onnx_engine(cpu_cores=_tier_detector._profile.cpu_cores)
        if not onnx_engine.has_any_onnx_models():
            logger.warning("No ONNX models loaded; falling back to tier3 behavior")
            active_tier = "tier3"
            _tier_detector._tier = "tier3"
            _tier_detector._reason = "onnx_fallback"
    # tier3: no engine pre-loading; recommender.load() handles it lazily

    # Existing OnlineLearner startup (tier1 only makes sense but kept for compat)
    if active_tier == "tier1":
        try:
            engine = get_apex_engine()
            _online_learner = OnlineLearner(lightgcn=engine.lightgcn)
            _online_learner.start()
            ...
        except Exception as exc:
            logger.critical("Failed to initialise OnlineLearner: %s", exc)
            _online_learner = None

    if _env_truthy("NOVA_BACKGROUND_RECOMMENDER_WARMUP"):
        _start_background_recommender_warmup()

    yield

    if _online_learner is not None:
        _online_learner.stop()
    await http_client.aclose()
```

A module-level variable holds the detector reference so `/health` can read it without blocking:

```python
_tier_detector: "TierDetector | None" = None
```

#### `/health` endpoint changes

The existing `HealthResponse` Pydantic model is extended:

```python
class HealthResponse(BaseModel):
    status: str
    movie_count: int
    app_version: Optional[str] = None
    app_commit: Optional[str] = None
    # NEW fields
    serving_tier: Optional[str] = None
    hardware_profile: Optional[dict] = None
    tier_selection_reason: Optional[str] = None
```

The handler reads from `_tier_detector` without blocking:

```python
@app.get("/health", response_model=HealthResponse)
async def health(...):
    ...
    # Tier info — non-blocking
    if _tier_detector is not None and _tier_detector._detected:
        p = _tier_detector._profile
        serving_tier = _tier_detector._tier
        hardware_profile = {
            "gpu_available": p.gpu_available,
            "ram_gb": round(p.ram_gb, 2),
            "cpu_cores": p.cpu_cores,
        }
        tier_selection_reason = _tier_detector._reason
    else:
        serving_tier = None
        hardware_profile = None
        tier_selection_reason = "detection_pending"

    return HealthResponse(
        status=...,
        movie_count=...,
        app_version=...,
        app_commit=...,
        serving_tier=serving_tier,
        hardware_profile=hardware_profile,
        tier_selection_reason=tier_selection_reason,
    )
```

---

## Data Flow

### Startup sequence

```
process start
  → lifespan() called
  → TierDetector.detect()        [< 5 s]
  → resolve_serving_tier()
  → tier1: get_apex_engine(device="cuda"), OnlineLearner
    tier2: get_onnx_engine(cpu_cores=N)
    tier3: nothing pre-loaded
  → background warmup (optional)
  → app ready
```

### Request path per tier

**Tier 1:**
```
/recommend → get_rec() → Recommender.load() (lazy)
           → get_apex_engine() → predict_ensemble() [GPU, compiled]
           → Redis session cache
```

**Tier 2:**
```
/recommend → get_rec() → Recommender.load() (lazy, no neural models)
           → get_onnx_engine() → predict_mmoe / predict_lightgcn / predict_hyperbolic
           → PostgreSQL backend
```

**Tier 3:**
```
/recommend → get_rec() → Recommender.load() (lazy, low_memory=True)
           → FAISS + TF-IDF only
           → SQLite backend
```

---

## Error Handling and Fallback Chain

| Condition | Fallback |
|---|---|
| Any hardware metric raises | Safe default substituted, warning logged |
| `NOVA_SERVING_TIER` invalid value | Auto-detection runs, error logged |
| `torch.compile` raises on a model | Model runs uncompiled, retry next request |
| Redis unreachable at tier1 startup | In-memory session cache, warning logged |
| ONNX model file missing | Warning logged, remaining models serve |
| No ONNX files at all | Tier 3 behavior activated, warning logged |
| `orjson` not installed | stdlib `json` used, warning at startup |
| `orjson` raises at runtime | stdlib `json` used for that call, warning logged |
| `_tier_detector` not yet set at `/health` | `serving_tier: null`, `reason: detection_pending` |

---

## Correctness Properties

### Property 1: HardwareProfile always has correct types

*For any* execution of `TierDetector.detect()`, the returned `HardwareProfile` satisfies: `gpu_available` is always `bool`, `ram_gb` is always a positive `float`, and `cpu_cores` is always an `int >= 1` — even when any individual hardware metric raises an exception (safe defaults are substituted).

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

---

### Property 2: Tier resolution is total and deterministic

*For any* `HardwareProfile` and any combination of `NOVA_SERVING_TIER` / `NOVA_SERVING_PROFILE` environment values, `resolve_serving_tier()` always returns a value in `{"tier1", "tier2", "tier3"}` — it never raises and never returns `None` or an unknown string.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

---

### Property 3: Auto-selection is a total function of RAM and GPU

*For any* `HardwareProfile` where `NOVA_SERVING_TIER` is unset and `NOVA_SERVING_PROFILE` is unset: `ram_gb < 8` always selects `tier3` regardless of `gpu_available`; `gpu_available=True` and `ram_gb >= 16` always selects `tier1`; all remaining cases (no GPU, or GPU with `8 <= ram_gb < 16`) always select `tier2`.

**Validates: Requirements 2.3, 2.4, 2.5**

---

### Property 4: ONNX thread count matches detected CPU cores

*For any* `cpu_cores` value `N >= 1`, an `ONNXEngine` constructed with `cpu_cores=N` configures every loaded ONNX Runtime session with `intra_op_num_threads == N`.

**Validates: Requirements 4.2**

---

### Property 5: orjson round-trip consistency

*For any* Python dict that is a valid recommendation response payload (all values are JSON-serializable), serializing with `_json_dumps` then deserializing with `_json_loads` produces an object equal to the original. Formally: `_json_loads(_json_dumps(payload)) == payload` for all valid payloads.

**Validates: Requirements 6.3, 6.4, 6.5**

---

### Property 6: /health tier fields are always present

*For any* GET `/health` response, the fields `serving_tier`, `hardware_profile`, and `tier_selection_reason` are always present in the response body — values may be `null` if detection is pending, but the keys are never absent.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.6**

---

## Components and Interfaces

### `backend/serving_tier.py` (new)

```python
@dataclass
class HardwareProfile:
    gpu_available: bool
    ram_gb: float
    cpu_cores: int

class TierDetector:
    def detect(self) -> HardwareProfile: ...
    def resolve(self) -> tuple[str, str]: ...  # (tier, reason)
    # Properties (read-only after resolve()):
    #   _profile: HardwareProfile | None
    #   _tier: str | None
    #   _reason: str | None
    #   _detected: bool

def get_tier_detector() -> TierDetector: ...
def resolve_serving_tier() -> tuple[str, str]: ...
```

### `backend/ensemble_engine.py` (modified interface)

```python
class ApexEnsembleEngine(nn.Module):
    def __init__(
        self,
        num_users: int = 1000,
        num_items: int = 10000,
        emb_dim: int = 16,
        device: str | None = None,   # NEW
    ): ...
    def _move_to_device(self) -> None: ...          # NEW
    def _try_compile(self, name: str) -> None: ...  # NEW
    def _try_compile_all(self) -> None: ...         # NEW
    # Existing methods unchanged

def get_apex_engine(
    num_users: int = 1000,
    num_items: int = 10000,
    device: str | None = None,   # NEW
) -> ApexEnsembleEngine: ...
```

### `backend/onnx_engine.py` (modified interface)

```python
class ONNXEngine:
    def __init__(self, cpu_cores: int = 0): ...  # CHANGED signature
    def has_any_onnx_models(self) -> bool: ...   # NEW
    # Existing methods unchanged

def get_onnx_engine(cpu_cores: int = 0) -> ONNXEngine: ...  # CHANGED signature
```

### `backend/recommender.py` (modified interface)

No public interface changes. `Recommender.load()` internally reads `resolve_serving_tier()` and adjusts `self._low_memory` and model loading accordingly.

### `backend/main.py` (modified interface)

```python
# New module-level variable
_tier_detector: TierDetector | None = None

# Extended Pydantic model
class HealthResponse(BaseModel):
    status: str
    movie_count: int
    app_version: Optional[str] = None
    app_commit: Optional[str] = None
    serving_tier: Optional[str] = None           # NEW
    hardware_profile: Optional[dict] = None      # NEW
    tier_selection_reason: Optional[str] = None  # NEW

# New JSON helpers
def _json_dumps(obj) -> str: ...
def _json_loads(s): ...
```

---

## Data Models

### `HardwareProfile`

| Field | Type | Description | Default on error |
|---|---|---|---|
| `gpu_available` | `bool` | Whether a CUDA-capable GPU is present | `False` |
| `ram_gb` | `float` | Total system RAM in gigabytes | `4.0` |
| `cpu_cores` | `int` | Logical CPU core count | `2` |

### Tier resolution output

| Field | Type | Values |
|---|---|---|
| `tier` | `str` | `"tier1"` / `"tier2"` / `"tier3"` |
| `reason` | `str` | `"explicit_override"` / `"legacy_profile_mapping"` / `"hardware_auto_detection"` / `"onnx_fallback"` |

### `/health` response extension

| Field | Type | Description |
|---|---|---|
| `serving_tier` | `str \| null` | Active tier or `null` if detection pending |
| `hardware_profile` | `object \| null` | `{gpu_available, ram_gb, cpu_cores}` or `null` if pending |
| `tier_selection_reason` | `str \| null` | Reason string or `"detection_pending"` |

---

## Error Handling

| Condition | Behavior |
|---|---|
| Any hardware metric raises during `detect()` | Safe default substituted; WARNING logged; detection continues |
| `NOVA_SERVING_TIER` set to invalid value | ERROR logged; auto-detection runs as fallback |
| `torch.compile` raises on a model | WARNING logged; model runs uncompiled; retry on next request |
| Redis unreachable at tier1 startup | In-memory session cache used; WARNING logged |
| Redis becomes unreachable after startup | Requests may fail; no automatic fallback (per Requirement 3.6a) |
| ONNX model file missing | WARNING logged; remaining models continue serving |
| No ONNX files present | Tier 3 behavior activated; WARNING logged |
| `orjson` not installed | stdlib `json` used for all calls; WARNING at startup |
| `orjson` raises at runtime | stdlib `json` used for that call only; WARNING logged |
| `_tier_detector` not yet set at `/health` | `serving_tier: null`, `tier_selection_reason: "detection_pending"` returned without blocking |

---

## Testing Strategy

### Unit tests

- `TierDetector.detect()` with mocked `torch.cuda`, `psutil`, `os.cpu_count` — including exception paths
- `TierDetector.resolve()` with all combinations of `NOVA_SERVING_TIER` and `NOVA_SERVING_PROFILE` env vars
- `ONNXEngine` with mocked `ort.InferenceSession` — verify `intra_op_num_threads` binding and `has_any_onnx_models()`
- `_json_dumps` / `_json_loads` with mocked `ImportError` and runtime exception paths
- `/health` endpoint with `_tier_detector` set and unset

### Property-based tests (Hypothesis)

- **Property 1** — `HardwareProfile` type invariants under arbitrary metric values and exceptions
- **Property 2 & 3** — Tier resolution totality and auto-selection boundary conditions
- **Property 4** — ONNX `intra_op_num_threads` binding for arbitrary `cpu_cores` values
- **Property 5** — orjson round-trip consistency for arbitrary recommendation payloads
- **Property 6** — `/health` tier fields always present in response

### Integration tests

- Full lifespan startup with `NOVA_SERVING_TIER=tier3` — verify no neural models loaded
- Full lifespan startup with `NOVA_SERVING_TIER=tier2` and empty `models/onnx/` — verify tier3 fallback

---

## File Change Summary

| File | Change type | Summary |
|---|---|---|
| `backend/serving_tier.py` | **New** | `HardwareProfile`, `TierDetector`, `resolve_serving_tier()` |
| `backend/ensemble_engine.py` | Modified | `device` param, `_move_to_device()`, `_try_compile_all()` |
| `backend/onnx_engine.py` | Modified | `cpu_cores` param, `has_any_onnx_models()` |
| `backend/recommender.py` | Modified | Read active tier in `load()`, skip neural models on tier3 |
| `backend/main.py` | Modified | orjson shim, lifespan tier init, `/health` extension |
