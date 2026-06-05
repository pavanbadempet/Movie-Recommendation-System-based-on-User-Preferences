# Implementation Plan: Adaptive Serving Tiers

## Overview

Implement hardware-aware serving tier selection by creating `backend/serving_tier.py`, wiring it into the lifespan hook in `main.py`, and modifying `ensemble_engine.py`, `onnx_engine.py`, and `recommender.py` to accept tier context. The orjson integration is independent and goes in first.

## Tasks

- [x] 1. Add orjson JSON shim to `backend/main.py`
  - [x] 1.1 Add orjson import with stdlib fallback at the top of `main.py`
    - Add `try: import orjson as _json_lib` / `except ImportError: import json as _json_lib` block
    - Define `_json_dumps(obj) -> str` and `_json_loads(s)` helpers with per-call fallback on orjson runtime errors
    - Log `INFO` when orjson is available, `WARNING` when falling back to stdlib
    - Replace all `json.dumps` / `json.loads` calls in the serving path with `_json_dumps` / `_json_loads`
    - _Requirements: 6.1, 6.2, 6.2a_

  - [ ]* 1.2 Write property test for orjson round-trip consistency
    - **Property 5: orjson round-trip consistency**
    - Generate arbitrary recommendation-shaped dicts (string keys, int/float/str/None/list values)
    - Assert `_json_loads(_json_dumps(payload)) == payload` for all generated payloads
    - Also assert output is identical to `json.dumps` / `json.loads` round-trip
    - **Validates: Requirements 6.3, 6.4, 6.5**

- [x] 2. Create `backend/serving_tier.py` — `HardwareProfile` and `TierDetector`
  - [x] 2.1 Implement `HardwareProfile` dataclass and `TierDetector.detect()`
    - Define `HardwareProfile(gpu_available: bool, ram_gb: float, cpu_cores: int)` dataclass
    - Implement `TierDetector.detect()` with per-metric try/except and safe defaults (False / 4.0 / 2)
    - Log a WARNING for each metric that falls back to its default
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5_

  - [ ]* 2.2 Write property test for HardwareProfile type invariants
    - **Property 1: HardwareProfile always has correct types**
    - Mock `torch.cuda.is_available`, `psutil.virtual_memory`, `os.cpu_count` with arbitrary valid and error-raising values
    - Assert `gpu_available` is always `bool`, `ram_gb` is always `float > 0`, `cpu_cores` is always `int >= 1`
    - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**

  - [x] 2.3 Implement `TierDetector.resolve()` and `resolve_serving_tier()`
    - Implement `_auto_select()`: ram_gb < 8 → tier3; gpu + ram >= 16 → tier1; else → tier2
    - Implement `resolve()`: check `NOVA_SERVING_TIER` override → legacy profile mapping → auto-select
    - Log invalid `NOVA_SERVING_TIER` as ERROR and fall back to auto-detection
    - Log resolved tier and reason at INFO level
    - Expose module-level `get_tier_detector()` singleton and `resolve_serving_tier()` convenience function
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_

  - [ ]* 2.4 Write property test for tier resolution totality and auto-selection rules
    - **Property 2: Tier resolution is total and deterministic**
    - **Property 3: Auto-selection is a total function of RAM and GPU**
    - Generate arbitrary `HardwareProfile` instances with varied `ram_gb` and `gpu_available` values
    - Assert result is always in `{"tier1", "tier2", "tier3"}` and never raises
    - Assert auto-selection boundary conditions: ram < 8 → tier3, gpu + ram >= 16 → tier1, else → tier2
    - **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

- [~] 3. Checkpoint — core detection module complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Modify `backend/onnx_engine.py` — wire cpu_cores into ONNX sessions
  - [~] 4.1 Add `cpu_cores` parameter to `ONNXEngine.__init__` and `get_onnx_engine()`
    - Change `ONNXEngine.__init__(self, cpu_cores: int = 0)` and store as `self._cpu_cores`
    - Replace hardcoded `intra_op_num_threads = 0` with `self._cpu_cores` in `load_model()`
    - Add `has_any_onnx_models(self) -> bool` method returning `len(self.sessions) > 0`
    - Update `get_onnx_engine(cpu_cores: int = 0)` singleton factory to pass `cpu_cores`
    - _Requirements: 4.1, 4.2, 4.4, 4.5_

  - [ ]* 4.2 Write property test for ONNX thread count binding
    - **Property 4: ONNX thread count matches detected CPU cores**
    - For arbitrary `cpu_cores` values (1–256), mock `ort.InferenceSession` and assert `intra_op_num_threads` equals the passed value
    - **Validates: Requirement 4.2**

- [ ] 5. Modify `backend/ensemble_engine.py` — device placement and torch.compile
  - [~] 5.1 Add `device` parameter to `ApexEnsembleEngine.__init__` and `get_apex_engine()`
    - Add `device: str | None = None` parameter; store as `self._device = device or "cpu"`
    - Add `self._compiled: dict[str, bool] = {}` to track per-model compile state
    - After existing model construction, call `self._move_to_device()` when device != "cpu"
    - Call `self._try_compile_all()` when device == "cuda"
    - Update `get_apex_engine(device: str | None = None)` to pass `device` to constructor
    - _Requirements: 3.2, 3.3_

  - [~] 5.2 Implement `_move_to_device()`, `_try_compile()`, and `_try_compile_all()`
    - `_move_to_device()`: iterate all 6 model attributes, call `.to(self._device)` with per-model try/except and WARNING on failure
    - `_try_compile(name)`: call `torch.compile(model)`, set `_compiled[name] = True` on success; on exception log WARNING and set `_compiled[name] = False`
    - `_try_compile_all()`: call `_try_compile(name)` for each model not yet compiled
    - In `predict_ensemble()`, move input tensors to `self._device` before forward passes
    - _Requirements: 3.2, 3.3, 3.4_

- [ ] 6. Modify `backend/recommender.py` — skip neural models on tier3
  - [~] 6.1 Read active tier in `Recommender.load()` and apply tier3 constraints
    - Import `resolve_serving_tier` from `backend.serving_tier` at the top of `load()`
    - Set `self._low_memory = self._low_memory or (active_tier == "tier3")`
    - Skip diffusion model loading when `is_tier3` is True
    - When `is_tier3`, cap `NOVA_TFIDF_MAX_FEATURES` at 12000 if currently higher
    - When `is_tier3`, ensure sparse index is deferred (do not call `_build_sparse_retrieval_index()` at load time)
    - _Requirements: 5.1, 5.2, 5.4, 5.5_

- [ ] 7. Modify `backend/main.py` — lifespan tier init and `/health` extension
  - [~] 7.1 Initialize `TierDetector` in lifespan and branch engine startup by tier
    - Add module-level `_tier_detector: TierDetector | None = None`
    - At the start of `lifespan()`, call `get_tier_detector().resolve()` to populate `_tier_detector`
    - For tier1: call `get_apex_engine(device="cuda" if gpu else "cpu")` and start `OnlineLearner`
    - For tier2: call `get_onnx_engine(cpu_cores=N)`; if `has_any_onnx_models()` is False, override tier to tier3 and log WARNING
    - For tier3: no engine pre-loading; recommender loads lazily on first request
    - _Requirements: 3.1, 3.2, 4.1, 4.5, 5.1_

  - [~] 7.2 Extend `HealthResponse` model and `/health` handler with tier fields
    - Add `serving_tier: Optional[str]`, `hardware_profile: Optional[dict]`, `tier_selection_reason: Optional[str]` to `HealthResponse`
    - In the `/health` handler, read `_tier_detector` without blocking: if not yet set or not yet detected, return `serving_tier=None` and `tier_selection_reason="detection_pending"`
    - Otherwise populate from `_tier_detector._profile`, `_tier_detector._tier`, `_tier_detector._reason`
    - Ensure all existing fields (`status`, `movie_count`, `app_version`, `app_commit`) remain unchanged
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6_

- [~] 8. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Task 1 (orjson) is fully independent and safe to merge first
- Tasks 2–3 (serving_tier.py) must complete before tasks 5–7 which import from it
- Task 4 (onnx_engine.py) is independent of task 5 (ensemble_engine.py) and can run in parallel
- Task 6 (recommender.py) depends on task 2 (serving_tier.py) but is independent of tasks 4 and 5
- Task 7 (main.py) depends on tasks 4, 5, and 6 all being complete
- Property tests use Hypothesis; run with `pytest backend/tests/ -x`

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "2.1"] },
    { "id": 1, "tasks": ["1.2", "2.2", "2.3"] },
    { "id": 2, "tasks": ["2.4", "4.1", "5.1"] },
    { "id": 3, "tasks": ["4.2", "5.2", "6.1"] },
    { "id": 4, "tasks": ["7.1"] },
    { "id": 5, "tasks": ["7.2"] }
  ]
}
```
