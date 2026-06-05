# Requirements Document

## Introduction

The APEX recommendation system currently runs a fixed 6-model ensemble regardless of the hardware it is deployed on. This creates two failure modes: under-resourced deployments (single on-prem servers, no GPU, limited RAM) crash or degrade silently, while over-resourced deployments (GPU clusters, Redis, Kafka) leave capability on the table because the serving stack is not wired to use it.

This feature introduces an **Adaptive Serving Tiers** architecture that auto-detects hardware at startup, selects the appropriate serving tier (Tier 1 Enterprise / Tier 2 Professional / Tier 3 Starter), and activates only the models and infrastructure components that the hardware can sustain — all without manual configuration. An explicit override via `NOVA_SERVING_TIER` is also supported for operators who want deterministic behavior.

The existing `NOVA_SERVING_PROFILE` env var (which today only controls FAISS loading) is superseded by this feature but remains backward-compatible.

---

## Glossary

- **Tier_Detector**: The startup component that reads hardware metrics and resolves the active serving tier.
- **Serving_Tier**: One of three named capability levels — `tier1`, `tier2`, or `tier3` — that governs which models and infrastructure components are activated.
- **Tier1_Engine**: The full 6-model APEX ensemble with GPU acceleration, `torch.compile`, and Redis session cache.
- **Tier2_Engine**: The ONNX-based CPU-optimized inference engine backed by PostgreSQL.
- **Tier3_Engine**: The lightweight FAISS + TF-IDF retrieval engine backed by SQLite with minimal memory footprint.
- **Hardware_Profile**: The struct produced by Tier_Detector containing detected GPU presence, total RAM (GB), and logical CPU core count.
- **NOVA_SERVING_TIER**: The environment variable that, when set to `tier1`, `tier2`, or `tier3`, overrides auto-detection.
- **NOVA_SERVING_PROFILE**: The legacy environment variable (`full`/`lite`/`low-memory`). Remains functional; its semantics are mapped to the nearest Serving_Tier when NOVA_SERVING_TIER is absent.
- **Health_Endpoint**: The existing `/health` HTTP endpoint in `backend/main.py`.
- **orjson**: A Rust-backed JSON library that is a drop-in replacement for the standard `json` module, providing faster serialization with no behavior change.
- **ONNX_Engine**: The existing `backend/onnx_engine.py` module, currently scaffolded but not wired into the serving path.
- **Session_Cache**: The Redis-backed or in-memory store used by the ensemble engine to cache per-user session sequences.

---

## Requirements

### Requirement 1: Hardware Auto-Detection at Startup

**User Story:** As a platform operator, I want the system to detect available hardware automatically at startup, so that I do not need to manually configure serving parameters for each deployment environment.

#### Acceptance Criteria

1. WHEN the application process starts, THE Tier_Detector SHALL detect whether a CUDA-capable GPU is available using `torch.cuda.is_available()`.
2. WHEN the application process starts, THE Tier_Detector SHALL measure total system RAM in gigabytes using `psutil.virtual_memory().total`.
3. WHEN the application process starts, THE Tier_Detector SHALL count logical CPU cores using `os.cpu_count()`.
4. WHEN hardware detection completes, THE Tier_Detector SHALL produce a Hardware_Profile containing `gpu_available` (bool), `ram_gb` (float), and `cpu_cores` (int).
5. IF hardware detection raises an exception for any individual metric, THEN THE Tier_Detector SHALL log a warning and substitute a safe default value (False for `gpu_available`, 4.0 for `ram_gb`, 2 for `cpu_cores`) rather than aborting startup.
6. THE Tier_Detector SHALL complete hardware detection within 5 seconds of process start.

---

### Requirement 2: Serving Tier Resolution

**User Story:** As a platform operator, I want the system to automatically select the correct serving tier based on detected hardware, so that every deployment runs at its optimal capability level without manual tuning.

#### Acceptance Criteria

1. WHEN `NOVA_SERVING_TIER` is set to `tier1`, `tier2`, or `tier3`, THE Tier_Detector SHALL use that value as the active Serving_Tier and skip hardware-based auto-selection.
2. IF `NOVA_SERVING_TIER` is set to a value other than `tier1`, `tier2`, or `tier3`, THEN THE Tier_Detector SHALL log an error and fall back to hardware-based auto-selection.
3. WHEN `NOVA_SERVING_TIER` is absent and a GPU is available and RAM is at least 16 GB, THE Tier_Detector SHALL select `tier1`.
4. WHEN `NOVA_SERVING_TIER` is absent and no GPU is available and RAM is at least 8 GB, THE Tier_Detector SHALL select `tier2`.
5. WHEN `NOVA_SERVING_TIER` is absent and RAM is less than 8 GB, THE Tier_Detector SHALL select `tier3` regardless of GPU availability.
6. WHEN `NOVA_SERVING_PROFILE` is set to `full` and `NOVA_SERVING_TIER` is absent, THE Tier_Detector SHALL treat it as equivalent to `tier1` during auto-selection.
7. WHEN `NOVA_SERVING_PROFILE` is set to `lite` or `low-memory` and `NOVA_SERVING_TIER` is absent, THE Tier_Detector SHALL treat it as equivalent to `tier3` during auto-selection.
8. THE Tier_Detector SHALL log the resolved Serving_Tier and the reason for selection (explicit override, legacy profile mapping, or hardware auto-detection) at INFO level during startup.

---

### Requirement 3: Tier 1 — Full Ensemble Engine (Enterprise)

**User Story:** As a Tier 1 Enterprise customer with a GPU cluster, I want the full 6-model APEX ensemble to run with GPU acceleration and Redis session caching, so that I receive the highest recommendation quality the system can produce.

#### Acceptance Criteria

1. WHILE the active Serving_Tier is `tier1`, THE Tier1_Engine SHALL execute all six ensemble models (LightGCN, Quantum, SASRec, KAN, Hyperbolic, Diffusion) during each recommendation request.
2. WHILE the active Serving_Tier is `tier1` and a CUDA GPU is available, THE Tier1_Engine SHALL move all model tensors to the GPU device before the first inference call.
3. WHILE the active Serving_Tier is `tier1` and a CUDA GPU is available, THE Tier1_Engine SHALL attempt `torch.compile` on each model in the ensemble before each inference call to enable kernel fusion.
4. IF `torch.compile` raises an exception for any model on any attempt, THEN THE Tier1_Engine SHALL log a warning and execute that model without compilation for that request, retrying compilation on the next request.
5. WHILE the active Serving_Tier is `tier1` and Redis is reachable, THE Tier1_Engine SHALL use Redis as the Session_Cache backend for per-user session sequences.
6. IF Redis is unreachable during `tier1` startup, THEN THE Tier1_Engine SHALL fall back to the existing in-memory session cache and log a warning.
6a. WHILE the active Serving_Tier is `tier1` and Redis becomes unreachable after startup, THE Tier1_Engine SHALL continue attempting Redis operations and SHALL allow requests to fail if Redis is unavailable.
7. WHILE the active Serving_Tier is `tier1`, THE Tier1_Engine SHALL load ensemble blend weights from `models/ensemble_weights.json` using the existing weight-loading logic.

---

### Requirement 4: Tier 2 — ONNX CPU Engine (Professional)

**User Story:** As a Tier 2 Professional customer with a capable CPU server and PostgreSQL, I want ONNX-optimized inference without requiring a GPU, so that I get fast recommendations within my hardware budget.

#### Acceptance Criteria

1. WHILE the active Serving_Tier is `tier2`, THE Tier2_Engine SHALL load ONNX model sessions from the `models/onnx/` directory using the existing `ONNXEngine` class in `backend/onnx_engine.py`.
2. WHILE the active Serving_Tier is `tier2`, THE Tier2_Engine SHALL configure ONNX Runtime sessions with `ORT_ENABLE_ALL` graph optimizations and `intra_op_num_threads` set to the detected CPU core count.
3. WHILE the active Serving_Tier is `tier2`, THE Tier2_Engine SHALL use only the ONNX-exported models (MMoE ranker, LightGCN, Hyperbolic) and SHALL NOT load the PyTorch-native Quantum, KAN, Diffusion, or SASRec models.
4. IF an ONNX model file is missing from `models/onnx/`, THEN THE Tier2_Engine SHALL log a warning for that model and continue serving with the remaining available ONNX models.
5. IF no ONNX model files are present, THEN THE Tier2_Engine SHALL fall back to Tier 3 behavior and log a warning that ONNX artifacts are unavailable.
6. WHILE the active Serving_Tier is `tier2`, THE Tier2_Engine SHALL use PostgreSQL as the primary database backend when `DATABASE_URL` points to a PostgreSQL instance.
7. WHILE the active Serving_Tier is `tier2` and PostgreSQL is unavailable, THE Tier2_Engine SHALL fall back to the existing SQLite backend using the existing graceful fallback logic.

---

### Requirement 5: Tier 3 — Lightweight Engine (Starter)

**User Story:** As a Tier 3 Starter customer running on a single on-prem server with limited RAM and no GPU, I want the system to serve recommendations using only FAISS and TF-IDF retrieval, so that the service runs reliably within my hardware constraints.

#### Acceptance Criteria

1. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL activate the existing low-memory serving profile, equivalent to setting `NOVA_LOW_MEMORY=true`.
2. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL use only FAISS vector retrieval and TF-IDF sparse retrieval for candidate generation and SHALL NOT load any PyTorch or ONNX neural models.
3. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL use SQLite as the database backend.
4. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL limit the TF-IDF vocabulary to a maximum of 12,000 features to constrain memory usage.
5. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL defer building the sparse retrieval index until the first search request rather than at startup.
6. WHILE the active Serving_Tier is `tier3`, THE Tier3_Engine SHALL NOT load the FAISS index into RAM if `NOVA_FORCE_VECTOR_ARTIFACTS` is not set and the index exceeds available memory.

---

### Requirement 6: orjson Integration (All Tiers)

**User Story:** As a platform operator, I want all JSON serialization to use orjson across every serving tier, so that API response latency is reduced without any change to response structure or behavior.

#### Acceptance Criteria

1. WHEN `orjson` is installed, THE System SHALL use `orjson` for all JSON serialization and deserialization operations in the serving path across all three tiers.
2. WHEN `orjson` is not installed, THE System SHALL fall back to the standard `json` module and log a warning at startup.
2a. IF `orjson` is installed but raises an exception during a runtime serialization call, THEN THE System SHALL fall back to the standard `json` module for that call and log a warning.
3. THE System SHALL produce byte-for-byte equivalent JSON output compared to the standard `json` module for all response payloads (same keys, same values, same structure).
4. THE System SHALL NOT change any existing API response schema, field names, or data types when switching from `json` to `orjson`.
5. FOR ALL valid recommendation response payloads, serializing with `orjson` then deserializing SHALL produce an object equal to the original (round-trip property).

---

### Requirement 7: Tier Information in /health Endpoint

**User Story:** As a platform operator or SRE, I want the `/health` endpoint to expose the active serving tier and hardware profile, so that I can verify the correct tier is running without inspecting logs or environment variables.

#### Acceptance Criteria

1. WHEN a GET request is made to `/health`, THE Health_Endpoint SHALL include a `serving_tier` field in the response body containing the active Serving_Tier value (`tier1`, `tier2`, or `tier3`).
2. WHEN a GET request is made to `/health`, THE Health_Endpoint SHALL include a `hardware_profile` object in the response body containing `gpu_available`, `ram_gb`, and `cpu_cores` as detected by the Tier_Detector.
3. WHEN a GET request is made to `/health`, THE Health_Endpoint SHALL include a `tier_selection_reason` field indicating whether the tier was set by `NOVA_SERVING_TIER` override, `NOVA_SERVING_PROFILE` mapping, or hardware auto-detection.
4. THE Health_Endpoint SHALL return the `serving_tier`, `hardware_profile`, and `tier_selection_reason` fields even when the recommender has not yet been loaded (i.e., when `NOVA_HEALTH_LOAD_RECOMMENDER=false`).
5. THE Health_Endpoint SHALL continue to return all existing fields (`status`, `movie_count`, `app_version`, `app_commit`) unchanged.
6. IF the Tier_Detector has not yet completed at the time of a `/health` request, THEN THE Health_Endpoint SHALL return `"serving_tier": null` and `"tier_selection_reason": "detection_pending"` rather than blocking the response, even when `NOVA_SERVING_TIER` is set.
