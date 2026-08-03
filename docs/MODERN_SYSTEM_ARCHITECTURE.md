# Modern System Architecture Principles

The **AI Recommendation System** is built on 5 fundamental **Modern System Architecture Principles** combining edge computing, dual-protocol microservices, heterogeneous hardware acceleration, Medallion Lakehouse storage, and data governance.

---

## 🏛️ System Architectural Pillars

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   5 SYSTEM ARCHITECTURAL PILLARS                                │
├─────────────────┬─────────────────┬───────────────────┬───────────────────┬─────────────────────┤
│ 1. EDGE COMPUTE │ 2. DUAL-PROTOCOL│ 3. HETEROGENEOUS  │ 4. MEDALLION ACID │ 5. GOVERNANCE &     │
│    & WASM       │    MICROSERVICES│    ACCELERATION   │    LAKEHOUSE      │    OPENLINEAGE      │
│  • Wasm 0ms     │  • gRPC HTTP/2  │  • Rust SIMD Core │  • Spark 4.1      │  • Unity Catalog    │
│  • HTTP/3 QUIC  │  • FastAPI REST │  • ONNX INT8 SIMD │  • Delta Lake 4.3 │  • OpenLineage DAG  │
└─────────────────┴─────────────────┴───────────────────┴───────────────────┴─────────────────────┤
```

---

## 1. 🌐 Edge Computing & Wasm Architecture
- **In-Browser Execution**: Quantized vector similarity searches execute directly inside the client browser via **WebAssembly (Wasm)**, achieving **`0ms` server latency** and zero network roundtrips.
- **HTTP/3 (QUIC over UDP)**: Public web traffic is routed through Cloudflare Edge using zero-RTT TLS 1.3 multiplexing without head-of-line blocking.

## 2. 📡 Dual-Protocol Microservices Architecture
- **gRPC over HTTP/2 (Port 50051)**: High-throughput binary Protocol Buffer (`.proto`) streams for inter-service communication (up to **10x faster** than JSON REST).
- **FastAPI 0.136+ ASGI**: Standard REST endpoints with `orjson` SIMD zero-copy JSON serialization for web clients.

## 3. ⚡ Heterogeneous Hardware Acceleration Architecture
- **Rust SIMD Core (`rust_core`)**: Universal PyO3 `abi3-py310` wheel executing SIMD AVX-512 / Neon candidate matrix scoring in **`< 0.3ms`**.
- **Quantized ONNX Runtime**: INT8/FP16 CPU SIMD execution eliminating PyTorch GIL locks during inference.
- **Multi-Armed Bandit Engine**: Thompson Sampling (Beta priors) & UCB1 dynamic exploration/exploitation.

## 4. 🥇 Medallion Lakehouse Storage Architecture
- **Apache Spark 4.1 (`pyspark 4.1.1`)**: Adaptive Query Execution (AQE) for petabyte distributed batch processing.
- **Polars 1.41+**: Rust-backed single-node micro-batch ingestion (`100,000+ items/sec`).
- **Delta Lake 4.3 (`delta-spark 4.3.1`)**: ACID transactions, Time Travel queries (`as_of_version`), and Z-Ordering `OPTIMIZE` compaction.
- **SCD Type 2**: Historical dimension record tracking with MD5 `record_hash`.

## 5. 🛡️ Governance & Observability Architecture
- **Unity Catalog 3-Level Namespace**: `main.recommendations.movies_curated` catalog metastore with granular RBAC grants (`GRANT SELECT`).
- **Data Privacy**: Cryptographic **SHA-256 PII Column Masking**.
- **OpenLineage 1.0**: Automated dataset provenance DAG tracking.
- **Telemetry**: Sentry SDK error monitoring + Prometheus metrics client + k6 Latency SLO tracking.
