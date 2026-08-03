# Modern Software Design Patterns Specification

The **AI Recommendation System** leverages enterprise software design patterns across Creational, Structural, and Behavioral categories to ensure high cohesion, low coupling, sub-millisecond performance, and strict type safety.

---

## 🎯 Design Patterns Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   ENTERPRISE DESIGN PATTERNS                                    │
├───────────────────────────────┬─────────────────────────────────┬───────────────────────────────┤
│     🏗️ CREATIONAL             │     🧩 STRUCTURAL               │     🔄 BEHAVIORAL             │
│  • Singleton & Factory        │  • Facade & Adapter             │  • Strategy & Chain of Resp.  │
│  • Builder (Spark/FastAPI)    │  • Decorator & Middleware       │  • Observer & Event Streaming │
└───────────────────────────────┴─────────────────────────────────┴───────────────────────────────┘
```

---

## 1. 🏗️ Creational Design Patterns

### A. Singleton & Factory Pattern
- **Implementation**: `get_recommender()`, `get_unity_catalog()`, `get_lineage_tracker()`
- **Role**: Lazy-initializes and reuses heavy singleton instances (e.g. FAISS indexes, PyTorch weights, and Unity Catalog metastores) across async FastAPI worker threads to prevent memory duplication.

```python
# etl/unity_catalog.py
_unity_catalog: Optional[UnityCatalogManager] = None

def get_unity_catalog() -> UnityCatalogManager:
    global _unity_catalog
    if _unity_catalog is None:
        _unity_catalog = UnityCatalogManager()
    return _unity_catalog
```

### B. Builder Pattern
- **Implementation**: `SparkSession.builder`, `RouterDeps`
- **Role**: Constructs complex Apache Spark 4.1 sessions with AQE, Arrow, and Delta Lake configs, as well as FastAPI dependency injection trees.

---

## 2. 🧩 Structural Design Patterns

### A. Adapter Pattern
- **Implementation**: `ONNXSBERTEncoder` (`backend/pipeline/recommender.py`)
- **Role**: Adapts high-speed ONNX Runtime C++ SIMD inference sessions to mimic the standard PyTorch `SentenceTransformer` `.encode()` method interface seamlessly without breaking callers.

```python
class ONNXSBERTEncoder:
    """Wrapper class that mimics SentenceTransformer interface but runs on ONNX Runtime for speed."""

    def encode(self, sentences: list[str], show_progress_bar: bool = False, batch_size: int = 32):
        # Executes ONNX INT8 quantized CPU SIMD session under the hood
        ...
```

### B. Facade Pattern
- **Implementation**: `Recommender` class
- **Role**: Provides a simple unified entrypoint (`recommend_by_id`, `search_by_title`) hiding the underlying complexity of FAISS ANN search, LightGCN graph embeddings, SASRec transformers, and Rust SIMD candidate scoring.

### C. Decorator & Middleware Pattern
- **Implementation**: `PlanEnforcerMiddleware`, `request_slo_middleware`, `@app.middleware("http")`
- **Role**: Attaches cross-cutting concerns (rate limiting, latency SLO tracking, Sentry error capturing) to HTTP and gRPC request handling without mutating business logic.

---

## 3. 🔄 Behavioral Design Patterns

### A. Strategy Pattern
- **Implementation**: `resolve_serving_tier()`, `ServingTier` (`backend/serving/serving_tier.py`)
- **Role**: Dynamically selects execution strategies (`LightweightTier`, `StandardTier`, `EnterpriseTier`) at startup based on host RAM, GPU detection, and environment variables.

### B. Chain of Responsibility Pattern
- **Implementation**: 10-Stage Recommendation Pipeline (`backend/pipeline/recommender_core.py`)
- **Role**: Candidate sets pass through a sequential pipeline chain:
  1. `FAISS / Vector Candidate Retrieval`
  2. `Collaborative Filtering Graph Scoring`
  3. `SASRec Sequential Transformer Re-ranking`
  4. `Rust SIMD Candidate Matrix Scoring`
  5. `Multi-Armed Bandit Exploration Reranking`
  6. `Maximal Marginal Relevance (MMR) Diversification`
  7. `Safety & Duplicate Filter`
  8. `LLM Explanation Generation`

### C. Observer & Event Stream Pattern
- **Implementation**: `DataLineageTracker` (`etl/data_lineage.py`), `StreamEvents` gRPC stream
- **Role**: Emits OpenLineage 1.0 JSON provenance specifications and ingests bi-directional telemetry streams asynchronously.
