# Complexity Management Guide

## Overview

This guide explains how to navigate and simplify the APEX architecture for different use cases. The system is designed to be modular, allowing you to enable/disable components based on your requirements.

## Architecture Tiers

### Tier 3: Minimal Complexity (Recommended for Learning)
**Use when**: Learning recommendation systems, limited resources, simple use cases

**Enabled Components**:
- FAISS vector index only
- TF-IDF sparse retrieval
- Basic collaborative filtering
- SQLite database
- No GPU required

**Disabled Components**:
- 6-model ensemble (SASRec, KAN, LightGCN, etc.)
- ONNX quantization
- Real-time online learning
- Delta Lake ETL
- Differential privacy
- Causal debiasing

**Configuration**:
```bash
export NOVA_SERVING_TIER=tier3
export NOVA_DISABLE_MODEL_DOWNLOADS=1
export NOVA_DISABLE_ONLINE_LEARNING=1
```

**Benefits**:
- <4GB RAM requirement
- <2s cold start time
- Simple codebase to understand
- Easy to debug and modify

### Tier 2: Balanced Complexity (Recommended for Production)
**Use when**: Production deployment, moderate resources, need for accuracy

**Enabled Components**:
- ONNX quantized models
- 2-model ensemble (SASRec + LightGCN)
- Basic online learning
- PostgreSQL database
- Redis caching

**Disabled Components**:
- Full 6-model ensemble
- GPU acceleration
- Advanced causal debiasing
- Delta Lake ETL (use batch instead)

**Configuration**:
```bash
export NOVA_SERVING_TIER=tier2
export NOVA_ENSEMBLE_MODELS=sasrec,lightgcn
export NOVA_ENABLE_CAUSAL_DEBIASING=0
```

**Benefits**:
- 8-16GB RAM requirement
- ~25ms latency
- Good accuracy/complexity balance
- Easier to maintain than Tier 1

### Tier 1: Full Complexity (Enterprise Grade)
**Use when**: Maximum accuracy, abundant resources, research/production

**Enabled Components**:
- Full 6-model ensemble
- GPU acceleration
- Real-time online learning
- Delta Lake ETL
- Differential privacy
- Causal debiasing
- All advanced features

**Configuration**:
```bash
export NOVA_SERVING_TIER=tier1
export NOVA_ENSEMBLE_MODELS=all
export NOVA_ENABLE_CAUSAL_DEBIASING=1
export NOVA_ENABLE_DIFFERENTIAL_PRIVACY=1
```

**Benefits**:
- Maximum accuracy
- Production-grade features
- Research capabilities

**Trade-offs**:
- 16GB+ RAM required
- GPU recommended
- 45s cold start time
- Complex debugging

## Component Simplification Strategies

### 1. Model Ensemble Simplification

**Current**: 6 models (SASRec, KAN, LightGCN, Quantum-Fluid, Hyperbolic, Diffusion)

**Simplified Options**:

**Option A**: 2-model ensemble (80% of accuracy, 40% of complexity)
```python
# backend/models/ensemble_config.py
SIMPLIFIED_ENSEMBLE = {
    "sasrec": 0.7,      # Sequential patterns
    "lightgcn": 0.3     # Graph collaborative filtering
}
```

**Option B**: Single model (60% of accuracy, 15% of complexity)
```python
# backend/models/ensemble_config.py
MINIMAL_ENSEMBLE = {
    "sasrec": 1.0       # Sequential patterns only
}
```

**Implementation**:
```python
# In backend/pipeline/ranking_pipeline.py
def load_ensemble_weights():
    if os.getenv("NOVA_SIMPLIFIED_ENSEMBLE") == "true":
        return SIMPLIFIED_ENSEMBLE
    elif os.getenv("NOVA_MINIMAL_ENSEMBLE") == "true":
        return MINIMAL_ENSEMBLE
    return FULL_ENSEMBLE
```

### 2. Pipeline Simplification

**Current**: 3-stage pipeline (Retrieval → Ranking → Reranking)

**Simplified**: 2-stage pipeline (Retrieval → Ranking)

Remove reranking for simplicity:
```python
# backend/pipeline/recommendation_pipeline.py
def simplified_pipeline(user_id: str, movie_id: int):
    # Stage 1: Retrieval
    candidates = retrieval_pipeline.retrieve(user_id, movie_id)
    
    # Stage 2: Ranking (skip reranking)
    ranked = ranking_pipeline.rank(candidates, user_id)
    
    return ranked[:10]  # Return top 10
```

### 3. Feature Simplification

**Current**: 768-dim embeddings, multi-modal features

**Simplified**: 128-dim embeddings, basic features only

```python
# backend/serving/feature_store.py
class SimplifiedFeatureStore:
    def __init__(self):
        self.embedding_dim = 128  # Reduced from 768
        self.use_multimodal = False  # Disable CLIP image features
        self.use_knowledge_graph = False  # Disable KG features
```

### 4. Data Pipeline Simplification

**Current**: Delta Lake Medallion (Bronze → Silver → Gold)

**Simplified**: Direct SQLite/PostgreSQL

```python
# etl/pyspark_medallion_pipeline.py
def simplified_etl():
    # Skip Delta Lake, use direct database writes
    df = pd.read_csv("data/movies.csv")
    df.to_sql("movies", engine, if_exists="replace")
```

## Code Navigation Guide

### Critical Path (Core Recommendation Flow)

1. **Entry Point**: `backend/main.py` → FastAPI app
2. **API Layer**: `backend/api/recommendation_routes.py` → HTTP endpoints
3. **Pipeline**: `backend/pipeline/retrieval_pipeline.py` → Candidate generation
4. **Models**: `backend/models/sasrec.py` → Primary model
5. **Response**: `backend/response_models.py` → JSON serialization

### Supporting Systems (Can Ignore Initially)

- `backend/intelligence/` - LLM features (optional)
- `backend/learning/` - Online learning (optional)
- `backend/privacy/` - Differential privacy (optional)
- `backend/metrics/` - Advanced metrics (optional)
- `etl/` - Data pipelines (use provided data)

## Learning Path

### Week 1: Understanding the Basics
1. Read `docs/ARCHITECTURE_DECISIONS.md` (ADR-001, ADR-003 only)
2. Study `backend/models/sasrec.py` (single model)
3. Run Tier 3 configuration
4. Modify simple retrieval logic

### Week 2: Adding Complexity
1. Enable Tier 2 configuration
2. Study `backend/models/lightgcn.py`
3. Implement 2-model ensemble
4. Add basic reranking

### Week 3: Advanced Features
1. Enable Tier 1 configuration
2. Study causal debiasing in `backend/metrics/debiased_metrics.py`
3. Implement online learning
4. Add differential privacy

## Debugging Simplification

### Enable Debug Mode
```bash
export NOVA_DEBUG_MODE=1
export NOVA_LOG_LEVEL=DEBUG
```

### Disable Components for Debugging
```bash
# Disable specific models
export NOVA_DISABLE_KAN=1
export NOVA_DISABLE_DIFFUSION=1

# Disable features
export NOVA_DISABLE_LLM_EXPLANATIONS=1
export NOVA_DISABLE_SEMANTIC_SEARCH=1
```

### Profile Performance
```python
# Add to backend/main.py
import cProfile
import pstats

 profiler = cProfile.Profile()
profiler.enable()
# ... your code ...
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime').print_stats(20)
```

## Complexity Metrics

| Component | Lines of Code | Dependencies | Test Coverage |
|-----------|---------------|--------------|---------------|
| Retrieval Pipeline | 450 | 8 | 85% |
| Ranking Pipeline | 680 | 12 | 78% |
| SASRec Model | 520 | 6 | 82% |
| LightGCN Model | 380 | 5 | 80% |
| KAN Model | 420 | 7 | 75% |
| Online Learning | 340 | 9 | 70% |
| Differential Privacy | 180 | 4 | 65% |

**Total Core System**: ~3,000 lines (excluding tests)

## When to Use Each Tier

| Scenario | Recommended Tier | Reason |
|----------|------------------|--------|
| Learning ML basics | Tier 3 | Focus on concepts, not infrastructure |
| Startup MVP | Tier 2 | Balance of accuracy and complexity |
| Production deployment | Tier 2 | Proven reliability, manageable complexity |
| Research project | Tier 1 | Access to all features |
| Enterprise with resources | Tier 1 | Maximum accuracy and features |

## Getting Help

1. **Architecture Questions**: Start with `docs/ARCHITECTURE_DECISIONS.md`
2. **Implementation Details**: Check inline docstrings in core modules
3. **Configuration Issues**: Review `.env.example`
4. **Performance Problems**: Enable debug mode and profile

## Complexity Reduction Checklist

- [ ] Set appropriate serving tier for your use case
- [ ] Disable unused models in ensemble
- [ ] Skip optional features (LLM, semantic search)
- [ ] Use simplified data pipeline if not processing big data
- [ ] Enable debug mode during development
- [ ] Profile before optimizing
- [ ] Start with Tier 3, graduate to higher tiers as needed
