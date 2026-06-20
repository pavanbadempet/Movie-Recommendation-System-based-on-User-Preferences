# Token Optimization Implementation

This document describes the token optimization techniques implemented in the Movie Recommendation System to reduce LLM API costs and improve efficiency.

## Implemented Optimizations

### 1. Prompt Optimization ✅
- **Tighter system prompts**: Reduced verbose instructions to concise directives
- **max_tokens parameter**: Limited output tokens to 50 for explanations (was unlimited)
- **Structured output requests**: Added "plain text only" instruction to reduce verbosity
- **Location**: `backend/intelligence/llm_explanations.py`

**Impact**: ~40-60% reduction in output tokens per request

### 1.5. Prompt Caching ✅ (NEW)
- **API-level prompt caching**: Enabled prompt caching for repeated system prompts
- **90% savings on cached prefixes**: System prompts are cached at the API level
- **Automatic cache management**: Handled by OpenRouter/compatible providers
- **Location**: `backend/intelligence/openrouter_client.py` - `enable_prompt_caching` parameter

**Impact**: ~90% cost reduction on repeated system prompts (explanation generation uses same system prompt repeatedly)

### 2. Context Compression ✅
- **Signal compression**: Reduced explanation tags from all tags to first 2 only
- **Compact formatting**: Changed "Genre Overlap Match: 85%" to "85% genre match"
- **User context truncation**: Limited user context to 100 characters max
- **Genre compression**: Long genre lists compressed to "Action, Comedy +3 more" format
- **Location**: `backend/intelligence/llm_explanations.py` - `_format_signals()` and `_compress_genres()` functions

**Impact**: ~50-70% reduction in input tokens per request

### 3. Model Routing ✅
- **Fast models for simple tasks**: Added `FAST_MODELS` list with cheaper/faster models
- **use_fast_model parameter**: Routes simple tasks (explanations) to fast models
- **Quality models for complex tasks**: Reserved high-quality models for complex reasoning
- **Location**: `backend/intelligence/openrouter_client.py`

**Impact**: ~40-70% cost reduction on simple tasks

### 4. Token Usage Monitoring ✅
- **Token tracking**: New `token_monitor.py` module tracks input/output tokens per feature
- **Accurate token counting**: Uses tiktoken library when available (cl100k_base encoding), falls back to estimation
- **Cost estimation**: Estimates API costs based on token usage
- **Statistics**: Provides per-feature and aggregate statistics
- **Persistence**: Can persist/load stats to/from JSON file
- **Location**: `backend/intelligence/token_monitor.py`

**Usage**:
```python
from backend.intelligence.token_monitor import track_token_usage, get_token_stats, log_token_summary

# Track usage (automatically called in llm_explanations.py)
track_token_usage("explanation", input_text, output_text, model="gpt-4")

# Get statistics
stats = get_token_stats("explanation")

# Log summary
log_token_summary()
```

### 5. Semantic Caching ✅
- **Similar query matching**: Caches responses for semantically similar queries
- **Normalization-based**: Uses query normalization for similarity detection
- **Backfilling**: Semantic cache hits backfill the exact cache
- **Persistence**: Can persist/load cache to/from JSON file
- **Location**: `backend/intelligence/semantic_cache.py`

**Impact**: ~60-90% reduction in API calls for repeated/similar queries

**Usage**:
```python
from backend.intelligence.semantic_cache import get_semantic_cache, set_semantic_cache, get_semantic_cache_stats

# Get cached response
cached = get_semantic_cache(prompt, context)

# Set cache
set_semantic_cache(prompt, response, context)

# Get stats
stats = get_semantic_cache_stats()
```

### 6. Structured Output Support ✅
- **response_format parameter**: Added support for structured output requests
- **JSON mode ready**: Can enable JSON mode for models that support it
- **Location**: `backend/intelligence/openrouter_client.py`

**Usage**:
```python
explanation = chat_completion(
    messages=messages,
    models=models,
    temperature=0.7,
    timeout_seconds=2.5,
    api_key=api_key,
    max_tokens=50,
    use_fast_model=True,
    response_format={"type": "json_object"}  # Optional
)
```

## Configuration

### Environment Variables

Add these to your `.env` file to control optimization behavior:

```bash
# Model routing
NOVA_EXPLANATION_MODELS=google/gemma-3-27b-it:free,qwen/qwen3-next-80b-a3b-instruct:free

# Cache sizes (modify in code if needed)
# _EXPLANATION_CACHE_MAX in llm_explanations.py
# _semantic_cache_max in semantic_cache.py
```

## Monitoring Token Usage

### View Current Statistics

```python
from backend.intelligence.token_monitor import get_token_stats, log_token_summary

# Get stats for specific feature
explanation_stats = get_token_stats("explanation")
print(explanation_stats)

# Get all stats
all_stats = get_token_stats()
print(all_stats)

# Log summary to console
log_token_summary()
```

### Persist Statistics

```python
from backend.intelligence.token_monitor import persist_token_stats, load_token_stats

# Save to file
persist_token_stats("data/token_stats.json")

# Load from file
stats = load_token_stats("data/token_stats.json")
```

## Expected Savings

Based on the implemented optimizations:

- **Prompt optimization**: 40-60% reduction in output tokens
- **Prompt caching**: 90% cost reduction on repeated system prefixes
- **Context compression**: 50-70% reduction in input tokens
- **Model routing**: 40-70% cost reduction on simple tasks
- **Semantic caching**: 60-90% reduction in API calls for similar queries
- **Combined effect**: Estimated 80-90% total cost reduction

## Advanced Optimizations (Future)

These optimizations are planned but not yet implemented:

1. **Vector-based semantic caching**: Use sentence-transformers for true semantic similarity (code included but commented out in semantic_cache.py)
2. **Hierarchical context compression**: Multi-stage compression for large documents
3. **Batch API usage**: Use batch APIs for non-urgent workloads (50% discount)
4. **Smaller model preprocessing**: Use tiny models for filtering before main inference
5. **Circuit breaker pattern**: Advanced retry logic with exponential backoff (current model fallback is sufficient for most cases)

## Testing

To verify optimizations are working:

```python
# Check that fast models are being used
import os
os.environ["NOVA_EXPLANATION_MODELS"] = "google/gemma-3-27b-it:free"

# Generate explanation and check logs
from backend.intelligence.llm_explanations import generate_explanation
explanation = generate_explanation("user123", {"id": 1, "title": "Movie", "genres": "Action"})

# Check token stats
from backend.intelligence.token_monitor import get_token_stats
print(get_token_stats())
```

## Troubleshooting

### High token usage still occurring

1. Check that `use_fast_model=True` is being used for simple tasks
2. Verify `max_tokens` is set appropriately
3. Review semantic cache hit rates
4. Check if context compression is working (log signal strings)

### Cache not working

1. Verify Redis is running if using Redis cache
2. Check cache size limits
3. Ensure cache keys are being generated correctly
4. Review logs for cache hit/miss messages

### Model routing not working

1. Verify `FAST_MODELS` list contains valid model IDs
2. Check that `use_fast_model=True` is passed to `chat_completion`
3. Review OpenRouter API key permissions
4. Check logs for model selection

## References

- [How I Reduced LLM Token Costs by 90%](https://medium.com/@ravityuval/how-i-reduced-llm-token-costs-by-90-using-prompt-rag-and-ai-agent-optimization-f64bd1b56d9f)
- [LLM Token Optimization: Cut Costs & Latency in 2026](https://redis.io/blog/llm-token-optimization-speed-up-apps/)
- [LLM Cost Optimization: 5 Levers to Cut API Spend 70-85%](https://www.morphllm.com/llm-cost-optimization)
- [token-optimizer GitHub Repository](https://github.com/alexgreensh/token-optimizer)
