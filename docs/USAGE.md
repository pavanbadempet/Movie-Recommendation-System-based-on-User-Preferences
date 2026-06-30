# Usage Guide - APEX

## API Endpoints

### Search
```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "sci-fi action movies",
    "top_k": 10,
    "use_dense": true
  }'
```

### Recommendations
```bash
curl -X POST http://localhost:8000/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "seed_id": "movie_123",
    "top_k": 5,
    "personalize": true,
    "user_id": "user_456"
  }'
```

### Events
```bash
curl -X POST http://localhost:8000/v1/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_type": "view",
    "user_id": "user_123",
    "content_id": "movie_456",
    "timestamp": "2025-01-27T10:30:00Z"
  }'
```

## UI Features

### React UI
- Semantic search
- Recommendation browsing
- Item details
- Search history
- Personalized feed

### Streamlit Console
- API testing
- Diagnostics
- Quality metrics
- Integration snippets
- Event explorer

## Advanced Features

### Hybrid Search
```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "best sci-fi",
    "use_sparse": true,
    "use_dense": true,
    "sparse_weight": 0.3,
    "dense_weight": 0.7
  }'
```

### Reranking
```bash
{
  "query": "action movies",
  "rerank": true,
  "rerank_top_k": 50
}
```

### Diversity (MMR)
```bash
{
  "query": "comedies",
  "top_k": 10,
  "diversity_weight": 0.5
}
```

### Personalization
```bash
{
  "seed_id": "movie_123",
  "user_id": "user_456",
  "personalize": true
}
```

## Batch Operations

### Bulk Upload Catalog
```bash
curl -X POST http://localhost:8000/v1/catalogs/upload \
  -F "file=@movies.csv"
```

### Bulk Events
```bash
curl -X POST http://localhost:8000/v1/events/bulk \
  -H "Content-Type: application/jsonl" \
  --data-binary @events.jsonl
```

## Diagnostics

### Per-Seed Diagnostics
```bash
curl http://localhost:8000/v1/diagnostics/recommendations/movie_123
```

Returns:
- Ranking stages
- Feature importance
- Lineage
- Benchmark results

### System Readiness
```bash
curl http://localhost:8000/v1/platform/readiness
```

Returns:
- Catalog status
- Artifact health
- Quality gates
- Serving dependencies

## Best Practices

1. **Search**: Start with semantic queries, not keywords
2. **Events**: Track all user interactions for better personalization
3. **Quality**: Use benchmarking tools to measure recommendation quality
4. **Caching**: Enable distributed cache for failover scenarios
5. **Monitoring**: Check SLOs regularly via `/v1/platform/slo`

---

See [FAQ.md](FAQ.md) for more information!
