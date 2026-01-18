# Architecture

Overview of the system design and key technical decisions.

## System overview

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend                             │
│                   Streamlit (Python)                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                        Backend                              │
│                   FastAPI + FAISS                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     External APIs                           │
│                   TMDB (posters, trailers)                  │
└─────────────────────────────────────────────────────────────┘
```

## Data pipeline

The ETL runs in three stages:

### 1. Ingest
- Source: Kaggle TMDB dataset (~1.1M rows)
- Filter: Remove movies with <50 votes, missing overviews
- Output: ~33K quality movies

### 2. Transform
- Combine title, plot, director, cast, genres into a single text blob
- Encode with SBERT (`all-mpnet-base-v2`, 768 dims)
- Normalize vectors for cosine similarity

### 3. Index
- Build FAISS IVF index with 100 clusters
- Trained on the full dataset for better cluster assignment

## Recommendation flow

1. User selects a movie
2. Look up the movie's embedding from the index
3. FAISS returns top 100 nearest neighbors
4. Re-rank using business logic (director, franchise, era, quality)
5. Apply MMR to diversify results
6. Return top 10

## Re-ranking logic

The raw FAISS scores are good but not perfect. A movie about "revenge" might match "vengeance" plots but miss that they're totally different genres. The re-ranker fixes this:

```python
final_score = faiss_score
if same_director:
    final_score += 0.10
if same_franchise:  # first word matches, 4+ chars
    final_score += 0.25
if no_genre_overlap:
    final_score -= 0.15
```

## MMR diversity

Without MMR, searching for "Avatar" returns Avatar 2, 3, 4, 5. MMR balances relevance vs diversity:

```
MMR = 0.7 * relevance - 0.3 * max_similarity_to_already_selected
```

λ=0.7 means 70% relevance, 30% diversity.

## Tech choices

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| Vector search | FAISS | Fast, battle-tested, works offline |
| Embeddings | SBERT MPNet | Best semantic similarity model |
| API | FastAPI | Async support, auto-generated docs |
| Frontend | Streamlit | Quick to build, easy to deploy |
| Data format | Parquet | Columnar, compressed |

## Scaling notes

Current setup handles 33K movies on a single server. For larger scale:

- Use FAISS IVF-PQ for 16x vector compression
- Move to Milvus or Pinecone for distributed search
- Add Redis caching for hot queries
- Put a CDN in front of poster images
