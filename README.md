# Movie Recommendation System

[![CI](https://github.com/pavanbadempet/Movie-Recommendation-System/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanbadempet/Movie-Recommendation-System/actions/workflows/ci.yml)
[![Demo](https://img.shields.io/badge/Demo-Live-brightgreen)](https://a-movie-recommendation-system.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Content-based movie recommendation engine. Uses SBERT embeddings for semantic similarity, FAISS for fast vector search, and a custom re-ranking layer for better results.

**[Try the demo](https://a-movie-recommendation-system.streamlit.app/)** | **[Architecture docs](docs/ARCHITECTURE.md)**

## What it does

- Finds similar movies based on plot, cast, director, and genre
- Searches 33,000+ movies in under 100ms
- Re-ranks results using franchise detection, director matching, and quality signals
- Uses MMR to avoid showing 5 sequels of the same movie

## 🚀 Quick Start (The Easy Way)

We have a unified management script to handle everything.

### 1. Setup
Install all dependencies with one command:
```bash
python manage.py setup
```

### 2. Run App
Starts both the **Backend API** and **Frontend UI** automatically:
```bash
python manage.py run
```
*   Frontend: http://localhost:8501
*   Backend: http://localhost:8000/docs

### 3. Run Data Pipeline
Run the ETL properly (Pandas or Spark):
```bash
# Standard (Local)
python manage.py etl

# Enterprise (Spark)
python manage.py etl --spark
```

### 4. Test
Run the full test suite:
```bash
python manage.py test
```

## How it works

1. **ETL pipeline** pulls movie data from TMDB (via Kaggle), cleans it, and generates embeddings
2. **SBERT (MPNet)** encodes movie metadata into 768-dim vectors
3. **FAISS index** enables fast approximate nearest neighbor search
4. **Re-ranking layer** boosts franchises, same-director films, and penalizes genre mismatches
5. **MMR diversity** prevents redundant results

## Project structure

```
backend/
  main.py          # FastAPI endpoints
  recommender.py   # FAISS search + re-ranking logic
etl/
  pipeline.py      # Orchestrates ingest → transform → index
  transform.py     # Feature engineering, SBERT encoding
  index.py         # FAISS index creation
models/            # faiss.index, embeddings
data/processed/    # Cleaned parquet files
streamlit_app.py   # Frontend
```

## Re-ranking factors

| Factor | Weight | Why |
|--------|--------|-----|
| Franchise match | +0.25 | Avatar → Avatar 2 |
| Same director | +0.10 | Nolan fans want more Nolan |
| Same era (±5 yrs) | +0.03 | Similar style/themes |
| Genre mismatch | -0.15 | Avoid semantic drift |
| Documentary filter | -0.15 | Hide "Making Of" unless relevant |

## Deployment

- **Backend**: Render.com (see `render.yaml`)
- **Frontend**: Streamlit Cloud

## Testing

```bash
pytest tests/ -v
```

## License

MIT
