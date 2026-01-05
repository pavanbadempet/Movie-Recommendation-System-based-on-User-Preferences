# 🎬 Movie Recommendation System

A **production-grade** content-based movie recommendation engine using SBERT semantic embeddings, FAISS similarity search, and intelligent multi-factor re-ranking.

[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen)](https://a-movie-recommendation-system.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Live Demo

**[▶️ Try it now: a-movie-recommendation-system.streamlit.app](https://a-movie-recommendation-system.streamlit.app/)**

---

## ✨ Features

### Core Recommendation Engine
- **MPNet Embeddings** (768 dimensions) — State-of-the-art sentence transformer
- **FAISS Similarity Search** — Sub-100ms nearest neighbor lookups
- **MMR Diversity** — Maximal Marginal Relevance prevents repetitive results
- **Multi-Factor Re-ranking** — Director, franchise, quality, era, language, genre consistency

### Explainability
Each recommendation includes human-readable explanations:
- *"Same franchise (Avatar)"*
- *"Same director (Christopher Nolan)"*
- *"Shared genres: Action, Sci-Fi"*
- *"Same era (2023)"*
- *"Critically acclaimed (8.5/10)"*

### Production Features
- **33,000+ Movies** — Comprehensive TMDB dataset
- **Real-time Enrichment** — Trailers, posters, cast via TMDB API
- **Streaming Providers** — Shows where to watch each movie
- **Premium Dark UI** — Glassmorphism, video backgrounds, smooth animations
- **Daily Data Refresh** — Automated GitHub Actions pipeline

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                         │
│   Premium Dark UI • Video Backgrounds • Streaming Providers     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│   Async Endpoints • TMDB Enrichment • Pydantic Validation       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Recommendation Engine                        │
│   SBERT (768d) → FAISS Search → Re-ranking → MMR Diversity      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       ETL Pipeline                              │
│   Kaggle Ingest → Transform → SBERT Embeddings → FAISS Index    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- TMDB API Key (free at [themoviedb.org](https://www.themoviedb.org/settings/api))

### Quick Start

```bash
# Clone repository
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env and add your TMDB_API_KEY

# Start backend
uvicorn backend.main:app --reload

# Start frontend (new terminal)
streamlit run streamlit_app.py
```

---

## 📁 Project Structure

```
Movie-Recommendation-System/
├── backend/
│   ├── main.py               # FastAPI endpoints
│   └── recommender.py        # Recommendation engine with re-ranking & MMR
├── etl/
│   ├── pipeline.py           # ETL orchestrator
│   ├── ingest.py             # Data loading & validation (Pandera)
│   ├── transform.py          # Feature engineering & SBERT embeddings
│   ├── index.py              # FAISS index building
│   └── pyspark_etl.py        # PySpark version for 1M+ scale
├── streamlit_app.py          # Premium Streamlit frontend
├── .github/workflows/
│   └── data-refresh.yml      # Daily Kaggle → SBERT → FAISS refresh
├── docker-compose.yml        # Full stack + Apache Airflow
├── data/
│   ├── raw/                  # Kaggle CSV files
│   └── processed/            # Parquet files
├── models/                   # FAISS index & SBERT embeddings
└── tests/                    # Unit & integration tests
```

---

## 🔬 Algorithm Details

### Feature Engineering
Each movie is represented as a semantic text combining:
- **Title** (weighted 2x for emphasis)
- **Tagline** (thematic hook)
- **Genres**
- **Plot/Overview** (main semantic content)
- **Director** (stylistic consistency)
- **Writers** (story patterns)
- **Cast** (top 10 actors)
- **Production Companies** (studio patterns)
- **Music Composer** (tone/style)

### Re-ranking Factors

| Factor | Boost | Description |
|--------|-------|-------------|
| Franchise | +0.25 | Same series (Avatar → Avatar 2) |
| Director | +0.10 | Same filmmaker |
| Same Era | +0.03 | Within 5 years |
| Recency | +0.02 | Recent releases |
| Quality | +0.02 | High ratings + vote confidence |
| Same Language | +0.02 | Non-English preference |
| Genre Mismatch | -0.15 | No shared genres |
| Documentary | -0.15 | Unless querying docs |
| Era Gap (30+ years) | -0.05 | Different generation |

### MMR Diversity
Maximal Marginal Relevance (λ=0.7) balances:
- **70% Relevance** — Similar to query movie
- **30% Diversity** — Different from already-selected results

---

## 🔄 Automated Data Refresh

### GitHub Actions (Daily)
The pipeline runs daily at 6 AM UTC via `.github/workflows/data-refresh.yml`:

1. **Download** latest TMDB dataset from Kaggle
2. **Transform** data and compute SBERT embeddings
3. **Build** new FAISS index
4. **Commit** updated artifacts back to repo

Requires secrets: `KAGGLE_USERNAME`, `KAGGLE_KEY`

### Local Refresh
```bash
python -m etl.pipeline --data data/raw/TMDB_movie_dataset_v11.csv
```

---

## 🐳 Docker Deployment

### Docker Compose (Full Stack + Airflow)
```bash
# Set environment variables
export KAGGLE_USERNAME=your_username
export KAGGLE_KEY=your_key
export TMDB_API_KEY=your_tmdb_key

# Start all services
docker-compose up -d

# Access:
# - Frontend: http://localhost:8501
# - Backend: http://localhost:8000
# - Airflow: http://localhost:8080 (admin/admin)
```

---

## ☁️ Cloud Deployment

### Streamlit Cloud (Frontend)
1. Connect GitHub repository
2. Set secrets: `TMDB_API_KEY`
3. Deploy!

### Render (Backend)
```yaml
# render.yaml included for one-click deploy
services:
  - type: web
    name: movie-recs-api
    runtime: python
    startCommand: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Movies indexed | 33,000+ |
| Embedding dimensions | 768 (MPNet) |
| Recommendation latency | <100ms |
| Index type | FAISS IVF |
| MMR diversity | λ=0.7 |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) — Movie data and API
- [Sentence Transformers](https://www.sbert.net/) — MPNet model
- [FAISS](https://github.com/facebookresearch/faiss) — Similarity search
- [Kaggle](https://www.kaggle.com/) — TMDB dataset hosting
- [Streamlit](https://streamlit.io/) — Frontend framework
