# 🎬 Movie Recommendation System

[![Live Demo](https://img.shields.io/badge/Demo-Live-brightgreen?style=for-the-badge)](https://a-movie-recommendation-system.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

A **production-grade** content-based movie recommendation engine using **SBERT semantic embeddings**, **FAISS vector search**, and intelligent multi-factor re-ranking.

---

## 🌟 Live Demo

**[▶️ Try it now: a-movie-recommendation-system.streamlit.app](https://a-movie-recommendation-system.streamlit.app/)**

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **MPNet Embeddings** | 768-dim state-of-the-art sentence transformer |
| **FAISS Search** | Sub-100ms nearest neighbor lookups |
| **MMR Diversity** | Maximal Marginal Relevance prevents repetitive results |
| **Multi-Factor Re-ranking** | Director, franchise, quality, era, language |
| **Explainability** | Human-readable recommendation explanations |
| **33,000+ Movies** | Comprehensive TMDB dataset |
| **Real-time Enrichment** | Trailers, posters, cast via TMDB API |

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
```

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System

# Setup
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your TMDB_API_KEY

# Run
uvicorn backend.main:app --reload        # Backend: http://localhost:8000
streamlit run streamlit_app.py           # Frontend: http://localhost:8501
```

---

## 📁 Project Structure

```
├── backend/
│   ├── main.py               # FastAPI endpoints
│   └── recommender.py        # SBERT + FAISS engine
├── etl/
│   ├── pipeline.py           # ETL orchestrator
│   ├── transform.py          # Feature engineering + embeddings
│   └── index.py              # FAISS index building
├── models/                   # FAISS index + embeddings
├── data/processed/           # Transformed Parquet files
├── streamlit_app.py          # Premium Streamlit frontend
├── docker-compose.yml        # Full stack deployment
└── render.yaml               # Render.com config
```

---

## 🔬 Algorithm Details

### Re-ranking Factors

| Factor | Boost | Description |
|--------|-------|-------------|
| Franchise Match | +0.25 | Same series (Avatar → Avatar 2) |
| Director Match | +0.10 | Same filmmaker |
| Same Era | +0.03 | Within 5 years |
| Quality | +0.02 | High ratings + vote confidence |
| Genre Mismatch | -0.15 | No shared genres |

### MMR Diversity (λ=0.7)
Balances **70% relevance** to query with **30% diversity** from already-selected results.

---

## 🐳 Deployment

### Render (Backend)
```bash
# One-click deploy with render.yaml
```

### Streamlit Cloud (Frontend)
1. Connect GitHub repository
2. Set `TMDB_API_KEY` in secrets
3. Deploy!

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| Movies indexed | 33,759 |
| Embedding dimensions | 768 |
| Query latency | <100ms |
| Index type | FAISS IVF |

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📝 License

[MIT](LICENSE) © Pavan Badempet

---

## 🙏 Acknowledgments

- [TMDB](https://www.themoviedb.org/) — Movie data and API
- [Sentence Transformers](https://www.sbert.net/) — MPNet model
- [FAISS](https://github.com/facebookresearch/faiss) — Facebook AI similarity search
