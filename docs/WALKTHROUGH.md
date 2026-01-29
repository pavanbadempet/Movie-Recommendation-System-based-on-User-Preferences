# Movie Recommendation System Walkthrough

## 🚀 Architecture Overview

We have built a fully automated, free-tier, GPU-accelerated recommendation pipeline.

```mermaid
graph TD
    A[Kaggle Notebook] -->|1. Download| B[TMDB Dataset]
    A -->|2. Encode (GPU)| C[SBERT Embeddings]
    A -->|3. Upload| D[Hugging Face Hub]
    D -->|4. Download| E[Render Backend]
    E -->|5. API| F[Streamlit Frontend]
```

## ✨ New Features

### 1. Massive Scale
- **235,000+ Movies** (up from ~40k)
- Truly deep search covering obscure, foreign, and niche films.

### 2. SOTA Accuracy (BGE-M3)
- Moved from `all-mpnet-base-v2` to **`BAAI/bge-m3`**
- **1024-dimensional embeddings** for richer semantic understanding.
- Handles multi-lingual queries much better.

### 3. GPU-Accelerated ETL
- **Free P100 GPU** on Kaggle processes 235k movies in **~4 minutes**.
- Fully automated script tracks metadata and vectors.

### 4. Zero-Cost Hosting
- **Model Storage**: Hugging Face (Unlimited Free)
- **Compute**: Kaggle (30hrs/week GPU Free)
- **Hosting**: Render & Streamlit (Free Tier)

## 🛠️ Components

### ETL Pipeline (`notebooks/kaggle_etl_pipeline.py`)
- Loads TMDB data
- cleans and tags content
- Encodes using BGE-M3
- Builds FAISS HNSW index (graph-based, fast)
- Uploads to HF Hub

### Backend Loader (`backend/model_loader.py`)
- Checks for local files
- Downloads missing artifacts from HF Hub
- Handles 1GB+ files robustly with progress logging

### Recommender (`backend/recommender.py`)
- Uses **FAISS HNSW** for millisecond-latency search over 235k items
- Implements custom re-ranking logic (Franchise, Director, Era boost)
- Diversity via MMR (Maximal Marginal Relevance)

## 📝 How to Update Data

1. Open the [Kaggle Notebook](https://www.kaggle.com/code/...)
2. Click **"Run All"**
3. Wait ~5 minutes
4. **Restart Render Service** (or wait for auto-deploy if configured)
5. Done! The app now has fresh data.
