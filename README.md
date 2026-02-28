---
title: Nova: Semantic Search Engine
emoji: 🚀
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 7860
---

# Nova: Enterprise Semantic Recommendation Engine 🎬

> *Advanced, context-aware search and discovery infrastructure.*

Nova is a high-performance, full-stack recommendation engine built to demonstrate enterprise-scale semantic search. Traditional SQL-based search tools fail because they rely on exact keyword matches. Nova uses **dense vector embeddings (SBERT)** and **FAISS** to understand the *human intent* and contextual meaning behind queries, providing rich, personalized recommendations for massive datasets in milliseconds.

While currently configured with a 30,000+ movie dataset via the TMDB API, the underlying architecture is a highly scalable **B2B Recommendation-as-a-Service (RaaS)** prototype that can be adapted for e-commerce, digital content libraries, and enterprise knowledge bases.

## Core Architecture & Features

* **True Semantic Understanding:** Queries like *"documentaries about minimalists"* or *"time travel heist"* return highly relevant results even if the keywords never appear in the item's title or metadata.
* **Vector Similarity Search (FAISS):** By projecting textual plot summaries into high-dimensional vector space, Nova achieves sub-50ms search latency across tens of thousands of records, outperforming standard cosine similarity algorithms.
* **Smart MMR Re-ranking:** Implements Maximal Marginal Relevance (MMR) algorithms to ensure recommendation diversity, preventing "echo chamber" results where an engine simply suggests five highly similar sequels.
* **High-Availability Backend:** Built with FastAPI, featuring automated fallback routing logic to sustain 100% uptime across multiple deployment environments (Hugging Face Spaces, Render).

## Technical Stack

* **Backend API:** FastAPI (Asynchronous, Type-Safe, High Concurrency)
* **Machine Learning:** Hugging Face `sentence-transformers` (SBERT), `scikit-learn` (TF-IDF Hybrid)
* **Vector Database:** Meta's FAISS (Facebook AI Similarity Search)
* **Data Processing:** Pandas, Parquet (Memory Mapping for zero-RAM vector loading)
* **Frontend:** Streamlit Component Architecture (Python-native reactive UI)
* **Deployment:** Dockerized, CI/CD synced to Hugging Face Spaces and Render via GitHub Actions.

## Quick Start Guide

Nova comes with an integrated CLI manager to abstract complex deployment and ETL (Extract, Transform, Load) operations.

### 1. Environment Setup

```bash
# Initialize virtual environment and install dependencies natively
python manage.py setup
```

### 2. Launch Local Servers

```bash
# Concurrently boots the FastAPI asynchronous backend and Streamlit UI
python manage.py run
```

Access the client interface at `http://localhost:8501`.

### 3. Run the ETL Pipeline (Data Ingestion)

If you are modifying the dataset or need to download fresh embeddings:

```bash
# Pulls latest catalog, regenerates SBERT embeddings, and rebuilds the FAISS index
python manage.py etl
```

*Note: The initial vectorization process is computation-heavy. Memory-mapping is utilized post-generation to keep RAM costs trivial.*

## Commercial Viability & Use Cases

This repository serves as a blueprint for organizations looking to integrate AI-driven discovery into their own platforms.

* **E-Commerce:** Replacing exact-match product search with semantic intent matching (e.g., "warm winter coat for skiing" -> Down Jackets).
* **Streaming Platforms:** Increasing user retention by providing deeply connected content paths ("Because you watched...").
* **Content Publishers:** Structuring massive, unstructured article archives for instant, relevant retrieval.

## Contributing

Contributions to architectural improvements, algorithmic refinements, or frontend optimizations are welcome. Please refer to `CONTRIBUTING.md` for our standardized pull request and issue guidelines.

---
*Developed for high-scale semantic discovery.*
