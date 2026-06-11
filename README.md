---
title: Movie Recommendation System
emoji: 🎬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🎬 APEX — Open-Source Causal Movie Recommendation Engine

> A high-performance, real-time recommendation engine combining sequential Transformers (SASRec), learnable edge networks (KAN), and graph collaboration (LightGCN) with causal popularity debiasing.

<div align="center">

<img src="docs/assets/hero-banner.svg" alt="APEX Banner" width="100%"/>

<br/>

<p align="center">
  <a href="https://github.com/pavanbadempet/Movie-Recommendation-System/stargazers"><img src="https://img.shields.io/github/stars/pavanbadempet/Movie-Recommendation-System?style=flat-square&color=f59e0b" alt="GitHub stars" /></a>
  <a href="https://github.com/pavanbadempet/Movie-Recommendation-System/network/members"><img src="https://img.shields.io/github/forks/pavanbadempet/Movie-Recommendation-System?style=flat-square&color=06b6d4" alt="GitHub forks" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/pavanbadempet/Movie-Recommendation-System?style=flat-square&color=22c55e" alt="MIT license" /></a>
</p>

<h3>
  <a href="#-quick-start"><strong>Quick Start</strong></a> &middot;
  <a href="#-core-features"><strong>Features</strong></a> &middot;
  <a href="#-system-architecture"><strong>Architecture</strong></a> &middot;
  <a href="#-reproducing-benchmarks"><strong>Evaluation</strong></a> &middot;
  <a href="#-rest-api-reference"><strong>API Reference</strong></a>
</h3>

</div>

---

## 🚀 Why APEX?

Most recommendation system tutorials teach you how to train a model in a Jupyter notebook, but leave out the hard part: **how to serve it in production**.

**APEX** is a complete, production-ready recommender engine. It combines **6 complementary ML architectures** into an ensemble that dynamically scales from free CPU servers to high-performance GPU instances. It integrates a **real-time streaming feedback loop** that updates candidate features instantly, and uses **causal debiasing** to ensure users discover new long-tail content, not just blockbusters.

---

## ⚡ Core Features

<table>
<tr>
<td width="33%" valign="top">

### 🤖 6-Model Ensemble
LightGCN (Graph), SASRec (Transformer), KAN (Kolmogorov-Arnold), Quantum-Fluid (Neural ODE), Hyperbolic, and Generative Latent Diffusion models.

</td>
<td width="33%" valign="top">

### ⚡ Dynamic Hardware Tiers
Auto-detects memory and hardware capabilities at startup: Tier 1 (Full GPU Ensemble) vs. Tier 2 (ONNX CPU) vs. Tier 3 (FAISS/TF-IDF lite).

</td>
<td width="33%" valign="top">

### 🔄 Streaming Feedback Loop
Clickstream rating feeds are ingested asynchronously. Sequential candidate vectors are updated in real-time without batch DB rebuilds.

</td>
</tr>
</table>

---

## 🏗 System Architecture

```mermaid
flowchart TD
    subgraph DataPipeline["BATCH & STREAMING DATA PIPELINES"]
        Kaggle["Kaggle Source"] --> ETL["PySpark ETL Jobs"]
        TMDB["TMDB API"] --> ETL
        ETL --> Bronze["Delta Lake Bronze (Raw)"]
        Bronze --> Silver["Delta Lake Silver (SCD Type 2)"]
        Silver --> Gold["Delta Lake Gold (Feature Store)"]
    end

    subgraph Serving["ADAPTIVE SERVING CONTAINER"]
        U["User Request"] --> API["FastAPI Endpoint"]
        API --> Tier["Tier Detection Module"]
        Tier -->|GPU / 16G RAM| T1["Tier 1: Full Ensemble"]
        Tier -->|No GPU / 8G RAM| T2["Tier 2: Quantized ONNX"]
        Tier -->|Lite / 4G RAM| T3["Tier 3: FAISS Index"]
    end

    subgraph Evaluation["CAUSAL MODEL GOVERNANCE"]
        T1 --> Rank["Ranking Engine"]
        Rank --> IPS["DR-IPS Debiased Weights"]
        Rank --> DP["Differential Privacy (ε-DP)"]
        Rank --> FA["Fairness Auditor (Gini & KL)"]
    end

    DataPipeline -->|Rebuild Artifacts| Artifacts[("Serving Indices (FAISS · TF-IDF)")]
    Artifacts --> Serving
```

---

## ⚖️ Causal Debiasing & Unbiased Evaluation

Standard recommendations suffer from **popularity bias**—inflating scores for blockbusters at the expense of niche content. APEX integrates an **Inverse Propensity Score (IPS)** and a **Doubly Robust (DR) Estimator** to optimize ensemble weights:

$$V_{DR}(\pi) = \frac{1}{n} \sum_{i=1}^n \left[ \hat{r}(x_i, a_i) + \frac{(r_i - \hat{r}(x_i, a_i)) \cdot \pi(a_i|x_i)}{p(a_i|x_i)} \right]$$

This ensures our offline metric matches real-world utility, boosting long-tail relevance while preserving accurate recommendations.

---

## 📈 Offline Benchmarks

| Model | HR@10 | NDCG@10 | DR-Optimized Weight | Paradigm |
| :--- | :---: | :---: | :---: | :--- |
| **Ensemble** | **0.785** | **0.542** | — | Weighted blend |
| SASRec | 0.761 | 0.520 | **0.659** | Sequential Transformer |
| KAN | 0.694 | 0.439 | **0.298** | Kolmogorov-Arnold Network |
| LightGCN | 0.672 | 0.411 | 0.005 | Graph Collaborative Filtering |
| Diffusion | 0.521 | 0.309 | 0.024 | Generative Latent Diffusion |
| Quantum-Fluid | 0.583 | 0.354 | 0.010 | Neural ODE + Complex Embeddings |
| Hyperbolic | 0.498 | 0.287 | 0.004 | Poincaré Ball Manifold |

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/pavanbadempet/Movie-Recommendation-System.git
cd Movie-Recommendation-System

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:
```ini
TMDB_API_KEY=your_tmdb_key_here
JWT_SECRET_KEY=generate_a_strong_random_secret
OPENROUTER_API_KEY=your_openrouter_key
REDIS_URL=redis://localhost:6379/0
```

### 3. Build Serving Indices & Run
```bash
# Build vector embeddings and FAISS indices
python scripts/rebuild_serving_artifacts.py

# Start FastAPI backend (Terminal 1)
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# Start React frontend (Terminal 2)
cd frontend
npm install
npm run dev
```

---

## 📡 REST API Reference

| Method | Endpoint | Description | Sample Query Parameters |
| :---: | :--- | :--- | :--- |
| `GET` | `/v1/recommendations/id/{movie_id}` | Collaborative filtering recommendation results. | `?n=10&explain=true` |
| `GET` | `/v1/recommendations/visually-similar/{movie_id}` | Visual affinity recommendations. | `?n=5` |
| `GET` | `/v1/search/semantic` | Vector semantic search over the movie catalog. | `?q=sci-fi space exploration` |
| `POST`| `/v1/events/rating` | Live user rating ingestion (triggers online learner).| `{"user_id": 1, "movie_id": 550, "rating": 5.0}` |

*Append `?explain=true` to recommendation endpoints to generate natural-language explanations powered by LLMs.*

---

## 🤝 Contributing & Tests

All tests must pass in CI before merging.

```bash
# Run backend tests
python -m pytest tests/ -v

# Run frontend tests
npm --prefix frontend run test
```

---

## 📄 License

MIT License — Copyright © 2026 **Pavan Badempet**. See [LICENSE](LICENSE) for details.

---

<div align="center">

### **If you find this project useful, give it a ⭐ star!**

</div>
