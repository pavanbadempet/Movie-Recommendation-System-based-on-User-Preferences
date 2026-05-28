# APEX: Advanced AI Recommendation Engine

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688.svg)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-1A2B3C.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

APEX is a production-grade, state-of-the-art recommendation system built from the ground up to mirror the enterprise architectures of Netflix, YouTube, and Amazon. It transcends traditional keyword matching and matrix factorization by implementing a **4-Layer Intelligence Stack** encompassing Deep Neural Retrieval, Multi-Modal Fusion (Text + Vision), Deep Reinforcement Learning, and Semantic Knowledge Graphs.

---

## 🧠 Architecture Overview

APEX is structured into 18 systematic phases across 4 intelligence layers:

### Layer 1: Data Platform & Streaming
* **High-Concurrency ETL Pipeline**: Processes massive datasets using Parquet and memory-mapped NumPy arrays.
* **In-Memory Feature Store**: Redis-backed cache for real-time user state (clicks, ratings, session velocity) with sub-millisecond retrieval.
* **Streaming Telemetry**: Kafka-style event appending and aggregation for real-time behavioral updates.

### Layer 2: Machine Learning Engine
* **Two-Tower Neural Retrieval**: SBERT-based dual encoders map users and items into a shared 768-dimensional latent space.
* **Vector Search (FAISS)**: Sub-millisecond ANN (Approximate Nearest Neighbor) retrieval across millions of dense vectors.
* **Multi-Task Learning (MMoE)**: A Multi-gate Mixture-of-Experts (MMoE) ranker that simultaneously predicts click-through rate (CTR) and user rating, dynamically weighting the loss.
* **LightGBM Ranker**: High-speed gradient boosting tree used as a fast fallback ranker.

### Layer 3: Advanced Aesthetics & Multi-Modal Understanding
* **Visual Encoders (CLIP)**: Uses OpenAI's CLIP model to extract 512-dimensional aesthetic embeddings from movie posters.
* **Multi-Modal FAISS Fusion**: Mathematically fuses SBERT text vectors (60%) and CLIP visual vectors (40%) into a unified 1280-dimensional search space for aesthetic + thematic matching.
* **Latent Diffusion Similarities**: Recommends items based on visual generative latent structures.

### Layer 4: Cognitive Intelligence & Compliance
* **Reinforcement Learning (A2C)**: An Actor-Critic neural network optimizing for long-term retention (7-day return probability) rather than cheap clickbait, trained via Conservative Q-Learning (CQL).
* **Deep Content Understanding (NLP)**: Uses HuggingFace Zero-Shot classification (`nli-distilroberta-base`) to extract abstract human concepts (Moral Dilemmas, Moods) and NER for entities.
* **Semantic Knowledge Graphs**: NetworkX-powered multi-hop reasoning (`User -> Liked Theme -> New Movie`).
* **LLM Personalization**: OpenRouter integration (GPT-4o / Llama 3) to dynamically generate personalized 1-sentence explanations ("Because you loved X, you'll enjoy Y").
* **Differential Privacy**: Mathematical bounding (Laplace/Gaussian noise) on user embeddings to guarantee GDPR / EU AI Act compliance.
* **Counterfactual Evaluation**: Inverse Propensity Scoring (IPS) to mathematically simulate model deployments offline before exposing them to users.

---

## ⚡ Quick Start

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
Create a `.env` file in the root directory:
```ini
TMDB_API_KEY=your_tmdb_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
REDIS_URL=redis://localhost:6379/0  # Optional for Layer 1 features
```

### 3. Data Pipeline & Model Generation
Run the core pipeline to generate all artifacts, neural embeddings, and FAISS indices:
```bash
python scripts/rebuild_serving_artifacts.py
```

### 4. Launch the API
Start the high-concurrency FastAPI backend:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📡 Core API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/recommendations/id/{movie_id}` | `GET` | Core Deep Neural Recommendation (MMoE Ranked) |
| `/v1/recommendations/visually-similar/{movie_id}` | `GET` | Multi-Modal (Text + Vision) Fusion Search |
| `/v1/recommendations/knowledge-graph/{movie_id}` | `GET` | Multi-Hop Semantic Reasoning Search |
| `/v1/search/semantic` | `GET` | Vector-based semantic search (handles misspellings & abstract concepts) |

*Append `?explain=true` to any recommendation endpoint to trigger the OpenRouter LLM for personalized natural-language justifications.*

---

## 🛡️ Enterprise Fairness & Compliance

APEX includes a rigorous `FairnessAuditor` (`scripts/fairness_audit.py`) that mathematically verifies:
1. **Popularity Bias**: Enforces a Gini Coefficient `< 0.70` to prevent the model from blindly surfacing blockbuster content and starving niche creators.
2. **Calibration (KL Divergence)**: Ensures the recommended item distributions perfectly mirror the user's organic taste distribution without forcing them into a filter bubble.
3. **Safety Filters**: The Reinforcement Learning architecture utilizes an absolute hard-boundary to guarantee the AI will never recommend content a user explicitly dislikes.

---

## 🧪 Testing

APEX maintains a rigorous testing suite covering neural network bounds, safety constraints, mathematical normalization, and offline replay evaluation.

```bash
python -m pytest backend/tests/ -v
```

---
*Built as a state-of-the-art reference architecture for large-scale applied AI engineering.*
