# 🏛️ Enterprise Multi-Tier AI Data Engineering & Neural Recommendation Engine

A production-grade, enterprise-scale **AI Data Engineering & Recommendation Platform** built with **Databricks Standard Serverless**, **Apache Spark (Photon Engine)**, **Delta Lake**, **Neon PostgreSQL (Multi-Shard Cluster)**, and **Kaggle NVIDIA T4 GPU Acceleration**.

---

## 🌟 Key Architectural Highlights & Engineering Achievements

### 1. Medallion Lakehouse Architecture (Delta Lake)
- **Bronze Layer (Raw Ingestion):** Automated Kaggle API pipeline ingesting **1,000,000+ TMDB movies** and **20,000,000+ MovieLens user interaction ratings** into Unity Catalog Volumes (`/Volumes/apex/default/secrets/raw_data`). Implements single-pass ingestion (`inferSchema=false`) with `_source_file` and `_ingested_at` data provenance metadata.
- **Silver Layer (Cleaned & Refined):** PySpark SQL schema enforcement, text normalization, JSON tag aggregation, and dead-letter queue (DLQ) isolation.
- **Gold Layer (Star Schema & Business Serving):** `apex.default.tmdb_gold_data` and `apex.default.tmdb_gold_with_embeddings`. Implements **SCD Type 2 (Slowly Changing Dimensions)** with `is_current`, `effective_date`, `end_date`, and `hash_diff` tracking. Automated Delta Lake compaction via `OPTIMIZE` Z-Ordering and `VACUUM`.

---

## 🏗️ System Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│              Databricks Standard Serverless (Photon C++ Engine)         │
│                                                                         │
│  [Step 00: Kaggle Ingestion]  -->  [Step 01: Pure PySpark SQL ETL]      │
│                                              │                          │
│                                              ▼                          │
│  [Step 02: Native Spark JDBC Export] <-- [Step 01c: Fast Vector Embeddings] │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│          Neon PostgreSQL Multi-Shard Serving Layer (~5ms Latency)        │
│  - 10-Shard Hashed Distribution (id % num_shards)                        │
│  - Covering B-Tree Indexes & 100% Full Float32 pgvector HNSW Graph Index│
└──────────────────────────────────▲──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│        Kaggle NVIDIA T4 GPU Acceleration Engine (30 Free Hrs/Wk)        │
│  - SentenceTransformer FP16 768-D Vector Encoding                        │
│  - 6 SOTA Neural Ensemble Models (SASRec, LightGCN, Neural ODE, KAN)    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ High-Throughput Storage & Serving Layer

### 2. Multi-Shard Neon PostgreSQL Serving Layer (~5ms Query Latency)
- **Multi-Shard Hashing:** Deterministic record distribution (`id % num_shards`) across 10 Neon project shards in AWS Singapore (`aws-ap-southeast-1`).
- **Zero-Memory Driver JDBC Export Engine:** Uses **Native Spark C++ Photon JDBC Writer** (`df_shard.write.format("jdbc")`), streaming data directly from Spark executor worker nodes to Neon PostgreSQL without driver-side socket streaming or memory overhead.
- **Uncompromised 100% Full Precision Float32 `pgvector` Indexing:**
  ```sql
  -- 100% Uncompressed Single-Precision Float32 Vector Storage
  ALTER TABLE movies 
    ALTER COLUMN embedding TYPE vector 
    USING embedding::vector;

  -- High-Throughput HNSW Cosine Similarity Index
  CREATE INDEX IF NOT EXISTS idx_movies_embedding_hnsw 
    ON movies USING hnsw (embedding vector_cosine_ops);
  ```

---

## 🤖 Deep Learning & Hybrid GPU Acceleration Engine

### 3. Kaggle NVIDIA T4 GPU Offloading Engine (`scripts/kaggle_gpu_runner.py`)
- Offloads heavy AI model training and FP16 SentenceTransformer vector encoding (`all-mpnet-base-v2`) to Kaggle's free GPU cluster (NVIDIA T4 16GB GPU with PyTorch FP16 Tensor Cores).
- Keeps all Databricks tasks on **Standard Serverless CPU** (<3s startup time, 0 GPU DBUs, 0 quota locks).

### 4. 6 State-of-the-Art Neural Recommendation Ensembles Trained
1. **SASRec:** Self-Attentive Sequential Transformer recommendation with BPR Loss.
2. **LightGCN:** Graph Convolutional Neural Network with normalized adjacency matrix propagation ($D^{-1/2} A D^{-1/2}$).
3. **Quantum Fluid Neural ODE:** Continuous-time dynamic intent differential equations (`torchdiffeq dopri5` adaptive step solver).
4. **Hyperbolic Poincaré Manifold:** Riemannian geometry space for hierarchical taxonomy representation ($||x|| < 1$ Poincaré ball constraint).
5. **Clifford Geometric Algebra:** Multivector rotation space for complex user-item interaction dynamics.
6. **KAN Ranker:** Kolmogorov-Arnold B-Spline learnable activation function ranker.

---

## 🛠️ Enterprise MLOps, CI/CD, & Observability
- **MLflow Tracking:** Complete experiment tracking logging hyperparameters, losses, metrics (`NDCG@10`, `Recall@10`, `MAP@10`), and trained PyTorch model checkpoints (`.pth`).
- **Centralized Doppler Secret Resolution:** Dynamic environment variable resolution (`dev`, `stg`, `prd`).
- **Automated Databricks REST API Integration:** Notebook workspace synchronization (`scripts/force_update_databricks_notebooks.py`) and job creation (`scripts/create_databricks_workflow_job.py`).

---

## 📁 Repository Structure

```
├── databricks_notebooks/
│   ├── 00_kaggle_download.py      # Automated Kaggle Dataset Ingestion (Bronze Layer)
│   ├── 01_pyspark_etl.py          # PySpark Medallion ETL, Star Schema, SCD Type 2 (Silver/Gold)
│   ├── 01c_gpu_embeddings.py      # Vector Embedding Generation & Delta Lake Table Registration
│   ├── 02_export_to_neon.py       # Native Spark JDBC Streaming to Neon PostgreSQL Multi-Shards
│   └── doppler_config.py          # Centralized Doppler Secret Resolution
├── kaggle_gpu_kernel/
│   ├── main.py                    # Standalone NVIDIA T4 GPU Embedding & Neural Training Script
│   └── kernel-metadata.json       # Kaggle GPU Kernel Manifest
├── scripts/
│   ├── train_apex_models.py       # Unified Training Script for All 6 Neural Ensemble Models
│   ├── kaggle_gpu_runner.py       # Kaggle API GPU Offloading Runner
│   ├── create_databricks_workflow_job.py  # Databricks Workflow Job REST API Creator
│   ├── force_update_databricks_notebooks.py# Databricks Workspace Notebook Rest API Sync
│   └── trigger_databricks_job_run.py     # Databricks Job Execution Trigger
└── ARCHITECTURE_SHOWCASE.md       # Full Enterprise Architectural Documentation
```
