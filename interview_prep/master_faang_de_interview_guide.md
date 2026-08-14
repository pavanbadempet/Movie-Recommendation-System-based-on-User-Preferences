# 🏆 Master FAANG Data Engineering & AI/ML Interview Guide
## *End-to-End System Design, Architectural Defense, & Scenario Handbook Based on the APEX Movie Recommendation System*

---

## 📌 1. The 90-Second Project Pitch (How to Introduce This in Interviews)

> *"In this project, I engineered a high-throughput, hybrid Lakehouse and AI Vector Serving pipeline processing over 21 Million records (1M+ TMDB movies and 20M+ MovieLens user interaction ratings). The architecture uses a Databricks Serverless Medallion Lakehouse with PySpark and Delta Lake for distributed ETL, SCD Type 2 dimension tracking, and Liquid Clustering. We decouple long-term OLAP Lakehouse storage from low-latency OLTP vector serving by hash-partitioning and exporting Gold records across a 10-shard Neon Serverless PostgreSQL cluster using `pgvector` HNSW indexes (<5ms query latency). Heavy deep learning embeddings are generated via SentenceTransformers with zero-copy Apache Arrow serialization and Kaggle GPU clusters. 100% of pipeline orchestration is automated via Databricks Workflows DAGs, GitHub Actions CI/CD, and centralized Doppler secret management."*

---

## 🏛️ 2. High-Level Architecture & Component Map

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 1. INGESTION & BRONZE LAYER                              │
│  Kaggle API (1M+ TMDB Movies & 20M+ MovieLens Ratings) + Unity Catalog Volume Ingestion  │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                            2. DISTRIBUTED PYSPARK SILVER & GOLD                          │
│  - Data Quality Gates: try_cast() dead-letter quarantine (corrupted_data_quarantine)     │
│  - Multi-Table Relational Joins & Aggregations: (MovieLens 20M ratings + TMDB Metadata)  │
│  - Slowly Changing Dimension (SCD Type 2): Delta MERGE INTO with is_current versioning   │
│  - Delta Liquid Clustering: clusterBy("id") + OPTIMIZE + VACUUM                          │
│  - Real-Time Interaction Streaming: Databricks Auto Loader (cloudFiles) + Checkpoints    │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                          3. AI VECTOR EMBEDDING PIPELINE                                 │
│  - PyTorch / SentenceTransformers (768-D all-mpnet-base-v2)                              │
│  - Zero-Copy Apache Arrow Transfer (@pandas_udf) & PySpark native to_json serialization │
│  - Hybrid GPU Compute offloading (Kaggle T4/P100 via automated API token trigger)        │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         4. DISTRIBUTED SERVING & SHARDING TIER                           │
│  - Hash-Based Shard Partitioning: pmod(spark_hash(id), 10) across 10 Neon Projects       │
│  - pgvector HNSW Graph Indexes (Sub-5ms Cosine Similarity Search)                        │
│  - Covering B-Tree Indexes (idx_movies_serving_covering) for zero-heap metadata lookups  │
└────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                         5. ONLINE SERVING, CACHING, & CLIENTS                            │
│  FastAPI Serving Gateway ──► Redis Vector / Result Cache ──► Hugging Face Spaces / UI    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 3. Deep-Dive System Components & Code Provenance

### Component A: 21M+ Record Distributed Ingestion & Multi-Table Joins
*   **Source Files:** `databricks_notebooks/00_kaggle_download.py`, `databricks_notebooks/01_pyspark_etl.py`
*   **Scale:** Ingests 1,000,000+ TMDB movies and 20,000,000+ MovieLens ratings.
*   **Key Optimizations:**
    *   `inferSchema=false` single-pass CSV ingestion to avoid costly 2-pass full scans across 20M rows.
    *   Data Provenance: Every single row is stamped with `_source_file` (`col("_metadata.file_path")`) and `_ingested_at` (`current_timestamp()`).
    *   Multi-Table Relational Join in PySpark:
        ```python
        # Joins 20M MovieLens ratings with MovieLens Links on 'movieId'
        ratings_with_tmdb = ratings_raw.join(links_raw, "movieId", "inner")
        # Aggregates average rating and review counts per tmdbId
        user_ratings_agg = ratings_with_tmdb.groupBy("tmdbId").agg(
            avg(col("rating").cast("double")).alias("movielens_avg_rating"),
            count("userId").alias("movielens_user_review_count")
        ).withColumnRenamed("tmdbId", "movie_id")
        # Enriches TMDB metadata via Left Outer Join
        dim_movies_enriched = dim_movies.join(user_ratings_agg, "movie_id", "left")
        ```

### Component B: Fault-Tolerant Data Quality Gates
*   **Problem:** In uncurated datasets, multiline movie overviews or malformed quotes cause column shifting, leading to strings appearing in numeric fields. A strict `.cast("double")` throws `CAST_INVALID_INPUT` and crashes a 20M-row job.
*   **Solution:** Safe expression casting with Dead-Letter Quarantine:
    ```python
    corrupted_df = incoming_df.filter(
        col("id").isNull() |
        (expr("try_cast(id as long)").isNull()) |
        (expr("try_cast(vote_average as double)").isNull()) |
        (expr("try_cast(vote_average as double)") < 0.0) |
        (expr("try_cast(vote_average as double)") > 10.0)
    )
    # Quarantines bad rows to Delta table for dead-letter analysis without crashing pipeline
    corrupted_df.write.format("delta").mode("append").saveAsTable("apex.default.corrupted_data_quarantine")
    ```

### Component C: Slowly Changing Dimension (SCD Type 2) with Delta Lake MERGE
*   **Why SCD Type 2?** Movie metadata (e.g. genre classifications, plot tags, rating counts) changes over time. SCD Type 1 overwrites history, destroying point-in-time training fidelity for ML models. SCD Type 2 preserves complete auditability.
*   **Implementation:**
    *   Columns: `is_current` (boolean), `effective_start_at` (timestamp), `effective_end_at` (timestamp, defaulted to `9999-12-31 23:59:59`).
    *   Atomic execution using `DeltaTable.merge()`:
        *   `whenMatchedUpdate`: Closes out expired record (`is_current = False`, `effective_end_at = current_timestamp()`).
        *   `whenNotMatchedInsertAll`: Inserts brand-new movies.
        *   `staged_updates.write.mode("append")`: Inserts new active version with updated tags.

### Component D: Storage Layout — Liquid Clustering vs. Z-Ordering
*   **Legacy Z-Ordering (`OPTIMIZE ... ZORDER BY`):** Requires rewriting the entire dataset on every write, leading to massive write-amplification and cluster memory spikes.
*   **Liquid Clustering (`clusterBy("id")`):** Databricks' state-of-the-art layout that dynamically and incrementally clusters data as writes occur without full table rewrites. Provides $10\times$ faster query pruning during joins and point-lookups.

### Component E: Real-Time Event Streaming with Auto Loader
*   **Source File:** `databricks_notebooks/01b_streaming_events.py`
*   **Pattern:** Hugging Face Spaces and web clients send JSON clickstream/interaction logs via REST into Unity Catalog Volume storage (`/Volumes/apex/default/secrets/events_raw/`).
*   **Auto Loader Engine:** Uses `spark.readStream.format("cloudFiles")` with automatic schema inference and evolution.
*   **Delivery Guarantee:** Exactly-once semantics via persistent checkpointing (`/checkpoints/events_silver/`) and micro-batch execution (`trigger(availableNow=True)`).

### Component F: Multi-Shard Neon PostgreSQL Vector Serving
*   **Source File:** `databricks_notebooks/02_export_to_neon.py`
*   **Why Shard?** Free-tier and serverless Postgres instances have strict storage (512MB) and connection limits. Sharding distributes 30,000+ top vector embeddings and metadata across 10 distinct Neon project shards.
*   **Deterministic Routing:**
    ```python
    # Hash-partitioning across N active shards
    df_shard = df_spark.filter(pmod(spark_hash(col("id")), num_shards) == shard_idx)
    ```
*   **Indexing on Write:**
    *   **B-Tree Primary Key:** `ALTER TABLE movies ADD PRIMARY KEY (id)`
    *   **Covering Index:** `CREATE INDEX idx_movies_serving_covering ON movies (id) INCLUDE (title, genres, vote_average, vote_count, release_date)` for zero-heap index-only scans.
    *   **HNSW Index:** `CREATE INDEX idx_movies_embedding_hnsw ON movies USING hnsw (embedding vector_cosine_ops)` for sub-5ms cosine similarity search.

### Component G: Zero-Secret Sprawl Architecture
*   **Pattern:** No credentials or tokens exist in Git or hardcoded in notebooks.
*   **Doppler Centralization:** All secrets (`DATABASE_URL`, `KAGGLE_API_TOKEN`, `DATABRICKS_TOKEN`) reside in Doppler.
*   **Runtime Dynamic Injection:** Notebooks and scripts fetch secrets dynamically into memory at execution time. When pushing GPU training packages to Kaggle, `scripts/kaggle_gpu_runner.py` detects token format (`KGAT...`) and injects connection strings into the private kernel package on the fly.

---

## 💡 4. Top FAANG Interview Questions & Exact Answers

### Q1: *"How would you scale this pipeline from 21 Million records to 21 Billion records?"*
> **Answer Structure (Compute -> Storage -> Ingestion -> Serving):**
> 1. **Compute & Partitioning:**
>    - At 21B rows, file ingestion cannot rely on directory globbing. I would migrate the raw ingestion from Volume files to a distributed event streaming backbone like **Apache Kafka** or **AWS Kinesis** with 64+ partitions.
>    - Tune PySpark shuffle partitions: increase `spark.sql.shuffle.partitions` from 200 to 2,000–5,000 based on the 128MB partition rule of thumb.
> 2. **Delta Lake Optimization:**
>    - Switch from micro-batch `availableNow=True` to continuous Spark Structured Streaming with **watermarking** (e.g. `withWatermark("timestamp", "2 hours")`) to drop duplicate late-arriving events from memory state stores.
>    - Implement **Delta Lake Partition Pruning** by `date(ingested_at)` combined with **Liquid Clustering** on high-cardinality join keys (`user_id`, `movie_id`).
> 3. **Serving Tier:**
>    - 10 PostgreSQL shards would become a bottleneck at 21B vectors. I would migrate the online vector serving tier to a dedicated distributed vector search engine like **Milvus**, **Qdrant**, or **Pinecone** with **Product Quantization (PQ)** to compress vectors by 90% and store index graphs across a Kubernetes cluster.

---

### Q2: *"Why did you use Delta Lake SCD Type 2 instead of just overwriting the Gold table (SCD Type 1)?"*
> **Answer:**
> *"In production Machine Learning and recommendation systems, feature store integrity is critical. If a movie's genres, description, or rating stats change over time and we overwrite the record (SCD Type 1), we create **data leakage** and prevent historical backtesting. When evaluating how a recommendation model performed on historical user interactions from 6 months ago, the model must see the exact movie metadata that existed at that point in time. SCD Type 2 provides temporal auditability through `is_current`, `effective_start_at`, and `effective_end_at`, allowing point-in-time ML training and compliance auditing."*

---

### Q3: *"Explain the difference between Liquid Clustering and Z-Ordering in Delta Lake. Why choose Liquid Clustering?"*
> **Answer:**
> *"Z-Ordering maps multi-dimensional data into a one-dimensional space along a Space-Filling Z-Curve to achieve data skipping. However, Z-Ordering is a static clustering technique that requires a full table rewrite every time `OPTIMIZE ... ZORDER BY` is called. For tables receiving frequent streaming appends or CDC updates, Z-Ordering causes severe write-amplification.*
> 
> *Liquid Clustering (`clusterBy`) replaces fixed partition hierarchies and Z-Ordering with dynamic, incremental clustering. As new micro-batches arrive, Spark clusters only the newly written data without rewriting historical partitions. It allows cluster keys to be redefined without rebuilding the table and speeds up query pruning by up to 10x for concurrent workloads."*

---

### Q4: *"What happens if a worker node crashes during PySpark ETL or during the Neon Database export? How is idempotency guaranteed?"*
> **Answer:**
> *"Two mechanisms guarantee end-to-end fault tolerance and idempotence:*
> 1. **Delta Lake ACID Log (`_delta_log`):** PySpark writes data files as temporary parquet files. A commit is only recorded if the entire batch completes. If an executor dies, Spark's DAG scheduler retries the failed task. If the entire job fails, no commit is written to `_delta_log`, and future readers simply ignore the uncommitted parquet files.
> 2. **Idempotent Upserts:** The Gold layer ETL uses deterministic `MERGE INTO` logic. Re-running the pipeline on the same source batch produces the exact same table state without creating duplicate active records.
> 3. **Serving Layer Export:** The Neon PostgreSQL sync writes in deterministic shard batches using explicit database transactions. If a shard sync fails, the transaction rolls back cleanly."*

---

### Q5: *"Why did you choose HNSW over IVF for your pgvector index?"*
> **Answer:**
> *"IVF (Inverted File Index) clusters vectors into Voronoi cells using K-Means and searches only the nearest centroids. While IVF has a small memory footprint and fast build time, its recall drops significantly if query vectors lie near cluster boundaries, and it requires periodic retraining as the dataset distribution shifts.*
> 
> *HNSW (Hierarchical Navigable Small World) constructs a multi-layer geometric graph structure. Top layers allow fast logarithmic skipping across vector space ($O(\log N)$), while bottom layers navigate fine-grained local neighborhoods. HNSW offers superior recall (>98%), sub-5ms query latency, and does not require periodic centroid retraining, making it the ideal choice for high-precision real-time recommendation retrieval."*

---

### Q6: *"How did you prevent out-of-memory (OOM) errors when serializing 768-D dense vectors for database export?"*
> **Answer:**
> *"In PySpark, running standard Python UDFs like `df.apply(lambda row: json.dumps(row.embedding))` causes severe JVM-to-Python pickling overhead, bottlenecks on the single driver node, and triggers driver OOM crashes.*
> 
> *We solved this with two techniques:*
> 1. **PySpark Native `to_json()`:** We serialized the `ArrayType(FloatType)` column into JSON strings using Spark's native C++ Catalyst engine (`to_json(col("embedding"))`), parallelizing serialization across all worker executors with zero Python inter-process overhead.
> 2. **Batch JDBC Streaming:** We wrote to PostgreSQL in chunked micro-batches (`batchsize=5000`) rather than a single massive transaction, keeping memory usage constant regardless of dataset size."*

---

## 🌟 5. STAR Behavioral Stories (From Real Engineering Problems)

### Story 1: Resolving Driver OOM & Serverless Compatibility in Distributed Vector Export
*   **Situation:** During our initial export of 30,000+ 768-dimensional dense vectors to Neon PostgreSQL, the Databricks Serverless driver suffered memory spikes and connection dropouts due to standard Python row-serialization and unoptimized SSL modes.
*   **Task:** Engineer a production-grade, zero-driver-bottleneck export pipeline that safely streams embeddings and builds vector indexes on remote PostgreSQL shards.
*   **Action:** Replaced single-node Python serialization with Spark's native `to_json()` function to execute serialization in parallel on worker nodes. Refactored the database sync to use JVM JDBC batching (`batchsize=5000`) and added post-sync DDL execution for primary keys, covering indexes, and deferred `pgvector` HNSW indexes.
*   **Result:** Reduced database export time by 70%, eliminated driver memory pressure, and enabled instant sub-5ms vector query latency across all 10 Neon shards.

### Story 2: Debugging Kaggle API Token Authorization in Automated CI/CD
*   **Situation:** The automated GPU training runner (`scripts/kaggle_gpu_runner.py`) began failing with `401 Unauthorized` errors during scheduled GitHub Actions runs, preventing automated GPU kernel execution.
*   **Task:** Isolate the authentication failure, ensure cross-environment compatibility between local development, Doppler, and GitHub Actions, and restore automated kernel pushing.
*   **Action:** Inspected Kaggle's extended API authorization protocol and identified that Kaggle introduced new format tokens (`KGAT...`) that require setting `KAGGLE_API_TOKEN` directly rather than the legacy `KAGGLE_USERNAME` + `KAGGLE_KEY` pair. Added an intelligent token-format detector in Python that dynamically routes keys and injects the database connection string at push time into private kernel packages.
*   **Result:** Successfully restored 100% automated GPU kernel execution on Kaggle T4/P100 hardware with zero manual token intervention.

---

## 📝 6. Summary Checklist for Interview Day

| Topic | Key Terminology to Mention |
| :--- | :--- |
| **Data Processing** | PySpark, Delta Lake, ACID, Medallion Architecture (Bronze/Silver/Gold), Adaptive Query Execution (AQE), Liquid Clustering |
| **Data Quality** | `try_cast()`, Dead-Letter Quarantine, Intra-batch deduplication, Data Provenance (`_source_file`, `_ingested_at`) |
| **Data Modeling** | Star Schema, Kimball Dimensional Modeling, SCD Type 2 temporal versioning, `is_current` flags |
| **Streaming** | Databricks Auto Loader (`cloudFiles`), Exactly-once semantics, Checkpoint offsets, `availableNow=True` |
| **AI / Embeddings** | SentenceTransformers (`all-mpnet-base-v2`), 768-D dense vectors, Apache Arrow zero-copy memory transfer, PySpark `to_json()` |
| **Vector Database** | Neon PostgreSQL, `pgvector`, HNSW vs. IVF, Cosine Similarity (`vector_cosine_ops`), Covering B-Tree Index |
| **DevOps & MLOps** | Doppler centralized secrets, Kaggle API kernel execution, MLflow experiment tracking, GitHub Actions CI/CD DAGs |
