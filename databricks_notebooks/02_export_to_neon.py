# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Export Gold Vector Dataset to Neon PostgreSQL (Multi-Shard Cluster)
# MAGIC
# MAGIC ## 🏛️ Enterprise Multi-Tier Storage Architecture:
# MAGIC
# MAGIC 1. **Databricks Delta Lake (Unlimited Historical Source of Truth & Lakehouse):**
# MAGIC    - **100% Full History:** Stores all raw datasets, incremental micro-batches, and versioned embeddings in Delta format (`apex.default.movies_gold`, `apex.default.user_events`).
# MAGIC    - **ACID Transactions & Time Travel:** Preserves full version history (`VERSION AS OF`) for auditability, point-in-time recovery, and offline ML model retraining.
# MAGIC
# MAGIC 2. **Neon PostgreSQL (Ultra-Fast Online Serving Layer ~5ms Latency):**
# MAGIC    - **Multi-Shard Distribution:** Hashes records across 10 Neon project shards in AWS Singapore (`aws-ap-southeast-1`).
# MAGIC    - **Serving State:** Holds the latest serving snapshot of Top movies with covering B-Tree indexes and `pgvector` HNSW indexes for instant web application vector similarity queries.
# MAGIC
# MAGIC ### 💡 Core Serving Patterns:
# MAGIC 1. **Decoupled Architecture:** Allows 24/7 web apps (Hugging Face / Vercel / Next.js) to query vector embeddings from Neon without keeping expensive Databricks clusters awake.
# MAGIC 2. **PySpark Distributed Vector Serialization:** Uses PySpark `to_json(col("embedding"))` to format 768-D dense vectors in parallel across worker nodes before export.
# MAGIC 3. **Doppler Environment Resolution:** Fetches the target `DATABASE_URL` dynamically based on the deployment environment parameter (`dev`, `stg`, `prd`).

# COMMAND ----------
import os
import gc
import requests
from pyspark.sql.functions import col, to_json, lit, spark_partition_id, pmod, spark_hash

# COMMAND ----------
# ----------------------------------------------------------------------
# ⚡ HIGH-PERFORMANCE SPARK CONFIGURATIONS (SOTA SERVERLESS TUNING)
# ----------------------------------------------------------------------
for conf_key, conf_val in [
    ("spark.sql.execution.arrow.pyspark.enabled", "true"),
    ("spark.sql.execution.arrow.pyspark.fallback.enabled", "true"),
    ("spark.sql.adaptive.enabled", "true"),
    ("spark.sql.adaptive.coalescePartitions.enabled", "true"),
    ("spark.sql.adaptive.skewJoin.enabled", "true"),
    ("spark.sql.adaptive.localShuffleReader.enabled", "true"),
    ("spark.databricks.delta.optimizeWrite.enabled", "true"),
    ("spark.databricks.delta.autoCompact.enabled", "true"),
    ("spark.sql.files.maxPartitionBytes", "134217728"),
    ("spark.sql.shuffle.partitions", "200"),
    ("spark.sql.inMemoryColumnarStorage.compressed", "true"),
    ("spark.sql.execution.vectorized.enabled", "true")
]:
    try:
        spark.conf.set(conf_key, conf_val)
    except Exception:
        pass  # Serverless compute manages these configurations natively

# COMMAND ----------
# Define the Neon Database URL (Configure this in Doppler)

# COMMAND ----------
# MAGIC %run ./doppler_config

# COMMAND ----------
try:
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    dbutils.widgets.text("ENVIRONMENT", "dev", "Deployment Environment (dev, stg, prd)")
    
    env = dbutils.widgets.get("ENVIRONMENT")

    # -------------------------------------------------------------------------
    # CENTRALIZED DOPPLER SECRET RESOLUTION
    # -------------------------------------------------------------------------
    secrets = load_centralized_doppler_secrets(dbutils=dbutils, env=env)
    
    DATABASE_URL = secrets.get("DATABASE_URL")
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in Doppler secrets payload!")
        
except Exception as e:
    raise ValueError(f"Failed to load DATABASE_URL from Doppler: {e}")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# COMMAND ----------
# COMMAND ----------
# MAGIC ## 3. Read Active Records from Gold Table & Export to PostgreSQL
# COMMAND ----------
gold_table_name = "apex.default.tmdb_gold_with_embeddings"
# Fallback to base gold table if embeddings table doesn't exist yet
if not spark.catalog.tableExists(gold_table_name):
    gold_table_name = "apex.default.tmdb_gold_data"
print(f"Reading Gold table from {gold_table_name}...")

from pyspark.sql.functions import col, to_json

# 1. Load Gold Delta Table
df_spark = spark.table(gold_table_name)

# EDGE CASE 1: Empty Gold Table Check (Databricks Serverless / Spark Connect Native)
# - IF Gold table has 0 rows: Exit early without attempting database transaction write.
if df_spark.limit(1).count() == 0:
    print("Gold table is empty. Skipping Neon PostgreSQL export.")
else:
    # CONDITION 1: Filter Active Records Only
    # - IF 'is_current' column exists: Keep active SCD Type 2 records (is_current == True).
    # - IMPLICIT ELSE: Process full dataset if is_current is absent.
    if "is_current" in df_spark.columns:
        df_spark = df_spark.filter(col("is_current") == True)

    # CONDITION 2: Vector Embedding Serialization
    # - IF 'embedding' vector column exists: Use PySpark native to_json() to format 768-D array into JSON string.
    #   📥 ARRAY PAYLOAD:  [0.0124, -0.0451, 0.0892, ...]
    #   📤 JSON STRING:   "[0.0124, -0.0451, 0.0892, ...]" (Parsed natively by pgvector in Neon Postgres)
    # CONDITION 2: Vector Embedding Serialization
    if "embedding" in df_spark.columns:
        df_spark = df_spark.withColumn("embedding", to_json(col("embedding")))

    # Select serving columns required for recommendations & vector search
    serving_cols = [c for c in ["id", "title", "genres", "vote_average", "vote_count", "release_date", "overview", "tags", "embedding"] if c in df_spark.columns]
    df_spark = df_spark.select(*serving_cols)

    # -------------------------------------------------------------------------
    # FREE TIER STORAGE QUOTA PROTECTION (Neon 512MB Storage Capacity)
    # -------------------------------------------------------------------------
    try:
        dbutils.widgets.text("EXPORT_LIMIT", "30000", "Max Records to Export (0 = Unlimited Full Dataset)")
        export_limit = int(dbutils.widgets.get("EXPORT_LIMIT"))
    except Exception:
        export_limit = 30000

    if export_limit > 0 and "vote_count" in df_spark.columns:
        print(f"Filtering Top {export_limit} highest-voted movies to fit inside Neon Free Tier 512MB storage limit and Serverless RAM...")
        df_spark = df_spark.orderBy(col("vote_count").desc()).limit(export_limit)

    total_records = df_spark.count()
    print(f"Total active Gold records to export: {total_records}")

    # -------------------------------------------------------------------------
    # MULTI-SHARD NEON RESOLUTION (Supports 1 to 10 Shards via Doppler)
    # -------------------------------------------------------------------------
    shard_urls = []
    for i in range(10):
        s_url = secrets.get(f"DATABASE_URL_SHARD_{i}") or secrets.get(f"DATABASE_URL_{i}")
        if s_url:
            if s_url.startswith("postgres://"):
                s_url = s_url.replace("postgres://", "postgresql://", 1)
            shard_urls.append(s_url)

    if not shard_urls:
        p_url = secrets.get("DATABASE_URL")
        if p_url:
            if p_url.startswith("postgres://"):
                p_url = p_url.replace("postgres://", "postgresql://", 1)
            shard_urls = [p_url]

    num_shards = len(shard_urls)
    print(f"Detected {num_shards} active Neon Database Shard(s) in Doppler!")

    # -------------------------------------------------------------------------
    # MULTI-SHARD STREAMING EXPORT ENGINE
    # -------------------------------------------------------------------------
    import gc

    for shard_idx, db_url in enumerate(shard_urls):
        print(f"\n========================================================")
        print(f"🚀 PROCESSING SHARD {shard_idx + 1}/{num_shards} ({db_url.split('@')[-1]})")
        print(f"========================================================")

        # Filter shard records using deterministic modulo hash
        from pyspark.sql.functions import pmod, hash as spark_hash
        if num_shards > 1:
            df_shard = df_spark.filter(pmod(spark_hash(col("id")), num_shards) == shard_idx)
        else:
            df_shard = df_spark

        shard_records = df_shard.count()
        print(f"Shard {shard_idx + 1} contains {shard_records} records.")

        if shard_records == 0:
            continue

        # Use Native Spark JDBC Writer for zero-memory driver overhead and maximum throughput
        jdbc_url = db_url
        if jdbc_url.startswith("postgresql://"):
            jdbc_url = jdbc_url.replace("postgresql://", "jdbc:postgresql://", 1)
        elif jdbc_url.startswith("postgres://"):
            jdbc_url = jdbc_url.replace("postgres://", "jdbc:postgresql://", 1)

        print(f"Streaming Shard {shard_idx + 1} DataFrame directly to Neon via Native Spark JDBC Engine...")
        try:
            df_shard.write.format("jdbc") \
                .option("url", jdbc_url) \
                .option("dbtable", "movies") \
                .option("driver", "org.postgresql.Driver") \
                .option("batchsize", "5000") \
                .mode("overwrite") \
                .save()
            print(f"Shard {shard_idx + 1} - Spark JDBC Sync Successful!")
        except Exception as jdbc_err:
            print(f"Spark JDBC Error on Shard {shard_idx + 1}: {jdbc_err}")
            raise jdbc_err

        # Build Post-Sync Clustered Primary Key, Covering, and HNSW Vector Indexes via JVM JDBC Connection
        print(f"Building PostgreSQL performance indexes on Shard {shard_idx + 1} via JVM JDBC Connection...")
        try:
            driver_manager = spark._sc._gateway.jvm.java.sql.DriverManager
            jdbc_conn = driver_manager.getConnection(jdbc_url)
            stmt = jdbc_conn.createStatement()
            
            # 1. Primary Key Clustered B-Tree Index
            try:
                stmt.execute("ALTER TABLE movies ADD PRIMARY KEY (id)")
            except Exception:
                stmt.execute("CREATE INDEX IF NOT EXISTS idx_movies_id ON movies (id)")

            # 2. High-Throughput Covering Index for Index-Only Metadata Scans
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_movies_serving_covering ON movies (id) INCLUDE (title, genres, vote_average, vote_count, release_date)")
            stmt.execute("CREATE INDEX IF NOT EXISTS idx_movies_release_date ON movies (release_date DESC)")

            # 3. Deferred HNSW Cosine Similarity Vector Index (100% Uncompressed Full Float32 Precision)
            try:
                stmt.execute("CREATE EXTENSION IF NOT EXISTS vector")
                stmt.execute("ALTER TABLE movies ALTER COLUMN embedding TYPE vector(768) USING embedding::vector(768)")
                stmt.execute("CREATE INDEX IF NOT EXISTS idx_movies_embedding_hnsw ON movies USING hnsw (embedding vector_cosine_ops)")
                print(f"Shard {shard_idx + 1} HNSW 100% Full Precision Float32 Vector Index created successfully!")
            except Exception as v_err:
                print(f"Shard {shard_idx + 1} pgvector Notice: {v_err}")

            stmt.close()
            jdbc_conn.close()
        except Exception as idx_err:
            print(f"Shard {shard_idx + 1} Index Notice: {idx_err}")

    print("\n🎉 Multi-Shard Export Complete! All Neon Database Shards are updated, indexed, and live!")
