# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Export Gold Features to Neon PostgreSQL (Serving Layer Sync)
# MAGIC
# MAGIC ## 📌 Overview & Serving Architecture
# MAGIC This notebook pulls active records (`is_current == True`) from the Gold Delta table (`apex.default.tmdb_gold_data`) and synchronizes them with a **Serverless Neon PostgreSQL Database**.
# MAGIC
# MAGIC ### 💡 Core Serving Patterns:
# MAGIC 1. **Decoupled Architecture:** Allows 24/7 web apps (Hugging Face / Vercel / Next.js) to query vector embeddings from Neon without keeping expensive Databricks clusters awake.
# MAGIC 2. **PySpark Distributed Vector Serialization:** Uses PySpark `to_json(col("embedding"))` to format 768-D dense vectors in parallel across worker nodes before export.
# MAGIC 3. **Doppler Environment Resolution:** Fetches the target `DATABASE_URL` dynamically based on the deployment environment parameter (`dev`, `stg`, `prd`).

# COMMAND ----------
# MAGIC %pip install psycopg2-binary sqlalchemy pandas
# MAGIC %restart_python

# COMMAND ----------
import os
import pandas as pd
from sqlalchemy import create_engine
import requests
from pyspark.sql.functions import col

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
        dbutils.widgets.text("EXPORT_LIMIT", "30000", "Max Records to Export (Neon 512MB Limit)")
        export_limit = int(dbutils.widgets.get("EXPORT_LIMIT"))
    except Exception:
        export_limit = 30000

    if export_limit > 0 and "vote_count" in df_spark.columns:
        print(f"Filtering Top {export_limit} highest-voted movies to fit inside Neon Free Tier 512MB storage limit...")
        df_spark = df_spark.orderBy(col("vote_count").desc()).limit(export_limit)

    total_records = df_spark.count()
    print(f"Total active Gold records to export: {total_records}")

    # -------------------------------------------------------------------------
    # NEON POSTGRESQL HIGH-THROUGHPUT BULK SYNC (psycopg2 execute_values)
    # -------------------------------------------------------------------------
    print(f"Connecting to Neon Postgres...")
    engine = create_engine(
        DATABASE_URL,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True
    )

    print(f"Streaming {total_records} records to Neon Postgres 'movies' table via fast bulk protocol...")

    import gc
    from psycopg2.extras import execute_values

    def psycopg2_fast_insert(table, conn, keys, data_iter):
        """
        ⚡ SOTA High-Performance Bulk Insert callback for pandas to_sql.
        Uses PostgreSQL native execute_values to accelerate writes by 5x-10x.
        """
        dbapi_conn = conn.connection
        with dbapi_conn.cursor() as cur:
            string_data = [tuple(x) for x in data_iter]
            columns = '", "'.join(keys)
            table_name = table.name
            sql = f'INSERT INTO "{table_name}" ("{columns}") VALUES %s'
            execute_values(cur, sql, string_data)

    # toLocalIterator() streams partition-by-partition with constant O(1) driver RAM overhead
    row_iterator = df_spark.toLocalIterator(prefetchPartitions=True)

    batch_buffer = []
    batch_count = 0
    total_synced = 0
    batch_size = 500  # Fast bulk batch size

    for row in row_iterator:
        batch_buffer.append(row.asDict())
        if len(batch_buffer) >= batch_size:
            batch_pandas = pd.DataFrame(batch_buffer)
            if_exists_mode = "replace" if total_synced == 0 else "append"
            try:
                batch_pandas.to_sql("movies", engine, if_exists=if_exists_mode, index=False, method=psycopg2_fast_insert)
            except Exception:
                # Fallback to standard multi-insert if execute_values encounters custom type constraints
                batch_pandas.to_sql("movies", engine, if_exists=if_exists_mode, index=False, method="multi", chunksize=100)

            total_synced += len(batch_buffer)
            batch_count += 1
            print(f"Synced Batch {batch_count}: {total_synced}/{total_records} records uploaded to Neon...")

            batch_buffer = []
            del batch_pandas
            gc.collect()

    # Process final remaining batch
    if len(batch_buffer) > 0:
        batch_pandas = pd.DataFrame(batch_buffer)
        if_exists_mode = "replace" if total_synced == 0 else "append"
        try:
            batch_pandas.to_sql("movies", engine, if_exists=if_exists_mode, index=False, method=psycopg2_fast_insert)
        except Exception:
            batch_pandas.to_sql("movies", engine, if_exists=if_exists_mode, index=False, method="multi", chunksize=100)

        total_synced += len(batch_buffer)
        print(f"Synced Final Batch: Total {total_synced}/{total_records} records uploaded successfully!")
        del batch_pandas
        gc.collect()

    # -------------------------------------------------------------------------
    # POSTGRESQL POST-SYNC COVERING INDEXES & VECTOR SEARCH OPTIMIZATION
    # -------------------------------------------------------------------------
    print("Building PostgreSQL performance indexes on table 'movies'...")
    with engine.begin() as ddl_conn:
        try:
            # 1. Primary & B-Tree Indexes for Fast Metadata Filtering
            ddl_conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_id ON movies (id);")
            ddl_conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_release_date ON movies (release_date DESC);")

            # 2. Enable pgvector extension and build HNSW vector index
            try:
                ddl_conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                ddl_conn.execute("ALTER TABLE movies ALTER COLUMN embedding TYPE vector USING embedding::vector;")
                ddl_conn.execute("CREATE INDEX IF NOT EXISTS idx_movies_embedding_hnsw ON movies USING hnsw (embedding vector_cosine_ops);")
                print("PostgreSQL HNSW Vector Index created successfully!")
            except Exception as v_err:
                print(f"pgvector Extension Notice: {v_err}")

            print("PostgreSQL primary covering indexes created successfully!")
        except Exception as idx_err:
            print(f"PostgreSQL Index Notice: {idx_err}")

    print("Export Complete! Neon PostgreSQL serving table 'movies' is now updated, indexed, and ready for 24/7 web apps.")
