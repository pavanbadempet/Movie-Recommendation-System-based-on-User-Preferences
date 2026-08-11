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
    if "embedding" in df_spark.columns:
        df_spark = df_spark.withColumn("embedding", to_json(col("embedding")))

    print(f"Converting Spark DataFrame to Pandas for PostgreSQL bulk export...")
    df_pandas = df_spark.toPandas()

    # -------------------------------------------------------------------------
    # NEON POSTGRESQL SYNC
    # -------------------------------------------------------------------------
    print(f"Connecting to Neon Postgres...")
    # EDGE CASE 2: SSL Mode Requirement for Serverless Postgres Connections
    engine = create_engine(
        DATABASE_URL,
        connect_args={"sslmode": "require"},
        pool_pre_ping=True
    )

    print(f"Exporting {len(df_pandas)} active records to Postgres table 'movies'...")
    df_pandas.to_sql("movies", engine, if_exists="replace", index=False)

    print("Export Complete! Neon PostgreSQL serving table 'movies' is now updated and ready for 24/7 web apps.")
