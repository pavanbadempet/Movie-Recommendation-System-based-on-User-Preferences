# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Automated Kaggle Data Ingestion (Bronze Layer)
# MAGIC
# MAGIC ## 📌 Overview & System Architecture
# MAGIC This notebook performs automated, production-grade ingestion of the daily TMDB Movies dataset directly into Databricks.
# MAGIC
# MAGIC ### 💡 Key Design Decisions & Speed Optimizations:
# MAGIC 1. **Doppler Secrets Management:** Fetches Kaggle API tokens securely via Unity Catalog Volumes (`/Volumes/apex/default/secrets/`).
# MAGIC 2. **Direct-to-Volume Download:** Downloads raw dataset archives directly into Unity Catalog Volumes (`/Volumes/apex/default/secrets/raw_data`), enabling PySpark to read raw files natively.
# MAGIC 3. **Single-Pass Fast Ingestion (`inferSchema=false`):** Avoids PySpark's default 2-pass full dataset scan, saving ~50% CPU ingestion time.
# MAGIC 4. **Data Lineage & Provenance:** Automatically stamps every raw row with `_source_file` (`_metadata.file_path`) and `_ingested_at` (`current_timestamp()`) for 100% auditability.
# MAGIC 5. **Append-Only Bronze Ledger:** Appends raw snapshots (`mode("append")`) to preserve full raw historical lineage.

# COMMAND ----------
%pip install kaggle

# COMMAND ----------
import os
import glob
import time
import logging
import requests

# Configure Structured Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("IngestionPipeline")

start_time = time.time()
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
    
    if not os.environ.get('KAGGLE_USERNAME') or not os.environ.get('KAGGLE_KEY'):
        raise ValueError("Kaggle credentials not found in Doppler secrets payload!")
        
except Exception as e:
    raise ValueError(f"Failed to load Kaggle credentials from Doppler: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Download the Dataset
# COMMAND ----------
KAGGLE_DATASET = "alanvourch/tmdb-movies-daily-updates"
volume_raw_dir = "/Volumes/apex/default/secrets/raw_data"

# Ensure target directory exists on Unity Catalog Volume
os.makedirs(volume_raw_dir, exist_ok=True)

# EDGE CASE 3: Stale CSV File Cleanup
# Removes previous CSV files to prevent duplicate reads during PySpark glob ingestion.
for old_csv in glob.glob(f"{volume_raw_dir}/*.csv"):
    try:
        os.remove(old_csv)
    except Exception:
        pass

from kaggle.api.kaggle_api_extended import KaggleApi

print(f"Authenticating Kaggle API & downloading {KAGGLE_DATASET} to {volume_raw_dir}...")
api = KaggleApi()
api.authenticate()

# Download and unzip directly into the Volume
api.dataset_download_files(KAGGLE_DATASET, path=volume_raw_dir, unzip=True)

print(f"Download and unzip complete! Found raw file: {downloaded_csvs[0]}")

MOVIELENS_DATASET = "grouplens/movielens-20m-dataset"
volume_movielens_dir = "/Volumes/apex/default/secrets/raw_movielens"
os.makedirs(volume_movielens_dir, exist_ok=True)
try:
    print(f"Downloading MovieLens 20M dataset ({MOVIELENS_DATASET}) to {volume_movielens_dir}...")
    api.dataset_download_files(MOVIELENS_DATASET, path=volume_movielens_dir, unzip=True)
    print("MovieLens 20M download complete!")
except Exception as ml_err:
    print(f"MovieLens download note: {ml_err}")

# COMMAND ----------
# MAGIC ## Save to Delta Lake Managed Table (Bronze Layer)
# COMMAND ----------
print(f"Reading raw CSV directly with PySpark from {volume_raw_dir}...")

# ----------------------------------------------------------------------
# ⚡ HIGH-PERFORMANCE SPARK CONFIGURATIONS (SAFE SERVERLESS TUNING)
# ----------------------------------------------------------------------
for conf_key, conf_val in [
    ("spark.sql.adaptive.enabled", "true"),
    ("spark.sql.adaptive.coalescePartitions.enabled", "true"),
    ("spark.databricks.delta.optimizeWrite.enabled", "true"),
    ("spark.databricks.delta.autoCompact.enabled", "true"),
    ("spark.sql.files.maxPartitionBytes", "134217728")
]:
    try:
        spark.conf.set(conf_key, conf_val)
    except Exception:
        pass  # Databricks Serverless manages these configurations natively

# 1-Pass Fast Ingestion (inferSchema=false saves 50% CPU overhead)
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("multiLine", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .load(f"{volume_raw_dir}/*.csv")

from pyspark.sql.functions import col, current_timestamp

# Add Data Lineage & Provenance metadata columns (Unity Catalog Standard)
# - _ingested_at: Timestamp when raw batch entered Bronze layer
# - _source_file: Exact Volume file path (col("_metadata.file_path"))
df = df.withColumn("_ingested_at", current_timestamp()) \
       .withColumn("_source_file", col("_metadata.file_path"))

print("Appending raw snapshot directly to Bronze Delta Lake Table 'apex.default.tmdb_raw_data'...")
df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("apex.default.tmdb_raw_data")

# Ingest MovieLens ratings and links into Bronze Delta tables if available
if os.path.exists(f"{volume_movielens_dir}/ratings.csv"):
    ratings_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").load(f"{volume_movielens_dir}/ratings.csv")
    ratings_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("apex.default.movielens_ratings_raw")
    print("Ingested MovieLens Ratings into 'apex.default.movielens_ratings_raw'")

if os.path.exists(f"{volume_movielens_dir}/links.csv"):
    links_df = spark.read.format("csv").option("header", "true").option("inferSchema", "false").load(f"{volume_movielens_dir}/links.csv")
    links_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("apex.default.movielens_links_raw")
    print("Ingested MovieLens Links into 'apex.default.movielens_links_raw'")

elapsed = round(time.time() - start_time, 2)
print(f"Data Ingestion Complete in {elapsed}s! Raw data is now persisted in Bronze Delta Lake format.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 📊 Visual Verification & Querying (Bronze Table)

# COMMAND ----------
# MAGIC %sql
# MAGIC SELECT _ingested_at, _source_file, count(1) AS raw_record_count
# MAGIC FROM apex.default.tmdb_raw_data
# MAGIC GROUP BY _ingested_at, _source_file
# MAGIC ORDER BY _ingested_at DESC
# MAGIC LIMIT 10;
