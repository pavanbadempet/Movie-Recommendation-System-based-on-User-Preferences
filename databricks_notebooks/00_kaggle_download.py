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
logger.info("Initializing Kaggle Data Ingestion Pipeline...")

try:
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    dbutils.widgets.text("ENVIRONMENT", "dev", "Deployment Environment (dev, stg, prd)")
    
    env = dbutils.widgets.get("ENVIRONMENT")
    doppler_token = None

    # 1. Try Unity Catalog Volume FIRST (Primary Production Method)
    for token_name in [f"{env}_doppler_token.txt", "doppler_token.txt"]:
        try:
            token_path = f"/Volumes/apex/default/secrets/{token_name}"
            doppler_token = dbutils.fs.head(token_path).strip()
            if doppler_token:
                print(f"Loaded token from Volume: {token_path}")
                break
        except Exception:
            pass

    # 2. Fallback to Job Parameter / Widget if Volume file not present
    if not doppler_token:
        doppler_token = dbutils.widgets.get("DOPPLER_TOKEN").strip()

    if not doppler_token:
        raise ValueError(f"DOPPLER_TOKEN is missing! Please upload '{env}_doppler_token.txt' to /Volumes/apex/default/secrets/")
        
    response = requests.get(
        "https://api.doppler.com/v3/configs/config/secrets",
        headers={"Authorization": f"Bearer {doppler_token}", "Accept": "application/json"}
    )
    response.raise_for_status()
    secrets = response.json()["secrets"]
    
    os.environ['KAGGLE_USERNAME'] = secrets.get("KAGGLE_USERNAME", {}).get("computed")
    os.environ['KAGGLE_KEY'] = secrets.get("KAGGLE_KEY", {}).get("computed")
    
    if not os.environ['KAGGLE_USERNAME'] or not os.environ['KAGGLE_KEY']:
        raise ValueError("Kaggle credentials not found in Doppler!")
        
except Exception as e:
    raise ValueError(f"Failed to retrieve Kaggle credentials from Doppler: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Download the Dataset

# COMMAND ----------
# -------------------------------------------------------------------------
# CONFIGURATION & CLEANUP
# -------------------------------------------------------------------------
KAGGLE_DATASET = "alanvourch/tmdb-movies-daily-updates"
volume_raw_dir = "/Volumes/apex/default/secrets/raw_data"

import glob
import time

start_time = time.time()
os.makedirs(volume_raw_dir, exist_ok=True)

# Clean up previous CSV files in the volume directory to prevent duplicate reads
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
print("Download and unzip complete!")

# COMMAND ----------
# MAGIC ## Save to Delta Lake Managed Table (Bronze Layer)
# COMMAND ----------
print(f"Reading raw CSV directly with PySpark from {volume_raw_dir}...")

# 1-Pass Ultra-Fast Ingestion (inferSchema=false saves 50% CPU time; strong typing happens in ETL)
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "false") \
    .option("multiLine", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .load(f"{volume_raw_dir}/*.csv")

from pyspark.sql.functions import col, current_timestamp

# Add Data Lineage & Provenance metadata columns (Unity Catalog Standard)
df = df.withColumn("_ingested_at", current_timestamp()) \
       .withColumn("_source_file", col("_metadata.file_path"))

print("Appending raw snapshot directly to Bronze Delta Lake Table 'apex.default.tmdb_raw_data'...")
df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("apex.default.tmdb_raw_data")

elapsed = round(time.time() - start_time, 2)
print(f"Data Ingestion Complete in {elapsed}s! Raw data is now persisted in Delta Lake format.")
