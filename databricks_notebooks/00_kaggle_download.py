# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Automated Kaggle Data Ingestion
# MAGIC This notebook acts as Task 0 in our automated pipeline. It securely connects to the Kaggle API, downloads the latest TMDB movie dataset, unzips it, and saves it to the Databricks FileStore (DBFS) so the next ETL steps have fresh data.

# COMMAND ----------
# MAGIC %pip install kaggle

# COMMAND ----------
import os
import zipfile
import shutil

# COMMAND ----------
# MAGIC %md
# MAGIC ## Authentication Setup
# MAGIC We grab the Kaggle API credentials from the notebook widgets (which you will pass in via the Job Parameters).

# COMMAND ----------
import requests
import os

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
# CONFIGURATION
# -------------------------------------------------------------------------
# The Kaggle dataset identifier
KAGGLE_DATASET = "alanvourch/tmdb-movies-daily-updates"

# Download directly into Unity Catalog Volume so PySpark can read it natively
volume_raw_dir = "/Volumes/apex/default/secrets/raw_data"
os.makedirs(volume_raw_dir, exist_ok=True)

from kaggle.api.kaggle_api_extended import KaggleApi

print(f"Authenticating Kaggle API & downloading {KAGGLE_DATASET} to {volume_raw_dir}...")
api = KaggleApi()
api.authenticate()

# Download and unzip directly into the Volume
api.dataset_download_files(KAGGLE_DATASET, path=volume_raw_dir, unzip=True)
print("Download and unzip complete!")

# COMMAND ----------
# MAGIC ## Save to Delta Lake Managed Table
# COMMAND ----------
print(f"Reading raw CSV directly with PySpark from {volume_raw_dir}...")

# Load directly with PySpark (No Pandas, pure distributed Spark read)
df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("quote", "\"") \
    .option("escape", "\"") \
    .load(f"{volume_raw_dir}/*.csv")

from pyspark.sql.functions import col, current_timestamp

# Add Data Lineage & Provenance metadata columns (Unity Catalog Standard)
df = df.withColumn("_ingested_at", current_timestamp()) \
       .withColumn("_source_file", col("_metadata.file_path"))

print("Appending raw snapshot directly to Bronze Delta Lake Table 'apex.default.tmdb_raw_data'...")
df.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable("apex.default.tmdb_raw_data")

print("Data Ingestion Complete! Raw data is now persisted in Delta Lake format.")
