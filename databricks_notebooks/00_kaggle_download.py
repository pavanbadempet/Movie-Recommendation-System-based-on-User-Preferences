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
    # 1. Try to get it from the Job Parameter / Widget first
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    dbutils.widgets.text("ENVIRONMENT", "dev", "Deployment Environment (dev, stg, prd)")
    
    doppler_token = dbutils.widgets.get("DOPPLER_TOKEN")
    env = dbutils.widgets.get("ENVIRONMENT")
    
    # 2. Try Unity Catalog Volume (Using your 'apex' catalog!)
    if not doppler_token:
        try:
            # Lightweight Databricks DBFS/Volume utility (instant, zero Spark overhead)
            token_path = f"/Volumes/apex/default/secrets/{env}_doppler_token.txt"
            doppler_token = dbutils.fs.head(token_path).strip()
        except Exception:
            pass
            
    # 3. Try Local Workspace File (Fallback)
    if not doppler_token:
        # Check multiple common Databricks working directories
        for path in ["doppler_token.txt", "databricks_notebooks/doppler_token.txt", "../doppler_token.txt"]:
            if os.path.exists(path):
                with open(path, "r") as f:
                    doppler_token = f.read().strip()
                if doppler_token:
                    break
            
    if not doppler_token:
        cwd = os.getcwd()
        raise ValueError(f"DOPPLER_TOKEN is missing! Please create a Volume at apex.default.secrets and upload doppler_token.txt there.")
        
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

# Define temporary local path and final DBFS path
local_download_dir = "/tmp/kaggle_data"
dbfs_raw_dir = "/dbfs/FileStore/apex/data/raw/tmdb"

print(f"Downloading {KAGGLE_DATASET} from Kaggle...")

# We import kaggle here AFTER setting the environment variables so it authenticates properly
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

# Download and unzip the files locally on the cluster
os.makedirs(local_download_dir, exist_ok=True)
api.dataset_download_files(KAGGLE_DATASET, path=local_download_dir, unzip=True)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Save to Managed Table (Data Lake)
# COMMAND ----------
# Since DBFS is disabled in Serverless Free Tier, we use PySpark to read the CSV from the local /tmp folder
# and save it directly as a Databricks Managed Table!

print("Reading raw CSV into Spark...")
# Find the downloaded CSV in /tmp/kaggle_data
import glob
csv_files = glob.glob(f"{local_download_dir}/*.csv")
if not csv_files:
    raise FileNotFoundError("No CSV files found in the Kaggle download!")

raw_csv_path = f"file:{csv_files[0]}"

# Load it into memory with Pandas first to bypass the Serverless 'file:/tmp' read ban
import pandas as pd
print("Reading CSV into memory using Pandas (bypassing Databricks Serverless storage limits)...")

# Read everything as string to prevent Spark Arrow inference errors on mixed types
pandas_df = pd.read_csv(csv_files[0], dtype=str)

# Convert Pandas DataFrame directly to Spark DataFrame in-memory
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
df = spark.createDataFrame(pandas_df)

print("Saving to Managed Table 'apex.default.tmdb_raw_data'...")
df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable("apex.default.tmdb_raw_data")

print("Data Ingestion Complete! The raw data is ready for the Medallion ETL.")
