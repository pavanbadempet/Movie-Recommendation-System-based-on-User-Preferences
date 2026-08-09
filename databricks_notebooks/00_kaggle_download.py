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
    doppler_token = dbutils.widgets.get("DOPPLER_TOKEN")
    
    # 2. Try Unity Catalog Volumes (New Serverless Standard)
    if not doppler_token:
        try:
            with open("/Volumes/main/default/secrets/doppler_token.txt", "r") as f:
                doppler_token = f.read().strip()
        except FileNotFoundError:
            pass
            
    # 3. Try DBFS (Older Workspaces)
    if not doppler_token:
        try:
            with open("/dbfs/FileStore/doppler_token.txt", "r") as f:
                doppler_token = f.read().strip()
        except FileNotFoundError:
            pass
            
    if not doppler_token:
        raise ValueError("DOPPLER_TOKEN is missing! Please store it in Unity Catalog or pass it as a Job Parameter.")
        
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
# MAGIC ## Move to DBFS (Data Lake)

# COMMAND ----------
print(f"Moving extracted files to DBFS at {dbfs_raw_dir}...")
os.makedirs(dbfs_raw_dir, exist_ok=True)

# Move all CSV/JSON files to the permanent DBFS location
for filename in os.listdir(local_download_dir):
    source_file = os.path.join(local_download_dir, filename)
    dest_file = os.path.join(dbfs_raw_dir, filename)
    
    if os.path.isfile(source_file):
        shutil.move(source_file, dest_file)
        print(f"Saved: {filename}")

print("Data Ingestion Complete! The raw data is ready for the Medallion ETL.")
