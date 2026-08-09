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

try:
    # In Databricks Serverless, we can't set global environment variables easily.
    # We pass the Doppler Token securely via a Job Parameter (Widget) at runtime.
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    doppler_token = dbutils.widgets.get("DOPPLER_TOKEN")
    
    if not doppler_token:
        raise ValueError("Please paste your DOPPLER_TOKEN in the text box at the top of the notebook!")
        
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
# Define the dataset. (Assuming the popular asaniczka/tmdb-movies-dataset-2023-10k-movies or similar)
# Change this string to whatever specific TMDB Kaggle dataset you are using!
KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-10k-movies"

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
