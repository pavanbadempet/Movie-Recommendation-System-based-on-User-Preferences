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
# Create widgets to accept Kaggle credentials securely
dbutils.widgets.text("KAGGLE_USERNAME", "", "Kaggle Username")
dbutils.widgets.text("KAGGLE_KEY", "", "Kaggle API Key")

os.environ['KAGGLE_USERNAME'] = dbutils.widgets.get("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = dbutils.widgets.get("KAGGLE_KEY")

if not os.environ['KAGGLE_USERNAME'] or not os.environ['KAGGLE_KEY']:
    raise ValueError("Kaggle credentials not provided! Please set them in the Job Parameters.")

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
