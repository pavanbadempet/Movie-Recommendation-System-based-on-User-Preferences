# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Export Gold Features to Neon Postgres
# MAGIC This script pulls the final Gold tables (which power the ML models) and exports them to a Serverless Neon Postgres database.
# MAGIC This allows the Hugging Face Space to read the data 24/7 without needing Databricks to be awake.

# COMMAND ----------
# MAGIC %pip install psycopg2-binary sqlalchemy pandas

# COMMAND ----------
import os
import pandas as pd
from sqlalchemy import create_engine
import requests

# COMMAND ----------
# Define the Neon Database URL (Configure this in Doppler)

try:
    # 1. Try to get it from the Job Parameter / Widget first
    dbutils.widgets.text("DOPPLER_TOKEN", "", "Doppler Service Token")
    doppler_token = dbutils.widgets.get("DOPPLER_TOKEN")
    
    # 2. Try Unity Catalog Volume (Using your 'apex' catalog!)
    if not doppler_token:
        try:
            with open("/Volumes/apex/default/secrets/doppler_token.txt", "r") as f:
                doppler_token = f.read().strip()
        except FileNotFoundError:
            pass
            
    # 3. Try Local Workspace File (Fallback)
    if not doppler_token:
        # Check multiple common Databricks working directories
        import os
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
    
    DATABASE_URL = secrets.get("DATABASE_URL", {}).get("computed")
    
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL not found in Doppler!")
        
except Exception as e:
    raise ValueError(f"Failed to retrieve DATABASE_URL from Doppler: {e}")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is not set!")

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# COMMAND ----------
# MAGIC ## 3. Read Data from Gold Table & Export
# COMMAND ----------

gold_table_name = "apex.default.tmdb_gold_data"
print(f"Reading Gold table from {gold_table_name}...")

# Load the Delta table into a Spark DataFrame
df_spark = spark.table(gold_table_name)
if "is_current" in df_spark.columns:
    df_spark = df_spark.filter(df_spark.is_current == True)

df_pandas = df_spark.toPandas()

# COMMAND ----------
print(f"Connecting to Neon Postgres...")
engine = create_engine(DATABASE_URL)

print(f"Exporting {len(df_pandas)} rows to Postgres table 'movies'...")
df_pandas.to_sql("movies", engine, if_exists="replace", index=False)

print("Export Complete! Hugging Face UI can now read the latest features.")
