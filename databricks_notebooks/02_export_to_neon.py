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
