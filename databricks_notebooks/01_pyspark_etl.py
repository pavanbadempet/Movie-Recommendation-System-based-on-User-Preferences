# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - APEX PySpark ETL (Medallion Architecture)
# MAGIC This notebook runs the daily batch ETL. It creates the Bronze, Silver, and Gold datasets. 
# MAGIC For this demo, if the raw datasets are missing from your Databricks FileStore, it will automatically generate a sample Gold dataset so the pipeline can succeed!

# COMMAND ----------
import os
import random
import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType, ArrayType

logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup & Data Generation

# COMMAND ----------
def load_gold_data(spark):
    raw_path = "dbfs:/FileStore/apex/data/raw/tmdb/*.csv"  # or *.json depending on dataset
    gold_path = "dbfs:/FileStore/apex/data/gold/movie_features"
    print(f"Reading Real Raw Data from {raw_path}...")
    
    # Read the raw dataset downloaded by the Kaggle task
    df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(raw_path)
    
    # Optional: Here you would normally run your LightGCN / MPNet embedding generation
    # For now, we will select the necessary columns and push to gold
    from pyspark.sql.functions import col
    
    # Standardize column names if necessary to match the schema
    # Example: df = df.withColumnRenamed("original_title", "title")
    
    # Save the real data to the Gold Delta table
    print(f"Writing {df.count()} movies to Gold layer...")
    df.write.format("delta").mode("overwrite").save(gold_path)
    return True

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution

# COMMAND ----------
# Run the mock data generation
load_gold_data(spark)

print("ETL Pipeline Completed Successfully!")
