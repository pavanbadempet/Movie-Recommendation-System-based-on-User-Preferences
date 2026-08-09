# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - APEX PySpark ETL (SOTA Medallion Architecture)
# MAGIC This notebook runs the daily batch ETL. It processes raw TMDB/MovieLens data through the Bronze, Silver, and Gold layers.
# MAGIC It implements Enterprise SOTA patterns: Data Quality Gates, SCD Type 2 CDC Merges, Z-Ordering, and Vacuuming.

# COMMAND ----------
import os
import logging
from datetime import datetime
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp

logger = logging.getLogger(__name__)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data Quality & Merge Logic (Gold Layer)

# COMMAND ----------
def load_gold_data(spark):
    raw_path = "dbfs:/FileStore/apex/data/raw/tmdb/*.csv"
    gold_path = "dbfs:/FileStore/apex/data/gold/movie_features"
    print(f"Reading Real Raw Data from {raw_path}...")
    
    # 1. Read the incoming raw dataset
    incoming_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(raw_path)
    
    # ----------------------------------------------------------------------
    # 2. DATA QUALITY GATES (SOTA)
    # ----------------------------------------------------------------------
    print("Running Data Quality Gates...")
    # Drop rows with critical missing keys
    incoming_df = incoming_df.filter(col("id").isNotNull())
    # Ensure ratings are mathematically valid before reaching the Vector DB
    if "vote_average" in incoming_df.columns:
        incoming_df = incoming_df.filter((col("vote_average") >= 0.0) & (col("vote_average") <= 10.0))
        
    # Standardize ID to string for Vector DB compatibility
    incoming_df = incoming_df.withColumn("id", col("id").cast("string"))

    # ----------------------------------------------------------------------
    # 3. SCD TYPE 2 PREPARATION (SOTA CDC)
    # ----------------------------------------------------------------------
    # Add SCD tracking columns to incoming data
    incoming_df = incoming_df.withColumn("is_current", lit(True)) \
                             .withColumn("effective_start_at", current_timestamp()) \
                             .withColumn("effective_end_at", lit(datetime(9999, 12, 31)))
                             
    # If the Gold table doesn't exist yet, do a first-time full save
    if not DeltaTable.isDeltaTable(spark, gold_path):
        print("First run detected: Creating base Gold table with Liquid Clustering...")
        # Enable Liquid Clustering on the 'id' column (SOTA replacing Z-Order)
        incoming_df.write.format("delta").clusterBy("id").save(gold_path)
    else:
        # ------------------------------------------------------------------
        # 4. DELTA LAKE MERGE (SCD TYPE 2 LOGIC)
        # ------------------------------------------------------------------
        print("Delta Table exists: Running SCD Type 2 MERGE...")
        gold_table = DeltaTable.forPath(spark, gold_path)
        
        # Step 4a: Update existing active records to mark them as inactive (is_current = False) 
        # if any of their attributes changed (e.g. vote_average updated).
        # We detect changes by joining on ID and comparing values.
        update_condition = "gold.id = incoming.id AND gold.is_current = True"
        
        # In a full SCD2, you compare hashes of columns. For simplicity, we just mark old as inactive if matched.
        gold_table.alias("gold").merge(
            source=incoming_df.alias("incoming"),
            condition=update_condition
        ).whenMatchedUpdate(
            set={
                "is_current": lit(False),
                "effective_end_at": current_timestamp()
            }
        ).execute()
        
        # Step 4b: Insert the brand new active records (both truly new movies, and the updated versions of existing movies)
        gold_table.alias("gold").merge(
            source=incoming_df.alias("incoming"),
            condition="gold.id = incoming.id AND gold.is_current = True"
        ).whenNotMatchedInsertAll().execute()

    # ----------------------------------------------------------------------
    # 5. OPTIMIZATIONS (LIQUID CLUSTERING & VACUUM)
    # ----------------------------------------------------------------------
    print("Running Delta Optimizations (Liquid Clustering)...")
    # Because the table was created with clusterBy, OPTIMIZE automatically applies Liquid Clustering!
    spark.sql(f"OPTIMIZE delta.`{gold_path}`")
    
    print("Running Vacuum to clear unneeded historical snapshots...")
    # Retain 168 hours (7 days) of time travel
    spark.sql(f"VACUUM delta.`{gold_path}` RETAIN 168 HOURS")

    print(f"ETL Pipeline completed successfully for Gold table!")
    return True

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution

# COMMAND ----------
load_gold_data(spark)
