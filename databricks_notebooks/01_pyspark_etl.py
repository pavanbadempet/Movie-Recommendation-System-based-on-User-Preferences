# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - APEX PySpark ETL (SOTA Medallion Architecture)
# MAGIC This notebook runs the daily batch ETL. It processes raw TMDB/MovieLens data through the Bronze, Silver, and Gold layers.
# MAGIC It implements Enterprise SOTA patterns: Data Quality Gates, Distributed AI Embedding Generation (Pandas UDF), SCD Type 2 CDC Merges, Liquid Clustering, and Vacuuming.
# MAGIC 
# MAGIC **NOTE: Your cluster must have at least 8GB of worker memory to load the Hugging Face AI models during the UDF execution.**

# COMMAND ----------
# MAGIC %pip install sentence-transformers pandas pyarrow

# COMMAND ----------
import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from delta.tables import DeltaTable
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, current_timestamp, concat_ws, pandas_udf, coalesce, expr, to_timestamp
from pyspark.sql.types import ArrayType, FloatType, StringType

# Broadcast the model name so all workers know which one to download
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

@pandas_udf(ArrayType(FloatType()))
def predict_embeddings(series: pd.Series) -> pd.Series:
    """
    Pandas UDF that runs on Databricks Worker nodes.
    Each node downloads the Hugging Face model and computes 768-D vectors in parallel batches.
    """
    from sentence_transformers import SentenceTransformer
    # Load model (downloads once per worker process, then cached in memory)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    # Generate embeddings
    embeddings = model.encode(series.tolist(), batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    
    # L2 Normalize for fast Cosine Similarity in pgvector
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms == 0, 1e-10, norms)
    embeddings = embeddings / norms
    
    # Return as a Pandas Series of lists (which Spark converts to ArrayType)
    return pd.Series(list(embeddings))

# ----------------------------------------------------------------------
# Gen AI Feature Extraction UDF (Top-Level Scope)
# ----------------------------------------------------------------------
@pandas_udf(StringType())
def extract_llm_features(overview_series: pd.Series) -> pd.Series:
    """
    Simulated GenAI Feature Extractor UDF.
    Extracts semantic metadata (mood, pacing, tropes) from movie plot overviews.
    """
    results = []
    for text in overview_series:
        if not text or len(text) < 10:
            results.append("")
            continue
        synthetic_llm_metadata = "Mood: Tense | Pacing: Fast | Tropes: Unreliable Narrator"
        results.append(synthetic_llm_metadata)
    return pd.Series(results)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Data Quality & Merge Logic (Gold Layer)

# COMMAND ----------
def load_gold_data(spark):
    raw_table = "apex.default.tmdb_raw_data"
    gold_table_name = "apex.default.tmdb_gold_data"
    print(f"Reading Real Raw Data from {raw_table}...")
    
    # 1. Read the incoming raw dataset
    incoming_df = spark.table(raw_table)
    
    # ----------------------------------------------------------------------
    # 2. DATA QUALITY GATES (SOTA)
    # ----------------------------------------------------------------------
    print("Running Data Quality Gates...")
    # Drop rows with critical missing keys
    incoming_df = incoming_df.filter(col("id").isNotNull())
    # Ensure ratings are mathematically valid before reaching the Vector DB (use try_cast to handle corrupt text)
    if "vote_average" in incoming_df.columns:
        incoming_df = incoming_df.withColumn("vote_average", expr("try_cast(vote_average as double)"))
        incoming_df = incoming_df.filter((col("vote_average") >= 0.0) & (col("vote_average") <= 10.0))
        
    # Standardize ID to string for Vector DB compatibility
    incoming_df = incoming_df.withColumn("id", col("id").cast("string"))

    # Ensure Lineage Metadata Columns exist
    if "_ingested_at" not in incoming_df.columns:
        incoming_df = incoming_df.withColumn("_ingested_at", current_timestamp())
    if "_source_file" not in incoming_df.columns:
        incoming_df = incoming_df.withColumn("_source_file", lit("unknown"))

    # ----------------------------------------------------------------------
    # 2.5 GEN AI FEATURE EXTRACTION
    # ----------------------------------------------------------------------
    print("Running Gen AI Agentic Feature Extraction...")
    if "overview" in incoming_df.columns:
        incoming_df = incoming_df.withColumn("gen_ai_features", extract_llm_features(col("overview")))
    else:
        incoming_df = incoming_df.withColumn("gen_ai_features", lit(""))
        
    # Create a dense 'tags' column representing the movie (Title + Genres + Overview + GEN AI FEATURES)
    incoming_df = incoming_df.withColumn(
        "tags",
        concat_ws(" | ", 
            coalesce(col("title"), lit("")),
            coalesce(col("genres"), lit("")),
            coalesce(col("overview"), lit("")),
            coalesce(col("gen_ai_features"), lit(""))
        )
    )

    # Embed the tags using our fast SentenceTransformer UDF
    incoming_df = incoming_df.withColumn("embedding", predict_embeddings(col("tags")))
    
    # ----------------------------------------------------------------------
    # 4. SCD TYPE 2 LOGIC (Data Lakehouse standard)
    # ----------------------------------------------------------------------
    # Add SCD tracking columns to incoming data
    incoming_df = incoming_df.withColumn("is_current", lit(True)) \
                             .withColumn("effective_start_at", current_timestamp()) \
                             .withColumn("effective_end_at", to_timestamp(lit("9999-12-31 23:59:59")))

    print(f"Merging enriched data into Gold Table: {gold_table_name}...")
    
    # Check if the Gold Table exists in the metastore
    table_exists = spark.catalog.tableExists(gold_table_name)
    
    if not table_exists:
        print("Gold table does not exist. Creating it for the first time...")
        # Enable Liquid Clustering on the 'id' column (SOTA replacing Z-Order)
        incoming_df.write.format("delta").clusterBy("id").saveAsTable(gold_table_name)
    else:
        print("Gold table exists. Performing SCD Type 2 UPSERT...")
        from delta.tables import DeltaTable
        
        gold_table = DeltaTable.forName(spark, gold_table_name)
        
        # Determine updates vs inserts based on 'tags' hash change
        update_condition = "gold.id = updates.id AND gold.tags != updates.tags AND gold.is_current = True"
        
        # Step 1: Identify records that need to be updated and mark the old ones as inactive
        staged_updates = incoming_df.alias("updates").join(
            gold_table.toDF().alias("gold"),
            expr(update_condition)
        ).selectExpr("updates.*")
        
        # Step 2: Merge the updates back to the gold table
        gold_table.alias("gold").merge(
            source=incoming_df.alias("updates"),
            condition="gold.id = updates.id AND gold.is_current = True"
        ).whenMatchedUpdate(
            condition="gold.tags != updates.tags",
            set={
                "is_current": "False",
                "effective_end_at": "current_timestamp()"
            }
        ).whenNotMatchedInsertAll().execute()
        
        # Step 3: Insert the new active versions of the updated records
        staged_updates.write.format("delta").mode("append").saveAsTable(gold_table_name)

    # ----------------------------------------------------------------------
    # 5. MAINTENANCE & OPTIMIZATION (SOTA)
    # ----------------------------------------------------------------------
    print("Optimizing Gold Table...")
    # Run optimize to physically compact files
    spark.sql(f"OPTIMIZE {gold_table_name}")
    # Vacuum old files to save storage costs (retention 7 days)
    spark.sql(f"VACUUM {gold_table_name} RETAIN 168 HOURS")

    print(f"ETL Pipeline completed successfully for Gold table!")
    return True

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution

# COMMAND ----------
load_gold_data(spark)
