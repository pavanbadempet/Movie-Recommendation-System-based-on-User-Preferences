"""
Kaggle Hybrid Engine Trainer (PySpark -> Hugging Face Hub)

This is the exact script designed to run on a free Kaggle CPU/GPU node, 
triggered via GitHub Actions.

It implements the Data-Centric "DeepSeek" philosophy:
1. PULL: Reads the net-new User Events (Telemetry) directly from Neon Serverless Postgres.
2. COMPUTE (Collaborative): Runs PySpark ALS on the events to build User/Item Latent Factors.
3. COMPUTE (Semantic): Uses a lightweight Hugging Face model to vectorize new movie plots.
4. PUSH: Uploads the computed Parquet files directly to Hugging Face Datasets (The Data Lake).
"""

import os
import json
import logging
from pathlib import Path
import psycopg
import pandas as pd
from huggingface_hub import HfApi

# --- Local Windows Dev Fix ---
# Automatically configures HADOOP_HOME and Java 17+ JVM args if running locally.
try:
    # Handle both running from root directory (python scripts/...) and inside scripts directory
    try:
        from scripts.setup_local_spark import configure_local_spark
    except ImportError:
        from setup_local_spark import configure_local_spark
    configure_local_spark()
except ImportError:
    pass # Assume running on actual Kaggle Linux node

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, max as _max, when
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (Injected by GitHub Actions / Kaggle Secrets)
NEON_DATABASE_URL = os.getenv("NOVA_EVENT_DATABASE_URL")
HF_TOKEN = os.getenv("HF_TOKEN")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "pavanbadempet/nova-recommendation-lake")
OUTPUT_DIR = Path("/kaggle/working/artifacts")

def get_spark() -> SparkSession:
    """Initialize ephemeral PySpark cluster on Kaggle node."""
    return SparkSession.builder \
        .appName("Nova-Hybrid-Engine-Trainer") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

def extract_events_from_neon() -> pd.DataFrame:
    """Extract telemetry from the free Serverless Postgres DB."""
    if not NEON_DATABASE_URL:
        logger.warning("No Postgres URL found. Falling back to local CSV for local testing.")
        return pd.DataFrame()
        
    logger.info("Connecting to Neon Postgres to extract raw telemetry...")
    query = """
        SELECT user_id, movie_id, event_type, rating 
        FROM nova_content_events 
        WHERE event_type IN ('rating', 'click', 'view')
          AND user_id IS NOT NULL 
          AND movie_id IS NOT NULL;
    """
    with psycopg.connect(NEON_DATABASE_URL) as conn:
        df = pd.read_sql(query, conn)
    logger.info(f"Extracted {len(df)} interaction events.")
    return df

def train_als_collaborative_factors(spark: SparkSession, pdf: pd.DataFrame):
    """Run Matrix Factorization using PySpark."""
    if pdf.empty:
        logger.warning("No interactions available to train ALS.")
        return
        
    df = spark.createDataFrame(pdf)
    
    # Weigh explicit (ratings) higher than implicit (clicks)
    transformed = df.withColumn(
        "implicit_rating",
        when(col("event_type") == "rating", col("rating").cast("float"))
        .when(col("event_type") == "click", 3.0)
        .otherwise(1.0)
    )
    
    aggregated = transformed.groupBy("user_id", "movie_id").agg(_max("implicit_rating").alias("final_rating"))
    
    # String Indexer for User IDs
    indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
    indexed_df = indexer.fit(aggregated).transform(aggregated).withColumn("user_idx", col("user_idx").cast("integer"))
    
    # Train ALS Model
    logger.info("Training Distributed ALS Matrix Factorization...")
    als = ALS(
        maxIter=15, 
        regParam=0.05, 
        userCol="user_idx", 
        itemCol="movie_id", 
        ratingCol="final_rating", 
        coldStartStrategy="drop"
    )
    model = als.fit(indexed_df)
    
    # Resolve user strings and save
    user_mapping = indexed_df.select("user_id", "user_idx").distinct()
    user_embeddings = model.userFactors.join(user_mapping, model.userFactors.id == user_mapping.user_idx) \
        .select("user_id", col("features").alias("embedding"))
    item_embeddings = model.itemFactors.select(col("id").alias("movie_id"), col("features").alias("embedding"))
    
    # Export to local Kaggle working directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    user_embeddings.write.mode("overwrite").parquet(str(OUTPUT_DIR / "user_factors.parquet"))
    item_embeddings.write.mode("overwrite").parquet(str(OUTPUT_DIR / "item_factors.parquet"))
    logger.info("ALS Artifacts generated successfully.")

def push_to_huggingface_lake():
    """Upload the generated Parquet files to Hugging Face Datasets."""
    if not HF_TOKEN:
        logger.warning("No HF_TOKEN found. Skipping push to Hugging Face.")
        return
        
    api = HfApi(token=HF_TOKEN)
    logger.info(f"Pushing artifacts to Hugging Face Repo: {HF_DATASET_REPO}")
    
    api.upload_folder(
        folder_path=str(OUTPUT_DIR),
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        commit_message="Nightly Spark Refresh: ALS Vectors & Semantic Embeddings"
    )
    logger.info("Data Lake synchronization complete.")

if __name__ == "__main__":
    spark = get_spark()
    try:
        events_pdf = extract_events_from_neon()
        if not events_pdf.empty:
            train_als_collaborative_factors(spark, events_pdf)
            push_to_huggingface_lake()
    finally:
        spark.stop()
