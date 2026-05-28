"""
Offline OLAP Pipeline: Matrix Factorization via Alternating Least Squares (ALS).

This Spark job bridges the gap between explicit/implicit user events (OLTP)
and the backend's fast-serving retrieval engine (OLAP output).

It performs the following Data Engineering pipeline:
1. INGEST: Reads raw interaction JSONL from the event firehose.
2. TRANSFORM: Cleanses and weighs implicit (clicks) vs explicit (ratings) signals.
3. MAP: StringIndex encodes UUID user_ids into integer vectors required by ALS.
4. MODEL: Trains the ALS (Collaborative Filtering) model across the sparse interaction matrix.
5. EXPORT: Serializes the dense user/item latent feature vectors (embeddings) to Parquet.
"""

import os
import logging
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, when, max as _max
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = REPO_ROOT / "data" / "events" / "movie_events.jsonl"
OUTPUT_DIR = REPO_ROOT / "data" / "als_embeddings"

def get_spark_session() -> SparkSession:
    return SparkSession.builder \
        .appName("Nova-ALS-Collaborative-Filtering") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

def extract_and_transform_events(spark: SparkSession, events_path: str) -> DataFrame | None:
    if not os.path.exists(events_path):
        logger.error(f"Event log not found at {events_path}. Cannot train ALS.")
        return None
        
    logger.info("Ingesting raw events...")
    raw_df = spark.read.json(events_path)
    
    if raw_df.count() == 0:
        logger.warning("Event log is empty.")
        return None

    # Filter for signals that indicate preference
    interactions = raw_df.filter(
        col("event_type").isin("rating", "click", "view")
    ).filter(col("user_id").isNotNull()).filter(col("movie_id").isNotNull())
    
    if interactions.count() == 0:
        logger.warning("No valid rating/click events found.")
        return None

    # Cast and weigh
    transformed = interactions.withColumn(
        "implicit_rating",
        when(col("event_type") == "rating", col("rating").cast("float"))
        .when(col("event_type") == "click", 3.0)
        .otherwise(1.0)
    ).select(
        "user_id",
        col("movie_id").cast("integer"),
        "implicit_rating"
    )
    
    # Deduplicate: if a user rated and clicked, keep the highest signal (max rating)
    aggregated = transformed.groupBy("user_id", "movie_id").agg(_max("implicit_rating").alias("final_rating"))
        
    return aggregated

def run_als_pipeline():
    spark = get_spark_session()
    
    try:
        interactions_df = extract_and_transform_events(spark, str(EVENTS_PATH))
        if interactions_df is None:
            return
            
        logger.info(f"Training on {interactions_df.count()} unique interactions...")
        
        # ALS requires integer IDs. Map string user_id -> integer user_idx
        indexer = StringIndexer(inputCol="user_id", outputCol="user_idx", handleInvalid="skip")
        indexer_model = indexer.fit(interactions_df)
        indexed_df = indexer_model.transform(interactions_df).withColumn("user_idx", col("user_idx").cast("integer"))
        
        # Build ALS Model
        als = ALS(
            maxIter=10,
            regParam=0.1,
            userCol="user_idx",
            itemCol="movie_id",
            ratingCol="final_rating",
            coldStartStrategy="drop",
            implicitPrefs=False
        )
        
        logger.info("Fitting ALS Model...")
        model = als.fit(indexed_df)
        
        # Extract embeddings (factors)
        user_factors = model.userFactors
        item_factors = model.itemFactors
        
        # Join user_factors back with the string user_id mapping
        user_mapping = indexed_df.select("user_id", "user_idx").distinct()
        user_embeddings = user_factors.join(user_mapping, user_factors.id == user_mapping.user_idx) \
            .select("user_id", col("features").alias("embedding"))
            
        movie_embeddings = item_factors.select(col("id").alias("movie_id"), col("features").alias("embedding"))
        
        # Export as Parquet
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        user_out = str(OUTPUT_DIR / "user_factors.parquet")
        item_out = str(OUTPUT_DIR / "item_factors.parquet")
        
        logger.info(f"Writing User Embeddings to {user_out}")
        user_embeddings.write.mode("overwrite").parquet(user_out)
        
        logger.info(f"Writing Item Embeddings to {item_out}")
        movie_embeddings.write.mode("overwrite").parquet(item_out)
        
        logger.info("OLAP Pipeline Complete! Backend can now serve Hybrid Recommendations.")
        
    finally:
        spark.stop()

if __name__ == "__main__":
    run_als_pipeline()
