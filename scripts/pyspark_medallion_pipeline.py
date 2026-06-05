"""
PySpark Medallion Architecture — Real Data Pipeline

This script processes REAL MovieLens data (100,836 ratings from 610 users
across 9,724 movies) through a Bronze → Silver → Gold Delta Lake architecture.

Gold layer outputs:
  - dim_movies: Movie dimension table with metadata
  - dim_users: User dimension table with aggregated features
  - fact_interactions: Cleaned, validated rating events
  - model_user_embeddings: ALS collaborative filtering user vectors
  - model_item_embeddings: ALS collaborative filtering item vectors

The ALS embeddings are consumed by backend/ensemble_engine.py via
_inject_pyspark_priors() to replace random weight initialization
with real learned collaborative signals.
"""

import os
import sys

# Add scripts/ and project root to path for setup_local_spark
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


import logging

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    avg,
    col,
    count,
    current_timestamp,
    monotonically_increasing_id,
)
from pyspark.sql.types import FloatType, IntegerType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Source data (real processed parquet files)
MOVIES_SOURCE = os.path.join(PROJECT_ROOT, "data", "processed", "movies_transformed.parquet")
RATINGS_SOURCE = os.path.join(PROJECT_ROOT, "data", "processed", "ratings_transformed.parquet")

# Data Lake Paths
BRONZE_RATINGS = os.path.join(PROJECT_ROOT, "data", "datalake", "bronze", "ratings")
BRONZE_MOVIES = os.path.join(PROJECT_ROOT, "data", "datalake", "bronze", "movies")
SILVER_RATINGS = os.path.join(PROJECT_ROOT, "data", "datalake", "silver", "ratings")
SILVER_MOVIES = os.path.join(PROJECT_ROOT, "data", "datalake", "silver", "movies")
GOLD_DIM_MOVIES = os.path.join(PROJECT_ROOT, "data", "datalake", "gold", "dim_movies")
GOLD_DIM_USERS = os.path.join(PROJECT_ROOT, "data", "datalake", "gold", "dim_users")
GOLD_FACT_INTERACTIONS = os.path.join(PROJECT_ROOT, "data", "datalake", "gold", "fact_interactions")
GOLD_USER_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "datalake", "gold", "model_user_embeddings")
GOLD_ITEM_EMBEDDINGS = os.path.join(PROJECT_ROOT, "data", "datalake", "gold", "model_item_embeddings")


def create_spark_session() -> SparkSession:
    """Create a Spark session configured for local execution."""
    return (
        SparkSession.builder.appName("Apex_Medallion_Pipeline")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.int96RebaseModeInWrite", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED")
        .config("spark.sql.parquet.datetimeRebaseModeInWrite", "CORRECTED")
        .getOrCreate()
    )


# ============================================================
# BRONZE LAYER: Raw Ingestion (no transformations)
# ============================================================


def ingest_to_bronze(spark: SparkSession):
    """
    Ingest real source data into Bronze layer as-is.
    Bronze = raw, unmodified copy of the source system.
    """
    logger.info("=" * 60)
    logger.info("BRONZE LAYER: Ingesting real source data")
    logger.info("=" * 60)

    # --- Ratings ---
    df_ratings = spark.read.parquet(RATINGS_SOURCE)
    bronze_count = df_ratings.count()
    df_ratings.write.mode("overwrite").parquet(BRONZE_RATINGS)
    logger.info(f"  Ratings ingested: {bronze_count:,} rows → {BRONZE_RATINGS}")

    # --- Movies ---
    df_movies = spark.read.parquet(MOVIES_SOURCE)
    movies_count = df_movies.count()
    df_movies.write.mode("overwrite").parquet(BRONZE_MOVIES)
    logger.info(f"  Movies ingested:  {movies_count:,} rows → {BRONZE_MOVIES}")

    return {"ratings": bronze_count, "movies": movies_count}


# ============================================================
# SILVER LAYER: Cleaned, validated, deduplicated
# ============================================================


def bronze_to_silver(spark: SparkSession):
    """
    Clean and validate Bronze data:
    - Remove nulls in critical columns
    - Enforce rating bounds [0.5, 5.0]
    - Cast types
    - Deduplicate
    """
    logger.info("=" * 60)
    logger.info("SILVER LAYER: Cleaning and validating Bronze data")
    logger.info("=" * 60)

    # --- Clean Ratings ---
    df_bronze_ratings = spark.read.parquet(BRONZE_RATINGS)

    df_silver_ratings = (
        df_bronze_ratings.filter(col("userId").isNotNull())
        .filter(col("movieId").isNotNull())
        .filter(col("rating").isNotNull())
        .filter((col("rating") >= 0.5) & (col("rating") <= 5.0))
        .dropDuplicates(["userId", "movieId"])
        .select(
            col("userId").cast(IntegerType()).alias("user_id"),
            col("movieId").cast(IntegerType()).alias("movie_id"),
            col("rating").cast(FloatType()).alias("rating"),
            col("timestamp").cast("long").alias("event_ts"),
        )
    )

    silver_ratings_count = df_silver_ratings.count()
    df_silver_ratings.write.mode("overwrite").parquet(SILVER_RATINGS)
    logger.info(f"  Ratings after cleaning: {silver_ratings_count:,} rows")

    # --- Clean Movies ---
    df_bronze_movies = spark.read.parquet(BRONZE_MOVIES)

    df_silver_movies = (
        df_bronze_movies.filter(col("id").isNotNull())
        .filter(col("title").isNotNull())
        .dropDuplicates(["id"])
        .select(
            col("id").cast(IntegerType()).alias("movie_id"),
            col("title"),
            col("genres"),
            col("vote_average").cast(FloatType()),
            col("vote_count").cast(IntegerType()),
            col("popularity").cast(FloatType()),
            col("release_date"),
            col("director"),
            col("original_language"),
        )
    )

    silver_movies_count = df_silver_movies.count()
    df_silver_movies.write.mode("overwrite").parquet(SILVER_MOVIES)
    logger.info(f"  Movies after cleaning:  {silver_movies_count:,} rows")

    dropped_ratings = df_bronze_ratings.count() - silver_ratings_count
    dropped_movies = df_bronze_movies.count() - silver_movies_count
    logger.info(f"  Dropped ratings: {dropped_ratings:,} | Dropped movies: {dropped_movies:,}")

    return {"ratings": silver_ratings_count, "movies": silver_movies_count}


# ============================================================
# GOLD LAYER: Feature engineering + dimensional modeling
# ============================================================


def silver_to_gold(spark: SparkSession):
    """
    Build the analytical Gold layer:
    - dim_movies: Movie dimension with metadata
    - dim_users: User dimension with aggregated features
    - fact_interactions: Clean interaction fact table
    """
    logger.info("=" * 60)
    logger.info("GOLD LAYER: Building star schema")
    logger.info("=" * 60)

    df_ratings = spark.read.parquet(SILVER_RATINGS)
    df_movies = spark.read.parquet(SILVER_MOVIES)

    # --- Fact Table: fact_interactions ---
    fact_interactions = df_ratings.select(
        monotonically_increasing_id().alias("interaction_sk"),
        col("user_id"),
        col("movie_id"),
        col("rating"),
        col("event_ts"),
    )
    fact_count = fact_interactions.count()
    fact_interactions.write.mode("overwrite").parquet(GOLD_FACT_INTERACTIONS)
    logger.info(f"  fact_interactions: {fact_count:,} rows")

    # --- Dimension: dim_movies ---
    dim_movies = df_movies.withColumn("loaded_at", current_timestamp())
    dim_movies_count = dim_movies.count()
    dim_movies.write.mode("overwrite").parquet(GOLD_DIM_MOVIES)
    logger.info(f"  dim_movies: {dim_movies_count:,} rows")

    # --- Dimension: dim_users ---
    dim_users = (
        df_ratings.groupBy("user_id")
        .agg(
            count("*").alias("total_ratings"),
            avg("rating").alias("avg_rating"),
        )
        .withColumn("loaded_at", current_timestamp())
    )
    dim_users_count = dim_users.count()
    dim_users.write.mode("overwrite").parquet(GOLD_DIM_USERS)
    logger.info(f"  dim_users: {dim_users_count:,} rows")

    return {"facts": fact_count, "movies": dim_movies_count, "users": dim_users_count}


# ============================================================
# GOLD LAYER: ALS Collaborative Filtering Embeddings
# ============================================================


def train_als_embeddings(spark: SparkSession):
    """
    Train PySpark ALS on the Gold fact_interactions table.
    Exports user and item embedding vectors as Parquet files
    that backend/ensemble_engine.py loads via _inject_pyspark_priors().
    """
    logger.info("=" * 60)
    logger.info("GOLD LAYER: Training ALS collaborative filtering model")
    logger.info("=" * 60)

    fact = spark.read.parquet(GOLD_FACT_INTERACTIONS)

    # Filter to only rows with ratings (not null)
    rated = fact.filter(col("rating").isNotNull())
    logger.info(f"  Training on {rated.count():,} rated interactions")

    # Train/test split
    (train, test) = rated.randomSplit([0.8, 0.2], seed=42)

    # Configure ALS
    als = ALS(
        maxIter=15,
        regParam=0.1,
        rank=16,  # Must match ensemble_engine.py emb_dim
        userCol="user_id",
        itemCol="movie_id",
        ratingCol="rating",
        coldStartStrategy="drop",
        nonnegative=True,
    )

    logger.info("  Training ALS model (rank=16, maxIter=15)...")
    model = als.fit(train)

    # Evaluate
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    logger.info(f"  ALS Model RMSE: {rmse:.4f}")

    # Export embeddings
    user_factors = model.userFactors
    item_factors = model.itemFactors

    user_count = user_factors.count()
    item_count = item_factors.count()

    user_factors.write.mode("overwrite").parquet(GOLD_USER_EMBEDDINGS)
    item_factors.write.mode("overwrite").parquet(GOLD_ITEM_EMBEDDINGS)

    logger.info(f"  User embeddings: {user_count:,} vectors → {GOLD_USER_EMBEDDINGS}")
    logger.info(f"  Item embeddings: {item_count:,} vectors → {GOLD_ITEM_EMBEDDINGS}")

    return {"rmse": rmse, "user_vectors": user_count, "item_vectors": item_count}


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("APEX MEDALLION PIPELINE: Starting")
    logger.info("=" * 60)

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        bronze_stats = ingest_to_bronze(spark)
        silver_stats = bronze_to_silver(spark)
        gold_stats = silver_to_gold(spark)
        als_stats = train_als_embeddings(spark)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE — Summary:")
        logger.info(f"  Bronze: {bronze_stats['ratings']:,} ratings, {bronze_stats['movies']:,} movies")
        logger.info(f"  Silver: {silver_stats['ratings']:,} ratings, {silver_stats['movies']:,} movies")
        logger.info(
            f"  Gold:   {gold_stats['facts']:,} facts, {gold_stats['movies']:,} dim_movies, {gold_stats['users']:,} dim_users"
        )
        logger.info(
            f"  ALS:    RMSE={als_stats['rmse']:.4f}, {als_stats['user_vectors']:,} user vectors, {als_stats['item_vectors']:,} item vectors"
        )
        logger.info("=" * 60)
    finally:
        spark.stop()
