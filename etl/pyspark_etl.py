"""
PySpark ETL Pipeline (Production)
---------------------------------
Scales the "Golden Prompt" logic to 1M+ records using Apache Spark + SBERT (GPU).

Prerequisites:
- pip install pyspark sentence-transformers torch
- setup_spark.ps1 (for winutils)
- setup_gpu.ps1 (for CUDA)
"""
import os
import sys
import logging

# Check for PySpark
try:
    # Fix for Windows: Workers need to know exactly which Python to use
    os.environ['PYSPARK_PYTHON'] = sys.executable
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import (
        col, lower, regexp_replace, concat, lit, 
        split, slice, array_join, pandas_udf, PandasUDFType, expr
    )
    from pyspark.sql.types import ArrayType, FloatType
except ImportError:
    print("PySpark not installed. Please run: pip install pyspark")
    sys.exit(1)

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pyspark_etl")

def get_spark_session():
    return SparkSession.builder \
        .appName("MovieRecs_Production") \
        .master("local[1]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()

# ...

def run():
    logger.info("Initializing Spark Session...")
    spark = get_spark_session()
    
    logger.info("1. Loading raw data...")
    # DROPMALFORMED skips rows unable to be parsed
    df = spark.read.option("mode", "DROPMALFORMED").csv("data/raw/TMDB_all_movies.csv", header=True, inferSchema=True)
    
    # 2. Filtering
    # Mirroring the robust logic: Vote Count >= 50
    if "vote_count" in df.columns:
        df = df.filter(expr("try_cast(vote_count as float) >= 50"))
        logger.info("   Filtered by vote_count >= 50")
    
    # 3. Build Semantic Feature String (Spark SQL)
    # Logic: Title -> Director -> Genres -> Plot -> Cast (Top 50)
    
    # Handle Nulls
    df = df.fillna("", subset=["title", "director", "genres", "overview", "cast"])
    
    # Clean Inputs First
    c_title = clean_col(col("title"))
    c_director = clean_col(col("director"))
    c_genres = clean_col(col("genres"))
    c_overview = clean_col(col("overview")) # Plot
    
    # Cast Logic: Split by comma, take top 50, join space, clean
    # Raw cast: "Actor A, Actor B"
    c_cast_raw = split(col("cast"), ",")
    c_cast_top50 = slice(c_cast_raw, 1, 50)
    c_cast_str = array_join(c_cast_top50, " ")
    c_cast = clean_col(c_cast_str)
    
    # Assemble Feature Components
    # "Title: {t}. {t}. Directed by {d}. Genres: {g}. Plot: {p}. Cast: {c}. Movie: {t} by {d}."
    
    feature_str = concat(
        lit("title "), c_title, lit(". "), c_title, lit("."), # Title x2
        lit(" directed by "), c_director, lit("."),
        lit(" genres "), c_genres, lit("."),
        lit(" plot "), c_overview, lit("."), # Middle (Critical)
        lit(" cast "), c_cast, lit("."),      # End (Truncated safely)
        lit(" movie "), c_title, lit(" by "), c_director # Tail
    )
    
    logger.info("3. Generating Semantic Features...")
    df = df.withColumn("tags", feature_str)
    
    # 4. Compute Embeddings
    logger.info("4. Computing Embeddings (Distributed SBERT)...")
    df_result = df.withColumn("vector", compute_embeddings_udf(col("tags")))
    
    # 5. Save
    output_path = "data/processed/movies_spark_1M.parquet"
    logger.info(f"5. Saving to {output_path}...")
    df_result.select("id", "title", "vote_count", "vote_average", "release_date", "tags", "vector") \
             .write.mode("overwrite").parquet(output_path)
             
    logger.info("DONE. Spark ETL Complete.")

if __name__ == "__main__":
    import traceback
    try:
        run()
    except Exception as e:
        logger.error(f"FATAL ERROR: {e}")
        with open("logs/spark_error.log", "w") as f:
            f.write(traceback.format_exc())
        sys.exit(1)
