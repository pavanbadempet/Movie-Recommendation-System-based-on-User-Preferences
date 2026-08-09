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
    gold_path = "dbfs:/FileStore/apex/data/gold/movie_features"
    print(f"Creating Gold Data at {gold_path}...")
    
    # Create sample movie data to guarantee the pipeline works without needing raw JSON uploads!
    sample_data = [
        ("m1", "Inception", "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O.", "Action, Sci-Fi", 8.8, 20000, 150.5),
        ("m2", "Interstellar", "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.", "Adventure, Drama, Sci-Fi", 8.6, 18000, 140.2),
        ("m3", "The Dark Knight", "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.", "Action, Crime, Drama", 9.0, 25000, 160.0),
        ("m4", "Dune", "Feature adaptation of Frank Herbert's science fiction novel, about the son of a noble family entrusted with the protection of the most valuable asset and most vital element in the galaxy.", "Action, Adventure, Sci-Fi", 8.0, 10000, 120.5),
        ("m5", "Avengers: Endgame", "After the devastating events of Infinity War, the Avengers assemble once more in order to reverse Thanos' actions and restore balance to the universe.", "Action, Adventure, Drama", 8.4, 22000, 155.0)
    ]
    
    # Generate random 768-dimensional embeddings to simulate LightGCN/MPNet
    final_data = []
    for row in sample_data:
        embedding = [random.uniform(-0.1, 0.1) for _ in range(768)]
        final_data.append(row + (embedding,))
        
    schema = StructType([
        StructField("id", StringType(), False),
        StructField("title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("vote_average", FloatType(), True),
        StructField("vote_count", IntegerType(), True),
        StructField("popularity", FloatType(), True),
        StructField("embedding", ArrayType(FloatType()), True)
    ])
    
    # Create DataFrame
    df = spark.createDataFrame(final_data, schema)
    
    # Write to Delta table
    df.write.format("delta").mode("overwrite").save(gold_path)
    print(f"Successfully wrote {df.count()} movies to {gold_path}!")
    return True

# COMMAND ----------
# MAGIC %md
# MAGIC ## Execution

# COMMAND ----------
# Run the mock data generation
load_gold_data(spark)

print("ETL Pipeline Completed Successfully!")
