"""
PySpark ETL - processes TMDB movie data using Spark.

Alternative to the Pandas pipeline for larger datasets.
"""
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, when, length

logger = logging.getLogger(__name__)


def create_spark_session():
    """Create a local Spark session with AQE enabled."""
    # MEMORY SAFETY for Machine Learning (SBERT runs off-heap)
    # Prevent Arrow batches from exploding memory during UDF transfer
    return SparkSession.builder \
        .appName("MovieETL") \
        .master("local[*]") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.adaptive.skewJoin.enabled", "true") \
        .config("spark.executor.memoryOverhead", "4g") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "1000") \
        .getOrCreate()


def run_spark_etl(input_path: str = "data/raw/TMDB_all_movies.csv", run_date: str = None, sink_format: str = "parquet"):
    """
    Run the Spark ETL pipeline.
    Args:
        input_path: Path to raw CSV
        run_date: Date string (YYYY-MM-DD) for partitioning
        sink_format: Output format ('parquet', 'delta', 'snowflake')
    """
    logger.info("Starting Spark ETL...")
    
    spark = create_spark_session()
    
    # Load data
    logger.info(f"Reading from {input_path}")
    df = spark.read.option("mode", "DROPMALFORMED") \
        .csv(input_path, header=True, inferSchema=True)
    
    initial_count = df.count()
    logger.info(f"Loaded {initial_count:,} rows")
    
    # ---------------------------------------------------------
    # DATA QUALITY GATES (Abort if data is garbage)
    # ---------------------------------------------------------
    if initial_count == 0:
        logger.error("DQ FAILURE: Input dataset is empty.")
        spark.stop()
        raise ValueError("Input dataset is empty")
        
    # Check null rate for critical columns
    null_titles = df.filter(col("title").isNull()).count()
    null_rate = null_titles / initial_count
    
    if null_rate > 0.5: # Hard limit: if >50% movies have no title, source is broken
        logger.error(f"DQ FAILURE: Null title rate {null_rate:.2%} exceeds 50% threshold.")
        spark.stop()
        raise ValueError(f"Data Quality Error: Too many null titles ({null_rate:.2%})")
        
    logger.info("DQ Success: Input data passed basic quality gates.")
    
    # Filter: Vote Count >= 50
    if "vote_count" in df.columns:
        df = df.filter(expr("try_cast(vote_count as float) >= 50"))
        logger.info("Filtered by vote_count >= 50")
    
    # Filter: Non-null title and overview
    df = df.filter(col("title").isNotNull() & col("overview").isNotNull())
    
    # Filter: Overview has at least 20 chars
    if "overview" in df.columns:
        df = df.filter(length(col("overview")) >= 20)
    
    # Create Tags Column (Simple concatenation for now, Spark SQL is fast)
    # Note: We duplicate simple tag generation here for full Spark pipeline
    df = df.withColumn("tags", 
        expr("concat_ws('. ', title, coalesce(overview, ''), 'Movie')")
    )
    
    # ---------------------------------------------------------
    # DISTRIBUTED MODEL INFERENCE (The "Pro" Move)
    # ---------------------------------------------------------
    logger.info("Generating Embeddings using Pandas UDF...")
    
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import ArrayType, FloatType
    import pandas as pd
    from sentence_transformers import SentenceTransformer
    
    # Broadcast model isn't efficient for large weights, better to load on executors once
    # We use a Scalar Iterator UDF to amortize model loading cost across a batch
    
    @pandas_udf(ArrayType(FloatType()))
    def predict_embeddings(iterator):
        # Load model once per partition/iterator
        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval() # Inference mode
        
        for series in iterator:
            # Series is a batch of strings
            # Encode batch
            embeddings = model.encode(
                series.tolist(), 
                batch_size=32, 
                show_progress_bar=False, 
                convert_to_numpy=True
            )
            # Normalize
            import numpy as np
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / norms
            
            yield pd.Series(list(embeddings))

    # Apply UDF
    # coalesce(10) to ensure we have enough parallelism but not too many small partitions
    df = df.repartition(10).withColumn("vector", predict_embeddings(col("tags")))
    
    final_count = df.count()
    logger.info(f"Generated embeddings for {final_count:,} rows")
    
    # ---------------------------------------------------------
    # DATA SINK (Enterprise Adapter)
    # ---------------------------------------------------------
    def write_sink(df, format_type="parquet", mode="overwrite", **kwargs):
        """
        Agnostic Data Sink.
        Supports:
        - 'parquet': Local/S3 (Standard)
        - 'delta': Databricks (Lakehouse)
        - 'snowflake': Snowflake Data Cloud
        """
        logger.info(f"Writing data to sink: format={format_type}")
        
        writer = df.write.mode(mode)
        
        if format_type == "delta":
            # DATABRICKS OPTIMIZATION
            output_path = "s3://my-datalake/movies_delta" # Example
            writer.format("delta").save(output_path)
            
        elif format_type == "snowflake":
            # SNOWFLAKE INTEGRATION
            # Requires spark-snowflake connector
            writer \
                .format("net.snowflake.spark.snowflake") \
                .options(**{
                    "sfUrl": kwargs.get("sfUrl"),
                    "sfUser": kwargs.get("sfUser"),
                    "sfPassword": kwargs.get("sfPassword"),
                    "sfDatabase": "MOVIE_DB",
                    "sfSchema": "PUBLIC",
                    "sfWarehouse": "COMPUTE_WH"
                }) \
                .option("dbtable", "MOVIES_PROCESSED") \
                .save()
                
        else:
            # DEFAULT: Parquet (Local/S3)
            # Partitioning support
            path = "data/processed/movies_spark_w_embeddings.parquet"
            if kwargs.get("run_date"):
                path += f"/run_date={kwargs.get('run_date')}"
            writer.parquet(path)
            return path

    # Execute Write
    write_sink(df, format_type=sink_format, run_date=run_date)
    logger.info("Data write complete.")
    
    # ---------------------------------------------------------
    # ARTIFACT GENERATION (Bridge to Backend)
    # ---------------------------------------------------------
    logger.info("Collecting vectors for FAISS index (Reference Architecture Pattern)...")
    
    # Collect to driver (Acceptable for <1M rows, otherwise use specialized tools)
    try:
        rows = df.select("id", "vector").collect()
        
        import numpy as np
        import faiss
        
        # Initializing arrays
        ids = [r['id'] for r in rows]
        vectors = np.array([r['vector'] for r in rows]).astype('float32')
        
        # Save for Backend
        np.save("models/sbert_embeddings.npy", vectors)
        logger.info("Saved models/sbert_embeddings.npy")
        
        # Build FAISS Index
        d = vectors.shape[1]
        index = faiss.IndexFlatIP(d)
        index.add(vectors)
        faiss.write_index(index, "models/faiss.index")
        logger.info("Saved models/faiss.index")
    except Exception as e:
        logger.warning(f"Could not build local artifacts (maybe running on pure cluster without shared FS?): {e}")
    
    spark.stop()
    return final_count




if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Spark ETL Pipeline")
    parser.add_argument("--date", type=str, help="Run date (YYYY-MM-DD)", default=None)
    parser.add_argument("--sink", type=str, help="Output format (parquet, delta, snowflake)", default="parquet")
    args = parser.parse_args()
    
    run_spark_etl(run_date=args.date, sink_format=args.sink)
