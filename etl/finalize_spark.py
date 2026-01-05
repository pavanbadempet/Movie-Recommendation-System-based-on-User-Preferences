import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finalize_spark")

def finalize():
    spark_output = Path("data/processed/movies_spark_1M.parquet")
    output_npy = Path("models/sbert_embeddings.npy")
    output_meta = Path("data/processed/movies_transformed.parquet")
    
    logger.info(f"Reading Spark output from {spark_output}...")
    # Pandas can read partitioned parquet directories automatically
    df = pd.read_parquet(spark_output)
    
    logger.info(f"Loaded {len(df):,} records.")
    
    # Extract Vectors
    logger.info("Extracting vectors...")
    # Spark arrays might be loaded as numpy arrays or lists
    vectors = np.stack(df["vector"].values)
    
    logger.info(f"Saving vectors shape {vectors.shape} to {output_npy}...")
    np.save(output_npy, vectors)
    
    # Drop vector column and save metadata
    df_meta = df.drop(columns=["vector"])
    logger.info(f"Saving metadata to {output_meta}...")
    df_meta.to_parquet(output_meta)
    
    logger.info("Finalization Complete.")

if __name__ == "__main__":
    finalize()
