"""
Real-World Data Acquisition Pipeline.

In order to prove the architectures (Diffusion, KAN, Hyperbolic, Quantum)
are completely universal, we must subject them to Deep Empirical Testing 
across various domains.

This script pulls open-source, real-world data from HuggingFace spanning:
1. E-Commerce (e.g. Amazon reviews or Instacart datasets)
2. Books / Text Media (e.g. Goodreads)
3. Entertainment (MovieLens)
"""

import os
import logging
from pathlib import Path
from datasets import load_dataset
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "real_world"

def fetch_datasets():
    os.makedirs(DATA_DIR, exist_ok=True)
    
    logger.info("=========================================================")
    logger.info("INITIATING REAL-WORLD MULTI-DOMAIN DATA ACQUISITION")
    logger.info("=========================================================")
    
    # 1. E-Commerce Domain (Amazon Product Reviews - Electronics)
    logger.info("Fetching Domain 1: E-Commerce (Amazon Electronics)...")
    try:
        # Load a small sample of the massive McAuley Amazon dataset
        amazon_ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Electronics", split="full[:10000]", trust_remote_code=True)
        amazon_df = amazon_ds.to_pandas()
        amazon_df.to_parquet(DATA_DIR / "amazon_electronics_sample.parquet")
        logger.info(f" -> Success! Downloaded {len(amazon_df)} raw e-commerce interactions.")
    except Exception as e:
        logger.error(f"Failed to fetch E-Commerce: {e}")
        
    # 2. Books Domain
    logger.info("Fetching Domain 2: Books (Goodreads)...")
    try:
        books_ds = load_dataset("zyztem99/goodreads-reviews", split="train[:10000]", trust_remote_code=True)
        books_df = books_ds.to_pandas()
        books_df.to_parquet(DATA_DIR / "goodreads_sample.parquet")
        logger.info(f" -> Success! Downloaded {len(books_df)} raw book interactions.")
    except Exception as e:
        logger.error(f"Failed to fetch Books: {e}")

    # 3. Social / Temporal Domain (Reddit or Similar)
    logger.info("Fetching Domain 3: Social Temporal (Reddit)...")
    try:
        reddit_ds = load_dataset("sentence-transformers/reddit-title-body", split="train[:10000]", trust_remote_code=True)
        reddit_df = reddit_ds.to_pandas()
        reddit_df.to_parquet(DATA_DIR / "reddit_sample.parquet")
        logger.info(f" -> Success! Downloaded {len(reddit_df)} social textual nodes.")
    except Exception as e:
        logger.error(f"Failed to fetch Social: {e}")

    logger.info("=========================================================")
    logger.info("DATA ACQUISITION COMPLETE. READY FOR HARMONIZATION PIPELINE.")
    logger.info("=========================================================")

if __name__ == "__main__":
    fetch_datasets()
