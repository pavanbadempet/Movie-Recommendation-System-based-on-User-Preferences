"""
Script to export Gold PySpark outputs to a free Neon/Supabase PostgreSQL database.
This is run by the GitHub Action after the ETL finishes.
"""

import logging
import os
import sys

import pandas as pd
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_to_postgres():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL is not set. Skipping export.")
        sys.exit(1)

    # Standardize the postgresql driver prefix for SQLAlchemy
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)

    logger.info("Connecting to PostgreSQL...")
    engine = create_engine(db_url)

    # Path where PySpark ETL saves the final Gold artifacts
    # Assuming 'data/gold/movies' or similar. Using 'artifacts/movies.parquet' as fallback.
    gold_path = os.path.join("data", "gold", "movie_features")

    if not os.path.exists(gold_path):
        logger.warning(f"Could not find Gold tables at {gold_path}. Ensure ETL ran successfully.")
        return

    logger.info(f"Loading data from {gold_path}")
    try:
        # Load parquet using pandas
        df = pd.read_parquet(gold_path)

        # Export to Postgres
        logger.info(f"Exporting {len(df)} records to Postgres table 'movies'...")
        df.to_sql("movies", engine, if_exists="replace", index=False)
        logger.info("Successfully exported Gold data to PostgreSQL serving layer!")

    except Exception as e:
        logger.error(f"Failed to export: {e}")
        sys.exit(1)


if __name__ == "__main__":
    export_to_postgres()
