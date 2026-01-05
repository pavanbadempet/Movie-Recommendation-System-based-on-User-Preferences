"""
Data ingestion module for loading and validating the TMDB movie dataset.
Handles Kaggle CSV loading with schema validation and quality checks.
"""
import logging
from pathlib import Path

import pandas as pd
import pandera as pa
from pandera import Column, Check, DataFrameSchema

from etl.config import paths, data_config

logger = logging.getLogger(__name__)


# Schema definition for raw movie data
MOVIE_SCHEMA = DataFrameSchema(
    {
        "id": Column(int, nullable=False, coerce=True),
        "title": Column(str, nullable=False),
        "overview": Column(str, nullable=True),
        "genres": Column(str, nullable=True),
        "vote_average": Column(float, Check.in_range(0, 10), nullable=True, coerce=True),
        "vote_count": Column(float, nullable=True, coerce=True),
        "popularity": Column(float, nullable=True, coerce=True),
        "release_date": Column(str, nullable=True),
        "poster_path": Column(str, nullable=True),
        # Add metadata columns for better recommendations
        "keywords": Column(str, nullable=True),
        "production_companies": Column(str, nullable=True),
        "cast": Column(str, nullable=True),
        "director": Column(str, nullable=True),
    },
    coerce=True,
    strict=False,  # Allow extra columns
)


def load_kaggle_data(file_path: Path | None = None) -> pd.DataFrame:
    """
    Load the TMDB movies dataset from Kaggle CSV.
    
    Args:
        file_path: Optional path to CSV file. Defaults to raw_data/TMDB_movie_dataset_v11.csv
        
    Returns:
        DataFrame with raw movie data
    """
    if file_path is None:
        file_path = paths.raw_data / "TMDB_movie_dataset_v11.csv"
    
    logger.info(f"Loading data from {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {file_path}. "
            "Please download from https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates"
        )
    
    # Load with chunking for memory efficiency on large files
    df = pd.read_csv(
        file_path,
        low_memory=False,
        on_bad_lines="warn",
    )
    
    logger.info(f"Loaded {len(df):,} movies from CSV")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate DataFrame against the expected schema.
    
    Args:
        df: Raw DataFrame to validate
        
    Returns:
        Validated DataFrame with correct types
    """
    logger.info("Validating schema...")
    
    try:
        validated_df = MOVIE_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed")
        return validated_df
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Schema validation warnings: {len(e.failure_cases)} issues found")
        # Log first few failures for debugging
        logger.debug(e.failure_cases.head(10))
        # Return original df, coercing types where possible
        return df


def run_quality_checks(df: pd.DataFrame) -> dict:
    """
    Run data quality checks and return metrics.
    
    Args:
        df: DataFrame to check
        
    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        "total_rows": len(df),
        "null_titles": df["title"].isna().sum(),
        "null_overviews": df["overview"].isna().sum(),
        "duplicate_ids": df["id"].duplicated().sum() if "id" in df.columns else 0,
        "movies_with_votes": (df["vote_count"] > 0).sum() if "vote_count" in df.columns else 0,
    }
    
    logger.info(f"Quality metrics: {metrics}")
    return metrics


def filter_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter movies based on quality thresholds.
    
    Removes:
    - Movies without titles or overviews
    - Movies below minimum vote count threshold
    - Adult content
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Filtered DataFrame
    """
    original_count = len(df)
    
    # Remove nulls in essential columns
    df = df.dropna(subset=["title", "overview"])
    
    # Filter by vote count (quality threshold)
    if "vote_count" in df.columns:
        df = df[df["vote_count"] >= data_config.min_vote_count]
    
    # Remove adult content if column exists
    if "adult" in df.columns:
        df = df[df["adult"] != True]
    
    # Apply max movies limit if configured
    if data_config.max_movies:
        df = df.head(data_config.max_movies)
    
    logger.info(f"Filtered from {original_count:,} to {len(df):,} movies")
    return df.reset_index(drop=True)


def save_to_parquet(df: pd.DataFrame, filename: str = "movies.parquet") -> Path:
    """
    Save DataFrame to Parquet format.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        
    Returns:
        Path to saved file
    """
    output_path = paths.processed_data / filename
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Saved {len(df):,} movies to {output_path}")
    return output_path


def ingest(file_path: Path | None = None) -> pd.DataFrame:
    """
    Main ingestion pipeline: load, validate, filter, and save.
    
    Args:
        file_path: Optional path to CSV file
        
    Returns:
        Processed DataFrame
    """
    logger.info("Starting data ingestion...")
    
    # Load raw data
    df = load_kaggle_data(file_path)
    
    # Validate schema
    df = validate_schema(df)
    
    # Run quality checks
    run_quality_checks(df)
    
    # Filter to high-quality movies
    df = filter_movies(df)
    
    # Save to Parquet
    save_to_parquet(df)
    
    logger.info("Ingestion complete")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingest()
