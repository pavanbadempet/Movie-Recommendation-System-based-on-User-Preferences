"""
Pandas-based ETL Pipeline.
Consolidated module for ingestion, transformation, and indexing.

Alternative to PySpark ETL for reliable local processing.
"""
import ast
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import faiss
from pandera import Column, Check, DataFrameSchema
from sentence_transformers import SentenceTransformer

from etl.config import paths, data_config

logger = logging.getLogger(__name__)

# ==========================================
# 1. INGESTION LOGIC
# ==========================================

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
    """Load the TMDB movies dataset from Kaggle CSV."""
    if file_path is None:
        file_path = paths.raw_data / "TMDB_all_movies.csv"
    
    logger.info(f"Loading data from {file_path}")
    
    if not file_path.exists():
        # Fallback to older filenames if specific one not found
        fallback = paths.raw_data / "TMDB_movie_dataset_v11.csv"
        if fallback.exists():
            file_path = fallback
        else:
            raise FileNotFoundError(
                f"Dataset not found at {file_path}. "
                "Please download from https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates"
            )
    
    # Load with chunking context if needed, but here simple read
    df = pd.read_csv(
        file_path,
        low_memory=False,
        on_bad_lines="warn",
    )
    
    logger.info(f"Loaded {len(df):,} movies from CSV")
    return df


def validate_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate DataFrame against the expected schema."""
    logger.info("Validating schema...")
    try:
        validated_df = MOVIE_SCHEMA.validate(df, lazy=True)
        logger.info("Schema validation passed")
        return validated_df
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Schema validation warnings: {len(e.failure_cases)} issues found")
        return df


def run_quality_checks(df: pd.DataFrame) -> dict:
    """Run data quality checks and return metrics."""
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
    Removes movies without titles/overviews, low votes, or adult content.
    """
    original_count = len(df)
    
    # Remove nulls
    df = df.dropna(subset=["title", "overview"])
    
    # Filter by vote count
    if "vote_count" in df.columns:
        df = df[df["vote_count"] >= data_config.min_vote_count]
    
    # Remove adult content
    if "adult" in df.columns:
        df = df[df["adult"] != True]
    
    # Max limit
    if data_config.max_movies:
        df = df.head(data_config.max_movies)
    
    logger.info(f"Filtered from {original_count:,} to {len(df):,} movies")
    return df.reset_index(drop=True)


def ingest(file_path: Path | None = None) -> pd.DataFrame:
    """Main ingestion pipeline."""
    logger.info("Starting ingestion...")
    df = load_kaggle_data(file_path)
    df = validate_schema(df)
    run_quality_checks(df)
    df = filter_movies(df)
    
    # Save intermediate if needed, but we usually stream in memory in pipeline
    # save_to_parquet(df) 
    return df


# ==========================================
# 2. TRANSFORMATION LOGIC
# ==========================================

def parse_json_column(value: str) -> list[str]:
    """Parse stringified JSON/list column to extract names."""
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [item.get("name", str(item)) for item in parsed if isinstance(item, dict)]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [s.strip() for s in str(value).split(",") if s.strip()]


def clean_text(text: str) -> str:
    """Clean text while PRESERVING punctuation for SBERT."""
    if pd.isna(text):
        return ""
    text = str(text)
    # Only remove truly problematic characters
    text = re.sub(r"[^\w\s.,;:!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def generate_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Generate unified 'tags' column for Semantic Search (Vectorized)."""
    logger.info("Generating tags (Vectorized)...")
    df = df.copy()
    
    # 1. Parse JSON columns (Keep apply here, hard to avoid for complex JSON parsing)
    for col_name in ["genres", "keywords", "production_companies"]:
        target = f"_{col_name}" if col_name != "production_companies" else "_companies"
        if col_name in df.columns:
            # Convert list of dicts to comma-separated string
            df[target] = df[col_name].apply(parse_json_column).str.join(", ")
        else:
            df[target] = ""
            
    # 2. Clean Overview
    df["_overview"] = df["overview"].fillna("").astype(str).apply(clean_text)
    
    # 3. Vectorized Concatenation
    # We build the tags string column-wise using vector operations
    # This is significantly faster than row-wise apply()
    
    # Start with Title
    tags = pd.Series("", index=df.index)
    title = df['title'].fillna("").astype(str)
    tags += "Title: " + title + ". " + title + ". "
    
    # Helper for conditional append
    def add_section(prefix, col_name, suffix="."):
        if col_name not in df.columns:
            return ""
        
        # Get series, fill NaNs
        s = df[col_name].fillna("").astype(str).str.strip()
        
        # Mask for valid content (not empty, not 'nan')
        mask = (s != "") & (s.str.lower() != "nan")
        
        # Vectorized "if condition"
        return np.where(mask, prefix + s + suffix + " ", "")

    tags += add_section("Tagline: ", "tagline")
    tags += add_section("Genres: ", "_genres")
    tags += add_section("Plot: ", "_overview", "") # Overview already has dot handled or we just append
    tags += add_section("Directed by ", "director")
    tags += add_section("Written by ", "writers")
    
    # Cast is special (limit to top 10)
    if "cast" in df.columns:
        s_cast = df['cast'].fillna("").astype(str).str.split(",").str[:10].str.join(", ")
        mask = s_cast != ""
        tags += np.where(mask, "Starring: " + s_cast + ". ", "")
        
    tags += add_section("Produced by ", "_companies")
    tags += add_section("Music by ", "music_composer")
    
    # Safe access for director in final string
    director = df['director'].fillna("") if 'director' in df.columns else pd.Series("", index=df.index)
    tags += "Movie: " + title + " by " + director + "."
    
    # Final cleanup to ensure SBERT friendly format
    df["tags"] = tags.apply(clean_text)
    
    # Cleanup temps
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df = df[df["tags"].str.len() > 10]
    
    return df.reset_index(drop=True)


def build_sbert_embeddings(tags: pd.Series) -> tuple[SentenceTransformer, np.ndarray]:
    """Build embeddings using sentence-transformers (all-mpnet-base-v2)."""
    model_name = 'all-mpnet-base-v2'
    logger.info(f"Loading SBERT model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(tags):,} movies...")
    embeddings = model.encode(
        tags.tolist(), 
        show_progress_bar=True, 
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Normalize for Cosine Similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return model, embeddings


def transform(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """Main transformation pipeline."""
    logger.info("Starting transformation...")
    
    if df is None:
        # If loading from ingest result
        parquet_path = paths.processed_data / "movies.parquet" # Or directly prompt ingest
        # But in unified pipeline we pass df directly. 
        # If called standalone, try loading:
        if (paths.processed_data / "movies.parquet").exists():
            df = pd.read_parquet(paths.processed_data / "movies.parquet")
        else:
            raise FileNotFoundError("No input DataFrame or parquet file found.")

    df = generate_tags(df)
    model, vectors = build_sbert_embeddings(df["tags"])
    
    # Save artifacts
    np.save(paths.models / "sbert_embeddings.npy", vectors)
    df.to_parquet(paths.processed_data / "movies_transformed.parquet", index=False)
    
    logger.info("Transformation complete")
    return df, vectors


# ==========================================
# 3. INDEXING LOGIC
# ==========================================

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build FAISS IVF index."""
    n_samples, n_features = vectors.shape
    logger.info(f"Building FAISS index for {n_samples:,} vectors...")
    
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    
    if n_samples < 10000:
        index = faiss.IndexFlatIP(n_features)
    else:
        nlist = min(data_config.faiss_nlist, n_samples // 39)
        quantizer = faiss.IndexFlatIP(n_features)
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(vectors)
    
    index.add(vectors)
    return index


def build_index(vectors: np.ndarray | None = None) -> faiss.Index:
    """Main indexing pipeline."""
    logger.info("Starting indexing...")
    
    if vectors is None:
        vectors = np.load(paths.models / "sbert_embeddings.npy")
        # Normalize just in case, though transform does it
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = vectors / norms
        
    index = build_faiss_index(vectors)
    faiss.write_index(index, str(paths.models / "faiss.index"))
    
    logger.info("Indexing complete")
    return index


# ==========================================
# 4. ORCHESTRATION
# ==========================================

class PipelineStage:
    """Context manager for timing."""
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        logger.info(f"--- Starting {self.name} ---")
        return self
    def __exit__(self, *args):
        logger.info(f"--- Completed {self.name} in {time.time() - self.start:.2f}s ---")


def run_pipeline(raw_data_path: Path | None = None, skip_ingest: bool = False) -> dict:
    """Execute complete ETL pipeline."""
    start_time = time.time()
    metrics = {"stages": {}}
    
    logger.info("STARTING PANDAS ETL PIPELINE")
    
    try:
        # 1. Ingest
        if not skip_ingest:
            with PipelineStage("INGEST"):
                df = ingest(raw_data_path)
                metrics["ingested_rows"] = len(df)
        else:
            logger.info("Skipping ingest, loading from transformed parquet if possible or erroring...")
            # Ideally we'd load raw parquet here if we had it, but we usually transform raw.
            # Simplified: we assume if skipping ingest we want to transform existing parquet?
            # Actually transform() handles loading if df is None.
            df = None 

        # 2. Transform
        with PipelineStage("TRANSFORM"):
            df, vectors = transform(df)
            metrics["final_rows"] = len(df)
        
        # 3. Index
        with PipelineStage("INDEX"):
            index = build_index(vectors)
            metrics["index_size"] = index.ntotal
            
        metrics["success"] = True
        logger.info(f"PIPELINE SUCCESS in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.exception("Pipeline failed")
        metrics["success"] = False
        metrics["error"] = str(e)
        raise
        
    return metrics


if __name__ == "__main__":
    import argparse
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="Pandas ETL Pipeline")
    parser.add_argument("--data", type=Path, help="Path to raw CSV")
    parser.add_argument("--index-only", action="store_true", help="Run only indexing stage")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")
    
    args = parser.parse_args()
    
    if args.index_only:
        # Just run indexing on existing embeddings
        build_index()
    else:
        run_pipeline(raw_data_path=args.data, skip_ingest=args.skip_ingest)
