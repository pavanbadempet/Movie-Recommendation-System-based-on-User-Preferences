"""
Feature transformation module for the Movie Recommendation System.
Handles text preprocessing, tag generation, and SBERT vectorization.
"""
import ast
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib

from etl.config import paths, data_config

logger = logging.getLogger(__name__)


def parse_json_column(value: str) -> list[str]:
    """
    Parse stringified JSON/list column to extract names.
    
    Handles formats like:
    - "[{'id': 18, 'name': 'Drama'}, {'id': 80, 'name': 'Crime'}]"
    - "Action, Comedy, Drama" (comma-separated)
    
    Args:
        value: String value from CSV
        
    Returns:
        List of extracted names
    """
    if pd.isna(value) or value == "":
        return []
    
    try:
        # Try parsing as JSON/Python literal
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [item.get("name", str(item)) for item in parsed if isinstance(item, dict)]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        # Fallback: treat as comma-separated string
        return [s.strip() for s in str(value).split(",") if s.strip()]


def clean_text(text: str) -> str:
    """
    Clean text while PRESERVING punctuation important for semantic understanding.
    
    SBERT models are trained on natural language with proper punctuation.
    We should NOT remove periods, commas, colons, etc. as they provide context.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text with preserved punctuation
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Only remove truly problematic characters (brackets, quotes, etc.)
    # KEEP: periods, commas, colons, semicolons, question marks, exclamation marks
    text = re.sub(r"[^\w\s.,;:!?-]", " ", text)
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def generate_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate unified 'tags' column for Semantic Search.
    
    Combines: Title (High Weight), Director, Cast, Keywords, Genres, Overview
    """
    logger.info("Generating tags from movie metadata...")
    df = df.copy()
    
    # Parse JSON columns
    if "genres" in df.columns:
        df["_genres"] = df["genres"].apply(parse_json_column)
    else:
        df["_genres"] = [[] for _ in range(len(df))]
    
    if "keywords" in df.columns:
        df["_keywords"] = df["keywords"].apply(parse_json_column)
    else:
        df["_keywords"] = [[] for _ in range(len(df))]
    
    if "production_companies" in df.columns:
        df["_companies"] = df["production_companies"].apply(parse_json_column)
    else:
        df["_companies"] = [[] for _ in range(len(df))]
        
    # Clean overview text
    df["_overview"] = df["overview"].apply(clean_text)
    
    # Combine all features into tags
    # PRODUCTION-GRADE FEATURE ENGINEERING
    # Optimized for SBERT semantic understanding
    def combine_tags(row):
        title = row.get("title", "")
        tagline = row.get("tagline", "")
        director = row.get("director", "")
        writers = row.get("writers", "")
        music_composer = row.get("music_composer", "")
        
        # Cast: Top 10 only (prevents SBERT truncation, focuses on star power)
        cast_raw = row.get("cast", "")
        cast_str = ""
        if cast_raw:
            cast_list = str(cast_raw).split(",")[:10]  # Reduced from 50
            cast_str = ", ".join(cast_list)
        
        # Production companies (studio style patterns)
        companies = ", ".join(row.get("_companies", []))
        
        genres = ", ".join(row.get("_genres", []))
        overview = row.get("_overview", "")
        
        parts = []
        
        # 1. Title (Highest weight - repeated for emphasis)
        parts.append(f"Title: {title}. {title}.")
        
        # 2. Tagline (Thematic essence in one line)
        if tagline and str(tagline).strip() and str(tagline).lower() != 'nan':
            parts.append(f"Tagline: {tagline}.")
        
        # 3. Genres (Critical for filtering)
        if genres:
            parts.append(f"Genres: {genres}.")
        
        # 4. Plot (Main semantic content - most important for similarity)
        if overview:
            parts.append(f"Plot: {overview}")
        
        # 5. Director (Stylistic consistency)
        if director:
            parts.append(f"Directed by {director}.")
        
        # 6. Writers (Story/dialogue patterns - NEW!)
        if writers and str(writers).strip() and str(writers).lower() != 'nan':
            parts.append(f"Written by {writers}.")
        
        # 7. Cast (Star power, reduced to top 10)
        if cast_str:
            parts.append(f"Starring: {cast_str}.")
        
        # 8. Production Companies (Studio patterns - NEW!)
        if companies:
            parts.append(f"Produced by {companies}.")
        
        # 9. Music Composer (Tone/style indicator - NEW!)
        if music_composer and str(music_composer).strip() and str(music_composer).lower() != 'nan':
            parts.append(f"Music by {music_composer}.")
        
        # Reinforcement tail (helps SBERT associate title with director)
        parts.append(f"Movie: {title} by {director}.")
        
        full_text = " ".join(parts)
        return clean_text(full_text)
    
    df["tags"] = df.apply(combine_tags, axis=1)
    
    # Drop temp columns
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    
    # Remove movies with empty tags
    original_count = len(df)
    df = df[df["tags"].str.len() > 10]
    logger.info(f"Generated tags for {len(df):,} movies (dropped {original_count - len(df):,} with insufficient data)")
    
    return df.reset_index(drop=True)


def build_sbert_embeddings(tags: pd.Series) -> tuple[SentenceTransformer, np.ndarray]:
    """
    Build Semantic Embeddings using SBERT.
    
    UPGRADED to 'all-mpnet-base-v2' (768 dimensions) for peak quality.
    MPNet is the best sentence transformer for semantic similarity.
    ~30% better than MiniLM on benchmarks.
    """
    # PEAK QUALITY: Use MPNet instead of MiniLM
    model_name = 'all-mpnet-base-v2'  # 768 dims, 110M params
    logger.info(f"Loading SBERT model: {model_name} (PEAK QUALITY)...")
    
    model = SentenceTransformer(model_name)
    
    logger.info(f"Encoding {len(tags):,} movies... (this may take a while)")
    embeddings = model.encode(
        tags.tolist(), 
        show_progress_bar=True, 
        batch_size=32,
        convert_to_numpy=True
    )
    
    # Normalize for Cosine Similarity (SBERT output is usually normalized, but good to ensure)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    logger.info(f"Created embeddings matrix with shape {embeddings.shape}")
    return model, embeddings


def transform(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Main transformation pipeline: generate tags and build SBERT embeddings.
    
    Args:
        df: Optional DataFrame. If None, loads from processed Parquet.
        
    Returns:
        Tuple of (transformed DataFrame, SBERT embeddings)
    """
    logger.info("Starting transformation...")
    
    # Load from Parquet if not provided
    if df is None:
        parquet_path = paths.processed_data / "movies.parquet"
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded {len(df):,} movies from {parquet_path}")
    
    # Generate tags
    df = generate_tags(df)
    
    # Build SBERT embeddings
    model, vectors = build_sbert_embeddings(df["tags"])
    
    # Save embeddings (model is standard, no need to save)
    output_path = paths.models / "sbert_embeddings.npy"
    np.save(output_path, vectors)
    logger.info(f"Saved SBERT embeddings with shape {vectors.shape} to {output_path}")
    
    # Save transformed DataFrame (with tags)
    output_path = paths.processed_data / "movies_transformed.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved transformed data to {output_path}")
    
    logger.info("Transformation complete")
    return df, vectors


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    transform()
