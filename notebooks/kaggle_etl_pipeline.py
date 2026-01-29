# Movie Recommendation System - GPU ETL Pipeline
# PySpark-Style Architecture using Pandas for GPU Compatibility
# Run this notebook on Kaggle with GPU accelerator enabled

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================
# CONFIGURATION
# ============================================

from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
HF_REPO = "pavanbadempet/movie-recs-models"

print(f"✅ HF Token loaded: {HF_TOKEN[:10]}..." if HF_TOKEN else "❌ No HF_TOKEN secret found!")

# ============================================
# SPARK-STYLE IMPORTS & SETUP
# ============================================

!pip install -q sentence-transformers faiss-cpu huggingface_hub

import ast
import re
import time
import logging
from pathlib import Path
from typing import Callable
from functools import reduce

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# GPU Check
print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# SPARK-STYLE DATAFRAME WRAPPER
# ============================================

class SparkStyleDF:
    """
    PySpark-style wrapper around Pandas DataFrame.
    Enables method chaining and functional transformations.
    """
    
    def __init__(self, df: pd.DataFrame):
        self._df = df
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    def withColumn(self, col_name: str, func: Callable) -> 'SparkStyleDF':
        """Add or replace a column using a function (like Spark's withColumn)."""
        self._df = self._df.copy()
        self._df[col_name] = func(self._df)
        return self
    
    def filter(self, condition: Callable) -> 'SparkStyleDF':
        """Filter rows based on condition (like Spark's filter/where)."""
        self._df = self._df[condition(self._df)].copy()
        return self
    
    def select(self, *cols) -> 'SparkStyleDF':
        """Select specific columns (like Spark's select)."""
        available = [c for c in cols if c in self._df.columns]
        self._df = self._df[available].copy()
        return self
    
    def dropDuplicates(self, subset=None) -> 'SparkStyleDF':
        """Drop duplicates (like Spark's dropDuplicates)."""
        self._df = self._df.drop_duplicates(subset=subset)
        return self
    
    def dropna(self, subset=None) -> 'SparkStyleDF':
        """Drop nulls (like Spark's dropna)."""
        self._df = self._df.dropna(subset=subset)
        return self
    
    def cache(self) -> 'SparkStyleDF':
        """Simulate Spark's cache (no-op in Pandas, but keeps API compatible)."""
        return self
    
    def count(self) -> int:
        """Count rows."""
        return len(self._df)
    
    def show(self, n: int = 5) -> None:
        """Display first n rows (like Spark's show)."""
        print(self._df.head(n).to_string())
    
    def transform(self, func: Callable) -> 'SparkStyleDF':
        """Apply a transformation function (like Spark's transform)."""
        return func(self)
    
    def toPandas(self) -> pd.DataFrame:
        """Convert to Pandas (identity in this case)."""
        return self._df.copy()

# ============================================
# UDF-STYLE FUNCTIONS (Like Spark UDFs)
# ============================================

def udf_parse_json(value):
    """UDF: Parse stringified JSON/list column to extract names."""
    if pd.isna(value) or value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return [item.get("name", str(item)) for item in parsed if isinstance(item, dict)]
        return [str(parsed)]
    except (ValueError, SyntaxError):
        return [s.strip() for s in str(value).split(",") if s.strip()]

def udf_clean_text(text):
    """UDF: Clean text while preserving punctuation for SBERT."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^\w\s.,;:!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ============================================
# TRANSFORMATION STAGES (Spark Pipeline Style)
# ============================================

def stage_ingest(path: str) -> SparkStyleDF:
    """Stage 1: Data Ingestion (like Spark read)."""
    logger.info(f"� STAGE: INGEST - Loading from {path}")
    df = pd.read_csv(path, low_memory=False)
    logger.info(f"   Loaded {len(df):,} records")
    return SparkStyleDF(df)

def stage_clean(sdf: SparkStyleDF) -> SparkStyleDF:
    """Stage 2: Data Cleaning & Filtering."""
    logger.info("🔧 STAGE: CLEAN - Applying filters")
    original = sdf.count()
    
    result = (sdf
        .dropna(subset=["title", "overview"])
        .filter(lambda df: df["vote_count"] >= 1)
        .filter(lambda df: df["popularity"] >= 0.5)
        .filter(lambda df: df["adult"] != True)
        .dropDuplicates(subset=["id"])
        .cache()
    )
    
    logger.info(f"   Filtered: {original:,} → {result.count():,} ({result.count()/original*100:.1f}%)")
    return result

def stage_feature_engineering(sdf: SparkStyleDF) -> SparkStyleDF:
    """Stage 3: Feature Engineering - Generate Tags."""
    logger.info("📝 STAGE: FEATURE ENGINEERING - Generating tags")
    
    df = sdf.df.copy()
    
    # Parse JSON columns (like Spark's from_json)
    for col_name in ["genres", "keywords", "production_companies"]:
        target = f"_{col_name}" if col_name != "production_companies" else "_companies"
        if col_name in df.columns:
            df[target] = df[col_name].apply(udf_parse_json).str.join(", ")
        else:
            df[target] = ""
    
    # Clean overview
    df["_overview"] = df["overview"].fillna("").astype(str).apply(udf_clean_text)
    
    # Build tags using concat (like Spark's concat_ws)
    def build_tag_row(row):
        parts = [
            f"Title: {row['title']}. {row['title']}.",
            f"Tagline: {row.get('tagline', '')}." if pd.notna(row.get('tagline')) else "",
            f"Genres: {row.get('_genres', '')}." if row.get('_genres') else "",
            f"Plot: {row.get('_overview', '')}",
            f"Directed by {row.get('director', '')}." if pd.notna(row.get('director')) else "",
            f"Starring: {','.join(str(row.get('cast', '')).split(',')[:10])}." if pd.notna(row.get('cast')) else "",
            f"Produced by {row.get('_companies', '')}." if row.get('_companies') else "",
            f"Movie: {row['title']} by {row.get('director', 'Unknown')}."
        ]
        return udf_clean_text(" ".join(p for p in parts if p))
    
    df["tags"] = df.apply(build_tag_row, axis=1)
    
    # Drop temp columns (like Spark's drop)
    df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
    df = df[df["tags"].str.len() > 10].reset_index(drop=True)
    
    logger.info(f"   Generated tags for {len(df):,} movies")
    return SparkStyleDF(df)

def stage_vectorize(sdf: SparkStyleDF) -> tuple:
    """Stage 4: Vectorization using SBERT (GPU Accelerated)."""
    logger.info("🚀 STAGE: VECTORIZE - Encoding with SBERT")
    
    df = sdf.df
    
    # Load model
    model = SentenceTransformer('all-mpnet-base-v2')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    logger.info(f"   Model loaded on: {device.upper()}")
    
    # Encode (like Spark ML's transform)
    start = time.time()
    BATCH_SIZE = 128 if torch.cuda.is_available() else 32
    
    embeddings = model.encode(
        df["tags"].tolist(),
        show_progress_bar=True,
        batch_size=BATCH_SIZE,
        convert_to_numpy=True,
        device=device
    )
    
    # Normalize (like Spark ML's Normalizer)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    elapsed = time.time() - start
    logger.info(f"   Encoded {len(df):,} vectors in {elapsed:.1f}s ({len(df)/elapsed:.0f}/sec)")
    
    return sdf, embeddings

def stage_index(embeddings: np.ndarray) -> faiss.Index:
    """Stage 5: Build FAISS Index (like Spark ML's LSH)."""
    logger.info("🔍 STAGE: INDEX - Building FAISS")
    
    n_samples, n_features = embeddings.shape
    embeddings_f32 = np.ascontiguousarray(embeddings.astype(np.float32))
    
    if n_samples < 10000:
        index = faiss.IndexFlatIP(n_features)
    else:
        nlist = min(256, n_samples // 39)
        quantizer = faiss.IndexFlatIP(n_features)
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings_f32)
    
    index.add(embeddings_f32)
    logger.info(f"   Built index with {index.ntotal:,} vectors")
    
    return index

def stage_save(sdf: SparkStyleDF, embeddings: np.ndarray, index: faiss.Index, output_dir: Path):
    """Stage 6: Save outputs (like Spark's write)."""
    logger.info(f"💾 STAGE: SAVE - Writing to {output_dir}")
    output_dir.mkdir(exist_ok=True)
    
    # Save embeddings
    emb_path = output_dir / "sbert_embeddings.npy"
    np.save(emb_path, embeddings)
    logger.info(f"   sbert_embeddings.npy: {emb_path.stat().st_size / 1e6:.1f} MB")
    
    # Save index
    idx_path = output_dir / "faiss.index"
    faiss.write_index(index, str(idx_path))
    logger.info(f"   faiss.index: {idx_path.stat().st_size / 1e6:.1f} MB")
    
    # Save movies (like Spark's write.parquet)
    movies_path = output_dir / "movies_transformed.parquet"
    essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 
                      'vote_count', 'popularity', 'release_date', 'poster_path',
                      'director', 'cast', 'original_language', 'tags']
    save_cols = [c for c in essential_cols if c in sdf.df.columns]
    sdf.df[save_cols].to_parquet(movies_path, index=False)
    logger.info(f"   movies_transformed.parquet: {movies_path.stat().st_size / 1e6:.1f} MB")
    
    return emb_path, idx_path, movies_path

def stage_upload(files: list, hf_repo: str, hf_token: str):
    """Stage 7: Upload to Hugging Face (like Spark's write to cloud)."""
    if not hf_token:
        logger.warning("⚠️ No HF_TOKEN - skipping upload")
        return
    
    logger.info(f"☁️ STAGE: UPLOAD - Uploading to {hf_repo}")
    api = HfApi()
    
    for file_path in files:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=file_path.name,
            repo_id=hf_repo,
            repo_type="model",
            token=hf_token
        )
        logger.info(f"   ✅ Uploaded {file_path.name}")
    
    logger.info(f"🎉 All files at: https://huggingface.co/{hf_repo}")

# ============================================
# MAIN PIPELINE (Spark Job Style)
# ============================================

def run_pipeline():
    """
    Execute the full ETL pipeline.
    Structured like a Spark application's main() function.
    """
    print("=" * 60)
    print("🎬 MOVIE RECOMMENDATION ETL PIPELINE")
    print("   PySpark-Style Architecture | GPU Accelerated")
    print("=" * 60)
    
    start_time = time.time()
    
    # Dataset path
    DATASET_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_movie_dataset_v11.csv"
    if not os.path.exists(DATASET_PATH):
        DATASET_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_all_movies.csv"
    
    OUTPUT_DIR = Path("/kaggle/working")
    
    # Execute pipeline stages
    sdf = stage_ingest(DATASET_PATH)
    sdf = stage_clean(sdf)
    sdf = stage_feature_engineering(sdf)
    sdf, embeddings = stage_vectorize(sdf)
    index = stage_index(embeddings)
    files = stage_save(sdf, embeddings, index, OUTPUT_DIR)
    stage_upload(files, HF_REPO, HF_TOKEN)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("📊 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"   Movies processed: {sdf.count():,}")
    print(f"   Embedding dimensions: {embeddings.shape}")
    print(f"   Total time: {elapsed:.1f}s")
    print(f"   GPU used: {torch.cuda.is_available()}")
    print("=" * 60)

# ============================================
# ENTRY POINT (Like Spark's spark-submit)
# ============================================

if __name__ == "__main__":
    run_pipeline()
