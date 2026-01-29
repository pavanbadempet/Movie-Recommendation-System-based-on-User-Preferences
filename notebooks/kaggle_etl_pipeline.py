# Movie Recommendation System - GPU ETL Pipeline
# Run this notebook on Kaggle with GPU accelerator enabled
# After running, embeddings are uploaded to Hugging Face automatically

# ============================================
# CONFIGURATION - Load from Kaggle Secrets
# ============================================

from kaggle_secrets import UserSecretsClient

# Load secrets from Kaggle
user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")  # Add this in Kaggle Settings → Secrets

HF_REPO = "pavanbadempet/movie-recs-models"  # Your Hugging Face repo

print(f"✅ HF Token loaded: {HF_TOKEN[:10]}..." if HF_TOKEN else "❌ No HF_TOKEN secret found!")

# ============================================
# SETUP
# ============================================

# Install required packages
!pip install -q sentence-transformers faiss-cpu huggingface_hub pandera

import os
import ast
import re
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check GPU availability
import torch
print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================
# LOAD KAGGLE DATASET
# ============================================

# The dataset is already available in Kaggle input
DATASET_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_movie_dataset_v11.csv"

if not os.path.exists(DATASET_PATH):
    # Alternative path
    DATASET_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_all_movies.csv"

print(f"📂 Loading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH, low_memory=False)
print(f"   Loaded {len(df):,} movies")

# ============================================
# DATA PREPROCESSING
# ============================================

# Configuration
MIN_VOTE_COUNT = 10
MAX_MOVIES = None  # Set to limit dataset size, e.g., 50000

# Filter movies
print("🔧 Filtering movies...")
original_count = len(df)

# Remove nulls
df = df.dropna(subset=["title", "overview"])

# Filter by vote count
if "vote_count" in df.columns:
    df = df[df["vote_count"] >= MIN_VOTE_COUNT]

# Remove adult content
if "adult" in df.columns:
    df = df[df["adult"] != True]

# Limit size if needed
if MAX_MOVIES:
    df = df.head(MAX_MOVIES)

df = df.reset_index(drop=True)
print(f"   Filtered from {original_count:,} to {len(df):,} movies")

# ============================================
# FEATURE ENGINEERING - Generate Tags
# ============================================

def parse_json_column(value):
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

def clean_text(text):
    """Clean text while preserving punctuation for SBERT."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"[^\w\s.,;:!?-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("📝 Generating tags...")

# Parse JSON columns
for col_name in ["genres", "keywords", "production_companies"]:
    target = f"_{col_name}" if col_name != "production_companies" else "_companies"
    if col_name in df.columns:
        df[target] = df[col_name].apply(parse_json_column).str.join(", ")
    else:
        df[target] = ""

# Clean overview
df["_overview"] = df["overview"].fillna("").astype(str).apply(clean_text)

# Build tags string using vectorized operations
tags = pd.Series("", index=df.index)
title = df['title'].fillna("").astype(str)
tags += "Title: " + title + ". " + title + ". "

def add_section(prefix, col_name, suffix="."):
    if col_name not in df.columns:
        return ""
    s = df[col_name].fillna("").astype(str).str.strip()
    mask = (s != "") & (s.str.lower() != "nan")
    return np.where(mask, prefix + s + suffix + " ", "")

tags += add_section("Tagline: ", "tagline")
tags += add_section("Genres: ", "_genres")
tags += add_section("Plot: ", "_overview", "")
tags += add_section("Directed by ", "director")

# Cast (limit to top 10)
if "cast" in df.columns:
    s_cast = df['cast'].fillna("").astype(str).str.split(",").str[:10].str.join(", ")
    mask = s_cast != ""
    tags += np.where(mask, "Starring: " + s_cast + ". ", "")

tags += add_section("Produced by ", "_companies")

# Final tag
director = df['director'].fillna("") if 'director' in df.columns else pd.Series("", index=df.index)
tags += "Movie: " + title + " by " + director + "."

df["tags"] = tags.apply(clean_text)

# Cleanup temp columns
df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
df = df[df["tags"].str.len() > 10].reset_index(drop=True)

print(f"   Generated tags for {len(df):,} movies")

# ============================================
# GENERATE EMBEDDINGS (GPU ACCELERATED)
# ============================================

print("🚀 Loading SBERT model...")
model = SentenceTransformer('all-mpnet-base-v2')

# Move to GPU if available
if torch.cuda.is_available():
    model = model.to('cuda')
    print("   ✅ Model loaded on GPU")

print(f"🔢 Encoding {len(df):,} movies...")
start_time = time.time()

embeddings = model.encode(
    df["tags"].tolist(),
    show_progress_bar=True,
    batch_size=64,  # Can be larger with GPU
    convert_to_numpy=True,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# Normalize for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

elapsed = time.time() - start_time
print(f"   ✅ Encoded in {elapsed:.1f}s ({len(df)/elapsed:.0f} movies/sec)")
print(f"   Shape: {embeddings.shape}")

# ============================================
# BUILD FAISS INDEX
# ============================================

print("🔍 Building FAISS index...")
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
print(f"   ✅ Built index with {index.ntotal:,} vectors")

# ============================================
# SAVE OUTPUTS
# ============================================

OUTPUT_DIR = Path("/kaggle/working")
OUTPUT_DIR.mkdir(exist_ok=True)

# Save embeddings
embeddings_path = OUTPUT_DIR / "sbert_embeddings.npy"
np.save(embeddings_path, embeddings)
print(f"💾 Saved embeddings: {embeddings_path} ({embeddings_path.stat().st_size / 1e6:.1f} MB)")

# Save FAISS index
index_path = OUTPUT_DIR / "faiss.index"
faiss.write_index(index, str(index_path))
print(f"💾 Saved FAISS index: {index_path} ({index_path.stat().st_size / 1e6:.1f} MB)")

# Save processed movie data
movies_path = OUTPUT_DIR / "movies_transformed.parquet"
essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 
                  'vote_count', 'popularity', 'release_date', 'poster_path',
                  'director', 'cast', 'original_language', 'tags']
save_cols = [c for c in essential_cols if c in df.columns]
df[save_cols].to_parquet(movies_path, index=False)
print(f"💾 Saved movies: {movies_path} ({movies_path.stat().st_size / 1e6:.1f} MB)")

# ============================================
# UPLOAD TO HUGGING FACE
# ============================================

if HF_TOKEN:
    print(f"☁️ Uploading to Hugging Face: {HF_REPO}")
    api = HfApi()
    
    # Upload embeddings
    api.upload_file(
        path_or_fileobj=str(embeddings_path),
        path_in_repo="sbert_embeddings.npy",
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print("   ✅ Uploaded sbert_embeddings.npy")
    
    # Upload FAISS index
    api.upload_file(
        path_or_fileobj=str(index_path),
        path_in_repo="faiss.index",
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print("   ✅ Uploaded faiss.index")
    
    # Upload processed movies
    api.upload_file(
        path_or_fileobj=str(movies_path),
        path_in_repo="movies_transformed.parquet",
        repo_id=HF_REPO,
        repo_type="model",
        token=HF_TOKEN
    )
    print("   ✅ Uploaded movies_transformed.parquet")
    
    print(f"\n🎉 All files uploaded to: https://huggingface.co/{HF_REPO}")
else:
    print("\n📁 Files saved locally in /kaggle/working/")
    print("   Download them and upload manually to Hugging Face if needed.")

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*50)
print("📊 PIPELINE COMPLETE")
print("="*50)
print(f"   Movies processed: {len(df):,}")
print(f"   Embedding dimensions: {embeddings.shape[1]}")
print(f"   Total processing time: {time.time() - start_time:.1f}s")
print(f"   GPU used: {torch.cuda.is_available()}")
