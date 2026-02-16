# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

"""
Movie Recommendation ETL Pipeline (Kaggle GPU)
Generates SBERT embeddings and FAISS index from TMDB dataset.
Uploads ALL artifacts to HuggingFace Hub.

Tag generation mirrors the local ETL (etl/pandas_etl.py) exactly
to ensure consistent recommendation quality.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
# Placeholder for CI injection. 
# If running on Kaggle without injection, this remains as the placeholder string.
HF_TOKEN = "HF_TOKEN_PLACEHOLDER"

if HF_TOKEN == "HF_TOKEN_PLACEHOLDER":
    try:
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        HF_TOKEN = secrets.get_secret("HF_TOKEN")
    except Exception as e:
        print(f"WARNING: Could not retrieve HF_TOKEN secret. Artifact uploads will be SKIPPED. Error: {e}")
        HF_TOKEN = None

HF_REPO = "pavanbadempet/movie-recs-models"

# Dependencies (sentence-transformers uses PyTorch CUDA automatically)
!pip install -q sentence-transformers faiss-cpu huggingface_hub

import ast
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi

# GPU check
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == 'cuda' else ""))

# ============================================================
# STEP 1: Load and filter data
# ============================================================
DATA_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_movie_dataset_v11.csv"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "/kaggle/input/tmdb-movies-daily-updates/TMDB_all_movies.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"Loaded {len(df):,} movies")

# Filter — stricter than before for quality
df = df.dropna(subset=["title", "overview"])
df = df[df["overview"].str.len() > 20]               # Must have real overview
if "vote_count" in df.columns:
    df = df[df["vote_count"] >= 5]                    # At least 5 votes (was 1)
if "popularity" in df.columns:
    df = df[df["popularity"] >= 1.0]                  # Meaningful popularity (was 0.5)
if "adult" in df.columns:
    df = df[df["adult"] != True]
if "id" in df.columns:
    df = df.drop_duplicates(subset=["id"])
df = df.reset_index(drop=True)
print(f"Filtered to {len(df):,} quality movies")


# ============================================================
# STEP 2: Generate rich tags (mirrors etl/pandas_etl.py exactly)
# ============================================================
def parse_json(val):
    """Parse stringified JSON/list column to extract names."""
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        return [x.get("name", str(x)) for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [s.strip() for s in str(val).split(",") if s.strip()]


def clean(text):
    """Clean text while preserving punctuation for SBERT."""
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s.,;:!?-]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


# Parse JSON columns
for col in ["genres", "keywords", "production_companies"]:
    key = "_companies" if col == "production_companies" else f"_{col}"
    df[key] = df[col].apply(parse_json).str.join(", ") if col in df.columns else ""

df["_overview"] = df["overview"].fillna("").astype(str).apply(clean)

# Build tags — same structure as pandas_etl.py generate_tags()
title = df['title'].fillna("").astype(str)

# Title repeated twice for emphasis (boosts sequel/franchise matching)
tags = "Title: " + title + ". " + title + ". "

def add(prefix, col, suffix="."):
    """Conditionally append a field to tags (vectorized)."""
    if col not in df.columns:
        return ""
    s = df[col].fillna("").astype(str).str.strip()
    mask = (s != "") & (s.str.lower() != "nan")
    return np.where(mask, prefix + s + suffix + " ", "")

# Tagline (curated human summary — high semantic value)
tags = tags + add("Tagline: ", "tagline")

# Genres
tags = tags + add("Genres: ", "_genres")

# Keywords (critical for thematic matching: "time travel", "alien invasion", etc.)
tags = tags + add("Keywords: ", "_keywords")

# Plot (overview)
tags = tags + add("Plot: ", "_overview", "")

# Director
tags = tags + add("Directed by ", "director")

# Writers (same writer = thematically similar films)
tags = tags + add("Written by ", "writers")

# Cast (top 10, prefix "Starring" to match local ETL)
if "cast" in df.columns:
    cast = df['cast'].fillna("").str.split(",").str[:10].str.join(", ")
    tags = tags + np.where(cast != "", "Starring: " + cast + ". ", "")

# Studio
tags = tags + add("Produced by ", "_companies")

# Music composer
tags = tags + add("Music by ", "music_composer")

# Final identity string: "Movie: Title by Director."
director = df['director'].fillna("") if 'director' in df.columns else pd.Series("", index=df.index)
tags = tags + "Movie: " + title + " by " + director + "."

# Clean and filter
df["tags"] = pd.Series(tags).apply(clean)
df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
df = df[df["tags"].str.len() > 10].reset_index(drop=True)
print(f"Generated rich tags for {len(df):,} movies")

# Show sample tag for verification
sample = df[df['title'] == 'Avatar']
if len(sample) > 0:
    print(f"\nSample tag (Avatar):\n{sample.iloc[0]['tags'][:300]}...")


# ============================================================
# STEP 3: Generate embeddings (GPU accelerated)
# ============================================================
# all-mpnet-base-v2 (768d) — same model as backend
MODEL_NAME = 'all-mpnet-base-v2'
print(f"\nLoading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)
batch_size = 128 if device == 'cuda' else 16

start = time.time()
embeddings = model.encode(df["tags"].tolist(), show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)

# L2 normalize for cosine similarity via inner product
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid division by zero
embeddings = embeddings / norms

print(f"Encoded {len(df):,} movies in {time.time()-start:.1f}s → shape: {embeddings.shape}")


# ============================================================
# STEP 4: Build FAISS HNSW index
# ============================================================
n, d = embeddings.shape
emb32 = np.ascontiguousarray(embeddings.astype(np.float32))

index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = 200
index.hnsw.efSearch = 128
index.add(emb32)
print(f"Built HNSW index: {index.ntotal:,} vectors")


# ============================================================
# STEP 5: ALIGNMENT CHECK (Critical!)
# ============================================================
assert len(df) == embeddings.shape[0] == index.ntotal, \
    f"ALIGNMENT MISMATCH! Movies: {len(df)}, Embeddings: {embeddings.shape[0]}, FAISS: {index.ntotal}"
print(f"ALIGNMENT VERIFIED: {len(df):,} movies = {embeddings.shape[0]:,} embeddings = {index.ntotal:,} FAISS vectors")


# ============================================================
# STEP 6: Save artifacts
# ============================================================
OUT = Path("/kaggle/working")
emb_path = OUT / "sbert_embeddings.npy"
idx_path = OUT / "faiss.index"
movies_path = OUT / "movies_transformed.parquet"

np.save(emb_path, embeddings)
faiss.write_index(index, str(idx_path))

cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count',
        'popularity', 'release_date', 'poster_path', 'director', 'cast',
        'original_language', 'tagline', 'keywords', 'tags']
df[[c for c in cols if c in df.columns]].to_parquet(movies_path, index=False)

print(f"Saved: embeddings ({emb_path.stat().st_size/1e6:.0f}MB), index ({idx_path.stat().st_size/1e6:.0f}MB), movies ({movies_path.stat().st_size/1e6:.0f}MB)")


# ============================================================
# STEP 7: Upload ALL artifacts to HuggingFace (atomic)
# ============================================================
if HF_TOKEN:
    api = HfApi()
    files = [
        (emb_path, "sbert_embeddings.npy"),
        (idx_path, "faiss.index"),
        (movies_path, "movies_transformed.parquet"),
    ]
    for path, name in files:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
        print(f"  Uploaded {name}")
    print(f"All artifacts uploaded to huggingface.co/{HF_REPO}")
else:
    print("No HF_TOKEN - files saved locally only")


# ============================================================
# STEP 8: Sanity check — Avatar recommendations
# ============================================================
avatar_idx = df[df['title'].str.lower() == 'avatar'].index
if len(avatar_idx) > 0:
    query_vec = emb32[avatar_idx[0]].reshape(1, -1)
    _, neighbors = index.search(query_vec, 11)
    print(f"\nSanity Check — 'Avatar' top 10 recommendations:")
    for i, idx in enumerate(neighbors[0][1:]):  # Skip self
        print(f"  {i+1}. {df.iloc[idx]['title']}")

print(f"\nPipeline complete: {len(df):,} movies, {d}d embeddings, model={MODEL_NAME}")
