"""
Movie Recommendation ETL Pipeline
Generates SBERT embeddings and FAISS index from TMDB dataset
Uploads ALL artifacts to HuggingFace Hub (atomic: parquet + embeddings + index)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Config
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
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

# Filter
df = df.dropna(subset=["title", "overview"])
if "vote_count" in df.columns:
    df = df[df["vote_count"] >= 1]
if "popularity" in df.columns:
    df = df[df["popularity"] >= 0.5]
if "adult" in df.columns:
    df = df[df["adult"] != True]
if "id" in df.columns:
    df = df.drop_duplicates(subset=["id"])
df = df.reset_index(drop=True)
print(f"Filtered to {len(df):,} movies")


# ============================================================
# STEP 2: Generate tags for each movie
# ============================================================
def parse_json(val):
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        return [x.get("name", str(x)) for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else [str(parsed)]
    except Exception:
        return [s.strip() for s in str(val).split(",") if s.strip()]


def clean(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\w\s.,;:!?-]", " ", str(text))
    return re.sub(r"\s+", " ", text).strip()


# Build tags
for col in ["genres", "keywords", "production_companies"]:
    key = "_companies" if col == "production_companies" else f"_{col}"
    df[key] = df[col].apply(parse_json).str.join(", ") if col in df.columns else ""

df["_overview"] = df["overview"].fillna("").astype(str).apply(clean)

title = df['title'].fillna("").astype(str)
tags = "Title: " + title + ". "

def add(prefix, col, suffix="."):
    if col not in df.columns:
        return ""
    s = df[col].fillna("").astype(str).str.strip()
    mask = (s != "") & (s.str.lower() != "nan")
    return np.where(mask, prefix + s + suffix + " ", "")

tags = tags + add("Genres: ", "_genres")
tags = tags + add("Plot: ", "_overview", "")
tags = tags + add("Directed by ", "director")

if "cast" in df.columns:
    cast = df['cast'].fillna("").str.split(",").str[:10].str.join(", ")
    tags = tags + np.where(cast != "", "Cast: " + cast + ". ", "")

tags = tags + add("Studio: ", "_companies")

df["tags"] = pd.Series(tags).apply(clean)
df = df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")
df = df[df["tags"].str.len() > 10].reset_index(drop=True)
print(f"Generated tags for {len(df):,} movies")


# ============================================================
# STEP 3: Generate embeddings (GPU accelerated)
# ============================================================
# Using all-mpnet-base-v2 (768d) - same model used by the backend
# This ensures consistency between Kaggle pipeline and local ETL
MODEL_NAME = 'all-mpnet-base-v2'
print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME, device=device)
batch_size = 128 if device == 'cuda' else 16

start = time.time()
embeddings = model.encode(df["tags"].tolist(), show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
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
        'popularity', 'release_date', 'poster_path', 'director', 'cast', 'original_language', 'tags']
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
# STEP 8: Quick sanity check
# ============================================================
# Test that Avatar recommendations make sense
avatar_idx = df[df['title'].str.lower() == 'avatar'].index
if len(avatar_idx) > 0:
    query_vec = emb32[avatar_idx[0]].reshape(1, -1)
    _, neighbors = index.search(query_vec, 6)
    print(f"\nSanity Check - 'Avatar' recommendations:")
    for i, idx in enumerate(neighbors[0][1:]):  # Skip self
        print(f"  {i+1}. {df.iloc[idx]['title']}")

print(f"\nPipeline complete: {len(df):,} movies, {d}d embeddings, model={MODEL_NAME}")
