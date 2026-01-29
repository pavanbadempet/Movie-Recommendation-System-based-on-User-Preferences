"""
Movie Recommendation ETL Pipeline
Generates SBERT embeddings and FAISS index from TMDB dataset
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

# Load data
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


# Tag generation helpers
def parse_json(val):
    if pd.isna(val) or val == "":
        return []
    try:
        parsed = ast.literal_eval(val)
        return [x.get("name", str(x)) for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else [str(parsed)]
    except:
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

# Embeddings - using BGE-M3 (SOTA for retrieval, much better than mpnet)
model = SentenceTransformer('BAAI/bge-m3', device=device)
batch_size = 64 if device == 'cuda' else 16  # BGE-M3 is larger, reduce batch

start = time.time()
embeddings = model.encode(df["tags"].tolist(), show_progress_bar=True, batch_size=batch_size, convert_to_numpy=True)
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
print(f"Encoded {len(df):,} movies in {time.time()-start:.1f}s")


# FAISS HNSW index (best for <1M vectors, no training needed, ~0.95+ recall)
n, d = embeddings.shape
emb32 = np.ascontiguousarray(embeddings.astype(np.float32))

index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)  # 32 = graph connectivity
index.hnsw.efConstruction = 200  # higher = better quality, slower build
index.hnsw.efSearch = 128  # higher = better recall at search time
index.add(emb32)
print(f"Built HNSW index: {index.ntotal:,} vectors")


# Save
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


# Upload to HuggingFace
if HF_TOKEN:
    api = HfApi()
    for path, name in [(emb_path, "sbert_embeddings.npy"), (idx_path, "faiss.index"), (movies_path, "movies_transformed.parquet")]:
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=HF_REPO, repo_type="model", token=HF_TOKEN)
    print(f"Uploaded to huggingface.co/{HF_REPO}")
else:
    print("No HF_TOKEN - files saved locally")

print(f"\nDone: {len(df):,} movies, {embeddings.shape[1]}d embeddings")
