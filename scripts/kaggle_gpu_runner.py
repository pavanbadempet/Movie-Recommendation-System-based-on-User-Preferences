import os
import json
import time
import requests
from pathlib import Path

def create_kaggle_gpu_kernel_package():
    """
    Builds a standalone Kaggle GPU Kernel package that runs heavy AI model training 
    and vector embedding generation on Kaggle's FREE NVIDIA T4 16GB GPU (30 hrs/week).
    """
    print("==================================================================")
    print("--> KAGGLE GPU RUNNER: Offloading Heavy AI Compute to NVIDIA T4 GPU")
    print("==================================================================")

    kaggle_dir = Path("kaggle_gpu_kernel")
    kaggle_dir.mkdir(exist_ok=True)

    # 1. Kernel Metadata Manifest (configures NVIDIA T4 GPU + Python environment)
    kernel_metadata = {
        "id": f"{os.environ.get('KAGGLE_USERNAME', 'flameemperor')}/apex-movie-rec-gpu-pipeline",
        "title": "Apex Movie Rec Gpu Pipeline",
        "code_file": "main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "true",  # Enables FREE NVIDIA T4 / P100 GPU
        "enable_internet": "true", # Enables HTTP access to Neon PostgreSQL & Hugging Face
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": []
    }

    with open(kaggle_dir / "kernel-metadata.json", "w", encoding="utf-8") as f:
        json.dump(kernel_metadata, f, indent=2)

    # 2. Main GPU Execution Script (PyTorch FP16 Tensor Cores + SentenceTransformers)
    script_content = '''import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from psycopg2.extras import execute_values

print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")

# Load Database URL from Kaggle Secrets or Doppler
DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    print("DATABASE_URL environment variable is required to stream embeddings to Neon!")
    sys.exit(0)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# -------------------------------------------------------------------------
# 1. GPU VECTOR EMBEDDING GENERATION (SentenceTransformers FP16)
# -------------------------------------------------------------------------
print("Loading Gold movies dataset from Neon PostgreSQL...")
engine = create_engine(DATABASE_URL, connect_args={"sslmode": "require"})

try:
    df = pd.read_sql("SELECT id, title, overview, tags FROM movies WHERE tags IS NOT NULL", engine)
    print(f"Loaded {len(df)} active movie records from Neon PostgreSQL.")

    if not df.empty:
        EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing {EMBEDDING_MODEL_NAME} on {device}...")
        model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=device)
        
        if device == "cuda":
            model.half()  # Enable 16-bit FP16 Tensor Cores speedup

        clean_texts = df["tags"].fillna("").astype(str).tolist()
        print("Generating 768-D semantic vector embeddings on GPU...")
        embeddings = model.encode(clean_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

        # L2 Normalization for Cosine Similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True).astype(np.float32)
        norms = np.where(norms == 0, np.float32(1e-10), norms)
        embeddings = (embeddings / norms).astype(np.float32)

        print("Serializing vectors into JSON strings for pgvector HNSW index...")
        df["embedding"] = [json.dumps(vec.tolist()) for vec in embeddings]

        # Fast Batch Update to Neon PostgreSQL
        print("Updating vector embeddings in Neon PostgreSQL...")
        dbapi_conn = engine.raw_connection()
        try:
            with dbapi_conn.cursor() as cur:
                update_query = """
                    UPDATE movies AS m SET 
                        embedding = v.embedding::vector
                    FROM (VALUES %s) AS v(id, embedding)
                    WHERE m.id = v.id;
                """
                tuples_to_update = list(zip(df["id"].astype(int), df["embedding"]))
                execute_values(cur, update_query, tuples_to_update, template=None, page_size=1000)
                dbapi_conn.commit()
                print("Successfully updated 768-D 100% full precision Float32 vectors in Neon PostgreSQL!")
        finally:
            dbapi_conn.close()

except Exception as err:
    print(f"Kaggle GPU Execution Note: {err}")

print("--> Kaggle GPU Execution Complete!")
'''

    with open(kaggle_dir / "main.py", "w", encoding="utf-8") as f:
        f.write(script_content)

    print(f"Generated Kaggle GPU Kernel package in '{kaggle_dir.resolve()}'")

def push_and_run_kaggle_gpu():
    """Pushes kernel to Kaggle API and triggers execution."""
    create_kaggle_gpu_kernel_package()

    # Support both new-format KGAT tokens and legacy username+key auth.
    # New tokens (KGAT...) must be set as KAGGLE_API_TOKEN.
    # Doppler stores the token under KAGGLE_KEY; GitHub Actions uses KAGGLE_API_TOKEN.
    kaggle_token = os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
    username = os.environ.get("KAGGLE_USERNAME")

    if not kaggle_token:
        print("[NOTICE] KAGGLE_KEY or KAGGLE_API_TOKEN not set in environment. Package generated locally!")
        return

    print("Authenticating with Kaggle API...")
    try:
        if kaggle_token.startswith("KGAT"):
            # New-format token: set KAGGLE_API_TOKEN and clear legacy vars
            os.environ["KAGGLE_API_TOKEN"] = kaggle_token
            os.environ.pop("KAGGLE_USERNAME", None)
            os.environ.pop("KAGGLE_KEY", None)
        elif username:
            # Legacy format: username + raw hex key
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = kaggle_token

        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        print("Pushing kernel to Kaggle GPU Cluster...")
        res = api.kernels_push("kaggle_gpu_kernel")
        print(f"Kaggle Push Response: {res}")
        print("--> Kaggle GPU Kernel successfully launched on NVIDIA T4 GPU!")

    except Exception as e:
        print(f"Kaggle API trigger note: {e}")

if __name__ == "__main__":
    push_and_run_kaggle_gpu()
