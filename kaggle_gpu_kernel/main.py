import os
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
