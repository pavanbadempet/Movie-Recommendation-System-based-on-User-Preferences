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
INCLUDE_ADULT_CONTENT = os.getenv("NOVA_INCLUDE_ADULT_CONTENT", "false").lower() in {"1", "true", "yes"}

# Dependencies (sentence-transformers uses PyTorch CUDA automatically)
!pip install -q sentence-transformers faiss-cpu huggingface_hub scikit-learn joblib

import ast
import hashlib
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from huggingface_hub import HfApi, hf_hub_download
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import ndcg_score

RUN_TS = datetime.now(timezone.utc).replace(microsecond=0)
RUN_ID = RUN_TS.strftime("%Y%m%dT%H%M%SZ")
RUN_DATE = RUN_TS.date().isoformat()

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
raw_row_count = len(df)
print(f"Loaded {len(df):,} movies")

# Filter — stricter than before for quality
df = df.dropna(subset=["title"])
df["title"] = df["title"].astype(str).str.strip()
df = df[df["title"] != ""]
if "overview" not in df.columns:
    df["overview"] = ""
df["overview"] = df["overview"].fillna("").astype(str)
adult_excluded_count = 0
if "adult" in df.columns and not INCLUDE_ADULT_CONTENT:
    adult_flag = df["adult"]
    if adult_flag.dtype == object:
        adult_flag = adult_flag.astype(str).str.lower().isin({"true", "1", "yes"})
    adult_excluded_count = int(adult_flag.sum())
    df = df[~adult_flag]
if "id" in df.columns:
    df = df.dropna(subset=["id"])
    sort_columns = [column for column in ["vote_count", "popularity"] if column in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=False, na_position="last")
    df = df.drop_duplicates(subset=["id"], keep="first")
df = df.reset_index(drop=True)
print(f"Retained {len(df):,} movies after identity gates ({adult_excluded_count:,} adult rows excluded for public demo)")


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


def add_catalog_coverage_features(frame):
    """Add quality features without deleting obscure long-tail titles."""
    frame = frame.copy()

    def text_len(column):
        if column not in frame.columns:
            return pd.Series(0, index=frame.index)
        return frame[column].fillna("").astype(str).str.strip().str.len()

    def numeric(column):
        if column not in frame.columns:
            return pd.Series(0.0, index=frame.index)
        return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    title_len = text_len("title")
    overview_len = text_len("overview")
    genres_len = text_len("genres")
    release_len = text_len("release_date")
    poster_len = text_len("poster_path")
    vote_count = numeric("vote_count")
    vote_average = numeric("vote_average")
    popularity = numeric("popularity")

    frame["metadata_completeness"] = (
        np.where(title_len > 0, 0.20, 0.0)
        + np.where(overview_len >= 20, 0.25, np.where(overview_len > 0, 0.10, 0.0))
        + np.where(genres_len > 0, 0.15, 0.0)
        + np.where(vote_count > 0, 0.15, 0.0)
        + np.where(popularity > 0, 0.10, 0.0)
        + np.where(release_len >= 4, 0.10, 0.0)
        + np.where(poster_len > 0, 0.05, 0.0)
    )
    vote_confidence = np.minimum(1.0, np.log1p(np.maximum(vote_count, 0.0)) / 8.0)
    popularity_norm = np.minimum(1.0, np.log1p(np.maximum(popularity, 0.0)) / 8.0)
    frame["content_quality_score"] = np.clip(
        frame["metadata_completeness"] * 0.45
        + (vote_average / 10.0) * vote_confidence * 0.30
        + popularity_norm * 0.25,
        0.0,
        1.0,
    )
    frame["quality_bucket"] = np.select(
        [
            frame["content_quality_score"] >= 0.70,
            frame["content_quality_score"] >= 0.45,
            frame["metadata_completeness"] >= 0.35,
        ],
        ["premium", "standard", "long_tail"],
        default="thin_metadata",
    )
    frame["searchable"] = title_len > 0
    frame["recommendable"] = (overview_len >= 20) | (genres_len > 0) | (frame["metadata_completeness"] >= 0.45)
    if "adult" in frame.columns:
        adult_flag = frame["adult"]
        if adult_flag.dtype == object:
            adult_flag = adult_flag.astype(str).str.lower().isin({"true", "1", "yes"})
        frame["is_adult_content"] = adult_flag.fillna(False).astype(bool)
    else:
        frame["is_adult_content"] = False
    frame["public_demo_eligible"] = ~frame["is_adult_content"]
    return frame


# Parse JSON columns
for col in ["genres", "keywords", "production_companies"]:
    key = "_companies" if col == "production_companies" else f"_{col}"
    df[key] = df[col].apply(parse_json).str.join(", ") if col in df.columns else ""

if "_genres" in df.columns:
    df["genres"] = df["_genres"]

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
df = add_catalog_coverage_features(df)
print(f"Generated rich tags for {len(df):,} movies")
print("Quality buckets:", df["quality_bucket"].value_counts(dropna=False).to_dict())

# Show sample tag for verification
sample = df[df['title'] == 'Avatar']
if len(sample) > 0:
    print(f"\nSample tag (Avatar):\n{sample.iloc[0]['tags'][:300]}...")


# ============================================================
# STEP 2B: Build SCD Type 2 movie dimension artifacts
# ============================================================
SCD_START_COL = "effective_start_at"
SCD_END_COL = "effective_end_at"
SCD_CURRENT_COL = "is_current"
SCD_HASH_COL = "record_hash"
SCD_HIGH_DATE = "9999-12-31T00:00:00Z"
SCD_TRACKED_COLUMNS = [
    "title",
    "overview",
    "genres",
    "vote_average",
    "vote_count",
    "popularity",
    "release_date",
    "poster_path",
    "director",
    "cast",
    "original_language",
]


def ensure_scd_columns(frame):
    """Ensure the SCD tracked columns exist before hashing."""
    frame = frame.copy()
    for column in SCD_TRACKED_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    return frame


def normalize_scd_value(value):
    """Normalize values before hashing so daily runs are deterministic."""
    if pd.isna(value):
        return "<NULL>"
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value).strip()


def add_record_hash(frame):
    """Add a stable hash for attributes that should create a new SCD version."""
    frame = ensure_scd_columns(frame)
    frame[SCD_HASH_COL] = frame[SCD_TRACKED_COLUMNS].apply(
        lambda row: hashlib.sha256(
            "||".join(normalize_scd_value(value) for value in row).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    return frame


def prepare_scd_versions(frame):
    """Prepare a latest snapshot as open-ended current SCD records."""
    versions = add_record_hash(frame)
    versions[SCD_START_COL] = f"{RUN_DATE}T00:00:00Z"
    versions[SCD_END_COL] = SCD_HIGH_DATE
    versions[SCD_CURRENT_COL] = True
    return versions


def load_existing_scd_history():
    """Download prior SCD history from Hugging Face when available."""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO,
            filename="movie_dimension_scd.parquet",
            repo_type="model",
            token=HF_TOKEN,
        )
        history = pd.read_parquet(path)
        print(f"Loaded existing SCD history from Hugging Face: {len(history):,} versions")
        return history
    except Exception as exc:
        print(f"No existing SCD history found; starting a new history table. Reason: {exc}")
        return None


def apply_scd_type2(existing, latest_snapshot):
    """Apply SCD Type 2 changes to the latest movie metadata snapshot."""
    incoming = prepare_scd_versions(latest_snapshot)

    if existing is None or len(existing) == 0:
        return incoming.reset_index(drop=True)

    existing = ensure_scd_columns(existing)
    if SCD_HASH_COL not in existing.columns:
        existing = add_record_hash(existing)
    if SCD_START_COL not in existing.columns:
        existing[SCD_START_COL] = f"{RUN_DATE}T00:00:00Z"
    if SCD_END_COL not in existing.columns:
        existing[SCD_END_COL] = SCD_HIGH_DATE
    if SCD_CURRENT_COL not in existing.columns:
        existing[SCD_CURRENT_COL] = True

    current = existing[existing[SCD_CURRENT_COL].astype(bool)].copy()
    current_by_id = current.set_index("id")[SCD_HASH_COL].to_dict()

    rows_to_insert = []
    ids_to_expire = set()

    for _, row in incoming.iterrows():
        movie_id = row["id"]
        current_hash = current_by_id.get(movie_id)
        if current_hash is None:
            rows_to_insert.append(row)
        elif current_hash != row[SCD_HASH_COL]:
            ids_to_expire.add(movie_id)
            rows_to_insert.append(row)

    if ids_to_expire:
        expire_mask = existing["id"].isin(ids_to_expire) & existing[SCD_CURRENT_COL].astype(bool)
        existing.loc[expire_mask, SCD_CURRENT_COL] = False
        existing.loc[expire_mask, SCD_END_COL] = f"{RUN_DATE}T00:00:00Z"

    if rows_to_insert:
        existing = pd.concat([existing, pd.DataFrame(rows_to_insert)], ignore_index=True, sort=False)

    return existing.reset_index(drop=True)


scd_input_columns = ["id"] + SCD_TRACKED_COLUMNS
scd_input = df[[column for column in scd_input_columns if column in df.columns]].copy()
scd_sort_columns = [column for column in ["vote_count", "popularity"] if column in scd_input.columns]
if scd_sort_columns:
    scd_input = scd_input.sort_values(scd_sort_columns, ascending=False, na_position="last")
scd_input = scd_input.drop_duplicates(subset=["id"], keep="first")

existing_scd = load_existing_scd_history() if HF_TOKEN else None
movie_dimension_scd = apply_scd_type2(existing_scd, scd_input)
movie_dimension_current = movie_dimension_scd[movie_dimension_scd[SCD_CURRENT_COL].astype(bool)].copy()

print(
    "SCD artifacts ready: "
    f"{len(movie_dimension_current):,} current movies, "
    f"{len(movie_dimension_scd):,} historical versions"
)


# ============================================================
# STEP 2B: Train free-tier learned ranker
# ============================================================
RANKER_FEATURE_COLUMNS = [
    "base_similarity",
    "dense_score",
    "sparse_score",
    "metadata_score",
    "behavior_score",
    "cross_encoder_score",
    "vote_average_norm",
    "vote_confidence",
    "popularity_norm",
    "release_year_norm",
    "is_recent",
]


def safe_float(value, default=0.0):
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def release_year(value):
    try:
        year = int(str(value or "")[:4])
        if 1800 <= year <= 2100:
            return year
    except Exception:
        return None
    return None


def catalog_quality_label(row):
    existing_score = row.get("content_quality_score")
    if existing_score is not None:
        try:
            existing_score = float(existing_score)
            if not math.isnan(existing_score):
                return max(0.0, min(1.0, existing_score))
        except Exception:
            pass
    vote_average = safe_float(row.get("vote_average"))
    vote_count = safe_float(row.get("vote_count"))
    popularity = safe_float(row.get("popularity"))
    quality = (vote_average / 10.0) * min(1.0, np.log1p(max(vote_count, 0.0)) / 8.0)
    popularity_score = min(1.0, np.log1p(max(popularity, 0.0)) / 8.0)
    return float(0.55 * popularity_score + 0.45 * quality)


def ranker_features(row, current_year=None):
    current_year = current_year or RUN_TS.year
    metadata_score = catalog_quality_label(row)
    year = release_year(row.get("release_date"))
    years_old = current_year - year if year else None
    vote_average = safe_float(row.get("vote_average"))
    vote_count = safe_float(row.get("vote_count"))
    popularity = safe_float(row.get("popularity"))
    return [
        metadata_score * 0.35,
        0.0,
        0.0,
        metadata_score,
        0.0,
        0.0,
        min(1.0, max(0.0, vote_average / 10.0)),
        min(1.0, math.log1p(max(vote_count, 0.0)) / 10.0),
        min(1.0, math.log1p(max(popularity, 0.0)) / 8.0),
        min(1.0, max(0.0, ((year or 1900) - 1900) / 140.0)),
        1.0 if years_old is not None and years_old <= 5 else 0.0,
    ]


def recall_at_k(labels, predictions, k=10, positive_threshold=0.2):
    positives = set(np.where(labels >= positive_threshold)[0])
    if not positives:
        return 0.0
    top_k = set(np.argsort(predictions)[::-1][:k])
    return round(len(positives & top_k) / len(positives), 6)


def train_catalog_ranker(movies_df, output_path):
    features = pd.DataFrame(
        [ranker_features(row) for _, row in movies_df.iterrows()],
        columns=RANKER_FEATURE_COLUMNS,
    )
    labels = np.asarray([catalog_quality_label(row) for _, row in movies_df.iterrows()], dtype=np.float32)
    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(features, labels)
    predictions = np.asarray(model.predict(features), dtype=np.float32)
    top_k = min(10, len(labels))
    ndcg = ndcg_score([labels], [predictions], k=top_k) if np.any(labels > 0) and top_k > 0 else 0.0
    metadata = {
        "training_mode": "catalog_bootstrap",
        "movie_count": int(len(movies_df)),
        "feedback_item_count": 0,
        "generated_at": RUN_TS.isoformat().replace("+00:00", "Z"),
        "evaluation": {
            "recall_at_k": recall_at_k(labels, predictions, k=top_k),
            "ndcg_at_k": round(float(ndcg), 6),
            "top_k": int(top_k),
            "prediction_min": round(float(predictions.min()), 6),
            "prediction_max": round(float(predictions.max()), 6),
        },
        "feature_importances": {
            column: round(float(importance), 6)
            for column, importance in zip(RANKER_FEATURE_COLUMNS, model.feature_importances_)
        },
    }
    joblib.dump(
        {
            "model": model,
            "feature_columns": RANKER_FEATURE_COLUMNS,
            "metadata": metadata,
        },
        output_path,
    )
    report = {
        "artifact_path": Path(output_path).name,
        "metadata": metadata,
    }
    report_path = Path(str(output_path) + ".metadata.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report, report_path


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
movie_ids = pd.to_numeric(df["id"], errors="raise").astype("int64").to_numpy()
assert len(movie_ids) == len(df), \
    f"MOVIE ID MAP MISMATCH! Movie IDs: {len(movie_ids)}, Movies: {len(df)}"
print(f"ALIGNMENT VERIFIED: {len(df):,} movies = {embeddings.shape[0]:,} embeddings = {index.ntotal:,} FAISS vectors")


def movie_id_sha256(ids):
    ids = np.asarray(ids, dtype=np.int64).astype("<i8", copy=False)
    return hashlib.sha256(ids.tobytes()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


SEMANTIC_STOPWORDS = {
    "about", "after", "again", "against", "also", "and", "are", "around", "back",
    "become", "becomes", "been", "before", "being", "between", "but", "can",
    "during", "each", "find", "finds", "for", "from", "has", "have", "her",
    "him", "his", "into", "its", "life", "lives", "man", "movie", "must",
    "new", "one", "only", "out", "own", "set", "she", "that", "the", "their",
    "them", "then", "they", "this", "through", "two", "when", "where", "who",
    "with", "years",
}

SEMANTIC_ARCS = {
    "adventure": {"adventure", "journey", "quest", "mission", "explore", "expedition"},
    "wonder": {"wonder", "magical", "mystical", "planet", "space", "future", "fantasy", "dream"},
    "survival": {"survival", "survive", "stranded", "escape", "disaster", "apocalypse", "wilderness"},
    "tension": {"war", "battle", "conflict", "enemy", "threat", "danger", "conspiracy", "chase"},
    "mystery": {"mystery", "detective", "secret", "investigation", "hidden", "murder", "missing"},
    "humor": {"comedy", "funny", "comic", "hilarious", "satire", "parody"},
    "romance": {"romance", "love", "relationship", "marriage", "heart", "couple"},
    "melancholy": {"grief", "loss", "lonely", "memory", "past", "regret", "death"},
    "fear": {"horror", "terror", "haunted", "monster", "demon", "nightmare", "killer"},
    "heroism": {"hero", "save", "protect", "rescue", "justice", "legend", "chosen"},
}

SEMANTIC_VIEWER_JOBS = {
    "escape_and_spectacle": {"adventure", "action", "space", "fantasy", "epic", "battle", "planet"},
    "world_immersion": {"world", "kingdom", "planet", "civilization", "future", "myth", "universe"},
    "intellectual_puzzle": {"mystery", "detective", "investigation", "conspiracy", "mind", "memory"},
    "emotional_catharsis": {"family", "love", "loss", "friendship", "grief", "home"},
    "adrenaline": {"chase", "fight", "war", "mission", "crime", "revenge", "explosion"},
    "comfort_laughs": {"comedy", "funny", "family", "romance", "friendship"},
    "dark_thrill": {"horror", "thriller", "killer", "haunted", "terror", "danger"},
}


def semantic_tokens(text):
    tokens = re.findall(r"[a-z][a-z0-9]{2,}", str(text or "").lower())
    return [token for token in tokens if token not in SEMANTIC_STOPWORDS and len(token) >= 3]


def parse_genre_labels(value):
    labels = []
    for item in re.split(r"[,|;/]", str(value or "")):
        label = item.strip().lower()
        if label:
            labels.append(label)
    return sorted(set(labels))


def labels_from_lexicon(tokens, lexicon):
    return [label for label, keywords in lexicon.items() if tokens & keywords]


def build_semantic_twin(row):
    genres = parse_genre_labels(row.get("genres"))
    text = " ".join(
        str(row.get(column) or "")
        for column in ["title", "tagline", "overview", "genres", "keywords", "director"]
    )
    tokens = semantic_tokens(text)
    title_tokens = set(semantic_tokens(row.get("title")))
    genre_tokens = set(semantic_tokens(row.get("genres")))
    counts = Counter(tokens)
    for token in title_tokens:
        counts[token] += 1.25
    for token in genre_tokens:
        counts[token] += 0.75
    concepts = [token for token, _ in counts.most_common(14)]
    token_set = set(tokens) | set(genres)
    risk_tags = []
    if len(str(row.get("overview") or "").strip()) < 30:
        risk_tags.append("thin_metadata")
    if float(row.get("vote_count") or 0) < 25:
        risk_tags.append("low_confidence")
    if "documentary" in token_set:
        risk_tags.append("documentary_spinoff")
    if token_set & {"sequel", "prequel", "reboot", "superhero", "mutant", "justice", "league"}:
        risk_tags.append("franchise_saturation")
    confidence = float(row.get("content_quality_score") or row.get("metadata_completeness") or 0)
    if confidence <= 0:
        confidence = min(1.0, 0.25 + 0.04 * len(concepts) + 0.08 * len(genres))
    return {
        "item_id": int(row.get("id")),
        "title": row.get("title"),
        "genres": genres,
        "concepts": concepts,
        "emotional_arcs": labels_from_lexicon(token_set, SEMANTIC_ARCS),
        "viewer_jobs": labels_from_lexicon(token_set, SEMANTIC_VIEWER_JOBS),
        "risk_tags": sorted(set(risk_tags)),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "generated_by": {
            "method": "deterministic_catalog_semantic_twin",
            "version": "1.0",
            "llm_in_hot_path": False,
        },
    }


def build_semantic_twin_artifacts(frame):
    rows = []
    concept_counts = Counter()
    arc_counts = Counter()
    job_counts = Counter()
    risk_counts = Counter()
    for row in frame.to_dict(orient="records"):
        twin = build_semantic_twin(row)
        concept_counts.update(twin["concepts"])
        arc_counts.update(twin["emotional_arcs"])
        job_counts.update(twin["viewer_jobs"])
        risk_counts.update(twin["risk_tags"])
        rows.append(
            {
                "id": int(row["id"]),
                "title": row.get("title"),
                "genres": json.dumps(twin["genres"], sort_keys=True),
                "concepts": json.dumps(twin["concepts"], sort_keys=True),
                "emotional_arcs": json.dumps(twin["emotional_arcs"], sort_keys=True),
                "viewer_jobs": json.dumps(twin["viewer_jobs"], sort_keys=True),
                "risk_tags": json.dumps(twin["risk_tags"], sort_keys=True),
                "confidence": float(twin["confidence"]),
                "semantic_twin_json": json.dumps(twin, sort_keys=True),
            }
        )
    twins = pd.DataFrame(rows)
    expected_ids = pd.to_numeric(frame["id"], errors="raise").astype("int64").to_numpy()
    actual_ids = pd.to_numeric(twins["id"], errors="raise").astype("int64").to_numpy()
    assert np.array_equal(expected_ids, actual_ids), "SEMANTIC TWIN ALIGNMENT MISMATCH"
    summary = {
        "artifact_version": 1,
        "run_id": RUN_ID,
        "run_date": RUN_DATE,
        "row_count": int(len(twins)),
        "avg_confidence": round(float(twins["confidence"].mean()), 6) if len(twins) else 0.0,
        "coverage": {
            "rows_with_concepts": int((twins["concepts"] != "[]").sum()),
            "rows_with_emotional_arcs": int((twins["emotional_arcs"] != "[]").sum()),
            "rows_with_viewer_jobs": int((twins["viewer_jobs"] != "[]").sum()),
            "rows_with_risk_tags": int((twins["risk_tags"] != "[]").sum()),
        },
        "top_concepts": dict(concept_counts.most_common(30)),
        "top_emotional_arcs": dict(arc_counts.most_common(20)),
        "top_viewer_jobs": dict(job_counts.most_common(20)),
        "risk_tags": dict(risk_counts.most_common(20)),
        "quality_gate": {
            "stage": "semantic_twins",
            "rows": int(len(twins)),
            "semantic_twin_rows": int(len(twins)),
            "id_order_matches_catalog": True,
        },
    }
    return twins, summary


# ============================================================
# STEP 6: Save artifacts
# ============================================================
OUT = Path("/kaggle/working")
emb_path = OUT / "sbert_embeddings.npy"
idx_path = OUT / "faiss.index"
movie_ids_path = OUT / "movie_ids.npy"
movies_path = OUT / "movies_transformed.parquet"
scd_path = OUT / "movie_dimension_scd.parquet"
current_dimension_path = OUT / "movie_dimension_current.parquet"
quality_path = OUT / "quality_report.json"
manifest_path = OUT / "pipeline_manifest.json"
ranker_path = OUT / "nova_ranker.joblib"
ranker_metadata_path = OUT / "nova_ranker.joblib.metadata.json"
semantic_twins_path = OUT / "semantic_twins.parquet"
semantic_summary_path = OUT / "semantic_twin_summary.json"

np.save(emb_path, embeddings)
np.save(movie_ids_path, movie_ids)
faiss.write_index(index, str(idx_path))

cols = ['id', 'title', 'overview', 'genres', 'vote_average', 'vote_count',
        'popularity', 'release_date', 'poster_path', 'director', 'cast',
        'original_language', 'tagline', 'keywords', 'tags',
        'metadata_completeness', 'content_quality_score', 'quality_bucket',
        'searchable', 'recommendable', 'is_adult_content', 'public_demo_eligible']
df[[c for c in cols if c in df.columns]].to_parquet(movies_path, index=False)
movie_dimension_scd.to_parquet(scd_path, index=False)
movie_dimension_current.to_parquet(current_dimension_path, index=False)
semantic_twins, semantic_summary = build_semantic_twin_artifacts(df)
semantic_twins.to_parquet(semantic_twins_path, index=False)
semantic_summary_path.write_text(json.dumps(semantic_summary, indent=2, sort_keys=True), encoding="utf-8")
ranker_report, generated_ranker_metadata_path = train_catalog_ranker(df, ranker_path)
ranker_metadata_path = generated_ranker_metadata_path

quality_report = {
    "run_id": RUN_ID,
    "run_date": RUN_DATE,
    "raw_rows": int(raw_row_count),
    "serving_rows": int(len(df)),
    "adult_excluded_rows": int(adult_excluded_count),
    "quality_buckets": {str(key): int(value) for key, value in df["quality_bucket"].value_counts(dropna=False).items()},
    "long_tail_rows": int((df["quality_bucket"] == "long_tail").sum()),
    "thin_metadata_rows": int((df["quality_bucket"] == "thin_metadata").sum()),
    "recommendable_rows": int(df["recommendable"].sum()),
    "searchable_rows": int(df["searchable"].sum()),
    "embedding_rows": int(embeddings.shape[0]),
    "faiss_index_size": int(index.ntotal),
    "movie_id_map_rows": int(len(movie_ids)),
    "movie_id_sha256": movie_id_sha256(movie_ids),
    "scd_current_rows": int(len(movie_dimension_current)),
    "scd_total_versions": int(len(movie_dimension_scd)),
    "ranker_training_mode": ranker_report["metadata"]["training_mode"],
    "ranker_feedback_item_count": ranker_report["metadata"]["feedback_item_count"],
    "semantic_twin_rows": int(len(semantic_twins)),
    "semantic_twin_avg_confidence": semantic_summary["avg_confidence"],
}
serving_contract = {
    "version": 1,
    "model_name": MODEL_NAME,
    "movie_rows": int(len(df)),
    "embedding_rows": int(embeddings.shape[0]),
    "embedding_dimensions": int(embeddings.shape[1]),
    "faiss_index_size": int(index.ntotal),
    "movie_id_map_rows": int(len(movie_ids)),
    "movie_id_sha256": quality_report["movie_id_sha256"],
}
manifest = {
    "run_id": RUN_ID,
    "run_date": RUN_DATE,
    "model_name": MODEL_NAME,
    "device": device,
    "hf_repo": HF_REPO,
    "artifacts": {
        "movies": movies_path.name,
        "embeddings": emb_path.name,
        "faiss_index": idx_path.name,
        "movie_ids": movie_ids_path.name,
        "movie_dimension_scd": scd_path.name,
        "movie_dimension_current": current_dimension_path.name,
        "quality_report": quality_path.name,
        "semantic_twins": semantic_twins_path.name,
        "semantic_twin_summary": semantic_summary_path.name,
        "ranker": ranker_path.name,
        "ranker_metadata": ranker_metadata_path.name,
    },
    "artifact_checksums": {},
    "serving_contract": serving_contract,
    "quality": quality_report,
    "semantic_twins": semantic_summary,
    "ranker": ranker_report["metadata"],
}
quality_path.write_text(json.dumps(quality_report, indent=2, sort_keys=True), encoding="utf-8")
for artifact_name in [
    emb_path,
    idx_path,
    movie_ids_path,
    movies_path,
    scd_path,
    current_dimension_path,
    quality_path,
    semantic_twins_path,
    semantic_summary_path,
    ranker_path,
    ranker_metadata_path,
]:
    manifest["artifact_checksums"][artifact_name.name] = {
        "sha256": file_sha256(artifact_name),
        "size_bytes": int(artifact_name.stat().st_size),
    }
manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

print(
    f"Saved: embeddings ({emb_path.stat().st_size/1e6:.0f}MB), "
    f"index ({idx_path.stat().st_size/1e6:.0f}MB), "
    f"movie ids ({movie_ids_path.stat().st_size/1e6:.1f}MB), "
    f"movies ({movies_path.stat().st_size/1e6:.0f}MB), "
    f"semantic twins ({semantic_twins_path.stat().st_size/1e6:.0f}MB), "
    f"SCD history ({scd_path.stat().st_size/1e6:.0f}MB), "
    f"ranker ({ranker_path.stat().st_size/1e6:.1f}MB)"
)


# ============================================================
# STEP 7: Upload ALL artifacts to HuggingFace (atomic)
# ============================================================
if HF_TOKEN:
    api = HfApi()
    files = [
        (emb_path, "sbert_embeddings.npy"),
        (idx_path, "faiss.index"),
        (movie_ids_path, "movie_ids.npy"),
        (movies_path, "movies_transformed.parquet"),
        (scd_path, "movie_dimension_scd.parquet"),
        (current_dimension_path, "movie_dimension_current.parquet"),
        (quality_path, "quality_report.json"),
        (semantic_twins_path, "semantic_twins.parquet"),
        (semantic_summary_path, "semantic_twin_summary.json"),
        (manifest_path, "pipeline_manifest.json"),
        (ranker_path, "nova_ranker.joblib"),
        (ranker_metadata_path, "nova_ranker.joblib.metadata.json"),
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
