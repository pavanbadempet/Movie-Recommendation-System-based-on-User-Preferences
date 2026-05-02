"""
Pandas-based ETL Pipeline.
Consolidated module for ingestion, transformation, and indexing.

Alternative to PySpark ETL for reliable local processing.
"""
import ast
import hashlib
import json
import logging
import platform
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import pandera.pandas as pa
from pandera.pandas import Column, Check, DataFrameSchema
from sentence_transformers import SentenceTransformer

from etl.config import paths, data_config

logger = logging.getLogger(__name__)
PIPELINE_NAME = "nova-pandas-etl"
PIPELINE_VERSION = "1.0"

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
    total_rows = len(df)
    null_titles = int(df["title"].isna().sum()) if "title" in df.columns else total_rows
    null_overviews = int(df["overview"].isna().sum()) if "overview" in df.columns else total_rows
    duplicate_ids = int(df["id"].duplicated().sum()) if "id" in df.columns else 0
    movies_with_votes = int((df["vote_count"] > 0).sum()) if "vote_count" in df.columns else 0
    vote_average_out_of_range = 0
    if "vote_average" in df.columns:
        vote_average = pd.to_numeric(df["vote_average"], errors="coerce")
        vote_average_out_of_range = int(((vote_average < 0) | (vote_average > 10)).sum())

    metrics = {
        "total_rows": int(total_rows),
        "null_titles": null_titles,
        "null_overviews": null_overviews,
        "duplicate_ids": duplicate_ids,
        "movies_with_votes": movies_with_votes,
        "vote_average_out_of_range": vote_average_out_of_range,
        "title_completeness": round((total_rows - null_titles) / total_rows, 6) if total_rows else 0.0,
        "overview_completeness": round((total_rows - null_overviews) / total_rows, 6) if total_rows else 0.0,
    }
    logger.info(f"Quality metrics: {metrics}")
    return metrics


def add_catalog_coverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add coverage-first quality features without dropping long-tail movies."""
    df = df.copy()
    title_len = df["title"].fillna("").astype(str).str.strip().str.len() if "title" in df.columns else pd.Series(0, index=df.index)
    overview_len = df["overview"].fillna("").astype(str).str.strip().str.len() if "overview" in df.columns else pd.Series(0, index=df.index)
    genres_len = df["genres"].fillna("").astype(str).str.strip().str.len() if "genres" in df.columns else pd.Series(0, index=df.index)
    release_len = df["release_date"].fillna("").astype(str).str.strip().str.len() if "release_date" in df.columns else pd.Series(0, index=df.index)
    poster_len = df["poster_path"].fillna("").astype(str).str.strip().str.len() if "poster_path" in df.columns else pd.Series(0, index=df.index)

    def numeric_column(column: str) -> pd.Series:
        if column not in df.columns:
            return pd.Series(0.0, index=df.index)
        return pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    vote_count = numeric_column("vote_count")
    vote_average = numeric_column("vote_average")
    popularity = numeric_column("popularity")

    df["metadata_completeness"] = (
        np.where(title_len > 0, 0.20, 0.0)
        + np.where(overview_len >= 20, 0.25, np.where(overview_len > 0, 0.10, 0.0))
        + np.where(genres_len > 0, 0.15, 0.0)
        + np.where(vote_count > 0, 0.15, 0.0)
        + np.where(popularity > 0, 0.10, 0.0)
        + np.where(release_len >= 4, 0.10, 0.0)
        + np.where(poster_len > 0, 0.05, 0.0)
    )
    vote_confidence = np.minimum(1.0, np.log1p(np.maximum(vote_count, 0)) / 8.0)
    popularity_norm = np.minimum(1.0, np.log1p(np.maximum(popularity, 0)) / 8.0)
    df["content_quality_score"] = (
        df["metadata_completeness"] * 0.45
        + (vote_average / 10.0) * vote_confidence * 0.30
        + popularity_norm * 0.25
    ).clip(lower=0.0, upper=1.0)
    df["quality_bucket"] = np.select(
        [
            df["content_quality_score"] >= 0.70,
            df["content_quality_score"] >= 0.45,
            df["metadata_completeness"] >= 0.35,
        ],
        ["premium", "standard", "long_tail"],
        default="thin_metadata",
    )
    df["searchable"] = title_len > 0
    df["recommendable"] = (overview_len >= 20) | (genres_len > 0) | (df["metadata_completeness"] >= 0.45)
    if "adult" in df.columns:
        adult_flag = df["adult"]
        if adult_flag.dtype == object:
            adult_flag = adult_flag.astype(str).str.lower().isin({"true", "1", "yes"})
        df["is_adult_content"] = adult_flag.fillna(False).astype(bool)
    else:
        df["is_adult_content"] = False
    df["public_demo_eligible"] = ~df["is_adult_content"]
    return df


def filter_movies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep catalog coverage while removing only rows that cannot be identified.
    Low-vote and low-popularity titles stay in the catalog and are ranked by
    coverage/quality features instead of being hard-filtered away.
    """
    original_count = len(df)
    
    # Remove rows that cannot be addressed or displayed.
    required = [column for column in ("id", "title") if column in df.columns]
    if required:
        df = df.dropna(subset=required)
    if "title" in df.columns:
        df = df[df["title"].astype(str).str.strip() != ""]

    # Keep one deterministic record per TMDB movie ID.
    # If a daily source refresh repeats an ID, the highest-signal row wins.
    if "id" in df.columns:
        sort_columns = [column for column in ("vote_count", "popularity") if column in df.columns]
        if sort_columns:
            df = df.sort_values(sort_columns, ascending=False, na_position="last")
        df = df.drop_duplicates(subset=["id"], keep="first")
    
    df = add_catalog_coverage_features(df)
    
    # Max limit
    if data_config.max_movies:
        df = df.head(data_config.max_movies)
    
    logger.info(f"Retained {len(df):,}/{original_count:,} movies after identity gates")
    return df.reset_index(drop=True)


def ingest(
    file_path: Path | None = None,
    return_quality: bool = False,
    return_raw: bool = False,
) -> pd.DataFrame | tuple:
    """Main ingestion pipeline."""
    logger.info("Starting ingestion...")
    df = load_kaggle_data(file_path)
    df = validate_schema(df)
    raw_df = df.copy()
    quality_metrics = run_quality_checks(df)
    df = filter_movies(df)
    
    # Save intermediate if needed, but we usually stream in memory in pipeline
    # save_to_parquet(df) 
    if return_quality and return_raw:
        return df, quality_metrics, raw_df
    if return_quality:
        return df, quality_metrics
    if return_raw:
        return df, raw_df
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
        # But in unified pipeline we pass df directly. 
        # If called standalone, try loading:
        if (paths.processed_data / "movies.parquet").exists():
            df = pd.read_parquet(paths.processed_data / "movies.parquet")
        else:
            raise FileNotFoundError("No input DataFrame or parquet file found.")

    df = generate_tags(df)
    model, vectors = build_sbert_embeddings(df["tags"])
    
    # Save serving artifacts atomically so failed runs do not corrupt the last good version.
    atomic_save_npy(vectors, paths.models / "sbert_embeddings.npy")
    atomic_write_parquet(df, paths.processed_data / "movies_transformed.parquet")
    
    logger.info("Transformation complete")
    return df, vectors


# ==========================================
# 3. INDEXING LOGIC
# ==========================================

def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    """Build FAISS HNSW index (matches Kaggle pipeline)."""
    n_samples, n_features = vectors.shape
    logger.info(f"Building FAISS HNSW index for {n_samples:,} vectors...")
    
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    
    # HNSW: best for <1M vectors, no training needed, ~0.95+ recall
    index = faiss.IndexHNSWFlat(n_features, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 200  # higher = better quality, slower build
    index.hnsw.efSearch = 128  # higher = better recall at search time
    
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
    atomic_write_faiss_index(index, paths.models / "faiss.index")
    
    logger.info("Indexing complete")
    return index


# ==========================================
# 4. RUN METADATA AND ARTIFACTS
# ==========================================

def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_cloud_path(path: object) -> bool:
    return isinstance(path, str) and path.startswith(("s3://", "gs://", "abfs://"))


def _local_artifact_dir(path_name: str, fallback_name: str) -> Path | None:
    path = getattr(paths, path_name, None)
    if path is None:
        logs_path = getattr(paths, "logs", None)
        if logs_path is None or _is_cloud_path(logs_path):
            return None
        path = Path(logs_path) / fallback_name
    if _is_cloud_path(path):
        return None
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_local_path(file_path: Path | str) -> Path:
    """Resolve local artifact paths and fail clearly for cloud URLs."""
    if _is_cloud_path(file_path):
        raise ValueError(f"Local atomic writes do not support cloud paths yet: {file_path}")
    return Path(file_path)


def _temp_artifact_path(output_path: Path) -> Path:
    """Build a same-directory temp path so replace is atomic on one filesystem."""
    return output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")


def _cleanup_temp_file(temp_path: Path) -> None:
    try:
        temp_path.unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary artifact %s", temp_path)


def atomic_write_parquet(df: pd.DataFrame, output_path: Path | str) -> Path:
    """Write a parquet file through a temporary file, then atomically replace."""
    output_path = _ensure_local_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_artifact_path(output_path)

    try:
        df.to_parquet(temp_path, index=False)
        temp_path.replace(output_path)
    except Exception:
        _cleanup_temp_file(temp_path)
        raise

    return output_path


def atomic_save_npy(array: np.ndarray, output_path: Path | str) -> Path:
    """Write a NumPy artifact atomically without np.save appending a suffix."""
    output_path = _ensure_local_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_artifact_path(output_path)

    try:
        with temp_path.open("wb") as handle:
            np.save(handle, array)
        temp_path.replace(output_path)
    except Exception:
        _cleanup_temp_file(temp_path)
        raise

    return output_path


def atomic_write_faiss_index(index: faiss.Index, output_path: Path | str) -> Path:
    """Write a FAISS index atomically."""
    output_path = _ensure_local_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_artifact_path(output_path)

    try:
        faiss.write_index(index, str(temp_path))
        temp_path.replace(output_path)
    except Exception:
        _cleanup_temp_file(temp_path)
        raise

    return output_path


def persist_stage_dataset(df: pd.DataFrame, path_name: str, file_stem: str, run_id: str) -> dict | None:
    """Persist a run-scoped stage dataset for lineage and replay."""
    stage_dir = _local_artifact_dir(path_name, path_name)
    if stage_dir is None:
        return None

    output_path = stage_dir / f"run_id={run_id}" / f"{file_stem}.parquet"
    atomic_write_parquet(df, output_path)
    return describe_file(output_path)


def persist_time_travel_snapshot(
    df: pd.DataFrame,
    path_name: str,
    table_name: str,
    run_id: str,
    run_date: str,
) -> dict | None:
    """Persist a manifest-backed table snapshot for local time travel."""
    table_root = _local_artifact_dir(path_name, path_name)
    if table_root is None:
        return None

    from etl.lakehouse import write_versioned_snapshot

    return write_versioned_snapshot(
        df=df,
        base_path=table_root,
        table_name=table_name,
        run_id=run_id,
        run_date=run_date,
    )


def assert_batch_invariants(
    df: pd.DataFrame | None,
    vectors: np.ndarray | None = None,
    index: faiss.Index | None = None,
    stage: str = "batch",
) -> dict:
    """Fail the batch run when core row-level contracts are broken."""
    if df is None or len(df) == 0:
        raise ValueError(f"{stage} produced no rows")

    required_columns = {"id", "title"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"{stage} missing required columns: {missing_columns}")

    if df["id"].isna().any():
        raise ValueError(f"{stage} contains null movie ids")
    if df["id"].duplicated().any():
        raise ValueError(f"{stage} contains duplicate movie ids")
    if df["title"].isna().any() or (df["title"].astype(str).str.strip() == "").any():
        raise ValueError(f"{stage} contains empty titles")

    result = {"stage": stage, "rows": int(len(df))}

    if vectors is not None:
        if len(vectors) != len(df):
            raise ValueError(f"{stage} vector count does not match row count")
        result["vector_rows"] = int(len(vectors))
        result["vector_dimensions"] = int(vectors.shape[1]) if len(vectors.shape) > 1 else 0

    if index is not None:
        if index.ntotal != len(df):
            raise ValueError(f"{stage} index size does not match row count")
        result["index_size"] = int(index.ntotal)

    return result


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def write_json_artifact(payload: dict, output_path: Path) -> Path:
    """Write a JSON artifact atomically for local filesystem runs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    temp_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(output_path)
    return output_path


def file_sha256(file_path: Path | str | None) -> str | None:
    """Return the SHA-256 checksum for a local file."""
    if file_path is None or _is_cloud_path(file_path):
        return None
    file_path = Path(file_path)
    if not file_path.exists() or not file_path.is_file():
        return None

    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(file_path: Path | str) -> dict:
    """Return path, size, and checksum metadata for a local artifact."""
    if _is_cloud_path(file_path):
        return {"path": file_path, "exists": None, "sha256": None, "size_bytes": None}

    file_path = Path(file_path)
    exists = file_path.exists()
    return {
        "path": str(file_path),
        "exists": exists,
        "size_bytes": int(file_path.stat().st_size) if exists and file_path.is_file() else 0,
        "sha256": file_sha256(file_path) if exists and file_path.is_file() else None,
    }


def persist_run_metadata(metrics: dict, manifest: dict, run_id: str) -> dict:
    """Persist quality and manifest artifacts when running on a local filesystem."""
    quality_dir = _local_artifact_dir("quality_reports", "quality")
    manifest_dir = _local_artifact_dir("manifests", "manifests")

    artifact_paths = {}
    if quality_dir is not None:
        quality_path = write_json_artifact(metrics.get("quality", {}), quality_dir / f"{run_id}.json")
        artifact_paths["quality_report"] = str(quality_path)

    if manifest_dir is not None:
        manifest_path = write_json_artifact(manifest, manifest_dir / f"{run_id}.json")
        artifact_paths["run_manifest"] = str(manifest_path)

    return artifact_paths


# ==========================================
# 5. ORCHESTRATION
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


def run_pipeline(
    raw_data_path: Path | None = None,
    skip_ingest: bool = False,
    run_id: str | None = None,
    run_date: str | None = None,
    write_metadata: bool = True,
) -> dict:
    """Execute complete ETL pipeline."""
    started_at = _utc_now()
    start_time = time.time()
    run_id = run_id or started_at.strftime("%Y%m%dT%H%M%SZ")
    run_date = run_date or started_at.date().isoformat()
    source_path = raw_data_path or (paths.raw_data / "TMDB_all_movies.csv")
    metrics = {
        "pipeline": PIPELINE_NAME,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": run_id,
        "run_date": run_date,
        "started_at": started_at.isoformat(),
        "stages": {},
        "quality_gates": {},
        "stage_artifacts": {},
        "time_travel_artifacts": {},
    }
    
    logger.info("STARTING PANDAS ETL PIPELINE")
    
    try:
        # 1. Ingest
        if not skip_ingest:
            with PipelineStage("INGEST"):
                df, quality_metrics, raw_df = ingest(raw_data_path, return_quality=True, return_raw=True)
                metrics["quality"] = quality_metrics
                metrics["raw_rows"] = quality_metrics["total_rows"]
                metrics["ingested_rows"] = len(df)
                metrics["filtered_rows"] = quality_metrics["total_rows"] - len(df)
                metrics["quality_gates"]["silver"] = assert_batch_invariants(df, stage="silver")
                bronze_artifact = persist_stage_dataset(raw_df, "bronze_data", "movies_raw", run_id)
                silver_artifact = persist_stage_dataset(df, "silver_data", "movies_curated", run_id)
                if bronze_artifact is not None:
                    metrics["stage_artifacts"]["bronze"] = bronze_artifact
                if silver_artifact is not None:
                    metrics["stage_artifacts"]["silver"] = silver_artifact
                bronze_snapshot = persist_time_travel_snapshot(raw_df, "bronze_data", "movies_raw", run_id, run_date)
                silver_snapshot = persist_time_travel_snapshot(df, "silver_data", "movies_curated", run_id, run_date)
                if bronze_snapshot is not None:
                    metrics["time_travel_artifacts"]["movies_raw"] = bronze_snapshot
                if silver_snapshot is not None:
                    metrics["time_travel_artifacts"]["movies_curated"] = silver_snapshot
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
            metrics["quality_gates"]["gold"] = assert_batch_invariants(df, vectors=vectors, stage="gold")
            gold_artifact = persist_stage_dataset(df, "gold_data", "movies_features", run_id)
            if gold_artifact is not None:
                metrics["stage_artifacts"]["gold"] = gold_artifact
            gold_snapshot = persist_time_travel_snapshot(df, "gold_data", "movies_features", run_id, run_date)
            if gold_snapshot is not None:
                metrics["time_travel_artifacts"]["movies_features"] = gold_snapshot
        
        # 3. Index
        with PipelineStage("INDEX"):
            index = build_index(vectors)
            metrics["index_size"] = index.ntotal
            metrics["quality_gates"]["serving"] = assert_batch_invariants(
                df,
                vectors=vectors,
                index=index,
                stage="serving",
            )

        metrics["success"] = True
        metrics["finished_at"] = _utc_now().isoformat()
        metrics["duration_seconds"] = round(time.time() - start_time, 3)
        metrics["artifacts"] = {
            "movies": describe_file(paths.processed_data / "movies_transformed.parquet"),
            "embeddings": describe_file(paths.models / "sbert_embeddings.npy"),
            "faiss_index": describe_file(paths.models / "faiss.index"),
        }

        manifest = {
            "pipeline": PIPELINE_NAME,
            "pipeline_version": PIPELINE_VERSION,
            "run_id": run_id,
            "run_date": run_date,
            "status": "success",
            "source": describe_file(source_path),
            "row_counts": {
                "raw_rows": metrics.get("raw_rows"),
                "curated_rows": metrics.get("ingested_rows"),
                "serving_rows": metrics.get("final_rows"),
                "index_size": metrics.get("index_size"),
            },
            "artifacts": metrics["artifacts"],
            "stage_artifacts": metrics["stage_artifacts"],
            "time_travel_artifacts": metrics["time_travel_artifacts"],
            "quality_gates": metrics["quality_gates"],
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
            "started_at": metrics["started_at"],
            "finished_at": metrics["finished_at"],
            "duration_seconds": metrics["duration_seconds"],
        }

        if write_metadata:
            metrics["metadata_artifacts"] = persist_run_metadata(metrics, manifest, run_id)

        logger.info(f"PIPELINE SUCCESS in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.exception("Pipeline failed")
        metrics["success"] = False
        metrics["error"] = str(e)
        metrics["finished_at"] = _utc_now().isoformat()
        metrics["duration_seconds"] = round(time.time() - start_time, 3)
        if write_metadata:
            failure_manifest = {
                "pipeline": PIPELINE_NAME,
                "pipeline_version": PIPELINE_VERSION,
                "run_id": run_id,
                "run_date": run_date,
                "status": "failed",
                "source": describe_file(source_path),
                "error": metrics["error"],
                "stage_artifacts": metrics.get("stage_artifacts", {}),
                "time_travel_artifacts": metrics.get("time_travel_artifacts", {}),
                "quality_gates": metrics.get("quality_gates", {}),
                "started_at": metrics["started_at"],
                "finished_at": metrics["finished_at"],
                "duration_seconds": metrics["duration_seconds"],
            }
            metrics["metadata_artifacts"] = persist_run_metadata(metrics, failure_manifest, run_id)
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
