"""
Configuration module for the Movie Recommendation ETL pipeline.
Centralizes all settings, paths, and environment variables.
"""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CLOUD_PATH_PREFIXES = ("s3://", "gs://", "abfs://")


def _normalize_storage_path(path: Path | str) -> Path | str:
    """Keep cloud URLs as strings and convert local paths to Path objects."""
    if isinstance(path, Path):
        return path
    if path.startswith(CLOUD_PATH_PREFIXES):
        return path.rstrip("/")
    return Path(path)


@dataclass
class Paths:
    """File and directory paths for the ETL pipeline.

    CLOUD-NATIVE READY:
    Can be overridden by env vars to point to S3/GCS paths.

    MEDALLION ARCHITECTURE:
    Supports Bronze (raw), Silver (cleaned), Gold (business) layers.
    """

    # Raw data (supports s3:// or local paths)
    raw_data: Path | str = os.getenv("RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw")

    # Processed local data used by the pandas ETL and existing model artifacts
    processed_data: Path | str = os.getenv("PROCESSED_DATA_PATH", PROJECT_ROOT / "data" / "processed")

    # Bronze layer - raw ingested data (immutable)
    bronze_data: Path | str = os.getenv("BRONZE_DATA_PATH", PROJECT_ROOT / "data" / "bronze")

    # Silver layer - cleaned, validated, filtered data
    silver_data: Path | str = os.getenv("SILVER_DATA_PATH", PROJECT_ROOT / "data" / "silver")

    # Gold layer - business-level aggregations and final datasets
    gold_data: Path | str = os.getenv("GOLD_DATA_PATH", PROJECT_ROOT / "data" / "gold")

    # Model artifacts
    models: Path | str = os.getenv("MODELS_PATH", PROJECT_ROOT / "models")

    # Logs directory
    logs: Path | str = os.getenv("LOGS_PATH", PROJECT_ROOT / "logs")

    # Data quality reports for pipeline runs
    quality_reports: Path | str = os.getenv("QUALITY_REPORTS_PATH", PROJECT_ROOT / "data" / "quality")

    # Run manifests for lineage, checksums, and output artifacts
    manifests: Path | str = os.getenv("MANIFESTS_PATH", PROJECT_ROOT / "data" / "manifests")

    def __post_init__(self):
        """Create local directories if they don't exist and are local paths."""
        path_names = (
            "raw_data",
            "processed_data",
            "bronze_data",
            "silver_data",
            "gold_data",
            "models",
            "logs",
            "quality_reports",
            "manifests",
        )
        for name in path_names:
            path = _normalize_storage_path(getattr(self, name))
            setattr(self, name, path)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""

    # Vote-count threshold used for quality scoring; do not hard-filter long-tail titles.
    min_vote_count: int = 50

    # Maximum number of movies to process (None = all)
    max_movies: int | None = None

    # Number of recommendations to return
    n_recommendations: int = 10

    # TF-IDF parameters
    tfidf_max_features: int = 5000
    tfidf_ngram_range: tuple = (1, 2)

    # FAISS index parameters
    faiss_nlist: int = 100  # Number of clusters for IVF index


@dataclass
class APIConfig:
    """Configuration for TMDB API (for poster/video fetching)."""

    api_key: str = os.getenv("TMDB_API_KEY")
    base_url: str = "https://api.themoviedb.org/3"
    image_base_url: str = "https://image.tmdb.org/t/p/w500"


# Global configuration instances
paths = Paths()
data_config = DataConfig()
api_config = APIConfig()
