"""
Configuration module for the Movie Recommendation ETL pipeline.
Centralizes all settings, paths, and environment variables.
"""
import os
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


@dataclass
class Paths:
    """File and directory paths for the ETL pipeline.
    
    CLOUD-NATIVE READY:
    Can be overridden by env vars to point to S3/GCS paths.
    """
    
    # Raw data (supports s3:// or local paths)
    raw_data: Path | str = os.getenv("RAW_DATA_PATH", PROJECT_ROOT / "data" / "raw")
    
    # Processed Parquet files
    processed_data: Path | str = os.getenv("PROCESSED_DATA_PATH", PROJECT_ROOT / "data" / "processed")
    
    # Model artifacts
    models: Path | str = os.getenv("MODELS_PATH", PROJECT_ROOT / "models")
    
    # Logs directory
    logs: Path | str = os.getenv("LOGS_PATH", PROJECT_ROOT / "logs")
    
    def __post_init__(self):
        """Create local directories if they don't exist and are local paths."""
        for path in [self.raw_data, self.processed_data, self.models, self.logs]:
            # Only mkdir if it's a local Path object, not a cloud URL string
            if isinstance(path, Path) or (isinstance(path, str) and not path.startswith(("s3://", "gs://", "abfs://"))):
                Path(path).mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Minimum vote count to filter movies (quality threshold)
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
