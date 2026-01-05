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
    """File and directory paths for the ETL pipeline."""
    
    # Raw data from Kaggle
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    
    # Processed Parquet files
    processed_data: Path = PROJECT_ROOT / "data" / "processed"
    
    # Model artifacts (FAISS index, vectorizer)
    models: Path = PROJECT_ROOT / "models"
    
    # Logs directory
    logs: Path = PROJECT_ROOT / "logs"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        for path in [self.raw_data, self.processed_data, self.models, self.logs]:
            path.mkdir(parents=True, exist_ok=True)


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
