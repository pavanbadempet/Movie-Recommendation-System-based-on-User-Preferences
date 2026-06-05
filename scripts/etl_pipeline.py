import logging
from pathlib import Path
import urllib.request
import zipfile

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Constants
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def extract():
    """Extracts the MovieLens dataset from the remote source."""
    logger.info("Starting Extraction Phase...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    zip_path = DATA_DIR / "ml-latest-small.zip"

    if not zip_path.exists():
        logger.info(f"Downloading dataset from {MOVIELENS_URL}...")
        urllib.request.urlretrieve(MOVIELENS_URL, zip_path)
        logger.info("Download complete.")
    else:
        logger.info("Dataset zip already exists. Skipping download.")

    extract_dir = DATA_DIR / "ml-latest-small"
    if not extract_dir.exists():
        logger.info("Unzipping dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATA_DIR)
        logger.info("Extraction complete.")
    else:
        logger.info("Dataset already extracted.")


def transform() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transforms the raw CSV files into normalized dataframes."""
    logger.info("Starting Transformation Phase...")

    movies_path = DATA_DIR / "ml-latest-small" / "movies.csv"
    ratings_path = DATA_DIR / "ml-latest-small" / "ratings.csv"

    logger.info("Loading CSVs into Pandas...")
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    logger.info(f"Loaded {len(movies_df)} movies and {len(ratings_df)} ratings.")

    # Transform Movies: Clean titles and genres
    # Replace "|" with ", " in genres to match our system's expected format
    movies_df["genres"] = movies_df["genres"].str.replace("|", ", ", regex=False)

    # Extract Year from Title if present
    movies_df["release_year"] = movies_df["title"].str.extract(r"\((\d{4})\)", expand=False)
    movies_df["title"] = movies_df["title"].str.replace(r"\(\d{4}\)", "", regex=True).str.strip()

    # Transform Ratings: Normalize between -1.0 and 1.0 for the Quantum Model phase/amplitude
    # Standard ratings are 0.5 to 5.0.
    # Mean centering
    mean_rating = ratings_df["rating"].mean()
    ratings_df["normalized_rating"] = (ratings_df["rating"] - mean_rating) / (5.0 - 0.5)

    logger.info("Transformation complete.")
    return movies_df, ratings_df


def load(movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
    """Loads the processed data into binary parquet format for high-speed I/O."""
    logger.info("Starting Load Phase...")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    movies_out = PROCESSED_DIR / "movies_transformed.parquet"
    ratings_out = PROCESSED_DIR / "ratings_transformed.parquet"

    logger.info("Writing to columnar Parquet format...")
    movies_df.to_parquet(movies_out, index=False)
    ratings_df.to_parquet(ratings_out, index=False)

    logger.info(f"Load complete. Artifacts saved to {PROCESSED_DIR}")


def build_interaction_matrix():
    """
    Constructs the sparse user-item interaction matrix required by
    the Quantum Fluid and FAISS pipelines.
    """
    logger.info("Building Sparse Interaction Matrix...")
    ratings_path = PROCESSED_DIR / "ratings_transformed.parquet"
    if not ratings_path.exists():
        logger.error("Transformed ratings not found. Run the ETL pipeline first.")
        return

    ratings_df = pd.read_parquet(ratings_path)

    # Create pivot table (User x Movie)
    # For large datasets, we would use scipy.sparse, but for small we can use pandas
    user_item_matrix = ratings_df.pivot(index="userId", columns="movieId", values="normalized_rating").fillna(0)

    matrix_out = PROCESSED_DIR / "user_item_matrix.parquet"
    user_item_matrix.to_parquet(matrix_out)

    logger.info(f"Matrix built with shape {user_item_matrix.shape}. Saved to {matrix_out}")


if __name__ == "__main__":
    logger.info("=== INITIALIZING DATA ENGINEERING ETL PIPELINE ===")
    extract()
    movies, ratings = transform()
    load(movies, ratings)
    build_interaction_matrix()
    logger.info("=== ETL PIPELINE SUCCESSFULLY COMPLETED ===")
