import os
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import logging
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Constants
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
POSTERS_DIR = PROJECT_ROOT / "data" / "posters"
POSTERS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

def fetch_and_download_poster(tmdb_id: str, movie_id: str) -> bool:
    """Fetch poster path from TMDB API and download the image."""
    if not tmdb_id or pd.isna(tmdb_id):
        return False
        
    save_path = POSTERS_DIR / f"{movie_id}.jpg"
    if save_path.exists():
        return True # Already downloaded

    try:
        # Get movie details from TMDB
        response = requests.get(
            f"{TMDB_BASE_URL}/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5
        )
        if response.status_code != 200:
            return False

        data = response.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return False

        # Download the image
        img_url = f"{TMDB_IMAGE_BASE_URL}{poster_path}"
        img_response = requests.get(img_url, stream=True, timeout=5)
        
        if img_response.status_code == 200:
            with open(save_path, "wb") as f:
                for chunk in img_response.iter_content(1024):
                    f.write(chunk)
            return True
    except Exception as e:
        logger.debug(f"Error downloading poster for TMDB {tmdb_id}: {e}")
    
    return False

def main():
    if not TMDB_API_KEY:
        logger.error("TMDB_API_KEY not found in environment variables.")
        return

    logger.info("Loading movie metadata...")
    # Load movies data which should contain TMDB IDs
    # If TMDB ID is not available, we can use the links.csv from MovieLens
    links_path = PROJECT_ROOT / "data" / "raw" / "links.csv"
    if not links_path.exists():
        logger.error(f"links.csv not found at {links_path}. We need tmdbId mappings.")
        return
        
    links_df = pd.read_csv(links_path)
    links_df = links_df.dropna(subset=['tmdbId'])
    
    total_movies = len(links_df)
    logger.info(f"Found {total_movies} movies with TMDB IDs.")
    
    success_count = 0
    
    # We use ThreadPoolExecutor to download posters in parallel to save time
    # Max workers kept relatively low to avoid rate-limiting from TMDB
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = []
        for _, row in links_df.iterrows():
            movie_id = str(int(row["movieId"]))
            tmdb_id = str(int(row["tmdbId"]))
            futures.append(executor.submit(fetch_and_download_poster, tmdb_id, movie_id))
            
        for i, future in enumerate(futures):
            if future.result():
                success_count += 1
            if (i + 1) % 500 == 0:
                logger.info(f"Processed {i + 1}/{total_movies} posters... (Success: {success_count})")

    logger.info("=" * 50)
    logger.info(f"Poster download complete! Successfully downloaded {success_count}/{total_movies} posters.")
    logger.info(f"Posters saved to: {POSTERS_DIR}")

if __name__ == "__main__":
    main()
