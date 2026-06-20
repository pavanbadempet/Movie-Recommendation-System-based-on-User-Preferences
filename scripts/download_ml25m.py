import os
import zipfile
import urllib.request
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
ML25_DIR = DATA_DIR / "ml-25m"
ZIP_URL = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_PATH = DATA_DIR / "ml-25m.zip"

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ML25_DIR.mkdir(parents=True, exist_ok=True)

    ratings_csv = ML25_DIR / "ratings.csv"
    if ratings_csv.exists():
        logger.info("ratings.csv already exists at %s. Skipping download.", ratings_csv)
        return

    logger.info("Downloading MovieLens 25M dataset from %s...", ZIP_URL)
    try:
        # Custom reporthook to print progress
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if count % 1000 == 0 or percent == 100:
                print(f"  Downloaded: {percent}% ({count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="\r")

        urllib.request.urlretrieve(ZIP_URL, ZIP_PATH, reporthook=progress_hook)
        print()  # Newline after progress
        logger.info("Download completed successfully!")
    except Exception as e:
        logger.error("Failed to download MovieLens 25M dataset: %s", e)
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        return

    logger.info("Extracting ratings.csv from zip file...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            # MovieLens zip contains files inside ml-25m/ directory
            # We want to extract ml-25m/ratings.csv
            target_file = "ml-25m/ratings.csv"
            
            # Extract to DATA_DIR so it goes to DATA_DIR / ml-25m / ratings.csv
            zip_ref.extract(target_file, DATA_DIR)
            logger.info("ratings.csv extracted successfully ✓")
    except Exception as e:
        logger.error("Failed to extract zip file: %s", e)
    finally:
        if ZIP_PATH.exists():
            logger.info("Cleaning up temporary zip file...")
            ZIP_PATH.unlink()
            logger.info("Cleanup complete.")

if __name__ == "__main__":
    download_and_extract()
