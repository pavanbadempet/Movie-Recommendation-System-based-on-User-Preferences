import logging
from pathlib import Path
import sys

import numpy as np
from tqdm import tqdm

# Setup path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend.vision_encoder import VisionEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTERS_DIR = PROJECT_ROOT / "data" / "posters"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    logger.info("=" * 50)
    logger.info("Starting Vision Embedding Generation Pipeline")
    logger.info("=" * 50)

    if not POSTERS_DIR.exists():
        logger.error(f"Posters directory not found at {POSTERS_DIR}. Run download_posters.py first.")
        return

    poster_files = list(POSTERS_DIR.glob("*.jpg"))
    total_posters = len(poster_files)

    if total_posters == 0:
        logger.error("No posters found to process.")
        return

    logger.info(f"Found {total_posters} posters to process.")

    # Initialize CLIP Encoder
    encoder = VisionEncoder()

    batch_size = 128 if encoder.device == "cuda" else 16
    logger.info(f"Using batch size: {batch_size} on {encoder.device}")

    movie_ids = []
    all_embeddings = []

    # Process in batches
    for i in tqdm(range(0, total_posters, batch_size), desc="Encoding Posters"):
        batch_paths = poster_files[i : i + batch_size]

        # Extract movie IDs from filenames (e.g., "123.jpg" -> 123)
        batch_ids = [int(p.stem) for p in batch_paths]
        movie_ids.extend(batch_ids)

        # Encode images
        embeddings = encoder.encode_images(batch_paths)
        # Move to CPU numpy
        all_embeddings.append(embeddings.cpu().numpy())

    # Concatenate all batches
    final_embeddings = np.vstack(all_embeddings)
    movie_ids_array = np.array(movie_ids)

    logger.info(f"Successfully generated embeddings shape: {final_embeddings.shape}")

    # Save artifacts
    emb_path = MODELS_DIR / "poster_embeddings.npy"
    ids_path = MODELS_DIR / "poster_movie_ids.npy"

    np.save(emb_path, final_embeddings)
    np.save(ids_path, movie_ids_array)

    logger.info(f"Vision artifacts saved to {MODELS_DIR}:")
    logger.info(f" -> poster_embeddings.npy ({emb_path.stat().st_size / 1024 / 1024:.2f} MB)")
    logger.info(f" -> poster_movie_ids.npy ({ids_path.stat().st_size / 1024:.2f} KB)")

    logger.info("=" * 50)
    logger.info("Phase 13.3 Complete.")


if __name__ == "__main__":
    main()
