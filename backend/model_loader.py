"""
Model file manager for downloading large model files from external storage.
Handles downloading embeddings from Hugging Face Hub to avoid Git LFS issues.
"""
import logging
import os
from pathlib import Path
import urllib.request
import shutil

logger = logging.getLogger(__name__)

# Model hosting configuration
# You can use any of these hosting options:
# 1. Hugging Face Hub: https://huggingface.co/{username}/{repo}/resolve/main/{filename}
# 2. GitHub Releases: https://github.com/{user}/{repo}/releases/download/{tag}/{filename}
# 3. Google Drive (with direct link): https://drive.google.com/uc?export=download&id={file_id}

MODEL_FILES = {
    "sbert_embeddings.npy": os.getenv(
        "EMBEDDINGS_URL",
        # Hugging Face Hub - Free unlimited storage for model files
        "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/sbert_embeddings.npy"
    ),
    "faiss.index": os.getenv(
        "FAISS_INDEX_URL",
        # Usually smaller, can stay in Git LFS
        ""
    ),
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> bool:
    """
    Download a file from URL to destination path.
    Shows progress for large files.
    """
    if not url:
        return False
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        logger.info(f"Downloading {dest_path.name} from {url[:50]}...")
        
        # Use urllib for simple, reliable downloads
        with urllib.request.urlopen(url, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(dest_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Log progress every 10MB
                    if total_size > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                        pct = (downloaded / total_size) * 100
                        logger.info(f"  Progress: {pct:.1f}% ({downloaded // (1024*1024)}MB / {total_size // (1024*1024)}MB)")
        
        logger.info(f"✓ Downloaded {dest_path.name} ({dest_path.stat().st_size // (1024*1024)}MB)")
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # Clean up partial download
        return False


def ensure_model_files(models_dir: Path) -> dict[str, bool]:
    """
    Ensure all required model files exist, downloading if necessary.
    
    Returns:
        Dict mapping filename to success status
    """
    results = {}
    
    for filename, url in MODEL_FILES.items():
        file_path = models_dir / filename
        
        # Skip if file already exists and is valid
        if file_path.exists() and file_path.stat().st_size > 1000:
            logger.info(f"✓ {filename} already exists ({file_path.stat().st_size // (1024*1024)}MB)")
            results[filename] = True
            continue
        
        # Try to download if URL is configured
        if url:
            results[filename] = download_file(url, file_path)
        else:
            # No URL configured, check if file exists in LFS
            if file_path.exists():
                # Might be an LFS pointer file - check size
                if file_path.stat().st_size < 1000:
                    logger.warning(f"⚠ {filename} appears to be an LFS pointer. Configure {filename.upper().replace('.', '_')}_URL in environment.")
                    results[filename] = False
                else:
                    results[filename] = True
            else:
                logger.warning(f"⚠ {filename} not found and no download URL configured")
                results[filename] = False
    
    return results


# Run on module import if in production
if os.getenv("RENDER") or os.getenv("STREAMLIT_RUNTIME"):
    MODELS_DIR = Path(__file__).parent.parent / "models"
    ensure_model_files(MODELS_DIR)
