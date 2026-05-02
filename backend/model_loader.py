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
    "sbert_embeddings.npy": {
        "url": os.getenv(
            "EMBEDDINGS_URL",
            "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/sbert_embeddings.npy"
        ),
        "dest": "sbert_embeddings.npy"
    },
    "faiss.index": {
        "url": os.getenv(
            "FAISS_INDEX_URL",
            "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/faiss.index"
        ),
        "dest": "faiss.index"
    },
    "movies_transformed.parquet": {
        "url": os.getenv(
            "MOVIES_DATA_URL",
            "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/movies_transformed.parquet"
        ),
        "dest": "../data/processed/movies_transformed.parquet"
    },
    "nova_ranker.joblib": {
        "url": os.getenv(
            "NOVA_RANKER_URL",
            "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/nova_ranker.joblib"
        ),
        "dest": "nova_ranker.joblib",
        "required": False,
    },
    "nova_ranker.joblib.metadata.json": {
        "url": os.getenv(
            "NOVA_RANKER_METADATA_URL",
            "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/nova_ranker.joblib.metadata.json"
        ),
        "dest": "nova_ranker.joblib.metadata.json",
        "required": False,
    }
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192, required: bool = True) -> bool:
    """
    Download a file from URL to destination path.
    Shows progress for large files.
    """
    if not url:
        return False
    
    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(f"{dest_path.name}.tmp")
    
    try:
        logger.info(f"Downloading {dest_path.name} from {url[:50]}...")
        
        # Use urllib for simple, reliable downloads
        with urllib.request.urlopen(url, timeout=300) as response:
            total_size = int(response.headers.get('content-length', 0))
            
            with open(temp_path, 'wb') as f:
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
        
        shutil.move(str(temp_path), str(dest_path))
        logger.info(f"Downloaded {dest_path.name} ({dest_path.stat().st_size // (1024*1024)}MB)")
        return True
        
    except Exception as e:
        log_fn = logger.error if required else logger.info
        log_fn(f"Failed to download {'required' if required else 'optional'} artifact {url}: {e}")
        if temp_path.exists():
            temp_path.unlink()
        return False


def ensure_model_files(models_dir: Path, selected_files: set[str] | list[str] | tuple[str, ...] | None = None) -> dict[str, bool]:
    """
    Ensure all required model files exist, downloading if necessary.
    
    Returns:
        Dict mapping filename to success status
    """
    results = {}
    force_refresh = os.getenv("FORCE_MODEL_REFRESH", "").lower() in {"1", "true", "yes"}
    selected = set(selected_files) if selected_files is not None else None
    
    for filename, config in MODEL_FILES.items():
        if selected is not None and filename not in selected:
            logger.info("Skipping %s; not required for this serving profile", filename)
            results[filename] = True
            continue

        # Handle flexible destination paths
        if isinstance(config, dict):
            url = config.get("url")
            dest_rel = config.get("dest", filename)
            required = bool(config.get("required", True))
            # Resolve relative paths against models_dir
            file_path = (models_dir / dest_rel).resolve()
        else:
            # Legacy support (just string URL)
            url = config
            required = True
            file_path = models_dir / filename
            
        # Skip if file already exists and is valid
        if file_path.exists() and file_path.stat().st_size > 1000 and not force_refresh:
            logger.info(f"{filename} already exists ({file_path.stat().st_size // (1024*1024)}MB)")
            results[filename] = True
            continue

        if file_path.exists() and file_path.stat().st_size > 1000 and force_refresh:
            # Check if remote file has changed (size-based cache invalidation)
            if url:
                try:
                    req = urllib.request.Request(url, method='HEAD')
                    with urllib.request.urlopen(req, timeout=10) as resp:
                        remote_size = int(resp.headers.get('content-length', 0))
                        local_size = file_path.stat().st_size
                        if remote_size > 0 and abs(remote_size - local_size) > 1024:
                            logger.info(f"⟳ {filename} changed remotely ({local_size//1024}KB → {remote_size//1024}KB), re-downloading...")
                        else:
                            logger.info(f"✓ {filename} is up-to-date ({local_size // (1024*1024)}MB)")
                            results[filename] = True
                            continue
                except Exception:
                    # If HEAD request fails, use cached file
                    logger.info(f"✓ {filename} already exists ({file_path.stat().st_size // (1024*1024)}MB)")
                    results[filename] = True
                    continue
            else:
                logger.info(f"✓ {filename} already exists ({file_path.stat().st_size // (1024*1024)}MB)")
                results[filename] = True
                continue
        
        # Try to download if URL is configured
        if url:
            results[filename] = download_file(url, file_path, required=required)
        else:
            # No URL configured, check if file exists locally
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


def default_artifacts_for_serving_profile() -> set[str]:
    """Return artifact names needed before the app starts in the current environment."""
    profile = os.getenv("NOVA_SERVING_PROFILE", "auto").strip().lower()
    low_memory = os.getenv("NOVA_LOW_MEMORY", "").strip().lower() in {"1", "true", "yes", "on"}
    render_like = any(
        os.getenv(name)
        for name in (
            "RENDER",
            "RENDER_SERVICE_ID",
            "RENDER_SERVICE_NAME",
            "RENDER_EXTERNAL_URL",
            "RENDER_EXTERNAL_HOSTNAME",
        )
    )

    if profile in {"lite", "light", "low-memory", "metadata"} or low_memory or (profile == "auto" and render_like):
        return {"movies_transformed.parquet", "nova_ranker.joblib", "nova_ranker.joblib.metadata.json"}

    return set(MODEL_FILES)


# Run on module import only for hosts that explicitly request eager artifact checks.
if os.getenv("NOVA_EAGER_MODEL_DOWNLOAD", "").strip().lower() in {"1", "true", "yes", "on"}:
    MODELS_DIR = Path(__file__).parent.parent / "models"
    ensure_model_files(MODELS_DIR, selected_files=default_artifacts_for_serving_profile())
