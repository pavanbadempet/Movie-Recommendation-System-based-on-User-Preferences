import asyncio
import logging
from pathlib import Path
import re
import sys

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = Path("data/video_cache")
MAX_CACHE_FILES = 15
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

# Dict of active download locks to avoid concurrent duplicate downloads
_download_locks: dict[str, asyncio.Lock] = {}
_lock = asyncio.Lock()


def init_cache_dir() -> None:
    """Ensure the cache directory exists."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def validate_youtube_id(youtube_id: str) -> str:
    """Return a normalized YouTube id or raise ValueError for unsafe input."""
    youtube_id = str(youtube_id or "").strip()
    if not YOUTUBE_ID_RE.fullmatch(youtube_id):
        raise ValueError("Invalid YouTube video id")
    return youtube_id


def cache_path_for_video(youtube_id: str) -> Path:
    """Return a cache path guaranteed to remain under CACHE_DIR."""
    safe_id = validate_youtube_id(youtube_id)
    cache_root = CACHE_DIR.resolve()
    target_path = (cache_root / f"{safe_id}.mp4").resolve()
    if target_path.parent != cache_root:
        raise ValueError("Invalid YouTube video id")
    return target_path


async def get_or_download_video(youtube_id: str) -> Path | None:
    """
    Get the path to the downloaded MP4 trailer file.
    If the file is not downloaded yet, download it on demand in low resolution.
    Returns the Path object or None if download failed.
    """
    init_cache_dir()
    youtube_id = validate_youtube_id(youtube_id)
    target_path = cache_path_for_video(youtube_id)

    # 1. Check if already cached
    if target_path.exists() and target_path.stat().st_size > 0:
        try:
            # Touch the file to update its modification time (used for LRU eviction)
            target_path.touch()
        except Exception:
            pass
        return target_path

    # 2. Get/create a lock for this specific youtube_id to prevent concurrent duplicate downloads
    async with _lock:
        if youtube_id not in _download_locks:
            _download_locks[youtube_id] = asyncio.Lock()
        video_lock = _download_locks[youtube_id]

    async with video_lock:
        # Check again in case another task completed the download while we waited
        if target_path.exists() and target_path.stat().st_size > 0:
            return target_path

        logger.info("Downloading YouTube trailer %s to cache...", youtube_id)
        # Format options:
        # -f best[height<=720][ext=mp4]/best[ext=mp4]/best - limits height to 720p (HD) if available for better visual quality, and ensures it's MP4 for browser compatibility.
        cmd = [
            sys.executable,
            "-m",
            "yt_dlp",
            "-f",
            "best[height<=720][ext=mp4]/best[ext=mp4]/best",
            "--no-playlist",
            "--no-warnings",
            "--quiet",
            "-o",
            str(target_path),
            f"https://www.youtube.com/watch?v={youtube_id}",
        ]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                logger.error("yt-dlp download failed for %s: %s", youtube_id, stderr.decode())
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                return None

            if not target_path.exists() or target_path.stat().st_size == 0:
                logger.error("yt-dlp completed but %s does not exist or is empty", target_path)
                if target_path.exists():
                    try:
                        target_path.unlink()
                    except Exception:
                        pass
                return None

            logger.info("Successfully downloaded YouTube trailer %s to cache", youtube_id)

            # Clean up the cache to remain within limits
            cleanup_cache()
            return target_path

        except Exception as e:
            logger.error("Error during download of video %s: %s", youtube_id, e)
            if target_path.exists():
                try:
                    target_path.unlink()
                except Exception:
                    pass
            return None
        finally:
            # Clean up the lock from our dictionary to release memory
            async with _lock:
                _download_locks.pop(youtube_id, None)


def cleanup_cache() -> None:
    """
    LRU Eviction. Sorts MP4 files by mtime (modification time) and deletes
    the oldest files until count <= MAX_CACHE_FILES.
    """
    try:
        files = list(CACHE_DIR.glob("*.mp4"))
        if len(files) <= MAX_CACHE_FILES:
            return

        # Sort by modification time (ascending, oldest first)
        files.sort(key=lambda x: x.stat().st_mtime)

        num_to_delete = len(files) - MAX_CACHE_FILES
        for i in range(num_to_delete):
            f = files[i]
            try:
                f.unlink()
                logger.info("Evicted video from cache: %s", f.name)
            except Exception as e:
                # E.g. PermissionError on Windows if currently open
                logger.warning("Failed to evict cached video %s: %s", f, e)
    except Exception as e:
        logger.error("Error cleaning up video cache: %s", e)
