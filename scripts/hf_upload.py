import logging
import sys

from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


SECRET_IGNORE_PATTERNS = [
    ".env",
    ".env.*",
    "**/.env",
    "**/.env.*",
    "frontend/.env",
    "frontend/.env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*secret*",
    "*credentials*",
    "**/*secret*",
    "**/*credentials*",
]


def build_ignore_patterns():
    """Return denylist patterns for files that must never be uploaded."""
    return list(SECRET_IGNORE_PATTERNS)


def main():
    repo_id = "pavanbadempet/movie-rec-api"
    repo_type = "space"

    logger.info(f"Starting upload of current directory to Hugging Face {repo_type} '{repo_id}'...")

    ignore_patterns = [
        # Git & env
        ".git",
        ".git/*",
        ".git/**",
        ".venv",
        ".venv/*",
        ".venv/**",
        ".venv_cuda",
        ".venv_cuda/*",
        ".venv_cuda/**",
        "venv",
        "venv/*",
        "venv/**",
        "ENV",
        "ENV/*",
        "ENV/**",
        # IDE & caches
        ".idea",
        ".vscode",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "__pycache__",
        "**/__pycache__",
        "**/__pycache__/*",
        "**/__pycache__/**",
        # Node modules & frontend builds
        "node_modules",
        "**/node_modules",
        "**/node_modules/*",
        "**/node_modules/**",
        "frontend/node_modules",
        "frontend/node_modules/*",
        "frontend/node_modules/**",
        "frontend/dist",
        "frontend/dist/*",
        "frontend/dist/**",
        "frontend/coverage",
        "frontend/coverage/*",
        "frontend/coverage/**",
        # Databases & Logs
        "*.db",
        "*.sqlite3",
        "*.sqlite3-*",
        "*.log",
        "frontend-vite*.log",
        "frontend-vite*.err.log",
        "logs",
        "logs/*",
        "logs/**",
        # Large data and models (which are downloaded at runtime)
        "test",
        "test/*",
        "test/**",
        "test2",
        "test2/*",
        "test2/**",
        "*.csv",
        "*.zip",
        "data",
        "data/*",
        "data/**",
        "models",
        "models/*",
        "models/**",
        "backend/models",
        "backend/models/*",
        "backend/models/**",
        "backend/data",
        "backend/data/*",
        "backend/data/**",
        "output",
        "output/*",
        "output/**",
    ]
    ignore_patterns.extend(build_ignore_patterns())

    api = HfApi()
    try:
        commit_info = api.upload_folder(
            folder_path=".",
            repo_id=repo_id,
            repo_type=repo_type,
            ignore_patterns=ignore_patterns,
            commit_message="perf: multi-layer performance, write-path, and database UUID optimizations",
            commit_description="Pushing local changes verified in tests, bypassing Git LFS/protocol limits.",
        )
        logger.info("Upload completed successfully!")
        logger.info(f"Commit: {commit_info}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
