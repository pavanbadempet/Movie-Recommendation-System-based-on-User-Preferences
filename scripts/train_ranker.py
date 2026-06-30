"""
Train APEX's learned recommendation ranker.

Usage:
    python scripts/train_ranker.py
    python scripts/train_ranker.py --events data/events/movie_events.jsonl --output models/nova_ranker.joblib
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.events import iter_events
from backend.pipeline.ranker import default_ranker_path
from backend.pipeline.ranker_training import train_nova_ranker

DATA_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"


def download_url(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as response:
        output_path.write_bytes(response.read())
    return output_path


def download_movies_from_hf(repo_id: str, repo_type: str, output_path: Path, token: str | None = None) -> Path:
    from huggingface_hub import hf_hub_download

    output_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        filename="movies_transformed.parquet",
        token=token,
        local_dir=str(output_path.parent),
    )
    return Path(downloaded_path)


def upload_ranker_to_hf(output_path: Path, repo_id: str, repo_type: str, token: str | None = None) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    files = [output_path, Path(str(output_path) + ".metadata.json")]
    for path in files:
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=repo_id,
                repo_type=repo_type,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train APEX's learned recommendation ranker.")
    parser.add_argument("--movies", default=str(DATA_DIR / "movies_transformed.parquet"))
    parser.add_argument("--events", default=None, help="JSONL behavior event log. Defaults to EVENT_LOG_PATH.")
    parser.add_argument(
        "--events-url", default=None, help="Optional remote JSONL event log URL to download before training."
    )
    parser.add_argument("--output", default=str(default_ranker_path(MODELS_DIR)))
    parser.add_argument(
        "--promotion-gate",
        action="store_true",
        help="Write a candidate artifact and promote only when offline metrics pass.",
    )
    parser.add_argument(
        "--production-output",
        default=str(default_ranker_path(MODELS_DIR)),
        help="Production ranker artifact path used by the promotion gate.",
    )
    parser.add_argument(
        "--download-movies-from-hf",
        action="store_true",
        help="Download movies_transformed.parquet from Hugging Face before training.",
    )
    parser.add_argument(
        "--upload-to-hf",
        action="store_true",
        help="Upload ranker artifact and metadata to Hugging Face after training.",
    )
    parser.add_argument("--hf-repo", default=os.getenv("HF_MODEL_REPO", "pavanbadempet/movie-recs-models"))
    parser.add_argument("--hf-repo-type", default=os.getenv("HF_MODEL_REPO_TYPE", "model"))
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()

    movies_path = Path(args.movies)
    if args.download_movies_from_hf:
        movies_path = download_movies_from_hf(
            repo_id=args.hf_repo,
            repo_type=args.hf_repo_type,
            output_path=movies_path,
            token=args.hf_token,
        )

    if not movies_path.exists():
        raise FileNotFoundError(f"Movie feature file not found: {movies_path}")

    events_path = args.events
    if args.events_url:
        events_path = str(download_url(args.events_url, REPO_ROOT / "data" / "events" / "remote_events.jsonl"))

    movies = pd.read_parquet(movies_path)
    events = list(iter_events(events_path))
    report = train_nova_ranker(
        movies=movies,
        events=events,
        output_path=args.output,
        promotion_gate=args.promotion_gate,
        production_path=args.production_output,
    )
    if args.upload_to_hf:
        upload_path = Path(report.get("promoted_artifact_path") or args.output)
        upload_ranker_to_hf(
            output_path=upload_path,
            repo_id=args.hf_repo,
            repo_type=args.hf_repo_type,
            token=args.hf_token,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
