"""Run the semantic recommendation benchmark against a serving catalog.

This script is intentionally low-memory. It can evaluate the sparse/metadata
fallback path from `movies_transformed.parquet` without loading FAISS or SBERT.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from backend.recommender import Recommender
from backend.semantic_benchmark import DEFAULT_BENCHMARK_PATH, evaluate_semantic_benchmark


def _download_movies_from_hf(repo_id: str, repo_type: str, token: str | None, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename="movies_transformed.parquet",
            repo_type=repo_type,
            token=token,
            cache_dir=cache_dir,
        )
    )


def build_offline_recommender(movies_path: Path) -> Recommender:
    """Create a recommender instance using only catalog metadata."""
    rec = Recommender()
    rec._movies = pd.read_parquet(movies_path)
    rec._vectors = None
    rec._index = None
    rec._low_memory = True
    rec._artifact_status = {
        "vector_artifacts_ready": False,
        "disabled_reason": "offline semantic benchmark uses metadata/sparse fallback",
    }
    rec._optimize_movie_frame()
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--movies-path", type=Path)
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--output", type=Path, default=Path("reports/semantic_benchmark_report.json"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--download-movies-from-hf", action="store_true")
    parser.add_argument("--hf-repo", default="pavanbadempet/movie-recs-models")
    parser.add_argument("--hf-repo-type", default="model")
    parser.add_argument("--hf-token")
    parser.add_argument("--max-bad-match-rate", type=float, default=0.25)
    parser.add_argument("--min-good-recall", type=float, default=0.0)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp)
        movies_path = args.movies_path
        if args.download_movies_from_hf:
            movies_path = _download_movies_from_hf(
                repo_id=args.hf_repo,
                repo_type=args.hf_repo_type,
                token=args.hf_token,
                cache_dir=cache_dir,
            )
        if movies_path is None:
            raise SystemExit("--movies-path is required unless --download-movies-from-hf is set")

        rec = build_offline_recommender(Path(movies_path))
        report = evaluate_semantic_benchmark(rec, benchmark_path=args.benchmark_path, k=args.k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    metrics = report.get("metrics") or {}
    print(json.dumps({
        "status": report.get("status"),
        "evaluated_case_count": report.get("evaluated_case_count"),
        "good_recall_at_k": metrics.get("good_recall_at_k"),
        "bad_match_rate_at_k": metrics.get("bad_match_rate_at_k"),
        "output": str(args.output),
    }, indent=2, sort_keys=True))

    if args.fail_on_threshold:
        bad_rate = float(metrics.get("bad_match_rate_at_k") or 0.0)
        good_recall = float(metrics.get("good_recall_at_k") or 0.0)
        if bad_rate > args.max_bad_match_rate:
            raise SystemExit(f"bad_match_rate_at_k {bad_rate} exceeds {args.max_bad_match_rate}")
        if good_recall < args.min_good_recall:
            raise SystemExit(f"good_recall_at_k {good_recall} below {args.min_good_recall}")


if __name__ == "__main__":
    main()
