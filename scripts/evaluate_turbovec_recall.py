"""Recall@K evaluation for the TurboVec production index.

Loads the production TurboVec index and a float32 embedding corpus, samples
``n_queries`` query vectors with a fixed seed (42), and measures Recall@K
versus a brute-force numpy inner-product baseline for each K in ``k_values``.

Usage:
    python scripts/evaluate_turbovec_recall.py \\
        --turbovec-path models/turbovec.tq \\
        --embeddings-path models/sbert_embeddings.npy

    # Override defaults
    python scripts/evaluate_turbovec_recall.py \\
        --turbovec-path models/turbovec.tq \\
        --embeddings-path models/sbert_embeddings.npy \\
        --n-queries 500 \\
        --k-values 10 50 100

Exit codes:
    0  — all quality gates passed (Recall@10 ≥ 0.90)
    1  — Recall@10 < 0.90, or any runtime error
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Quality gate threshold for Recall@10 — exit code 1 if not met.
RECALL_AT_10_THRESHOLD = 0.90


def evaluate_recall(
    turbovec_path: Path | str,
    embeddings_path: Path | str,
    n_queries: int = 1000,
    k_values: tuple[int, ...] = (10, 50),
) -> dict[int, float]:
    """Measure Recall@K for the TurboVec index against a brute-force baseline.

    Samples ``n_queries`` vectors from the corpus using a fixed RNG seed of 42
    so results are reproducible across runs.  For each K the function counts
    how many of the TurboVec top-K results overlap with the brute-force top-K
    results (inner-product order) and averages the overlap fraction over all
    queries.

    Parameters
    ----------
    turbovec_path:
        Path to the serialised ``TurboQuantIndex`` (``.tq``) file.
    embeddings_path:
        Path to a ``.npy`` file containing the float32 embedding matrix,
        shape ``(N, dim)``.
    n_queries:
        Number of query vectors to sample from the corpus (default: 1000).
    k_values:
        Tuple of K values for which Recall@K is computed (default: (10, 50)).

    Returns
    -------
    dict[int, float]
        Mapping from each K value to its Recall@K score in ``[0.0, 1.0]``.

    Raises
    ------
    FileNotFoundError
        If ``turbovec_path`` or ``embeddings_path`` does not exist.
    ValueError
        If ``n_queries`` exceeds the number of available corpus vectors.
    """
    from turbovec import TurboQuantIndex

    turbovec_path = Path(turbovec_path)
    embeddings_path = Path(embeddings_path)

    if not turbovec_path.exists():
        raise FileNotFoundError(f"TurboVec index not found: {turbovec_path}")
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    # --- Load the full embedding corpus (memory-mapped to avoid RAM pressure) ---
    logger.info("Loading embeddings from %s ...", embeddings_path)
    vectors: np.ndarray = np.load(embeddings_path, mmap_mode="r").astype(np.float32)
    n_corpus, dim = vectors.shape
    logger.info("Corpus: %s vectors, %d dimensions", f"{n_corpus:,}", dim)

    if n_queries > n_corpus:
        raise ValueError(
            f"n_queries ({n_queries}) exceeds corpus size ({n_corpus}). "
            "Reduce --n-queries or use a larger embedding corpus."
        )

    # --- Sample query indices with a fixed seed for reproducibility ---
    rng = np.random.default_rng(42)
    query_indices = rng.choice(n_corpus, size=n_queries, replace=False)
    queries: np.ndarray = np.array(vectors[query_indices], dtype=np.float32)
    logger.info("Sampled %d query vectors (seed=42)", n_queries)

    # --- Load the TurboVec production index ---
    logger.info("Loading TurboVec index from %s ...", turbovec_path)
    index = TurboQuantIndex.load(str(turbovec_path))
    logger.info("TurboVec index loaded: %s vectors", f"{len(index):,}")

    # --- Compute Recall@K for each requested K ---
    recall: dict[int, float] = {}
    for k in k_values:
        logger.info("Computing Recall@%d ...", k)

        # TurboVec ANN results — shape (n_queries, k)
        _, tq_indices = index.search(queries, k)

        # Brute-force inner-product baseline — shape (n_queries, N)
        # Computed in batches implicitly via numpy matmul to keep memory bounded.
        scores_bf: np.ndarray = queries @ np.array(vectors, dtype=np.float32).T
        # Sort descending; take top-k column indices — shape (n_queries, k)
        bf_indices: np.ndarray = np.argsort(scores_bf, axis=1)[:, ::-1][:, :k]

        # Recall = fraction of BF top-K results that appear in TurboVec top-K,
        # averaged across all queries.
        hits: float = sum(len(set(tq_indices[i].tolist()) & set(bf_indices[i].tolist())) / k for i in range(n_queries))
        recall[k] = hits / n_queries
        logger.info("Recall@%d = %.4f", k, recall[k])

    return recall


def _print_summary(recall: dict[int, float]) -> None:
    """Print a formatted recall summary table to stdout."""
    print()
    print("TurboVec Recall Evaluation Summary")
    print("-" * 40)
    for k, score in sorted(recall.items()):
        gate = ""
        if k == 10:
            gate = "  ✓ PASS" if score >= RECALL_AT_10_THRESHOLD else "  ✗ FAIL (threshold: 0.90)"
        print(f"  Recall@{k:<4d}: {score:.4f}{gate}")
    print("-" * 40)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--turbovec-path",
        type=Path,
        default=Path("models/turbovec.tq"),
        help="Path to the TurboVec index file (default: models/turbovec.tq)",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("models/sbert_embeddings.npy"),
        help="Path to the float32 embeddings .npy file (default: models/sbert_embeddings.npy)",
    )
    parser.add_argument(
        "--n-queries",
        type=int,
        default=1000,
        help="Number of query vectors to sample from the corpus (default: 1000)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[10, 50],
        metavar="K",
        help="K values for Recall@K evaluation (default: 10 50)",
    )
    args = parser.parse_args()

    k_values: tuple[int, ...] = tuple(sorted(set(args.k_values)))

    try:
        recall = evaluate_recall(
            turbovec_path=args.turbovec_path,
            embeddings_path=args.embeddings_path,
            n_queries=args.n_queries,
            k_values=k_values,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Recall evaluation failed: %s", exc)
        sys.exit(1)

    _print_summary(recall)

    # --- Quality gate: exit 1 if Recall@10 < 0.90 ---
    recall_at_10 = recall.get(10)
    if recall_at_10 is None:
        logger.warning("K=10 was not included in k_values; skipping Recall@10 quality gate.")
    elif recall_at_10 < RECALL_AT_10_THRESHOLD:
        logger.error(
            "Quality gate FAILED: Recall@10 = %.4f < %.2f threshold. The TurboVec index may need to be rebuilt.",
            recall_at_10,
            RECALL_AT_10_THRESHOLD,
        )
        sys.exit(1)
    else:
        logger.info(
            "Quality gate PASSED: Recall@10 = %.4f ≥ %.2f threshold.",
            recall_at_10,
            RECALL_AT_10_THRESHOLD,
        )


if __name__ == "__main__":
    main()
