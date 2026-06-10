"""One-time migration utility: converts faiss.index → turbovec.tq.

Usage:
    python scripts/migrate_faiss_to_turbovec.py \\
        --faiss-path models/faiss.index \\
        --output-path models/turbovec.tq

The script loads the source FAISS index, extracts raw float32 vectors via
``reconstruct_n``, builds a ``TurboQuantIndex`` with ``bit_width=4``, verifies
that the row counts match, writes the ``.tq`` file, and prints a summary of
source/output file sizes and row count.

Exit codes:
    0  — migration completed successfully
    1  — row count mismatch or any other error
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


def migrate(faiss_path: Path, output_path: Path) -> dict:
    """Load a FAISS index, rebuild as a TurboVec index, and write to disk.

    Parameters
    ----------
    faiss_path:
        Path to the source ``faiss.index`` file.
    output_path:
        Destination path for the output ``turbovec.tq`` file.

    Returns
    -------
    dict
        Summary with keys: ``source_path``, ``source_size_bytes``,
        ``output_path``, ``output_size_bytes``, ``row_count``,
        ``dimensions``.

    Raises
    ------
    ValueError
        If the row count of the TurboVec index does not match the source
        FAISS index after adding all vectors.
    FileNotFoundError
        If ``faiss_path`` does not exist on disk.
    """
    import faiss
    from turbovec import TurboQuantIndex

    faiss_path = Path(faiss_path)
    output_path = Path(output_path)

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    # --- Load source FAISS index ---
    logger.info("Loading FAISS index from %s", faiss_path)
    faiss_index = faiss.read_index(str(faiss_path))
    n: int = faiss_index.ntotal
    dim: int = faiss_index.d
    logger.info("FAISS index: %d vectors, %d dimensions", n, dim)

    # --- Extract raw float32 vectors ---
    logger.info("Extracting raw vectors via reconstruct_n ...")
    vectors = np.zeros((n, dim), dtype=np.float32)
    faiss_index.reconstruct_n(0, n, vectors)

    # --- Build TurboVec index ---
    logger.info("Building TurboQuantIndex (bit_width=4) ...")
    tq_index = TurboQuantIndex(dim, bit_width=4)
    if n > 0:
        tq_index.add(vectors)

    # --- Verify row count ---
    if len(tq_index) != n:
        raise ValueError(f"Row count mismatch after migration: faiss.ntotal={n}, turbovec.len={len(tq_index)}")

    # --- Persist TurboVec index ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing TurboVec index to %s ...", output_path)
    tq_index.write(str(output_path))
    logger.info("TurboVec index written successfully.")

    return {
        "source_path": str(faiss_path),
        "source_size_bytes": int(faiss_path.stat().st_size),
        "output_path": str(output_path),
        "output_size_bytes": int(output_path.stat().st_size),
        "row_count": n,
        "dimensions": dim,
    }


def _print_summary(summary: dict) -> None:
    """Print a human-readable migration summary to stdout."""
    source_mb = summary["source_size_bytes"] / (1024 * 1024)
    output_mb = summary["output_size_bytes"] / (1024 * 1024)
    print()
    print("Migration summary")
    print("-" * 48)
    print(f"  Source file : {summary['source_path']}")
    print(f"  Source size : {summary['source_size_bytes']:,} bytes ({source_mb:.2f} MB)")
    print(f"  Output file : {summary['output_path']}")
    print(f"  Output size : {summary['output_size_bytes']:,} bytes ({output_mb:.2f} MB)")
    print(f"  Row count   : {summary['row_count']:,}")
    print(f"  Dimensions  : {summary['dimensions']}")
    print("-" * 48)
    print("Migration completed successfully.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faiss-path",
        type=Path,
        default=Path("models/faiss.index"),
        help="Path to the source FAISS index file (default: models/faiss.index)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("models/turbovec.tq"),
        help="Destination path for the TurboVec index (default: models/turbovec.tq)",
    )
    args = parser.parse_args()

    try:
        summary = migrate(faiss_path=args.faiss_path, output_path=args.output_path)
        _print_summary(summary)
    except ValueError as exc:
        logger.error("Row count mismatch — migration aborted: %s", exc)
        sys.exit(1)
    except FileNotFoundError as exc:
        logger.error("Source file not found: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("Migration failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
