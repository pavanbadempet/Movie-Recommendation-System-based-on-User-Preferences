"""
ETL Pipeline entry point for GitHub Actions.
Wraps pandas_etl for backward compatibility with workflow commands.
"""

import argparse
import logging
from pathlib import Path

from etl.pandas_etl import build_index, run_pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="ETL Pipeline")
    parser.add_argument("--data", type=Path, help="Path to raw CSV")
    parser.add_argument("--index-only", action="store_true", help="Run only indexing stage")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion")

    args = parser.parse_args()

    if args.index_only:
        build_index()
    else:
        run_pipeline(raw_data_path=args.data, skip_ingest=args.skip_ingest)
