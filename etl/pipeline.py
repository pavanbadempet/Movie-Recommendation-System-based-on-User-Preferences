"""
Pipeline orchestrator for the Movie Recommendation ETL.
Runs the complete pipeline: ingest → transform → index.
"""
import logging
import time
from datetime import datetime
from pathlib import Path

from etl.config import paths
from etl import ingest, transform, index

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(paths.logs / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"),
    ],
)
logger = logging.getLogger(__name__)


class PipelineStage:
    """Context manager for timing pipeline stages."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        logger.info(f"{'=' * 50}")
        logger.info(f"Starting stage: {self.name}")
        logger.info(f"{'=' * 50}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"Completed {self.name} in {elapsed:.2f}s")
        else:
            logger.error(f"Failed {self.name} after {elapsed:.2f}s: {exc_val}")
        return False


def run_pipeline(raw_data_path: Path | None = None, skip_ingest: bool = False) -> dict:
    """
    Execute the complete ETL pipeline.
    
    Args:
        raw_data_path: Optional path to raw CSV file
        skip_ingest: Skip ingestion (use existing Parquet)
        
    Returns:
        Dictionary with pipeline metrics
    """
    start_time = time.time()
    metrics = {"stages": {}}
    
    logger.info("=" * 60)
    logger.info("MOVIE RECOMMENDATION SYSTEM - ETL PIPELINE")
    logger.info(f"Started at: {datetime.now():%Y-%m-%d %H:%M:%S}")
    logger.info("=" * 60)
    
    try:
        # Stage 1: Ingest
        if not skip_ingest:
            with PipelineStage("INGEST") as stage:
                df = ingest.ingest(raw_data_path)
                metrics["stages"]["ingest"] = {
                    "rows": len(df),
                    "elapsed_s": time.time() - stage.start_time,
                }
        else:
            logger.info("Skipping ingest stage, loading from Parquet...")
            df = None
        
        # Stage 2: Transform
        with PipelineStage("TRANSFORM") as stage:
            df, vectors = transform.transform(df)
            metrics["stages"]["transform"] = {
                "rows": len(df),
                "vector_shape": vectors.shape,
                "elapsed_s": time.time() - stage.start_time,
            }
        
        # Stage 3: Index
        with PipelineStage("INDEX") as stage:
            faiss_index = index.build_index(vectors)
            metrics["stages"]["index"] = {
                "index_size": faiss_index.ntotal,
                "elapsed_s": time.time() - stage.start_time,
            }
        
        # Summary
        total_time = time.time() - start_time
        metrics["total_elapsed_s"] = total_time
        metrics["success"] = True
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Final movie count: {faiss_index.ntotal:,}")
        logger.info("=" * 60)
        
    except Exception as e:
        total_time = time.time() - start_time
        metrics["total_elapsed_s"] = total_time
        metrics["success"] = False
        metrics["error"] = str(e)
        
        logger.exception(f"Pipeline failed after {total_time:.2f}s")
        raise
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Movie Recommendation ETL pipeline")
    parser.add_argument("--data", type=Path, help="Path to raw CSV file")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion stage")
    
    args = parser.parse_args()
    
    metrics = run_pipeline(raw_data_path=args.data, skip_ingest=args.skip_ingest)
    print(f"\nPipeline metrics: {metrics}")
