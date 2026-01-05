"""
Automated Daily Refresh Pipeline
--------------------------------
Orchestrates the full data update lifecycle:
1. Download latest data from Kaggle.
2. Run Transformation (PySpark or Pandas).
3. Rebuild FAISS Index.
4. Notify completion.

Usage:
    python refresh.py --spark   (Use PySpark - Fastest for 1M+ rows)
    python refresh.py --pandas  (Use Pandas - Most reliable on Windows)
"""
import os
import sys
import shutil
import argparse
import logging
import subprocess
from datetime import datetime
from pathlib import Path

# Setup Logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"refresh_{datetime.now():%Y%m%d_%H%M%S}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE),
    ],
)
logger = logging.getLogger("refresh")

# Config
KAGGLE_DATASET = "alanvourch/tmdb-movies-daily-updates"
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

def run_cmd(cmd: list, cwd=None):
    logger.info(f"EXEC: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        logger.error(f"FAILED: {result.stderr}")
        raise RuntimeError(f"Command failed: {result.stderr}")
    logger.info(f"OUTPUT: {result.stdout[-500:]}") # Log tail

def download_data():
    logger.info("--- Step 1: Downloading from Kaggle ---")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check credentials
    if not (Path.home() / ".kaggle/kaggle.json").exists() and not os.environ.get("KAGGLE_KEY"):
        logger.warning("No Kaggle credentials found. Skipping download (using existing data).")
        return

    # Try to find 'kaggle' executable
    kaggle_exe = shutil.which("kaggle")
    if not kaggle_exe:
        # Fallback to checking typical script path
        script_path = Path(sys.executable).parent / "Scripts" / "kaggle.exe"
        if script_path.exists():
            kaggle_exe = str(script_path)
    
    if not kaggle_exe:
        logger.warning("'kaggle' executable not found. Install it with `pip install kaggle`.")
        return

    cmd = [
        kaggle_exe, "datasets", "download",
        "-d", KAGGLE_DATASET,
        "-p", str(RAW_DIR),
        "--unzip", "--force"
    ]
    run_cmd(cmd)
    logger.info("Download complete.")
    
    # Normalize filename to TMDB_all_movies.csv for consistency
    # (Kaggle dataset might change version names, e.g., v11.csv)
    for file in RAW_DIR.glob("TMDB_movie_dataset_*.csv"):
        target = RAW_DIR / "TMDB_all_movies.csv"
        # If target exists and is different, remove it first
        if target.exists() and target != file:
            target.unlink()
        if target != file:
            file.rename(target)
            logger.info(f"Renamed {file.name} to {target.name}")
            break

def run_pyspark_etl():
    logger.info("--- Step 2: Running PySpark Transformation ---")
    
    # 1. Set Hadoop Env (Essential for Windows)
    env = os.environ.copy()
    if 'HADOOP_HOME' not in env:
        # Try default path we set up earlier
        hadoop_path = r"C:\Users\pavan\hadoops"
        if os.path.exists(hadoop_path):
            env['HADOOP_HOME'] = hadoop_path
            env['PATH'] = f"{env['PATH']};{hadoop_path}\\bin"
    
    # 2. Run Spark Job
    # pyspark_etl.py reads 'data/raw/TMDB_all_movies.csv' by default
    cmd = [sys.executable, "etl/pyspark_etl.py"]
    logger.info(f"EXEC with Hadoop Env: {' '.join(cmd)}")
    
    # We use subprocess directly to pass env
    result = subprocess.run(cmd, env=env, text=True) # Stream output directly
    if result.returncode != 0:
        raise RuntimeError("PySpark ETL Failed.")
        
    # 3. Finalize Output (Convert Partitions to NPY)
    logger.info("Finalizing Spark Output...")
    run_cmd([sys.executable, "finalize_spark.py"])

def run_pandas_etl():
    logger.info("--- Step 2: Running Pandas Transformation (Reliable Fallback) ---")
    # Point to the normalized file and DO NOT skip ingest
    data_path = RAW_DIR / "TMDB_all_movies.csv"
    run_cmd([sys.executable, "-m", "etl.pipeline", "--data", str(data_path)])

def rebuild_index():
    logger.info("--- Step 3: Rebuilding FAISS Index ---")
    run_cmd([sys.executable, "-m", "etl.index"])

def main():
    parser = argparse.ArgumentParser(description="Movie Recs Daily Refresh")
    parser.add_argument("--spark", action="store_true", help="Use PySpark for transformation")
    parser.add_argument("--pandas", action="store_true", help="Use Pandas for transformation")
    args = parser.parse_args()

    # Default to Pandas on Windows unless requested otherwise
    use_spark = args.spark

    try:
        download_data()
        
        if use_spark:
            try:
                run_pyspark_etl()
            except Exception as e:
                logger.error(f"PySpark failed: {e}. Falling back to Pandas.")
                run_pandas_etl()
        else:
            run_pandas_etl()
            
        rebuild_index()
        
        logger.info("="*30)
        logger.info("SUCCESS: System Refreshed")
        logger.info("="*30)
        
    except Exception as e:
        logger.error(f"Refresh FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
