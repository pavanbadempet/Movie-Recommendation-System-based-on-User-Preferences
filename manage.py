#!/usr/bin/env python
"""
Unified Management CLI for Movie Recommendation System.
Usage:
    python manage.py setup      # Install dependencies
    python manage.py etl        # Run Data Pipeline
    python manage.py run        # Start Backend + Frontend
    python manage.py test       # Run all tests
    python manage.py clean      # Remove artifacts/cache
    python manage.py docker     # Run with Docker Compose
"""
import argparse
import subprocess
import sys
import time
import os
import platform
from pathlib import Path

# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def log(msg, color=Colors.OKBLUE):
    print(f"{color}{Colors.BOLD}[MANAGER] {msg}{Colors.ENDC}")

def run_cmd(cmd, cwd=None, background=False):
    """Run a shell command."""
    log(f"Running: {cmd}")
    if background:
        return subprocess.Popen(cmd, shell=True, cwd=cwd)
    
    try:
        subprocess.check_call(cmd, shell=True, cwd=cwd)
    except subprocess.CalledProcessError:
        log("Command failed.", Colors.FAIL)
        sys.exit(1)

def check_env():
    """Check for .env file."""
    if not os.path.exists(".env"):
        log("No .env file found. Creating from template...", Colors.WARNING)
        with open(".env", "w") as f:
            f.write("TMDB_API_KEY=your_key_here\n")
            f.write("API_URL=http://localhost:8000\n")
        log("Created .env. Please edit it with your API keys!", Colors.WARNING)

def setup():
    """Install dependencies."""
    log("Installing dependencies...")
    run_cmd(f"{sys.executable} -m pip install --upgrade pip")
    run_cmd(f"{sys.executable} -m pip install -r requirements.txt")
    
    check_env()
    log("Setup complete!", Colors.OKGREEN)

def etl(spark=False):
    """Run ETL Pipeline."""
    log("Running ETL Pipeline...")
    if spark:
        log("Using PySpark (Enterprise)...")
        run_cmd(f"{sys.executable} etl/pyspark_etl.py")
    else:
        log("Using Pandas (Local)...")
        run_cmd(f"{sys.executable} -m etl.pandas_etl")
    log("ETL Complete!", Colors.OKGREEN)

def test():
    """Run Tests."""
    log("Running Tests...")
    run_cmd(f"{sys.executable} -m pytest tests/ -v")
    log("All tests passed!", Colors.OKGREEN)

def run_app():
    """Run Backend and Frontend concurrently."""
    check_env()
    
    log("Starting Backend (Port 8000)...")
    backend = run_cmd(f"{sys.executable} -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload", background=True)
    
    # Wait for backend to start
    time.sleep(3)
    
    log("Starting Frontend (Port 8501)...")
    frontend = run_cmd(f"{sys.executable} -m streamlit run streamlit_app.py", background=True)
    
    try:
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        log("Stopping services...", Colors.WARNING)
        backend.terminate()
        frontend.terminate()

def docker_run():
    """Run with Docker Compose."""
    log("Starting Docker Containers...")
    run_cmd("docker-compose up --build -d")
    log("Services running at http://localhost:8501 (Frontend) and http://localhost:8080 (Airflow)", Colors.OKGREEN)

def clean():
    """Clean artifacts."""
    log("Cleaning up...", Colors.WARNING)
    paths_to_clean = [
        "data/processed/movies_transformed.parquet",
        "models/faiss.index",
        "models/sbert_embeddings.npy",
        "__pycache__",
        ".pytest_cache"
    ]
    if platform.system() == "Windows":
        run_cmd("del /s /q __pycache__")
    else:
        run_cmd("find . -name '__pycache__' -exec rm -rf {} +")
        
    log("Clean complete.")

def main():
    parser = argparse.ArgumentParser(description="Project Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("setup", help="Install dependencies")
    
    etl_parser = subparsers.add_parser("etl", help="Run ETL pipeline")
    etl_parser.add_argument("--spark", action="store_true", help="Use PySpark instead of Pandas")
    
    subparsers.add_parser("run", help="Run App (Backend + Frontend)")
    subparsers.add_parser("test", help="Run Tests")
    subparsers.add_parser("clean", help="Clean artifacts")
    subparsers.add_parser("docker", help="Run with Docker")
    
    args = parser.parse_args()
    
    if args.command == "setup":
        setup()
    elif args.command == "etl":
        etl(spark=args.spark)
    elif args.command == "test":
        test()
    elif args.command == "run":
        run_app()
    elif args.command == "clean":
        clean()
    elif args.command == "docker":
        docker_run()

if __name__ == "__main__":
    main()
