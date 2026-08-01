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
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


# Colors for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def log(msg, color=Colors.OKBLUE):
    print(f"{color}{Colors.BOLD}[MANAGER] {msg}{Colors.ENDC}", flush=True)


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


def lakehouse(format_type="text", as_of=None, compare_from=None, compare_to=None):
    """Inspect local medallion snapshots and SCD history."""
    cmd = f"{sys.executable} scripts/inspect_lakehouse.py --format {format_type}"
    if as_of:
        cmd += f' --as-of "{as_of}"'
    if compare_from:
        cmd += f' --compare-from "{compare_from}"'
    if compare_to:
        cmd += f' --compare-to "{compare_to}"'
    if format_type == "json":
        subprocess.check_call(cmd, shell=True)
        return

    log("Inspecting lakehouse snapshots...")
    run_cmd(cmd)


def rebuild_serving(
    movies_path="data/processed/movies_transformed.parquet",
    models_dir="models",
    processed_dir="data/processed",
    batch_size=32,
    upload_to_hf=False,
    hf_repo="pavanbadempet/movie-recs-models",
    hf_repo_type="model",
):
    """Rebuild aligned vector serving artifacts from the current catalog."""
    log("Rebuilding serving artifacts...")
    cmd = (
        f"{sys.executable} scripts/rebuild_serving_artifacts.py "
        f'--movies-path "{movies_path}" '
        f'--models-dir "{models_dir}" '
        f'--processed-dir "{processed_dir}" '
        f"--batch-size {batch_size} "
        f'--hf-repo "{hf_repo}" '
        f'--hf-repo-type "{hf_repo_type}"'
    )
    if upload_to_hf:
        cmd += " --upload-to-hf"
    run_cmd(cmd)


def run_app():
    """Run Backend and Frontend concurrently."""
    check_env()

    log("Starting FastAPI Backend on http://localhost:8000 ...")
    backend = run_cmd(
        f"{sys.executable} -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload",
        background=True,
    )

    # Give backend a moment to start
    time.sleep(3)

    # Prefer React frontend if node_modules exist, else fall back to Streamlit
    frontend_dir = Path("frontend")
    node_modules = frontend_dir / "node_modules"

    if node_modules.exists():
        log("Starting React Frontend on http://localhost:5173 ...")
        bun_cmd = "bun.cmd" if platform.system() == "Windows" else "bun"
        frontend = run_cmd(f"{bun_cmd} run dev", cwd=str(frontend_dir), background=True)
        log("App running!", Colors.OKGREEN)
        log("  Backend API : http://localhost:8000", Colors.OKGREEN)
        log("  Frontend UI : http://localhost:5173", Colors.OKGREEN)
        log("  API Docs    : http://localhost:8000/docs", Colors.OKGREEN)
    else:
        log("React node_modules not found. Run 'cd frontend && bun install' first.", Colors.WARNING)
        log("Falling back to Streamlit frontend on http://localhost:8501 ...", Colors.WARNING)
        frontend = run_cmd(
            f"{sys.executable} -m streamlit run frontend/streamlit_app.py",
            background=True,
        )
        log("App running!", Colors.OKGREEN)
        log("  Backend API : http://localhost:8000", Colors.OKGREEN)
        log("  Frontend UI : http://localhost:8501", Colors.OKGREEN)
        log("  API Docs    : http://localhost:8000/docs", Colors.OKGREEN)

    log("Press Ctrl+C to stop all services.", Colors.WARNING)
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

    if platform.system() == "Windows":
        run_cmd("del /s /q __pycache__")
    else:
        run_cmd("find . -name '__pycache__' -exec rm -rf {} +")

    log("Clean complete.")


def deploy():
    """Commit and push changes to trigger deployment."""
    log("Deploying updates to Git...", Colors.WARNING)

    # 1. Add specific artifacts
    log("Adding artifacts...")
    artifacts = ["data/processed/movies_transformed.parquet", "models/sbert_embeddings.npy", "models/turbovec.tq"]
    for art in artifacts:
        if os.path.exists(art):
            run_cmd(f"git add {art}")

    # 2. Add docs/configs if changed
    run_cmd("git add README.md docs/ manage.py etl/")

    # 3. Commit
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        run_cmd(f'git commit -m "chore: update models and data ({current_time})"')

        # 4. Push
        log("Pushing to remote to trigger Render/Streamlit...", Colors.OKBLUE)
        run_cmd("git push")
        log("Deployment triggered! Check Render dashboard.", Colors.OKGREEN)

    except Exception:
        log("Nothing to commit or push failed.", Colors.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Project Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("setup", help="Install dependencies")
    subparsers.add_parser("frontend-install", help="Install React frontend dependencies")

    etl_parser = subparsers.add_parser("etl", help="Run ETL pipeline")
    etl_parser.add_argument("--spark", action="store_true", help="Use PySpark instead of Pandas")

    subparsers.add_parser("run", help="Run App (Backend + Frontend)")
    subparsers.add_parser("test", help="Run Tests")
    subparsers.add_parser("clean", help="Clean artifacts")
    subparsers.add_parser("docker", help="Run with Docker (App + Airflow)")
    subparsers.add_parser("airflow", help="Start Airflow Orchestration (via Docker)")
    subparsers.add_parser("deploy", help="Push changes to Git (Triggers Render/Streamlit)")

    lakehouse_parser = subparsers.add_parser("lakehouse", help="Inspect local medallion snapshots and SCD history")
    lakehouse_parser.add_argument("--format", choices=("text", "json"), default="text")
    lakehouse_parser.add_argument("--as-of", dest="as_of")
    lakehouse_parser.add_argument("--compare-from")
    lakehouse_parser.add_argument("--compare-to")

    rebuild_serving_parser = subparsers.add_parser(
        "rebuild-serving", help="Rebuild aligned serving artifacts from movies_transformed.parquet"
    )
    rebuild_serving_parser.add_argument("--movies-path", default="data/processed/movies_transformed.parquet")
    rebuild_serving_parser.add_argument("--models-dir", default="models")
    rebuild_serving_parser.add_argument("--processed-dir", default="data/processed")
    rebuild_serving_parser.add_argument("--batch-size", type=int, default=32)
    rebuild_serving_parser.add_argument("--upload-to-hf", action="store_true")
    rebuild_serving_parser.add_argument("--hf-repo", default="pavanbadempet/movie-recs-models")
    rebuild_serving_parser.add_argument("--hf-repo-type", default="model")

    args = parser.parse_args()

    if args.command == "setup":
        setup()
    elif args.command == "frontend-install":
        npm_cmd = "npm.cmd" if platform.system() == "Windows" else "npm"
        run_cmd(f"{npm_cmd} install", cwd="frontend")
        log("Frontend dependencies installed.", Colors.OKGREEN)
    elif args.command == "etl":
        etl(spark=args.spark)
    elif args.command == "test":
        test()
    elif args.command == "run":
        run_app()
    elif args.command == "clean":
        clean()
    elif args.command in ["docker", "airflow"]:
        docker_run()
    elif args.command == "deploy":
        deploy()
    elif args.command == "lakehouse":
        lakehouse(
            format_type=args.format,
            as_of=args.as_of,
            compare_from=args.compare_from,
            compare_to=args.compare_to,
        )
    elif args.command == "rebuild-serving":
        rebuild_serving(
            movies_path=args.movies_path,
            models_dir=args.models_dir,
            processed_dir=args.processed_dir,
            batch_size=args.batch_size,
            upload_to_hf=args.upload_to_hf,
            hf_repo=args.hf_repo,
            hf_repo_type=args.hf_repo_type,
        )


if __name__ == "__main__":
    main()
