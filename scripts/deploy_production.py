#!/usr/bin/env python3
"""Build and deploy APEX with the repository's Dockerfile and Helm chart."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHART_PATH = Path("k8s/helm/apex")


class DeploymentConfig:
    ENVIRONMENTS = {
        "staging": {
            "release": "apex-staging",
            "namespace": "staging",
            "replicas": 2,
            "cpu_request": "500m",
            "memory_request": "1Gi",
            "cpu_limit": "2000m",
            "memory_limit": "4Gi",
        },
        "production": {
            "release": "apex-production",
            "namespace": "production",
            "replicas": 4,
            "cpu_request": "1000m",
            "memory_request": "2Gi",
            "cpu_limit": "4000m",
            "memory_limit": "8Gi",
        },
    }


def _default_image_tag() -> str:
    configured = os.getenv("APEX_IMAGE_TAG", "").strip()
    if configured:
        return configured
    result = subprocess.run(
        ["git", "rev-parse", "--short=12", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or "local"


class DeploymentManager:
    """Execute a reproducible Docker + Helm deployment."""

    def __init__(
        self,
        environment: str,
        dry_run: bool = False,
        skip_checks: bool = False,
        image_repository: str | None = None,
        image_tag: str | None = None,
        push_image: bool = False,
        backup_dir: Path | str = PROJECT_ROOT / ".deploy-backups",
    ):
        if environment not in DeploymentConfig.ENVIRONMENTS:
            raise ValueError(f"Unknown environment: {environment}")
        self.environment = environment
        self.config = DeploymentConfig.ENVIRONMENTS[environment]
        self.dry_run = dry_run
        self.skip_checks = skip_checks
        self.image_repository = image_repository or os.getenv(
            "APEX_IMAGE_REPOSITORY", "ghcr.io/pavanbadempet/apex-backend"
        )
        self.image_tag = image_tag or _default_image_tag()
        self.push_image = push_image
        self.backup_dir = Path(backup_dir)

    @property
    def image(self) -> str:
        return f"{self.image_repository}:{self.image_tag}"

    def deployment_commands(self) -> list[list[str]]:
        """Return the exact mutation commands in execution order."""
        commands = [["docker", "build", "-f", "Dockerfile", "-t", self.image, "."]]
        if self.push_image:
            commands.append(["docker", "push", self.image])
        commands.append(
            [
                "helm",
                "upgrade",
                "--install",
                self.config["release"],
                str(CHART_PATH).replace("\\", "/"),
                "--namespace",
                self.config["namespace"],
                "--create-namespace",
                "--atomic",
                "--wait",
                "--timeout",
                "10m",
                "--set-string",
                f"image.repository={self.image_repository}",
                "--set-string",
                f"image.tag={self.image_tag}",
                "--set",
                f"replicaCount={self.config['replicas']}",
                "--set-string",
                f"resources.requests.cpu={self.config['cpu_request']}",
                "--set-string",
                f"resources.requests.memory={self.config['memory_request']}",
                "--set-string",
                f"resources.limits.cpu={self.config['cpu_limit']}",
                "--set-string",
                f"resources.limits.memory={self.config['memory_limit']}",
            ]
        )
        return commands

    def _run(self, command: list[str]) -> None:
        logger.info("Running: %s", subprocess.list2cmdline(command))
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    def _capture(self, command: list[str]) -> str:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout

    def pre_deployment_checks(self) -> bool:
        """Validate local files, tools, tests, cluster access, and secret refs."""
        if not (PROJECT_ROOT / "Dockerfile").is_file() or not (PROJECT_ROOT / CHART_PATH / "Chart.yaml").is_file():
            logger.error("Root Dockerfile or Helm chart is missing")
            return False
        for tool in ("docker", "kubectl", "helm"):
            if shutil.which(tool) is None:
                logger.error("Required deployment tool is unavailable: %s", tool)
                return False
        if self.environment == "production":
            dirty = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            ).stdout.strip()
            if dirty:
                logger.error("Production deployment requires a clean Git worktree")
                return False
            if self.image_tag in {"latest", "local"}:
                logger.error("Production deployment requires an immutable image tag")
                return False

        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_api.py", "tests/test_deploy_production.py", "-q"],
            cwd=PROJECT_ROOT,
            check=False,
            timeout=600,
        )
        if test_result.returncode != 0:
            logger.error("Deployment gate tests failed")
            return False
        if subprocess.run(["kubectl", "cluster-info"], cwd=PROJECT_ROOT, check=False).returncode != 0:
            logger.error("Cannot connect to the configured Kubernetes cluster")
            return False
        secret_check = subprocess.run(
            ["kubectl", "get", "secret", "apex-secrets", "-n", self.config["namespace"]],
            cwd=PROJECT_ROOT,
            check=False,
        )
        if secret_check.returncode != 0:
            logger.error("Required Kubernetes Secret apex-secrets is unavailable")
            return False
        return True

    def create_backup_receipt(self) -> dict | None:
        """Persist the current Helm revision and values for auditable rollback."""
        try:
            history = json.loads(
                self._capture(
                    [
                        "helm",
                        "history",
                        self.config["release"],
                        "--namespace",
                        self.config["namespace"],
                        "--output",
                        "json",
                    ]
                )
            )
            deployed = [entry for entry in history if str(entry.get("status", "")).lower() == "deployed"]
            if not deployed:
                return None
            revision = int(deployed[-1]["revision"])
            values = self._capture(
                [
                    "helm",
                    "get",
                    "values",
                    self.config["release"],
                    "--namespace",
                    self.config["namespace"],
                    "--all",
                    "--output",
                    "yaml",
                ]
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError, ValueError):
            logger.info("No existing Helm release found; deployment will be an install")
            return None

        self.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{self.config['release']}-{timestamp}-revision-{revision}.json"
        receipt = {
            "filename": filename,
            "created_at": datetime.now(UTC).isoformat(),
            "release": self.config["release"],
            "namespace": self.config["namespace"],
            "revision": revision,
            "values_yaml": values,
        }
        (self.backup_dir / filename).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        return receipt

    def rollback(self, receipt: dict | None) -> None:
        if receipt is None:
            logger.warning("No prior deployed revision exists for explicit rollback")
            return
        self._run(
            [
                "helm",
                "rollback",
                self.config["release"],
                str(receipt["revision"]),
                "--namespace",
                self.config["namespace"],
                "--wait",
                "--timeout",
                "10m",
            ]
        )

    def smoke_test(self) -> bool:
        """Port-forward the deployed service and require a healthy API response."""
        local_port = 18000
        process = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "--namespace",
                self.config["namespace"],
                f"service/{self.config['release']}-apex",
                f"{local_port}:8000",
            ],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            for _ in range(20):
                if process.poll() is not None:
                    return False
                try:
                    response = requests.get(f"http://127.0.0.1:{local_port}/health", timeout=2)
                    if response.status_code == 200 and response.json().get("status") in {"healthy", "degraded"}:
                        return True
                except requests.RequestException:
                    pass
                time.sleep(1)
            return False
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    def deploy(self) -> bool:
        if not self.skip_checks and not self.pre_deployment_checks():
            return False
        if self.skip_checks:
            logger.warning("Pre-deployment checks explicitly skipped")
        commands = self.deployment_commands()
        if self.dry_run:
            for command in commands:
                logger.info("DRY RUN: %s", subprocess.list2cmdline(command))
            return True

        receipt = self.create_backup_receipt()
        try:
            for command in commands:
                self._run(command)
            if not self.smoke_test():
                raise RuntimeError("Post-deployment smoke test failed")
            return True
        except Exception:
            logger.exception("Deployment failed")
            self.rollback(receipt)
            return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment", choices=sorted(DeploymentConfig.ENVIRONMENTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-checks", action="store_true")
    parser.add_argument("--push", action="store_true", help="Push the built image before Helm deployment")
    parser.add_argument("--image-repository", default=os.getenv("APEX_IMAGE_REPOSITORY"))
    parser.add_argument("--image-tag", default=os.getenv("APEX_IMAGE_TAG"))
    parser.add_argument(
        "--confirm-production",
        action="store_true",
        help="Required for non-dry-run production deployment.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.environment == "production" and not args.dry_run and not args.confirm_production:
        logger.error("Non-dry-run production deployment requires --confirm-production")
        return 2
    manager = DeploymentManager(
        args.environment,
        dry_run=args.dry_run,
        skip_checks=args.skip_checks,
        image_repository=args.image_repository,
        image_tag=args.image_tag,
        push_image=args.push,
    )
    return 0 if manager.deploy() else 1


if __name__ == "__main__":
    raise SystemExit(main())
