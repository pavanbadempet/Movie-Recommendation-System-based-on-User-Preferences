"""
Abstract Factory Pattern for Data Lake Storage.

This module provides a vendor-agnostic interface for artifact management,
allowing the system to seamlessly switch between:
1. Zero-Cost / Development: Hugging Face Datasets & Models
2. Enterprise Production: AWS S3 / MinIO

By abstracting `StorageManager`, the PySpark pipeline and FastAPI backend 
require zero code changes when migrating from a free Kaggle environment 
to an AWS EMR / Databricks production environment.
"""

from abc import ABC, abstractmethod
import os
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)

class StorageManager(ABC):
    """Abstract Base Class defining the strict contract for our Data Lake."""
    
    @abstractmethod
    def upload_artifact(self, local_path: str | Path, remote_path: str):
        """Upload a file or directory to the Data Lake."""
        pass
        
    @abstractmethod
    def download_artifact(self, remote_path: str, local_path: str | Path):
        """Download a file or directory from the Data Lake."""
        pass

class HuggingFaceStorage(StorageManager):
    """
    Zero-Cost Storage Layer acting as a Data Lake.
    Leverages Hugging Face Datasets for massive parquet storage.
    """
    def __init__(self, repo_id: str = None, token: str = None):
        try:
            from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        except ImportError:
            raise RuntimeError("huggingface_hub is required for HuggingFaceStorage.")
            
        self.api = HfApi(token=token or os.getenv("HF_TOKEN"))
        self.repo_id = repo_id or os.getenv("HF_DATASET_REPO", "pavanbadempet/nova-recommendation-lake")
        
    def upload_artifact(self, local_path: str | Path, remote_path: str):
        local_path = Path(local_path)
        logger.info(f"[HF Storage] Uploading {local_path} to {self.repo_id}/{remote_path}")
        
        if local_path.is_dir():
            self.api.upload_folder(
                folder_path=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Automated Sync: {remote_path}"
            )
        else:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"Automated Sync: {remote_path}"
            )

    def download_artifact(self, remote_path: str, local_path: str | Path):
        from huggingface_hub import hf_hub_download, snapshot_download
        logger.info(f"[HF Storage] Downloading {remote_path} from {self.repo_id} to {local_path}")
        
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine if remote is a file or folder based on extension heuristic
        if "." in remote_path.split("/")[-1] and not remote_path.endswith("/"):
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                filename=remote_path,
                local_dir=str(local_path.parent)
            )
            # HF Hub downloads sometimes maintain the repo structure, ensure it maps to local_path
            shutil.move(downloaded_path, local_path)
        else:
            snapshot_download(
                repo_id=self.repo_id,
                repo_type="dataset",
                allow_patterns=f"{remote_path}/*",
                local_dir=str(local_path)
            )

class AWSS3Storage(StorageManager):
    """
    Enterprise Storage Layer using AWS S3 / MinIO.
    """
    def __init__(self, bucket_name: str = None):
        try:
            import boto3
        except ImportError:
            raise RuntimeError("boto3 is required for AWSS3Storage.")
            
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL"), # Used for MinIO
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
        )
        self.bucket = bucket_name or os.getenv("AWS_S3_BUCKET", "nova-recommendation-lake")

    def upload_artifact(self, local_path: str | Path, remote_path: str):
        local_path = Path(local_path)
        logger.info(f"[S3 Storage] Uploading {local_path} to s3://{self.bucket}/{remote_path}")
        
        if local_path.is_dir():
            for filepath in local_path.rglob("*"):
                if filepath.is_file():
                    s3_path = os.path.join(remote_path, filepath.relative_to(local_path).as_posix())
                    self.s3.upload_file(str(filepath), self.bucket, s3_path)
        else:
            self.s3.upload_file(str(local_path), self.bucket, remote_path)

    def download_artifact(self, remote_path: str, local_path: str | Path):
        local_path = Path(local_path)
        logger.info(f"[S3 Storage] Downloading s3://{self.bucket}/{remote_path} to {local_path}")
        
        # Simple heuristic for file vs dir
        if "." in remote_path.split("/")[-1] and not remote_path.endswith("/"):
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3.download_file(self.bucket, remote_path, str(local_path))
        else:
            paginator = self.s3.get_paginator('list_objects_v2')
            local_path.mkdir(parents=True, exist_ok=True)
            for page in paginator.paginate(Bucket=self.bucket, Prefix=remote_path):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                    rel_path = os.path.relpath(key, remote_path)
                    target_file = local_path / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    self.s3.download_file(self.bucket, key, str(target_file))

def get_storage_manager() -> StorageManager:
    """Factory method to inject the correct storage provider."""
    provider = os.getenv("STORAGE_PROVIDER", "huggingface").strip().lower()
    if provider == "aws" or provider == "s3" or provider == "minio":
        return AWSS3Storage()
    elif provider == "huggingface" or provider == "hf":
        return HuggingFaceStorage()
    else:
        raise ValueError(f"Unknown STORAGE_PROVIDER: {provider}")
