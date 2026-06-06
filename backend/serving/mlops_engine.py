"""
MLOps Validation, Statistical Data Drift Detection, and Run Lineage Registry.

This module provides the core logic to detect feature distribution shift (KS-test),
semantic embedding drift, log metadata lineage, and govern model promotion flags.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class MLOpsEngine:
    """Governs ML run validation, feature drift analysis, and model promo/rollback lineage."""

    def __init__(self, registry_path: Path) -> None:
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

    def load_registry(self) -> List[Dict[str, Any]]:
        """Load history logs from the registry JSON file."""
        if not self.registry_path.exists():
            return []
        try:
            raw = self.registry_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except Exception as exc:
            logger.warning("Failed to read MLOps lineage registry: %s — resetting.", exc)
        return []

    def save_registry(self, history: List[Dict[str, Any]]) -> None:
        """Write history logs to the registry JSON file."""
        try:
            self.registry_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.error("Failed to save MLOps registry: %s", exc)

    def compute_ks_drift(
        self, baseline_df: pd.DataFrame, new_df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Run Kolmogorov-Smirnov test on numerical features to detect distribution shifts."""
        results = {}
        for col in columns:
            if col not in baseline_df.columns or col not in new_df.columns:
                logger.warning("Column '%s' missing from baseline or new DataFrame. Skipping.", col)
                continue

            base_vals = pd.to_numeric(baseline_df[col], errors="coerce").dropna().to_numpy()
            new_vals = pd.to_numeric(new_df[col], errors="coerce").dropna().to_numpy()

            if len(base_vals) == 0 or len(new_vals) == 0:
                logger.warning("Insufficient numeric values for column '%s' drift check.", col)
                continue

            # Run two-sample Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(base_vals, new_vals)
            results[col] = {"statistic": float(statistic), "p_value": float(p_value)}

        return results

    def compute_embedding_drift(
        self, baseline_embeds: np.ndarray, new_embeds: np.ndarray
    ) -> Dict[str, float]:
        """Compute cosine similarity alignment shift between baseline and new embedding matrices."""
        if baseline_embeds.shape[1] != new_embeds.shape[1]:
            logger.warning(
                "Embedding dimensions mismatch: baseline=%s, new=%s. Skipping semantic drift.",
                baseline_embeds.shape,
                new_embeds.shape,
            )
            return {"mean_alignment_shift": 1.0, "is_drifted": 1.0}

        # Normalize rows to unit vectors
        norm_base = baseline_embeds / np.linalg.norm(baseline_embeds, axis=1, keepdims=True).clip(min=1e-8)
        norm_new = new_embeds / np.linalg.norm(new_embeds, axis=1, keepdims=True).clip(min=1e-8)

        # Average global semantic centroid cosine difference
        base_centroid = norm_base.mean(axis=0)
        new_centroid = norm_new.mean(axis=0)

        base_centroid /= np.linalg.norm(base_centroid).clip(min=1e-8)
        new_centroid /= np.linalg.norm(new_centroid).clip(min=1e-8)

        cos_sim = float(np.dot(base_centroid, new_centroid))
        alignment_shift = 1.0 - cos_sim  # 0.0 means identical distribution centroid, 2.0 opposite

        return {
            "mean_alignment_shift": alignment_shift,
            "centroid_cosine_similarity": cos_sim,
        }

    def compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        try:
            with Path(file_path).open("rb") as f:
                while chunk := f.read(4 * 1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return "unknown_hash"

    def validate_and_register_run(
        self,
        run_id: str,
        new_df: pd.DataFrame,
        new_embeds: Optional[np.ndarray] = None,
        turbovec_path: Optional[Path] = None,
        baseline_df: Optional[pd.DataFrame] = None,
        baseline_embeds: Optional[np.ndarray] = None,
        numerical_columns: Optional[List[str]] = None,
        ks_p_value_threshold: float = 0.05,
        max_embedding_shift: float = 0.15,
    ) -> Dict[str, Any]:
        """Validate current run against a baseline, assess data drift, log metrics, and record run lineage."""
        if numerical_columns is None:
            numerical_columns = ["popularity", "vote_count", "content_quality_score"]

        # Run drift validations if baseline is available
        ks_results = {}
        emb_results = {}
        drift_detected = False
        drift_reasons = []

        if baseline_df is not None:
            ks_results = self.compute_ks_drift(baseline_df, new_df, numerical_columns)
            for col, metrics in ks_results.items():
                # If p-value < threshold, the null hypothesis that samples are from the same distribution is rejected.
                if metrics["p_value"] < ks_p_value_threshold:
                    drift_detected = True
                    drift_reasons.append(
                        f"Numeric feature drift in '{col}': p-value {metrics['p_value']:.4f} < {ks_p_value_threshold}"
                    )

        if baseline_embeds is not None and new_embeds is not None:
            emb_results = self.compute_embedding_drift(baseline_embeds, new_embeds)
            if emb_results["mean_alignment_shift"] > max_embedding_shift:
                drift_detected = True
                drift_reasons.append(
                    f"Semantic embedding drift: centroid shift {emb_results['mean_alignment_shift']:.4f} > {max_embedding_shift}"
                )

        # Retrieve model hashes
        turbovec_hash = "none"
        if turbovec_path is not None and Path(turbovec_path).exists():
            turbovec_hash = self.compute_checksum(turbovec_path)

        # Decide promotion state
        promo_status = "promoted"
        if drift_detected:
            promo_status = "needs_review"
            logger.warning(
                "MLOps Engine: Run '%s' flagged with statistical drift: %s. Setting status to 'needs_review'.",
                run_id,
                "; ".join(drift_reasons),
            )
        else:
            logger.info("MLOps Engine: Run '%s' passed all statistical drift gates. Status set to 'promoted'.", run_id)

        # Prepare lineage record
        run_record = {
            "run_id": run_id,
            "timestamp": time.time(),
            "metrics": {
                "movie_rows": len(new_df),
                "embedding_rows": len(new_embeds) if new_embeds is not None else 0,
            },
            "hashes": {
                "turbovec_index": turbovec_hash,
            },
            "drift_analysis": {
                "numerical_drift": ks_results,
                "semantic_drift": emb_results,
                "drift_detected": drift_detected,
                "drift_reasons": drift_reasons,
            },
            "promotion_status": promo_status,
        }

        # Log into history registry
        history = self.load_registry()
        # Keep registry bounded to last 100 runs
        history.append(run_record)
        history = history[-100:]
        self.save_registry(history)

        return run_record
