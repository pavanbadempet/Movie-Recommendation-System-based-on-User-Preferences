"""
Artifact integrity validation for the APEX Movie Recommendation System.

This module provides SHA-256 checksum verification, row-alignment validation,
and manifest-based batch validation for serving artifacts (FAISS index, ONNX
weights, embedding arrays, etc.).

Typical usage
-------------
>>> from pathlib import Path
>>> from backend.artifact_validator import ArtifactValidator, create_manifest
>>>
>>> # Build a manifest from a list of artifact files
>>> manifest = create_manifest(
...     [Path("models/faiss.index"), Path("models/sbert_embeddings.npy")],
...     output_path=Path("models/artifact_manifest.json"),
... )
>>>
>>> # Validate artifacts at startup
>>> validator = ArtifactValidator(Path("models/artifact_manifest.json"))
>>> validator.validate(Path("models/faiss.index"))   # True or raises
>>> results = validator.validate_all()               # {"faiss.index": True, ...}
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Read files in 8 MiB chunks to keep memory usage bounded for large artifacts.
_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB


def _compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file, reading in chunks.

    Parameters
    ----------
    path:
        Path to the file whose checksum is to be computed.

    Returns
    -------
    str
        Lowercase hex-encoded SHA-256 digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(_CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


class ArtifactValidator:
    """Validates serving artifacts using SHA-256 checksums and row alignment.

    Parameters
    ----------
    manifest_path:
        Path to a JSON file that maps artifact names (relative or absolute
        path strings) to their expected SHA-256 hex digests.  If the file
        does not exist the manifest is treated as empty and all artifacts
        are considered *unknown* (they pass validation with a WARNING).

    Attributes
    ----------
    manifest:
        The loaded manifest dictionary ``{artifact_name: expected_sha256}``.
    """

    def __init__(self, manifest_path: Path) -> None:
        self.manifest: dict[str, str] = self._load_manifest(manifest_path)
        logger.debug(
            "ArtifactValidator initialised with %d manifest entries from '%s'",
            len(self.manifest),
            manifest_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self, manifest_path: Path) -> dict[str, str]:
        """Load the JSON manifest from *manifest_path*.

        Parameters
        ----------
        manifest_path:
            Path to the manifest JSON file.

        Returns
        -------
        dict[str, str]
            Mapping of artifact name → expected SHA-256 hex digest.
            Returns an empty dict if the file does not exist (graceful
            degradation — the validator will warn rather than crash).
        """
        path = Path(manifest_path)
        if not path.exists():
            logger.warning(
                "Manifest file '%s' does not exist; all artifacts will be treated as unknown.",
                path,
            )
            return {}

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error(
                "Failed to load manifest from '%s': %s — treating manifest as empty.",
                path,
                exc,
            )
            return {}

        if not isinstance(data, dict):
            logger.error(
                "Manifest at '%s' is not a JSON object (got %s) — treating manifest as empty.",
                path,
                type(data).__name__,
            )
            return {}

        logger.info("Loaded artifact manifest from '%s' (%d entries).", path, len(data))
        return {str(k): str(v) for k, v in data.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, artifact_path: Path) -> bool:
        """Validate a single artifact file against the manifest.

        Steps
        -----
        1. Raise :exc:`FileNotFoundError` if the file does not exist.
        2. Compute the SHA-256 checksum of the file contents (chunked I/O).
        3. Look up the expected checksum in the manifest.

           * If the artifact is **not in the manifest**: log a WARNING and
             return ``True`` (unknown artifacts pass by default).
           * If the checksums **match**: log INFO and return ``True``.
           * If the checksums **do not match**: raise :exc:`ValueError` with
             a message identifying the path and both checksums.

        Parameters
        ----------
        artifact_path:
            Path to the artifact file to validate.

        Returns
        -------
        bool
            ``True`` when validation passes.

        Raises
        ------
        FileNotFoundError
            If *artifact_path* does not exist.
        ValueError
            If the computed checksum does not match the manifest entry.
        """
        path = Path(artifact_path)

        if not path.exists():
            logger.error("Artifact not found: '%s'", path)
            raise FileNotFoundError(f"Artifact not found: '{path}'")

        logger.debug("Computing SHA-256 for '%s' …", path)
        actual = _compute_sha256(path)

        # Use the string representation of the path as the manifest key,
        # but also try the filename stem so callers can use either form.
        key = str(path)
        if key not in self.manifest:
            key = path.name  # fall back to bare filename

        if key not in self.manifest:
            logger.warning(
                "Artifact '%s' is not listed in the manifest; skipping checksum check.",
                path,
            )
            return True

        expected = self.manifest[key]
        if actual != expected:
            msg = f"Checksum mismatch for {path}: expected {expected}, got {actual}"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Artifact '%s' passed checksum validation.", path)
        return True

    def validate_row_alignment(
        self,
        embeddings: np.ndarray,
        movie_df: pd.DataFrame,
    ) -> bool:
        """Assert that *embeddings* and *movie_df* have the same number of rows.

        Parameters
        ----------
        embeddings:
            NumPy array whose first dimension is the number of movies.
        movie_df:
            DataFrame whose length is the number of movies.

        Returns
        -------
        bool
            ``True`` when the row counts match.

        Raises
        ------
        ValueError
            If ``embeddings.shape[0] != len(movie_df)``, with a message
            reporting both counts.
        """
        n = embeddings.shape[0]
        m = len(movie_df)

        if n != m:
            msg = f"Row alignment mismatch: embeddings has {n} rows but movie_df has {m} rows"
            logger.error(msg)
            raise ValueError(msg)

        logger.info("Row alignment check passed: embeddings and movie_df both have %d rows.", n)
        return True

    def validate_all(self) -> dict[str, bool]:
        """Validate every artifact listed in the manifest.

        Iterates over all entries in :attr:`manifest` and calls
        :meth:`validate` for each one.  Exceptions are caught per-artifact
        and recorded as ``False`` rather than propagated, so a single bad
        artifact does not abort the entire validation run.

        Returns
        -------
        dict[str, bool]
            Mapping of ``{artifact_name: validation_passed}``.  ``True``
            means the artifact exists and its checksum matches; ``False``
            means validation failed (file missing, checksum mismatch, or
            unexpected I/O error).
        """
        results: dict[str, bool] = {}

        if not self.manifest:
            logger.warning("validate_all() called but manifest is empty; nothing to validate.")
            return results

        for artifact_name in self.manifest:
            path = Path(artifact_name)
            try:
                results[artifact_name] = self.validate(path)
            except FileNotFoundError as exc:
                logger.error("validate_all: artifact '%s' not found — %s", artifact_name, exc)
                results[artifact_name] = False
            except ValueError as exc:
                logger.error("validate_all: artifact '%s' failed checksum — %s", artifact_name, exc)
                results[artifact_name] = False
            except OSError as exc:
                logger.error("validate_all: I/O error reading artifact '%s' — %s", artifact_name, exc)
                results[artifact_name] = False

        passed = sum(v for v in results.values())
        logger.info("validate_all complete: %d/%d artifacts passed.", passed, len(results))
        return results


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def create_manifest(
    artifact_paths: list[Path],
    output_path: Path,
) -> dict[str, str]:
    """Compute SHA-256 checksums for a list of files and write a manifest JSON.

    The manifest maps each artifact's path string (as provided in
    *artifact_paths*) to its SHA-256 hex digest.  The JSON file is written
    to *output_path*; parent directories are created if they do not exist.

    Parameters
    ----------
    artifact_paths:
        List of :class:`~pathlib.Path` objects pointing to the artifact files
        to include in the manifest.
    output_path:
        Destination path for the manifest JSON file.

    Returns
    -------
    dict[str, str]
        The manifest dictionary ``{artifact_path_str: sha256_hex}`` that was
        written to *output_path*.

    Raises
    ------
    FileNotFoundError
        If any path in *artifact_paths* does not exist.
    """
    manifest: dict[str, str] = {}

    for path in artifact_paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cannot create manifest: artifact not found: '{path}'")
        logger.debug("Hashing '%s' for manifest …", path)
        checksum = _compute_sha256(path)
        manifest[str(path)] = checksum
        logger.info("Manifest entry: '%s' → %s", path, checksum)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    logger.info("Manifest written to '%s' (%d entries).", output_path, len(manifest))
    return manifest
