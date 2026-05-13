"""
Recommendation engine.
This isn't just a database wrapper; it loads the FAISS index and handles the "fuzzy" logic 
of making recommendations feel personalized.
"""
import logging
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any
import gc

import numpy as np
import pandas as pd
import os
import json

# Import model loader to handle external model downloads
from backend.model_loader import ensure_model_files
from backend.openrouter_client import chat_completion, configured_models, openrouter_api_key
from backend.query_understanding import intent_score, parse_query_intent
from backend.ranker import load_ranker
from backend.semantic_twin import build_semantic_twin, compare_semantic_twins

logger = logging.getLogger(__name__)

# Resolve paths relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _render_like_environment() -> bool:
    """Detect constrained PaaS runtimes where the full vector stack can exceed memory."""
    return any(
        os.getenv(name)
        for name in (
            "RENDER",
            "RENDER_SERVICE_ID",
            "RENDER_SERVICE_NAME",
            "RENDER_EXTERNAL_URL",
            "RENDER_EXTERNAL_HOSTNAME",
        )
    )


def _serving_profile() -> str:
    """Resolve the serving profile for this process."""
    profile = os.getenv("NOVA_SERVING_PROFILE", "auto").strip().lower()
    if profile in {"full", "lite", "light", "low-memory", "metadata"}:
        return profile
    return "auto"


def _low_memory_serving_enabled() -> bool:
    """Return true when serving should avoid loading heavyweight vector artifacts."""
    if _env_truthy("NOVA_LOW_MEMORY"):
        return True
    if os.getenv("NOVA_LOW_MEMORY", "").strip().lower() in {"0", "false", "no", "off"}:
        return False

    profile = _serving_profile()
    if profile in {"lite", "light", "low-memory", "metadata"}:
        return True
    if profile == "full":
        return False

    return _render_like_environment()

class Recommender:
    """
    The brain of the operation.
    It manages the FAISS index (for speed) and the metadata (for context).
    """
    
    def __init__(self):
        self._index: Any | None = None
        self._vectorizer: Any | None = None
        self._item_vectorizer: Any | None = None
        self._movies: pd.DataFrame | None = None
        self._vectors: np.ndarray | None = None
        self._artifact_movie_ids: np.ndarray | None = None
        self._artifact_manifest: dict[str, Any] | None = None
        self._content_text: pd.Series | None = None
        self._tfidf_matrix = None
        self._item_tfidf_matrix = None
        self._query_encoder = None
        self._cross_encoder = None
        self._learned_ranker = None
        self._behavior_features: dict[str, Any] = {}
        self._behavior_features_refreshed_at: datetime | None = None
        self._semantic_twin_cache: dict[int, dict[str, Any]] = {}
        self._low_memory = _low_memory_serving_enabled()
        self._artifact_status: dict[str, Any] = {"vector_artifacts_ready": False}
    
    def load(self) -> "Recommender":
        """
        Loads the heavy artifacts.
        We use memory-mapping for the vectors so we don't blow up the RAM on the free tier.
        """
        logger.info("Loading recommendation engine...")
        selected_artifacts = {
            "movies_transformed.parquet",
            "semantic_twins.parquet",
            "semantic_twin_summary.json",
            "pipeline_manifest.json",
            "nova_ranker.joblib",
            "nova_ranker.joblib.metadata.json",
        }
        if not self._low_memory or _env_truthy("NOVA_FORCE_VECTOR_ARTIFACTS"):
            selected_artifacts.update({"sbert_embeddings.npy", "faiss.index", "movie_ids.npy"})
        ensure_model_files(MODELS_DIR, selected_files=selected_artifacts)
        if self._low_memory:
            logger.info("Using low-memory serving profile; FAISS/SBERT artifacts are optional.")
        
        # Load FAISS index
        index_path = MODELS_DIR / "faiss.index"
        if self._low_memory and not _env_truthy("NOVA_FORCE_VECTOR_ARTIFACTS"):
            logger.info("Skipping FAISS index load in low-memory serving profile.")
        elif not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run the ETL pipeline first.")
        else:
            import faiss

            self._index = faiss.read_index(str(index_path))
            logger.info(f"Loaded FAISS index with {self._index.ntotal:,} vectors")
        
        # Load SBERT embeddings with memory-mapping (reads from disk, not RAM)
        vectors_path = MODELS_DIR / "sbert_embeddings.npy"
        if self._low_memory and not _env_truthy("NOVA_FORCE_VECTOR_ARTIFACTS"):
            logger.info("Skipping embedding matrix load in low-memory serving profile.")
        elif vectors_path.exists():
            # Memory-mapped mode: doesn't load entire array into RAM
            self._vectors = np.load(vectors_path, mmap_mode='r')
            logger.info(f"Loaded SBERT embeddings with shape {self._vectors.shape} (memory-mapped)")
        else:
            # Fallback to TF-IDF if SBERT not found
            vectors_path = MODELS_DIR / "tfidf_vectors.npy"
            if vectors_path.exists():
                self._vectors = np.load(vectors_path, mmap_mode='r')
                logger.warning("SBERT embeddings not found, using TF-IDF vectors.")
            else:
                logger.warning("No vectors found.")

        movie_ids_path = MODELS_DIR / "movie_ids.npy"
        if movie_ids_path.exists():
            self._artifact_movie_ids = np.load(movie_ids_path, mmap_mode="r")
            logger.info("Loaded vector movie id map with %s ids", len(self._artifact_movie_ids))

        manifest_path = MODELS_DIR / "pipeline_manifest.json"
        if manifest_path.exists():
            try:
                self._artifact_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self._artifact_status.update(
                    {
                        "manifest_run_id": self._artifact_manifest.get("run_id"),
                        "manifest_run_date": self._artifact_manifest.get("run_date"),
                    }
                )
            except Exception as exc:
                logger.warning("Could not read pipeline manifest %s: %s", manifest_path, exc)
        
        # Load movie metadata - only essential columns to save memory
        movies_path = DATA_DIR / "movies_transformed.parquet"
        if not movies_path.exists():
            movies_path = DATA_DIR / "movies.parquet"
        
        if movies_path.exists():
            # Only load columns we actually need for recommendations
            essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average',
                            'vote_count', 'popularity', 'release_date', 'poster_path',
                            'director', 'original_language', 'tagline', 'runtime',
                            'metadata_completeness', 'content_quality_score',
                            'quality_bucket', 'searchable', 'recommendable',
                            'public_demo_eligible']
            if not self._low_memory:
                essential_cols.append('cast')
            try:
                self._movies = pd.read_parquet(movies_path, columns=essential_cols)
            except (KeyError, ValueError):
                # Fallback if some columns don't exist
                self._movies = pd.read_parquet(movies_path)
            self._optimize_movie_frame()
            self._validate_vector_artifacts()
            logger.info(f"Loaded {len(self._movies):,} movies")
        else:
            raise FileNotFoundError("Movie data not found. Run the ETL pipeline first.")

        if self._low_memory and not _env_truthy("NOVA_BUILD_SPARSE_ON_LOAD"):
            logger.info("Deferring sparse retrieval index build until first AI search.")
        else:
            self._build_sparse_retrieval_index()
        self._learned_ranker = load_ranker(models_dir=MODELS_DIR)

        self.refresh_behavior_features(force=True)
        
        return self
    
    @property
    def movies(self) -> pd.DataFrame:
        """Get movie metadata DataFrame."""
        if self._movies is None:
            raise RuntimeError("Recommender not loaded. Call load() first.")
        return self._movies

    def refresh_behavior_features(self, force: bool = False) -> dict[str, Any]:
        """Refresh aggregated behavior features used as a light ranking signal."""
        ttl_seconds = int(os.getenv("BEHAVIOR_FEATURE_TTL_SECONDS", "60"))
        now = datetime.now(UTC)

        if (
            not force
            and self._behavior_features_refreshed_at is not None
            and (now - self._behavior_features_refreshed_at).total_seconds() < ttl_seconds
        ):
            return self._behavior_features

        try:
            from backend.events import aggregate_behavior_features

            self._behavior_features = aggregate_behavior_features(limit=100)
            self._behavior_features_refreshed_at = now
        except Exception as exc:
            logger.warning("Behavior feature refresh skipped: %s", exc)
            self._behavior_features = {}
            self._behavior_features_refreshed_at = now

        return self._behavior_features

    def _optimize_movie_frame(self) -> None:
        """Reduce the in-memory footprint of the serving catalog."""
        if self._movies is None:
            return

        for column in ("id", "vote_count"):
            if column in self._movies.columns:
                self._movies[column] = pd.to_numeric(self._movies[column], errors="coerce", downcast="integer")

        for column in (
            "vote_average",
            "popularity",
            "metadata_completeness",
            "content_quality_score",
        ):
            if column in self._movies.columns:
                self._movies[column] = pd.to_numeric(self._movies[column], errors="coerce", downcast="float")

        for column in ("searchable", "recommendable", "public_demo_eligible"):
            if column in self._movies.columns:
                self._movies[column] = self._movies[column].fillna(False).astype(bool)

        for column in ("quality_bucket", "original_language"):
            if column in self._movies.columns:
                self._movies[column] = self._movies[column].fillna("").astype("category")

    @staticmethod
    def _clean_response_value(value: Any) -> Any:
        """Convert pandas/numpy missing values and scalars to JSON-safe values."""
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _clean_response_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a movie record before FastAPI response validation."""
        return {key: cls._clean_response_value(value) for key, value in record.items()}

    def _disable_vector_artifacts(self, reason: str) -> None:
        """Disable vector serving when artifacts violate row-alignment contracts."""
        logger.warning("Disabling FAISS/SBERT recommendations: %s", reason)
        self._artifact_status.update(
            {
                "vector_artifacts_ready": False,
                "disabled_reason": reason,
            }
        )
        self._index = None
        self._vectors = None
        gc.collect()

    @staticmethod
    def _movie_id_sha256(movie_ids: np.ndarray) -> str:
        """Hash the exact ordered movie-id vector used by FAISS row positions."""
        ids = np.asarray(movie_ids, dtype=np.int64).astype("<i8", copy=False)
        return hashlib.sha256(ids.tobytes()).hexdigest()

    def _expected_manifest_contract(self) -> dict[str, Any]:
        """Extract row-level expectations from the optional pipeline manifest."""
        manifest = self._artifact_manifest or {}
        contract = manifest.get("serving_contract") or {}
        quality = manifest.get("quality") or {}
        return {
            "movie_count": contract.get("movie_rows") or quality.get("serving_rows"),
            "vector_count": contract.get("embedding_rows") or quality.get("embedding_rows"),
            "index_count": contract.get("faiss_index_size") or quality.get("faiss_index_size"),
            "id_map_count": contract.get("movie_id_map_rows") or quality.get("movie_id_map_rows"),
            "movie_id_sha256": contract.get("movie_id_sha256") or quality.get("movie_id_sha256"),
        }

    def _validate_vector_artifacts(self) -> None:
        """Validate that vector artifacts are row-aligned with the serving catalog."""
        if self._movies is None:
            return

        movie_count = len(self._movies)
        vector_count = int(self._vectors.shape[0]) if self._vectors is not None and len(self._vectors.shape) == 2 else None
        index_count = int(getattr(self._index, "ntotal", 0)) if self._index is not None else None
        id_map_count = int(len(self._artifact_movie_ids)) if self._artifact_movie_ids is not None else None

        self._artifact_status.update(
            {
                "movie_count": movie_count,
                "vector_count": vector_count,
                "faiss_index_count": index_count,
                "movie_id_map_count": id_map_count,
            }
        )

        expected = self._expected_manifest_contract()
        for key, actual in (
            ("movie_count", movie_count),
            ("vector_count", vector_count),
            ("index_count", index_count),
            ("id_map_count", id_map_count),
        ):
            expected_value = expected.get(key)
            if expected_value is not None and actual is not None and int(expected_value) != int(actual):
                self._disable_vector_artifacts(
                    f"pipeline manifest {key} ({expected_value}) != loaded artifact count ({actual})"
                )
                return

        if self._vectors is None and self._index is None:
            return
        if self._vectors is None or self._index is None:
            self._disable_vector_artifacts("both embedding matrix and FAISS index must be present")
            return
        if vector_count != index_count:
            self._disable_vector_artifacts(f"embedding rows ({vector_count}) != FAISS rows ({index_count})")
            return

        if self._artifact_movie_ids is None:
            if _env_truthy("NOVA_ALLOW_LEGACY_ROW_ALIGNED_VECTORS"):
                if vector_count != movie_count:
                    self._disable_vector_artifacts(
                        f"vector rows ({vector_count}) != serving catalog rows ({movie_count}) and no movie_ids.npy map exists"
                    )
                    return
                self._artifact_status.update(
                    {
                        "vector_artifacts_ready": True,
                        "disabled_reason": None,
                        "legacy_row_alignment": True,
                    }
                )
                return

            self._disable_vector_artifacts(
                "movie_ids.npy is required for vector serving; row count alone cannot prove FAISS/catalog alignment"
            )
            return

        if id_map_count != vector_count:
            self._disable_vector_artifacts(f"movie id map rows ({id_map_count}) != vector rows ({vector_count})")
            return

        artifact_id_values = np.asarray(self._artifact_movie_ids, dtype=np.int64)
        actual_id_hash = self._movie_id_sha256(artifact_id_values)
        self._artifact_status["movie_id_sha256"] = actual_id_hash
        expected_id_hash = expected.get("movie_id_sha256")
        if expected_id_hash and expected_id_hash != actual_id_hash:
            self._disable_vector_artifacts("movie_ids.npy checksum does not match the pipeline manifest")
            return

        artifact_ids = pd.Series(artifact_id_values).astype("int64")
        if artifact_ids.duplicated().any():
            self._disable_vector_artifacts("movie_ids.npy contains duplicate movie ids")
            return

        movie_ids = pd.to_numeric(self._movies["id"], errors="coerce")
        if movie_ids.isna().any():
            self._disable_vector_artifacts("serving catalog contains non-numeric movie ids")
            return

        current_ids = movie_ids.astype("int64").to_numpy()
        if len(current_ids) != len(artifact_id_values):
            self._disable_vector_artifacts(
                f"serving catalog rows ({len(current_ids)}) != vector id rows ({len(artifact_id_values)})"
            )
            return

        if not np.array_equal(current_ids, artifact_id_values):
            current_set = set(current_ids.tolist())
            artifact_set = set(artifact_id_values.tolist())
            if current_set != artifact_set:
                self._disable_vector_artifacts("vector id map does not contain the same movie ids as the catalog")
                return

            logger.warning("Reordering serving catalog to match vector movie id map.")
            reordered = (
                self._movies.assign(_movie_id_for_alignment=current_ids)
                .drop_duplicates(subset=["_movie_id_for_alignment"], keep="first")
                .set_index("_movie_id_for_alignment")
                .loc[artifact_id_values]
                .reset_index(drop=True)
            )
            self._movies = reordered

        self._artifact_status.update({"vector_artifacts_ready": True, "disabled_reason": None})

    def _build_sparse_retrieval_index(self) -> None:
        """Build a TF-IDF recall index for hybrid search and cold-start resilience."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        text_parts = []
        for column in ("title", "overview", "genres", "director", "cast", "original_language"):
            if column in self._movies.columns:
                text_parts.append(self._movies[column].fillna("").astype(str))

        if not text_parts:
            self._content_text = pd.Series([""] * len(self._movies), index=self._movies.index)
        else:
            content_text = text_parts[0]
            for part in text_parts[1:]:
                content_text = content_text + ". " + part
            self._content_text = content_text

        default_features = "12000" if self._low_memory else "50000"
        max_features = int(os.getenv("NOVA_TFIDF_MAX_FEATURES", default_features))
        ngram_range = (1, 1) if self._low_memory else (1, 2)
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            min_df=1,
            dtype=np.float32,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._content_text)
        logger.info("Built sparse TF-IDF retrieval index with %s features", len(self._vectorizer.vocabulary_))
        self._content_text = None

    def _ensure_sparse_retrieval_index(self) -> None:
        """Build sparse retrieval lazily when a request path needs AI search."""
        if self._tfidf_matrix is None or self._vectorizer is None:
            self._build_sparse_retrieval_index()

    def _build_item_retrieval_index(self) -> None:
        """Build a plot/genre-focused sparse index for item-to-item recommendations."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._movies is None:
            return

        def text_column(column: str) -> pd.Series:
            if column not in self._movies.columns:
                return pd.Series([""] * len(self._movies), index=self._movies.index)
            return self._movies[column].fillna("").astype(str)

        overview = text_column("overview")
        tagline = text_column("tagline")
        genres = text_column("genres").str.replace(",", " ", regex=False)
        language = text_column("original_language")

        # Item recommendations should be about story, tone, and genre. Title is
        # deliberately omitted so "Avatar" does not mostly retrieve making-of
        # documentaries just because they repeat the title.
        item_text = (
            overview
            + ". "
            + tagline
            + ". Genres "
            + genres
            + ". Language "
            + language
        )

        default_features = "18000" if self._low_memory else "40000"
        max_features = int(os.getenv("NOVA_ITEM_TFIDF_MAX_FEATURES", default_features))
        self._item_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=1,
            dtype=np.float32,
        )
        self._item_tfidf_matrix = self._item_vectorizer.fit_transform(item_text)
        logger.info("Built item-to-item sparse retrieval index with %s features", len(self._item_vectorizer.vocabulary_))

    def _ensure_item_retrieval_index(self) -> None:
        """Build item recommendation sparse retrieval lazily."""
        if self._item_tfidf_matrix is None or self._item_vectorizer is None:
            self._build_item_retrieval_index()

    def _dense_query_enabled(self) -> bool:
        """Return whether online dense query encoding should be attempted."""
        value = os.getenv("NOVA_ENABLE_DENSE_QUERY", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _cross_encoder_enabled(self) -> bool:
        """Return whether optional cross-encoder reranking should be attempted."""
        value = os.getenv("NOVA_ENABLE_CROSS_ENCODER", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _llm_rerank_enabled(self) -> bool:
        """Return whether slow OpenRouter LLM reranking should run in the hot path."""
        value = os.getenv("NOVA_ENABLE_LLM_RERANK", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _get_query_encoder(self):
        """Load the query bi-encoder lazily when the deployment opts in."""
        if self._query_encoder is None:
            from sentence_transformers import SentenceTransformer

            model_name = os.getenv("NOVA_QUERY_ENCODER_MODEL", "all-mpnet-base-v2")
            self._query_encoder = SentenceTransformer(model_name)
            logger.info("Loaded query encoder: %s", model_name)
        return self._query_encoder

    def _get_cross_encoder(self):
        """Load an optional lightweight cross-encoder reranker lazily."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            model_name = os.getenv("NOVA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._cross_encoder = CrossEncoder(model_name)
            logger.info("Loaded cross-encoder reranker: %s", model_name)
        return self._cross_encoder

    def _popularity_quality_score(self, movie: dict[str, Any]) -> float:
        """Small bounded business score from popularity and quality."""
        if movie.get("content_quality_score") is not None:
            try:
                score = float(movie.get("content_quality_score"))
                if not np.isnan(score):
                    return max(0.0, min(1.0, score))
            except (TypeError, ValueError):
                pass
        completeness = float(movie.get("metadata_completeness") or 0.0)
        popularity = float(movie.get("popularity") or 0)
        rating = float(movie.get("vote_average") or 0)
        votes = float(movie.get("vote_count") or 0)
        popularity_score = min(1.0, np.log1p(max(popularity, 0)) / 8.0)
        confidence = min(1.0, np.log1p(max(votes, 0)) / 8.0)
        quality_score = (rating / 10.0) * confidence if rating > 0 else 0.0
        return float(0.35 * popularity_score + 0.35 * quality_score + 0.30 * completeness)

    @staticmethod
    def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
        """Min-max normalize a sparse score map into [0, 1]."""
        if not scores:
            return {}
        values = np.array(list(scores.values()), dtype=np.float32)
        min_value = float(values.min())
        max_value = float(values.max())
        if max_value <= min_value:
            return {key: 1.0 for key in scores}
        return {
            key: float((value - min_value) / (max_value - min_value))
            for key, value in scores.items()
        }

    def _genre_set(self, movie: dict[str, Any]) -> set[str]:
        return {part.strip().lower() for part in str(movie.get("genres") or "").split(",") if part.strip()}

    def _semantic_twin_for_index(self, movie_idx: int) -> dict[str, Any]:
        """Build/cache the deterministic semantic twin for a catalog row."""
        if movie_idx not in self._semantic_twin_cache:
            self._semantic_twin_cache[movie_idx] = build_semantic_twin(self.get_movie_by_index(movie_idx))
        return self._semantic_twin_cache[movie_idx]

    def get_semantic_twin_by_id(self, movie_id: int) -> dict[str, Any] | None:
        """Return the semantic item twin for a movie ID."""
        matches = self._movies[self._movies["id"] == movie_id].index
        if len(matches) == 0:
            return None
        return self._semantic_twin_for_index(int(matches[0]))

    def _semantic_affinity_for_indices(self, query_idx: int, candidate_idx: int) -> dict[str, Any]:
        """Compare query/candidate semantic twins and return serializable signals."""
        affinity = compare_semantic_twins(
            self._semantic_twin_for_index(query_idx),
            self._semantic_twin_for_index(candidate_idx),
        )
        return affinity.as_dict()

    def _apply_query_mmr(
        self,
        candidates: list[dict],
        n: int,
        lambda_param: float = 0.72,
    ) -> list[dict]:
        """Diversify query search results using candidate vectors where available."""
        if len(candidates) <= n or self._vectors is None:
            return candidates[:n]

        selected: list[dict] = []
        remaining = candidates.copy()
        selected.append(remaining.pop(0))

        while remaining and len(selected) < n:
            best_idx = 0
            best_score = -float("inf")
            for idx, candidate in enumerate(remaining):
                candidate_indices = self._movies[self._movies["id"] == candidate.get("id")].index
                if len(candidate_indices) == 0:
                    continue
                candidate_idx = int(candidate_indices[0])
                relevance = float(candidate.get("similarity_score") or 0)

                max_similarity = 0.0
                for chosen in selected:
                    chosen_indices = self._movies[self._movies["id"] == chosen.get("id")].index
                    if len(chosen_indices) == 0:
                        continue
                    chosen_idx = int(chosen_indices[0])
                    candidate_vector = np.asarray(self._vectors[candidate_idx], dtype=np.float32)
                    chosen_vector = np.asarray(self._vectors[chosen_idx], dtype=np.float32)
                    max_similarity = max(max_similarity, float(np.dot(candidate_vector, chosen_vector)))

                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_similarity
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))
        return selected[:n]

    def _apply_learned_ranker(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Apply the trained ranker artifact when available."""
        if self._learned_ranker is None or not candidates or not self._learned_ranker_enabled():
            return candidates
        try:
            return self._learned_ranker.rerank(candidates)
        except Exception as exc:
            logger.warning("Learned ranker skipped: %s", exc)
            return candidates

    def _quality_gate_item_recommendations(
        self,
        candidates: list[dict[str, Any]],
        query_movie: dict[str, Any],
        n: int,
    ) -> list[dict[str, Any]]:
        """Drop obvious low-quality or genre-drift candidates when enough alternatives exist."""
        if len(candidates) <= n:
            return candidates

        query_genres = self._genre_set(query_movie)
        gated: list[dict[str, Any]] = []
        for candidate in candidates:
            rating = float(candidate.get("vote_average") or 0.0)
            votes = float(candidate.get("vote_count") or 0.0)
            candidate_genres = self._genre_set(candidate)
            shared_genres = query_genres & candidate_genres
            signals = candidate.get("retrieval_signals") or {}
            semantic_score = float(signals.get("semantic_twin") or 0.0)

            if votes >= 500 and 0 < rating < 5.5:
                continue
            if "science fiction" in query_genres and "science fiction" not in candidate_genres:
                if len(shared_genres) < 2 and semantic_score < 0.62:
                    continue

            gated.append(candidate)

        return gated if len(gated) >= n else candidates

    def _learned_ranker_enabled(self) -> bool:
        """Return whether the learned ranker has enough signal to influence serving."""
        value = os.getenv("NOVA_ENABLE_LEARNED_RANKER", "auto").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False

        metadata = getattr(self._learned_ranker, "metadata", {}) or {}
        training_mode = str(metadata.get("training_mode") or "").lower()
        try:
            feedback_count = int(metadata.get("feedback_item_count") or 0)
        except (TypeError, ValueError):
            feedback_count = 0
        min_feedback = int(os.getenv("NOVA_MIN_RANKER_FEEDBACK_ITEMS", "100"))

        if training_mode == "catalog_bootstrap" and feedback_count < min_feedback:
            return False
        return feedback_count >= min_feedback

    def _behavior_boost(self, movie_id: Any) -> tuple[float, list[str]]:
        """Return a bounded score nudge from recent product behavior."""
        if movie_id is None:
            return 0.0, []

        try:
            movie_key = str(int(movie_id))
        except (TypeError, ValueError):
            return 0.0, []

        trending_movies = self._behavior_features.get("trending_movies", {})
        if not isinstance(trending_movies, dict):
            return 0.0, []

        stats = trending_movies.get(movie_key)
        if not isinstance(stats, dict):
            return 0.0, []

        event_count = int(stats.get("event_count") or 0)
        views = int(stats.get("views") or 0)
        clicks = int(stats.get("clicks") or 0)
        ratings = int(stats.get("ratings") or 0)
        avg_rating = stats.get("avg_rating")

        boost = min(0.08, event_count * 0.005 + views * 0.003 + clicks * 0.01)
        if avg_rating is not None and ratings > 0 and float(avg_rating) >= 4.0:
            boost += min(0.02, ratings * 0.005)
        boost = min(0.10, boost)

        reasons = []
        if event_count:
            reasons.append(f"Trending with viewers ({event_count} recent events)")
        if avg_rating is not None and ratings > 0:
            reasons.append(f"Audience signal ({float(avg_rating):.1f}/5)")

        return boost, reasons[:2]

    @staticmethod
    def _event_recency_decay(event_ts: Any, half_life_days: float = 14.0) -> float:
        """Return a bounded recency weight for personalization events."""
        if not event_ts:
            return 0.65
        try:
            parsed = datetime.fromisoformat(str(event_ts).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            age_days = max(0.0, (datetime.now(UTC) - parsed).total_seconds() / 86400.0)
            return float(0.5 ** (age_days / half_life_days))
        except Exception:
            return 0.65

    def _genre_affinity_from_profile(self, profile: dict[str, Any]) -> dict[str, float]:
        """Build genre affinity weights from positive user events."""
        affinity: dict[str, float] = {}
        for event in profile.get("recent_events") or []:
            if event.get("negative"):
                continue
            movie = self.get_movie_by_id(int(event.get("movie_id"))) if event.get("movie_id") is not None else None
            if not movie:
                continue
            weight = float(event.get("weight") or 1.0) * self._event_recency_decay(event.get("event_ts"))
            for genre in self._genre_set(movie):
                affinity[genre] = affinity.get(genre, 0.0) + weight

        if not affinity:
            return {}
        max_weight = max(affinity.values())
        if max_weight <= 0:
            return {}
        return {genre: round(weight / max_weight, 4) for genre, weight in affinity.items()}

    def recommend_for_user_profile(self, profile: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
        """Blend seed-item, search-intent, genre-affinity, and trending signals for a user."""
        result_limit = max(1, min(int(n), 50))
        negative_ids = {int(movie_id) for movie_id in profile.get("negative_movie_ids") or []}
        seed_events = [
            event for event in (profile.get("recent_events") or [])
            if event.get("movie_id") is not None and not event.get("negative")
        ]
        genre_affinity = self._genre_affinity_from_profile(profile)

        scored: dict[int, dict[str, Any]] = {}

        def add_candidate(candidate: dict[str, Any], score: float, reason: str, stage: str) -> None:
            candidate_id = candidate.get("id")
            if candidate_id is None:
                return
            try:
                candidate_id = int(candidate_id)
            except (TypeError, ValueError):
                return
            if candidate_id in negative_ids:
                return

            item = dict(candidate)
            genre_boost = 0.0
            for genre in self._genre_set(item):
                genre_boost += genre_affinity.get(genre, 0.0)
            genre_boost = min(0.15, genre_boost * 0.045)
            final_score = float(score) + genre_boost
            current = scored.get(candidate_id)
            if current is None or final_score > float(current.get("similarity_score") or 0):
                explanations = list(item.get("explanation") or [])
                explanations.insert(0, reason)
                if genre_boost > 0:
                    explanations.insert(1, "matches your genre affinity")
                item["similarity_score"] = final_score
                item["retrieval_stage"] = stage
                item["retrieval_signals"] = {
                    **(item.get("retrieval_signals") or {}),
                    "personalization": round(final_score, 4),
                    "genre_affinity": round(genre_boost, 4),
                }
                item["explanation"] = explanations[:5]
                item["explanation_text"] = " | ".join(item["explanation"])
                scored[candidate_id] = item

        seen_seed_ids = set()
        for event_rank, event in enumerate(seed_events[:8]):
            seed_movie_id = int(event["movie_id"])
            seen_seed_ids.add(seed_movie_id)
            event_weight = float(event.get("weight") or 1.0)
            recency = self._event_recency_decay(event.get("event_ts"))
            seed_weight = event_weight * recency / (event_rank + 1)
            for candidate in self.recommend_by_id(seed_movie_id, n=min(30, max(result_limit * 4, 12))):
                if candidate.get("id") in seen_seed_ids:
                    continue
                base_score = float(candidate.get("similarity_score") or 0.0)
                add_candidate(
                    candidate,
                    score=base_score * seed_weight,
                    reason=f"personalized from recent {event.get('event_type', 'interaction')}",
                    stage="personalized_v2_seed_blend",
                )

        for search in (profile.get("top_searches") or [])[:3]:
            query_text = str(search.get("query_text") or "").strip()
            if not query_text:
                continue
            count_weight = min(1.0, float(search.get("count") or 1) / 3.0)
            for candidate in self.ai_search(query_text, n=min(12, max(result_limit * 2, 8))):
                add_candidate(
                    candidate,
                    score=float(candidate.get("similarity_score") or 0.0) * 0.42 * count_weight,
                    reason=f"matches your search intent: {query_text}",
                    stage="personalized_v2_search_blend",
                )

        if not scored:
            behavior = self.refresh_behavior_features()
            for item in (behavior.get("trending_movies") or {}).values():
                movie_id = item.get("movie_id")
                if movie_id is None:
                    continue
                movie = self.get_movie_by_id(int(movie_id)) if isinstance(movie_id, int) else None
                if movie:
                    add_candidate(
                        movie,
                        score=min(1.0, float(item.get("event_count") or 0) / 20.0),
                        reason=f"trending with viewers ({item.get('event_count')} recent events)",
                        stage="personalized_v2_trending_fallback",
                    )

        results = sorted(scored.values(), key=lambda item: float(item.get("similarity_score") or 0), reverse=True)
        return results[:result_limit]
    
    def get_movie_by_id(self, movie_id: int) -> dict | None:
        """Get movie details by TMDB ID."""
        matches = self._movies[self._movies["id"] == movie_id]
        if len(matches) == 0:
            return None
        return self._clean_response_record(matches.iloc[0].to_dict())
    
    def get_movie_by_index(self, idx: int) -> dict:
        """Get movie details by DataFrame index."""
        return self._clean_response_record(self._movies.iloc[idx].to_dict())
        
    def get_all_titles(self, limit: int = 5000) -> list[dict]:
        """
        Return a lightweight list of movie IDs and Titles for autocomplete.
        """
        if self._movies is None:
            return []
        
        # Extract necessary columns
        cols = ["id", "title"]
        if "release_date" in self._movies.columns:
            cols.append("release_date")
        if "popularity" in self._movies.columns:
            cols.append("popularity")
        if "genres" in self._movies.columns:
            cols.append("genres")
            
        titles_df = self._movies[cols].copy()
        
        # Append release year to the title for disambiguation
        if "release_date" in titles_df.columns:
            years = pd.to_datetime(titles_df["release_date"], errors="coerce").dt.year
            mask = years.notna() & (years > 0)
            titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " (" + years[mask].astype(int).astype(str) + ")"
            
        # Append genres to the title for extra context
        if "genres" in titles_df.columns:
            # Handle NaN/None in genres
            mask = titles_df["genres"].notna() & (titles_df["genres"] != "")
            # Take only the first 2 genres to keep it clean, if it's a comma-separated string
            def get_top_genres(g_str):
                try:
                    parts = str(g_str).split(",")
                    return ", ".join(p.strip() for p in parts[:2])
                except Exception:
                    return str(g_str)
            
            top_genres = titles_df.loc[mask, "genres"].apply(get_top_genres)
            titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " - " + top_genres
        
        # Sort by popularity so famous movies appear at the top instead of garbage punctuation
        if "popularity" in titles_df.columns:
            titles_df = titles_df.sort_values("popularity", ascending=False)
        else:
            titles_df = titles_df.sort_values("title")
            
        # Limit to the top N most popular movies to save bandwidth and browser memory
        if limit and limit > 0:
            titles_df = titles_df.head(limit)
        
        # Return only id and title

        return titles_df[["id", "title"]].to_dict(orient="records")
    
    def search_movies(self, query: str, limit: int = 20) -> list[dict]:
        """
        Standard text search, but with a few tweaks to make it feel smarter.
        We prioritize Titles, but also peek at Genres and Overviews so you can search for "action aliens".
        """
        """
        Search movies by title, overview, and genres (Deep Search).
        
        Args:
            query: Search query string
            limit: Maximum results to return
            
        Returns:
            List of matching movie dictionaries sorted by relevance
        """
        if not query:
            return []
        if self._movies is None:
            return []
            
        q_lower = query.lower().strip()

        def text_column(column: str) -> pd.Series:
            if column not in self._movies.columns:
                return pd.Series("", index=self._movies.index, dtype="string")
            return self._movies[column].fillna("").astype(str)

        def numeric_column(column: str) -> pd.Series:
            if column not in self._movies.columns:
                return pd.Series(0.0, index=self._movies.index, dtype="float32")
            return pd.to_numeric(self._movies[column], errors="coerce").fillna(0.0)

        titles = text_column("title")
        overviews = text_column("overview")
        genres = text_column("genres")
        
        # 1. Title Match (Weight: 10)
        mask_title = titles.str.lower().str.contains(q_lower, regex=False, na=False)
        
        # 2. Overview Match (Weight: 3) - Allows searching by plot concepts
        mask_overview = overviews.str.lower().str.contains(q_lower, regex=False, na=False)
        
        # 3. Genre Match (Weight: 5)
        mask_genre = genres.str.lower().str.contains(q_lower, regex=False, na=False)
        
        # Combine matches
        matches = self._movies[mask_title | mask_overview | mask_genre].copy()
        
        if len(matches) == 0:
            return []
            
        matches["relevance"] = 0.0
        
        # Title Factors
        m_title = text_column("title").loc[matches.index].str.lower()
        exact_title = m_title == q_lower
        starts_with_boundary = (
            exact_title
            | m_title.str.startswith(f"{q_lower} ", na=False)
            | m_title.str.startswith(f"{q_lower}:", na=False)
            | m_title.str.startswith(f"{q_lower}-", na=False)
        )
        starts_with_prefix = m_title.str.startswith(q_lower, na=False) & ~starts_with_boundary
        matches.loc[m_title == q_lower, "relevance"] += 50.0
        matches.loc[starts_with_boundary, "relevance"] += 20.0
        matches.loc[starts_with_prefix, "relevance"] += 8.0
        matches.loc[m_title.str.contains(q_lower, regex=False), "relevance"] += 10.0
        
        # Other Factors
        # Note: We use the masks subsetted by the matches index
        matches.loc[mask_genre[matches.index], "relevance"] += 5.0
        matches.loc[mask_overview[matches.index], "relevance"] += 3.0
        
        # Ranking intent: exact title should find the canonical title first, but
        # weak duplicate-title records should not bury major franchise entries.
        popularity = numeric_column("popularity").loc[matches.index].clip(lower=0)
        vote_count = numeric_column("vote_count").loc[matches.index].clip(lower=0)
        matches["relevance"] += np.log1p(popularity) * 2.0
        matches["relevance"] += np.log1p(vote_count) * 0.8

        strong_exact_exists = bool((exact_title & ((vote_count >= 500) | (popularity >= 20))).any())
        if strong_exact_exists:
            weak_exact_duplicate = exact_title & (vote_count < 100) & (popularity < 15)
            matches.loc[weak_exact_duplicate, "relevance"] -= 55.0

        franchise_continuation = (
            (
                m_title.str.startswith(f"{q_lower}: ", na=False)
                | m_title.str.startswith(f"{q_lower} - ", na=False)
            )
            & ((vote_count >= 250) | (popularity >= 20))
        )
        matches.loc[franchise_continuation, "relevance"] += 16.0
        
        # Sort by relevance
        matches = matches.sort_values("relevance", ascending=False).head(limit)
        
        return [self._clean_response_record(record) for record in matches.to_dict(orient="records")]

    def _metadata_recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
        """Content-based fallback recommender when vector artifacts are unavailable."""
        if self._movies is None or movie_idx < 0 or movie_idx >= len(self._movies):
            return []

        self.refresh_behavior_features()
        try:
            self._ensure_item_retrieval_index()
            content_scores = self._item_tfidf_matrix[movie_idx].dot(self._item_tfidf_matrix.T).toarray().ravel()
        except Exception as exc:
            logger.warning("Item sparse similarity unavailable; using metadata-only scoring: %s", exc)
            content_scores = np.zeros(len(self._movies), dtype=np.float32)

        query_movie = self.get_movie_by_index(movie_idx)
        query_twin = self._semantic_twin_for_index(movie_idx)
        q_genres = self._genre_set(query_movie)
        q_director = str(query_movie.get("director") or "").strip().lower()
        q_language = str(query_movie.get("original_language") or "").strip().lower()
        query_votes = float(query_movie.get("vote_count") or 0)
        query_runtime = float(query_movie.get("runtime") or 0)
        q_title_tokens = {
            token
            for token in str(query_movie.get("title") or "").lower().replace(":", " ").replace("-", " ").split()
            if len(token) >= 4 and token not in {"movie", "part", "chapter"}
        }

        scores = np.asarray(content_scores, dtype=np.float32) * 0.76

        def numeric_array(column: str, default: float = 0.0) -> np.ndarray:
            if column not in self._movies.columns:
                return np.full(len(self._movies), default, dtype=np.float32)
            return pd.to_numeric(
                self._movies[column],
                errors="coerce",
            ).fillna(default).to_numpy(dtype=np.float32)

        genre_overlap = np.zeros(len(self._movies), dtype=np.float32)
        if q_genres and "genres" in self._movies.columns:
            genres = self._movies["genres"].fillna("").astype(str).str.lower()
            for genre in q_genres:
                genre_mask = genres.str.contains(genre, regex=False, na=False).to_numpy()
                genre_overlap += genre_mask.astype(np.float32)
            genre_ratio = genre_overlap / max(len(q_genres), 1)
            scores += genre_ratio * 0.12
            scores += np.minimum(genre_overlap, 2) * 0.02

        if q_director and q_director != "unknown" and "director" in self._movies.columns:
            directors = self._movies["director"].fillna("").astype(str).str.lower()
            director_mask = directors.eq(q_director).to_numpy()
            scores += director_mask.astype(np.float32) * 0.08

        if q_language and "original_language" in self._movies.columns:
            languages = self._movies["original_language"].astype(str).str.lower()
            language_mask = languages.eq(q_language).to_numpy()
            scores += language_mask.astype(np.float32) * 0.025

        if "content_quality_score" in self._movies.columns:
            quality = numeric_array("content_quality_score")
            scores += np.clip(quality, 0, 1) * 0.12
        else:
            vote_average = numeric_array("vote_average")
            vote_count = numeric_array("vote_count")
            confidence = np.minimum(1.0, np.log1p(np.maximum(vote_count, 0)) / 8.0)
            scores += np.clip(vote_average / 10.0, 0, 1) * confidence * 0.10

        if "vote_count" in self._movies.columns:
            vote_count = numeric_array("vote_count")
            scores += np.minimum(1.0, np.log1p(np.maximum(vote_count, 0)) / 10.0) * 0.06

        if "popularity" in self._movies.columns:
            popularity = numeric_array("popularity")
            scores += np.minimum(1.0, np.log1p(np.maximum(popularity, 0)) / 8.0) * 0.08

        scores[movie_idx] = -np.inf
        if len(scores) <= 1:
            return []

        candidate_count = min(max(n * 40, 250), len(scores) - 1)
        candidate_indices = np.argpartition(scores, -candidate_count)[-candidate_count:]
        candidate_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

        results = []
        for idx in candidate_indices:
            idx = int(idx)
            if len(results) >= n:
                break
            if not np.isfinite(scores[idx]) or scores[idx] <= 0:
                continue

            movie = self.get_movie_by_index(idx)
            movie_genres = self._genre_set(movie)
            content_score = float(content_scores[idx]) if len(content_scores) > idx else 0.0
            shared_genres = q_genres & movie_genres
            candidate_votes = float(movie.get("vote_count") or 0)
            candidate_runtime = float(movie.get("runtime") or 0)
            semantic_affinity = compare_semantic_twins(query_twin, self._semantic_twin_for_index(idx)).as_dict()
            semantic_score = float(semantic_affinity["score"])

            if movie.get("public_demo_eligible") is False:
                continue
            if movie.get("recommendable") is False:
                continue
            if query_runtime >= 60 and 0 < candidate_runtime < 60:
                continue
            if "documentary" in movie_genres and "documentary" not in q_genres:
                continue
            if "tv movie" in movie_genres and "tv movie" not in q_genres:
                continue
            if not shared_genres and content_score < 0.10:
                continue
            min_votes = 500 if query_votes >= 5000 else 50
            if candidate_votes < min_votes and content_score < 0.16:
                continue
            if {"animation", "family"} & movie_genres and not ({"animation", "family"} & q_genres):
                scores[idx] -= 0.12
            if "comedy" in movie_genres and "comedy" not in q_genres and content_score < 0.16:
                continue

            scores[idx] += semantic_score * 0.18

            title_tokens = {
                token
                for token in str(movie.get("title") or "").lower().replace(":", " ").replace("-", " ").split()
                if len(token) >= 4
            }
            if q_title_tokens & title_tokens and "documentary" not in movie_genres:
                scores[idx] += 0.14

            reasons = []
            if shared_genres:
                reasons.append(f"Shared genres: {', '.join(sorted(g.title() for g in shared_genres)[:2])}")
            if content_score >= 0.18:
                reasons.append("Similar story and setting")
            reasons.extend(semantic_affinity.get("reasons") or [])
            if q_director and str(movie.get("director") or "").strip().lower() == q_director:
                reasons.append(f"Same director ({movie.get('director')})")
            if q_language and str(movie.get("original_language") or "").strip().lower() == q_language:
                reasons.append("Same catalog language")
            if float(movie.get("vote_average") or 0) >= 7.5:
                reasons.append(f"Strong audience rating ({float(movie.get('vote_average') or 0):.1f}/10)")
            if not reasons:
                reasons.append("Closest content and catalog-quality match")

            behavior_boost, behavior_reasons = self._behavior_boost(movie.get("id"))
            score = float(scores[idx] + behavior_boost)
            movie["similarity_score"] = score
            movie["retrieval_stage"] = "content_sparse_fallback"
            movie["retrieval_signals"] = {
                "content_sparse": round(content_score, 4),
                "semantic_twin": round(semantic_score, 4),
                "semantic_twin_details": semantic_affinity,
                "genre_overlap": round(float(len(shared_genres) / max(len(q_genres), 1)), 4),
                "metadata": round(float(scores[idx]), 4),
                "behavior": round(behavior_boost, 4),
                "vector_artifacts_loaded": False,
                "vector_artifact_status": self._artifact_status,
            }
            movie["explanation"] = (reasons + behavior_reasons)[:5]
            movie["explanation_text"] = " | ".join(movie["explanation"])
            results.append(movie)

        results.sort(key=lambda item: float(item.get("similarity_score") or 0), reverse=True)
        results = self._apply_learned_ranker(results)
        return self._quality_gate_item_recommendations(results, query_movie, n)[:n]
    
    def recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its DataFrame index.
        
        Args:
            movie_idx: Index of the movie in the DataFrame
            n: Number of recommendations
            
            Returns:
            List of recommended movie dictionaries with similarity scores
        """
        if self._vectors is None or self._index is None:
            logger.info("Vector artifacts not loaded/aligned; using content sparse recommendation fallback.")
            return self._metadata_recommend_by_index(movie_idx, n=n)

        self.refresh_behavior_features()
        
        # Get query vector
        query_vector = self._vectors[movie_idx].reshape(1, -1).astype(np.float32)
        query_vector = np.ascontiguousarray(query_vector)
        
        # Search (Fetch 100 candidates for re-ranking)
        # We fetch more than N to allow the business logic to re-order them
        fetch_k = min(100, getattr(self._index, "ntotal", len(self._movies)))
        
        # Configure IVF search
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = min(50, getattr(self._index, "nlist", 10))
            
        # Configure HNSW search (efSearch > k helps recall)
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = 200
        
        distances, indices = self._index.search(query_vector, fetch_k)
        
        # Get Query Metadata for Re-Ranking
        query_movie = self.get_movie_by_index(movie_idx)
        query_twin = self._semantic_twin_for_index(movie_idx)
        q_director = query_movie.get("director")
        q_title_tokens = set(query_movie["title"].lower().split())
        stop_words = {"the", "a", "an", "of", "and", "in", "to", "part", "vol", "volume", "chapter"}
        q_title_tokens -= stop_words
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == movie_idx or idx < 0 or idx >= len(self._movies):
                continue
            
            cand = self.get_movie_by_index(idx)
            raw_score = float(dist)
            final_score = raw_score
            semantic_affinity = compare_semantic_twins(query_twin, self._semantic_twin_for_index(int(idx))).as_dict()
            semantic_score = float(semantic_affinity["score"])
            final_score += semantic_score * 0.14
            
            # --- BUSINESS LOGIC RE-RANKING ---
            
            # Director Match (+0.10) - Strong signal for stylistic similarity
            if q_director and cand.get("director") == q_director:
                final_score += 0.10
                
            # We are removing franchise string-matching heuristics because they overly heavily bias
            # the FAISS pool. The Hugging Face Llama-3 Reranker is now smart enough to detect true franchises
            # based on plot semantics and metadata without brute forcing scores.
            pass
            
            # Popularity Nudge (Log Scale)
            votes = cand.get("vote_count", 0)
            if votes > 0:
                final_score += 0.02 * np.log10(votes)

            # Genre Consistency Check
            # If the candidate shares NO genres with the query, it's likely a semantic drift (e.g. word match).
            # "Avatar" (Sci-Fi) vs "The Aviator" (Drama) -> No overlap.
            q_genres_str = str(query_movie.get("genres", "")).lower()
            cand_genres_str = str(cand.get("genres", "")).lower()
            
            # Simple set parsing (assuming comma separated)
            q_genre_set = {g.strip() for g in q_genres_str.split(",") if g.strip()}
            cand_genre_set = {g.strip() for g in cand_genres_str.split(",") if g.strip()}
            
            # Penalize if Disjoint (and query actually has genres)
            if q_genre_set and cand_genre_set.isdisjoint(q_genre_set):
                final_score -= 0.15 

            # Documentary Penalty (Unless Query is also a Documentary)
            # Users usually don't want "Making Of" videos when searching for feature films.
            is_query_doc = "documentary" in q_genres_str
            is_cand_doc = "documentary" in cand_genres_str
            
            if is_cand_doc and not is_query_doc:
                final_score -= 0.15 # Strong penalty to push them down
            
            # Quality-based score adjustments
            
            # Quality Boost (Favor well-rated films)
            cand_rating = cand.get("vote_average", 0) or 0
            cand_votes = cand.get("vote_count", 0) or 0
            if cand_rating > 0 and cand_votes > 100:
                # Combines rating quality with vote confidence
                quality_score = (cand_rating / 10) * np.log10(max(cand_votes, 1))
                final_score += 0.02 * quality_score  # Subtle but effective
            
            # Era Matching (Penalize large time gaps)
            try:
                q_year = int(str(query_movie.get("release_date", ""))[:4])
                c_year = int(str(cand.get("release_date", ""))[:4])
                year_gap = abs(q_year - c_year)
                
                if year_gap <= 5:
                    final_score += 0.03  # Same era boost
                elif year_gap >= 30:
                    final_score -= 0.05  # Different generation penalty
            except (ValueError, TypeError, IndexError):
                pass  # Skip if dates are invalid
            
            # Recency Boost (Slight preference for newer films)
            try:
                c_year = int(str(cand.get("release_date", ""))[:4])
                current_year = datetime.now().year
                years_old = current_year - c_year
                if years_old <= 5:
                    final_score += 0.02  # Recent film boost
            except (ValueError, TypeError, IndexError):
                pass
            
            # Same Language Preference
            q_lang = str(query_movie.get("original_language", "en")).lower()
            c_lang = str(cand.get("original_language", "en")).lower()
            if q_lang == c_lang:
                final_score += 0.02  # Same language slight boost

            behavior_boost, behavior_reasons = self._behavior_boost(cand.get("id"))
            final_score += behavior_boost
            
            # === EXPLAINABILITY (Why was this recommended?) ===
            explanation_tags = []
            

            
            # Director match
            if q_director and cand.get("director") == q_director:
                explanation_tags.append(f"Same director ({q_director})")
            
            # Genre overlap
            shared_genres = q_genre_set & cand_genre_set
            if shared_genres:
                top_genres = list(shared_genres)[:2]
                explanation_tags.append(f"Shared genres: {', '.join(g.title() for g in top_genres)}")

            explanation_tags.extend(semantic_affinity.get("reasons") or [])
            
            # Era match
            try:
                q_year = int(str(query_movie.get("release_date", ""))[:4])
                c_year = int(str(cand.get("release_date", ""))[:4])
                if abs(q_year - c_year) <= 5:
                    explanation_tags.append(f"Same era ({c_year})")
            except (ValueError, TypeError, IndexError):
                pass
            
            # High quality
            if cand_rating >= 7.5 and cand_votes >= 1000:
                explanation_tags.append(f"Critically acclaimed ({cand_rating}/10)")
            
            # Same language (if not English - more notable)
            if q_lang == c_lang and q_lang != "en":
                explanation_tags.append(f"Same language ({c_lang.upper()})")

            explanation_tags.extend(behavior_reasons)
            
            # Default if no specific reasons found
            if not explanation_tags:
                explanation_tags.append("Similar themes and plot")
                
            cand["similarity_score"] = final_score
            cand["retrieval_stage"] = "vector_semantic_ranked"
            cand["retrieval_signals"] = {
                "dense": round(raw_score, 4),
                "semantic_twin": round(semantic_score, 4),
                "semantic_twin_details": semantic_affinity,
                "behavior": round(behavior_boost, 4),
            }
            cand["explanation"] = explanation_tags  # NEW: Add explanation
            cand["explanation_text"] = " | ".join(explanation_tags)  # Human-readable
            results.append(cand)
        
        # Sort by boosted score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = self._apply_learned_ranker(results)
        results = self._quality_gate_item_recommendations(results, query_movie, n)
        llm_window = int(os.getenv("NOVA_LLM_RERANK_CANDIDATES", "12"))
        top_candidates = results[: max(1, min(20, llm_window))]
        
        # Optional LLM-as-judge reranking. Keep this off by default because free
        # OpenRouter models can be rate-limited and should never block serving.
        if self._llm_rerank_enabled():
            try:
                llm_results = self._rerank_with_llm(query_movie, top_candidates, n)
                if llm_results and len(llm_results) > 0:
                    return llm_results
            except Exception as e:
                logger.warning("LLM reranking skipped; falling back to FAISS/MMR. Error: %s", e)
        
        # === MMR DIVERSITY (Maximal Marginal Relevance) ===
        # Prevents returning 5 nearly identical movies
        if len(results) > n and self._vectors is not None:
            diverse_results = self._apply_mmr(results, movie_idx, n, lambda_param=0.7)
            return diverse_results
        
        return results[:n]
    
    def _rerank_with_llm(self, query_movie: dict, candidates: list[dict], n: int = 10) -> list[dict]:
        """
        Uses OpenRouter API to semantically rerank candidates.
        """
        openrouter_key = openrouter_api_key()
        if not openrouter_key:
            logger.warning("OPENROUTER_API_KEY missing. Skipping LLM reranking and falling back to FAISS/MMR.")
            raise ValueError("OPENROUTER_API_KEY environment variable is not set or accessible.")
        
        # Prepare candidates for prompt
        cand_text = ""
        for i, c in enumerate(candidates):
            cand_text += f"[{i}] Title: {c.get('title')}, Genres: {c.get('genres')}, Plot: {c.get('overview', 'N/A')}\n"
            
        prompt = f"""You are an expert film critic. I will give you a QUERY MOVIE and a list of CANDIDATE MOVIES.
Your job is to select the {n} absolute best recommendations based on deep aesthetic similarity, themes, tropes, target audience, and vibe.
Ignore generic keyword matches. Focus on the actual experience of watching the movie.

QUERY MOVIE:
Title: {query_movie.get('title')}
Genres: {query_movie.get('genres')}
Plot: {query_movie.get('overview')}

CANDIDATE MOVIES:
{cand_text}

Output strictly in valid JSON format like this:
{{
  "recommendations": [
    {{"index": <candidate_index_from_above>, "explanation": "<a one-sentence explanation of why this matches the query aesthetic>"}}
  ]
}}
Do not write any other text except the JSON object.
"""
        cleaned_response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            models=configured_models("OPENROUTER_RERANK_MODELS"),
            temperature=0.1,
            timeout_seconds=float(os.getenv("OPENROUTER_RERANK_TIMEOUT_SECONDS", "8")),
            api_key=openrouter_key,
        )
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        data = json.loads(cleaned_response)
        
        reranked_results = []
        for item in data.get("recommendations", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                movie = candidates[idx]
                movie["explanation_text"] = "LLM Reranked: " + str(item.get("explanation", "Highly similar aesthetic vibe."))
                # Keep original similarity score but sort by the LLM's chosen order
                reranked_results.append(movie)
                
        if not reranked_results:
            raise ValueError(f"LLM returned no valid recommendations. Raw text: {cleaned_response[:100]}")
            
        return reranked_results[:n]
    
    def _apply_mmr(self, candidates: list[dict], query_idx: int, n: int, lambda_param: float = 0.7) -> list[dict]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
        
        λ = 0.7 means 70% relevance, 30% diversity
        """
        if len(candidates) <= n:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # First pick: highest score (most relevant)
        selected.append(remaining.pop(0))
        
        while len(selected) < n and remaining:
            best_mmr = -float('inf')
            best_idx = 0
            
            for i, cand in enumerate(remaining):
                # Get candidate index in original DataFrame
                cand_matches = self._movies[self._movies["id"] == cand["id"]].index
                if len(cand_matches) == 0:
                    continue
                cand_idx = cand_matches[0]
                
                relevance = cand["similarity_score"]
                
                # Calculate max similarity to already selected
                max_sim_to_selected = 0
                for sel in selected:
                    sel_matches = self._movies[self._movies["id"] == sel["id"]].index
                    if len(sel_matches) == 0:
                        continue
                    sel_idx = sel_matches[0]
                    
                    # Cosine similarity between candidate and selected
                    # Cast to float32 for precision/speed (essential if vectors are float16)
                    v_cand = self._vectors[cand_idx].astype(np.float32)
                    v_sel = self._vectors[sel_idx].astype(np.float32)
                    sim = float(np.dot(v_cand, v_sel))
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def recommend_by_id(self, movie_id: int, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its TMDB ID.
        
        Args:
            movie_id: TMDB movie ID
            n: Number of recommendations
            
        Returns:
            List of recommended movie dictionaries
        """
        # Find index of the movie
        matches = self._movies[self._movies["id"] == movie_id].index
        if len(matches) == 0:
            return []
        
        movie_idx = matches[0]
        return self.recommend_by_index(movie_idx, n)
    
    def recommend_by_title(self, title: str, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its title.
        
        Args:
            title: Movie title (case-insensitive)
            n: Number of recommendations
            
        Returns:
            List of recommended movie dictionaries
        """
        title_lower = title.lower()
        matches = self._movies[self._movies["title"].str.lower() == title_lower].index
        
        if len(matches) == 0:
            # Try partial match
            matches = self._movies[
                self._movies["title"].str.lower().str.contains(title_lower, na=False)
            ].index
        
        if len(matches) == 0:
            return []
        
        movie_idx = matches[0]
        return self.recommend_by_index(movie_idx, n)

    def ai_search(self, query: str, n: int = 10, fetch_k: int = 80) -> list[dict]:
        """
        Multi-stage AI search for the product API.

        Stages:
        1. Sparse TF-IDF recall for exact names, entities, and cold-start text.
        2. Dense vector recall when online query encoding is enabled.
        3. Metadata, popularity, quality, and behavior scoring.
        4. Optional cross-encoder reranking for high-precision top-window ranking.
        5. MMR diversification so the list is not repetitive.
        """
        if not query or self._movies is None:
            return []

        self.refresh_behavior_features()
        query_intent = parse_query_intent(query)
        fetch_k = max(n, min(fetch_k, len(self._movies)))
        dense_scores: dict[int, float] = {}
        sparse_scores: dict[int, float] = {}
        dense_error = None

        self._ensure_sparse_retrieval_index()
        if self._tfidf_matrix is not None and self._vectorizer is not None:
            from sklearn.metrics.pairwise import cosine_similarity

            query_sparse = self._vectorizer.transform([query])
            sparse_similarities = cosine_similarity(query_sparse, self._tfidf_matrix).ravel()
            sparse_indices = np.argsort(sparse_similarities)[::-1][:fetch_k]
            sparse_scores = {
                int(idx): float(sparse_similarities[int(idx)])
                for idx in sparse_indices
                if sparse_similarities[int(idx)] > 0
            }

        if self._dense_query_enabled() and self._index is not None and self._vectors is not None:
            try:
                encoder = self._get_query_encoder()
                query_embedding = encoder.encode([query], convert_to_numpy=True)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                query_embedding = query_embedding.astype(np.float32)
                distances, indices = self._index.search(query_embedding, fetch_k)
                dense_scores = {
                    int(idx): float(distances[0][rank])
                    for rank, idx in enumerate(indices[0])
                    if idx >= 0 and idx < len(self._movies)
                }
            except Exception as exc:
                dense_error = str(exc)
                logger.warning("Dense query retrieval skipped: %s", exc)

        normalized_sparse = self._normalize_score_map(sparse_scores)
        normalized_dense = self._normalize_score_map(dense_scores)
        candidate_indices = set(normalized_sparse) | set(normalized_dense)
        if not candidate_indices:
            return self.search_movies(query, limit=n)

        alpha = 0.62 if normalized_dense else 0.0
        ranked_candidates = []
        for idx in candidate_indices:
            movie = self._clean_response_record(self._movies.iloc[idx].to_dict())
            sparse_score = normalized_sparse.get(idx, 0.0)
            dense_score = normalized_dense.get(idx, 0.0)
            metadata_score = self._popularity_quality_score(movie)
            behavior_boost, behavior_reasons = self._behavior_boost(movie.get("id"))
            intent_boost, intent_reasons = intent_score(movie, query_intent)
            hybrid_score = (
                alpha * dense_score
                + (1 - alpha) * sparse_score
                + 0.10 * metadata_score
                + behavior_boost
                + intent_boost
            )

            explanation = []
            if dense_score > 0:
                explanation.append("semantic meaning match")
            if sparse_score > 0:
                explanation.append("keyword/entity match")
            if metadata_score > 0.5:
                explanation.append("strong catalog quality signal")
            explanation.extend(intent_reasons)
            explanation.extend(behavior_reasons)
            if dense_error and not normalized_dense:
                explanation.append("dense query model unavailable; sparse fallback used")
            if not explanation:
                explanation.append("best available catalog match")

            movie["similarity_score"] = float(hybrid_score)
            movie["retrieval_stage"] = "hybrid" if normalized_dense else "sparse_metadata"
            movie["retrieval_signals"] = {
                "dense": round(dense_score, 4),
                "sparse": round(sparse_score, 4),
                "metadata": round(metadata_score, 4),
                "behavior": round(behavior_boost, 4),
                "intent": round(intent_boost, 4),
                "intent_features": query_intent,
            }
            movie["explanation"] = explanation[:4]
            movie["explanation_text"] = " | ".join(movie["explanation"])
            ranked_candidates.append(movie)

        ranked_candidates.sort(key=lambda item: item["similarity_score"], reverse=True)

        if self._cross_encoder_enabled() and len(ranked_candidates) > 1:
            try:
                reranker = self._get_cross_encoder()
                rerank_window = ranked_candidates[: min(len(ranked_candidates), int(os.getenv("NOVA_RERANK_WINDOW", "30")))]
                pairs = [
                    [
                        query,
                        f"{item.get('title', '')}. {item.get('genres', '')}. {item.get('overview', '')}",
                    ]
                    for item in rerank_window
                ]
                rerank_scores = reranker.predict(pairs)
                for item, rerank_score in zip(rerank_window, rerank_scores):
                    item["retrieval_signals"]["cross_encoder"] = round(float(rerank_score), 4)
                    item["similarity_score"] = 0.75 * float(item["similarity_score"]) + 0.25 * float(rerank_score)
                    item["retrieval_stage"] = f"{item['retrieval_stage']}_cross_encoder"
                    item["explanation"] = ["neural reranker selected this match"] + item["explanation"][:3]
                    item["explanation_text"] = " | ".join(item["explanation"])
                ranked_candidates = rerank_window + ranked_candidates[len(rerank_window):]
                ranked_candidates.sort(key=lambda item: item["similarity_score"], reverse=True)
            except Exception as exc:
                logger.warning("Cross-encoder reranking skipped: %s", exc)

        ranked_candidates = self._apply_learned_ranker(ranked_candidates)
        return self._apply_query_mmr(ranked_candidates, n=n)

    def semantic_search(self, query: str, n: int = 10) -> list[dict]:
        """
        Search movies by semantic meaning using the SBERT model + FAISS index.
        
        Unlike search_movies() which does text matching on titles,
        this encodes the query with the same model used for embeddings
        and searches the FAISS index directly.
        
        Args:
            query: Natural language query (e.g. "movies about space exploration")
            n: Number of results to return
            
        Returns:
            List of movie dictionaries with similarity scores
        """
        return self.ai_search(query, n=n)


# Lazy-loaded SBERT model for semantic search queries
_sbert_model = None

def _get_sbert_model():
    """Get or load the SBERT model (lazy singleton)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("Loaded SBERT model for semantic search")
    return _sbert_model


# Global singleton instance (lazy loaded)
_recommender: Recommender | None = None
_recommender_lock = Lock()


def get_recommender() -> Recommender:
    """Get or create the global Recommender instance."""
    global _recommender
    if _recommender is None:
        with _recommender_lock:
            if _recommender is None:
                _recommender = Recommender().load()
    return _recommender

