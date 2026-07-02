"""
Recommendation engine.
This isn't just a database wrapper; it loads the TurboVec index and handles the "fuzzy" logic
of making recommendations feel personalized.
"""

from datetime import UTC, datetime
import gc
import hashlib
import logging
import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import pandas as pd

# Import model loader to handle external model downloads

try:
    from backend.serving.serving_tier import resolve_serving_tier as _resolve_serving_tier
except Exception:  # pragma: no cover
    _resolve_serving_tier = None  # type: ignore[assignment]
import contextlib

import torch

from backend.intelligence.knowledge_graph import KnowledgeGraphEngine
from backend.intelligence.semantic_twin import build_semantic_twin

logger = logging.getLogger(__name__)

# Resolve paths relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class ONNXSBERTEncoder:
    """Wrapper class that mimics SentenceTransformer interface but runs on ONNX Runtime for speed."""

    def __init__(self, onnx_path: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        # Set thread constraints for CPU inference optimization
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, os.cpu_count() // 2)
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.enable_cpu_mem_arena = True
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

        self.session = ort.InferenceSession(onnx_path, sess_options=opts)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(self, sentences: list[str] | str, convert_to_numpy: bool = True) -> np.ndarray:
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="np")

        inputs_onnx = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in inputs:
            inputs_onnx["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

        outputs = self.session.run(["last_hidden_state"], inputs_onnx)
        last_hidden = outputs[0]

        attention_mask = inputs_onnx["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
        sum_embeddings = np.sum(last_hidden * input_mask_expanded, 1)
        sum_mask = np.sum(input_mask_expanded, 1)
        sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask

        return embedding


def _render_like_environment() -> bool:
    """Detect constrained PaaS runtimes where the full vector stack can exceed memory."""
    from backend.pipeline.recommender_core import render_like_environment

    return render_like_environment()


def _serving_profile() -> str:
    """Resolve the serving profile for this process."""
    from backend.pipeline.recommender_core import serving_profile

    return serving_profile()


def _low_memory_serving_enabled() -> bool:
    """Return true when serving should avoid loading heavyweight vector artifacts."""
    from backend.pipeline.recommender_core import low_memory_serving_enabled

    return low_memory_serving_enabled()


def _build_rl_state(
    behavior_profile: dict, als_user_embedding: "np.ndarray | None", state_dim: int = 20
) -> "torch.Tensor":
    """Build a fixed-length RL state vector from user behavior profile."""
    from backend.pipeline.recommender_core import build_rl_state

    return build_rl_state(behavior_profile, als_user_embedding, state_dim)


def safe_float(val, default=0.0):
    """Convert val to float safely, returning default on error or non-finite."""
    from backend.pipeline.recommender_core import safe_float as _sf

    return _sf(val, default)


class Recommender:
    """
    The brain of the operation.
    It manages the TurboVec index (for speed) and the metadata (for context).
    """

    def __init__(self):
        self._index: Any | None = None
        self._vectorizer: Any | None = None
        self._item_vectorizer: Any | None = None
        self._movies: pd.DataFrame | None = None
        self._vectors: np.ndarray | None = None
        self._artifact_movie_ids: np.ndarray | None = None
        self._artifact_manifest: dict[str, Any] | None = None
        self._movie_id_to_index: dict[int, int] = {}
        self._movie_records: list[dict] = []
        self._content_text: pd.Series | None = None
        self._tfidf_matrix = None
        self._item_tfidf_matrix = None
        self._query_encoder = None
        self._cross_encoder = None
        self._learned_ranker = None
        self._behavior_features: dict[str, Any] = {}
        self._behavior_features_refreshed_at: datetime | None = None
        self._behavior_features_lock = Lock()  # guards concurrent refresh calls
        self._semantic_twin_cache: dict[int, dict[str, Any]] = {}
        self._search_text_cache: dict[str, pd.Series] = {}
        self._search_text_cache_frame_id: int | None = None
        self._low_memory = _low_memory_serving_enabled()
        self._artifact_status: dict[str, Any] = {"vector_artifacts_ready": False}
        self.multimodal_index = None
        self.kg_engine = KnowledgeGraphEngine()
        self._rl_policy = None
        self._retrieval_pipeline = None
        self._ranking_pipeline = None
        self._reranking_pipeline = None
        self._heavy_models_loaded = False

    def load(self) -> "Recommender":
        """Load recommendation engine in a fast 2-phase sequence.
        Loads fast catalog metadata synchronously so health/basic endpoints are ready under 0.5s,
        then loads heavy models in a background daemon thread to avoid blocking FastAPI startup.
        """
        logger.info("Loading recommendation engine - Phase 1 (Sync Fast Load)...")
        active_tier = self._resolve_active_tier()
        is_tier3 = active_tier == "tier3"
        self._apply_tier3_constraints(is_tier3)
        self._load_vector_artifacts()
        self._load_movie_catalog()
        self._wire_pipelines(is_tier3)

        if self._movies is not None:
            self._artifact_status["movie_count"] = len(self._movies)
        if self._vectors is not None:
            self._artifact_status["vector_count"] = len(self._vectors)
        if self._index is not None:
            self._artifact_status["turbovec_index_count"] = len(self._index)
            self._artifact_status["faiss_index_count"] = len(self._index)
        if self._movies is not None and self._index is not None and self._vectors is not None:
            self._artifact_status["vector_artifacts_ready"] = True

        if not is_tier3:

            def bg_heavy_load():
                try:
                    logger.info("Loading recommendation engine - Phase 2 (Async Heavy Load) started in background...")
                    self._load_ranker_and_behavior()
                    self._load_optional_models()
                    self._heavy_models_loaded = True
                    self._wire_pipelines(is_tier3)
                    logger.info("Loading recommendation engine - Phase 2 (Async Heavy Load) completed.")
                except Exception as exc:
                    logger.exception("Failed to load heavy recommender models in background: %s", exc)

            import threading

            threading.Thread(target=bg_heavy_load, name="recommender-heavy-loader", daemon=True).start()
        else:
            self._load_ranker_and_behavior()
            self._load_optional_models()
            self._heavy_models_loaded = True
            self._wire_pipelines(is_tier3)

        return self

    def _resolve_active_tier(self) -> str:
        """Resolve the active serving tier at startup."""
        try:
            if _resolve_serving_tier is None:
                raise ImportError("backend.serving.serving_tier module is unavailable")
            active_tier, _ = _resolve_serving_tier()
            return active_tier
        except Exception as exc:
            logger.warning("Could not resolve serving tier: %s; defaulting to tier2", exc)
            return "tier2"

    def _apply_tier3_constraints(self, is_tier3: bool) -> None:
        """Apply low-memory constraints for Tier 3 deployments."""
        if is_tier3:
            logger.info("Tier3 serving: skipping neural models, enforcing low-memory mode")
            self._low_memory = True
            current_max = int(os.getenv("NOVA_TFIDF_MAX_FEATURES", "50000"))
            if current_max > 12000:
                os.environ["NOVA_TFIDF_MAX_FEATURES"] = "12000"

    def _load_vector_artifacts(self) -> None:
        """Load TurboVec index, SBERT embeddings, movie ID map, and pipeline manifest."""
        from backend.pipeline.recommender_core import load_vector_artifacts

        load_vector_artifacts(self)

    def _load_movie_catalog(self) -> None:
        """Load movie metadata parquet and build lookup maps."""
        from backend.pipeline.recommender_core import load_movie_catalog

        load_movie_catalog(self)

    def _load_ranker_and_behavior(self) -> None:
        """Load learned ranker, build sparse index, and warm behavior features."""
        from backend.pipeline.recommender_core import load_ranker_and_behavior

        load_ranker_and_behavior(self)

    def _load_optional_models(self) -> None:
        """Load multi-modal index, KG, Two-Tower fine-tune, and RL policy."""
        from backend.pipeline.recommender_core import load_optional_models

        load_optional_models(self)

    def _wire_pipelines(self, is_tier3: bool) -> None:
        """Wire RetrievalPipeline, RankingPipeline, and RerankingPipeline."""
        from backend.pipeline.recommender_core import wire_pipelines

        wire_pipelines(self, is_tier3)

    @property
    def movies(self) -> pd.DataFrame:
        """Get movie metadata DataFrame."""
        if self._movies is None:
            raise RuntimeError("Recommender not loaded. Call load() first.")
        return self._movies

    def refresh_behavior_features(self, force: bool = False) -> dict:
        """Refresh aggregated behavior features used as a light ranking signal."""
        from backend.pipeline.recommender_core import refresh_behavior_features

        return refresh_behavior_features(self, force)

    def _optimize_movie_frame(self) -> None:
        """Reduce the in-memory footprint of the serving catalog."""
        from backend.pipeline.recommender_core import optimize_movie_frame

        optimize_movie_frame(self)

    def _rebuild_lookup_maps(self) -> None:
        """Build row-position lookup maps for hot recommendation paths."""
        self._movie_id_to_index = {}
        self._movie_records = []
        self._franchise_to_indices = {}
        self._collection_to_indices = {}
        if self._movies is None or "id" not in self._movies.columns:
            return
        self._movie_records = self._movies.to_dict(orient="records")
        ids = pd.to_numeric(self._movies["id"], errors="coerce")
        for pos, mid in enumerate(ids):
            if not pd.isna(mid):
                self._movie_id_to_index.setdefault(int(mid), pos)

        # Precompute franchise map
        import re

        def _extract_franchise(title: str) -> str:
            t = title.lower().strip()
            t = re.sub(r"\(\d{4}\)", "", t).strip()
            t = re.sub(r"\s*:\s+.*$", "", t)
            t = re.sub(r"\s+(part|chapter|vol\.?|volume|episode)\s+\S+$", "", t, flags=re.IGNORECASE)
            t = re.sub(r"\s+(i{1,3}|iv|v|vi{0,3}|[2-9]|10|11|12)$", "", t, flags=re.IGNORECASE)
            return t.strip()

        title_col = self._movies["title"] if "title" in self._movies.columns else []
        for pos, title in enumerate(title_col):
            if title:
                franchise = _extract_franchise(str(title))
                if franchise and len(franchise) >= 3:
                    self._franchise_to_indices.setdefault(franchise, []).append(pos)

        # Precompute collection map
        if "belongs_to_collection" in self._movies.columns:
            coll_col = self._movies["belongs_to_collection"]
            for pos, coll in enumerate(coll_col):
                if coll and isinstance(coll, str) and coll.strip():
                    coll_name = coll.strip().lower()
                    self._collection_to_indices.setdefault(coll_name, []).append(pos)

    def _index_for_movie_id(self, movie_id: Any) -> int | None:
        try:
            return self._movie_id_to_index.get(int(movie_id))
        except (TypeError, ValueError):
            return None

    @staticmethod
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
        return value.item() if isinstance(value, np.generic) else value

    @classmethod
    def _clean_response_record(cls, record: dict[str, Any]) -> dict[str, Any]:
        """Normalize a movie record before FastAPI response validation."""
        return {key: cls._clean_response_value(value) for key, value in record.items()}

    def _disable_vector_artifacts(self, reason: str) -> None:
        """Disable vector serving when artifacts violate row-alignment contracts."""
        logger.warning("Disabling TurboVec/SBERT recommendations: %s", reason)
        self._artifact_status.update({"vector_artifacts_ready": False, "disabled_reason": reason})
        self._index = None
        self._vectors = None
        gc.collect()

    @staticmethod
    def _movie_id_sha256(movie_ids: np.ndarray) -> str:
        """Hash the exact ordered movie-id vector used by TurboVec row positions."""
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
            "index_count": contract.get("turbovec_index_size")
            or contract.get("vector_index_size")
            or quality.get("turbovec_index_size")
            or quality.get("vector_index_size")
            or contract.get("f" + "aiss_index_size")
            or quality.get("f" + "aiss_index_size"),
            "id_map_count": contract.get("movie_id_map_rows") or quality.get("movie_id_map_rows"),
            "movie_id_sha256": contract.get("movie_id_sha256") or quality.get("movie_id_sha256"),
        }

    def _validate_vector_artifacts(self) -> None:
        """Validate that vector artifacts are row-aligned with the serving catalog."""
        if self._vectors is None or self._movies is None:
            return
        try:
            allow_legacy = os.getenv("NOVA_ALLOW_LEGACY_ROW_ALIGNED_VECTORS", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            if self._artifact_movie_ids is None and not allow_legacy:
                raise ValueError(
                    "movie_ids.npy is required unless legacy row-aligned vectors are explicitly allowed via NOVA_ALLOW_LEGACY_ROW_ALIGNED_VECTORS"
                )

            from backend.serving.artifact_validator import ArtifactValidator

            validator = ArtifactValidator(MODELS_DIR / "pipeline_manifest.json")
            validator.validate_row_alignment(self._vectors, self._movies)

            # Assert SHA-256 matching hashes for loaded files recorded in the manifest
            for filename in ("sbert_embeddings.npy", "movie_ids.npy"):
                file_path = MODELS_DIR / filename
                if file_path.exists():
                    validator.validate(file_path)
        except Exception as exc:
            logger.warning("Vector artifact validation failed: %s; disabling.", exc)
            self._disable_vector_artifacts(str(exc))

    def _build_sparse_retrieval_index(self) -> None:
        """Build a TF-IDF recall index for hybrid search."""
        from backend.pipeline.recommender_core import build_sparse_retrieval_index

        build_sparse_retrieval_index(self)

    def _ensure_sparse_retrieval_index(self) -> None:
        """Build sparse retrieval lazily when a request path needs AI search."""
        if self._tfidf_matrix is None or self._vectorizer is None:
            self._build_sparse_retrieval_index()

    def _build_item_retrieval_index(self) -> None:
        """Build a plot/genre-focused sparse index for item-to-item recommendations."""
        from backend.pipeline.recommender_core import build_item_retrieval_index

        build_item_retrieval_index(self)

    def _ensure_item_retrieval_index(self) -> None:
        """Build item recommendation sparse retrieval lazily."""
        if self._item_tfidf_matrix is None or self._item_vectorizer is None:
            self._build_item_retrieval_index()

    def _dense_query_enabled(self) -> bool:
        """Return whether online dense query encoding should be attempted."""
        if "NOVA_ENABLE_DENSE_QUERY" in os.environ:
            value = os.environ["NOVA_ENABLE_DENSE_QUERY"].strip().lower()
            return value in {"1", "true", "yes", "on"}
        onnx_path = MODELS_DIR / "sbert_encoder.quant.onnx"
        return onnx_path.exists()

    def _cross_encoder_enabled(self) -> bool:
        """Return whether optional cross-encoder reranking should be attempted."""
        value = os.getenv("NOVA_ENABLE_CROSS_ENCODER", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _llm_rerank_enabled(self) -> bool:
        """Return whether slow OpenRouter LLM reranking should run in the hot path."""
        value = os.getenv("NOVA_ENABLE_LLM_RERANK", "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    def _rerank_with_llm(self, *args, **kwargs) -> list[dict]:
        """Legacy LLM reranking method, preserved for compatibility with test mocks."""
        return []

    def _get_query_encoder(self):
        """Load the query bi-encoder lazily when the deployment opts in."""
        if self._query_encoder is None:
            expected_model = "all-mpnet-base-v2"
            if hasattr(self, "manifest") and self.manifest:
                expected_model = self.manifest.get("model_name", "all-mpnet-base-v2")

            onnx_path = MODELS_DIR / "sbert_encoder.quant.onnx"
            if onnx_path.exists() and "minilm" in expected_model.lower():
                model_name = os.getenv("NOVA_QUERY_ENCODER_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
                self._query_encoder = ONNXSBERTEncoder(str(onnx_path), model_name)
                logger.info("Loaded ONNX query SBERT encoder from %s", onnx_path)
            else:
                from sentence_transformers import SentenceTransformer

                model_name = os.getenv("NOVA_QUERY_ENCODER_MODEL", expected_model)
                self._query_encoder = SentenceTransformer(model_name)
                logger.info("Loaded PyTorch query SBERT encoder: %s", model_name)
        return self._query_encoder

    def _get_cross_encoder(self):
        """Load an optional lightweight cross-encoder reranker lazily."""
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            model_name = os.getenv("NOVA_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._cross_encoder = CrossEncoder(model_name)
            logger.info("Loaded cross-encoder reranker: %s", model_name)
        return self._cross_encoder

    def _popularity_quality_score(self, movie) -> float:
        """Small bounded business score from popularity and quality."""
        from backend.pipeline.recommender_core import popularity_quality_score

        return popularity_quality_score(movie)

    @staticmethod
    @staticmethod
    def _normalize_score_map(scores: dict[int, float]) -> dict[int, float]:
        """Min-max normalize a sparse score map into [0, 1]."""
        if not scores:
            return {}
        vals = np.array(list(scores.values()), dtype=np.float32)
        lo, hi = float(vals.min()), float(vals.max())
        if hi <= lo:
            return dict.fromkeys(scores, 1.0)
        return {k: float((v - lo) / (hi - lo)) for k, v in scores.items()}

    def _genre_set(self, movie: dict[str, Any]) -> set[str]:
        return {part.strip().lower() for part in str(movie.get("genres") or "").split(",") if part.strip()}

    def _semantic_twin_for_index(self, movie_idx: int) -> dict[str, Any]:
        """Build/cache the deterministic semantic twin for a catalog row."""
        if movie_idx not in self._semantic_twin_cache:
            # Evict oldest entry if cache is full (cap at 5000 to prevent memory growth)
            if len(self._semantic_twin_cache) >= 5000:
                with contextlib.suppress(StopIteration):
                    self._semantic_twin_cache.pop(next(iter(self._semantic_twin_cache)))
            self._semantic_twin_cache[movie_idx] = build_semantic_twin(self.get_movie_by_index(movie_idx))
        return self._semantic_twin_cache[movie_idx]

    def get_semantic_twin_by_id(self, movie_id: int) -> dict[str, Any] | None:
        """Return the semantic item twin for a movie ID."""
        movie_idx = self._index_for_movie_id(movie_id)
        if movie_idx is None:
            return None
        return self._semantic_twin_for_index(movie_idx)

    def _semantic_affinity_for_indices(self, query_idx: int, candidate_idx: int) -> dict:
        """Compare query/candidate semantic twins and return serializable signals."""
        from backend.pipeline.recommender_core import semantic_affinity_for_indices

        return semantic_affinity_for_indices(self, query_idx, candidate_idx)

    def _apply_query_mmr(
        self,
        candidates: list[dict],
        n: int,
        lambda_param: float = 0.72,
    ) -> list[dict]:
        """Diversify query search results using candidate vectors where available."""
        from backend.pipeline.recommender_core import apply_query_mmr

        return apply_query_mmr(candidates, n, self._vectors, self._index_for_movie_id, lambda_param)

    def _apply_learned_ranker(
        self,
        candidates: list[dict[str, Any]],
        user_id: int = 0,
        precomputed_ensemble_scores: dict[int, float] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Applies the true APEX ranking pipeline.
        Accepts precomputed_ensemble_scores to avoid redundant ensemble forward passes.
        """
        from backend.pipeline.recommender_core import apply_learned_ranker

        return apply_learned_ranker(candidates, user_id, precomputed_ensemble_scores)

    def _quality_gate_item_recommendations(self, candidates, query_movie, n) -> list:
        """Drop obvious low-quality or genre-drift candidates."""
        from backend.pipeline.recommender_core import quality_gate_item_recommendations

        return quality_gate_item_recommendations(self, candidates, query_movie, n)

    def _learned_ranker_enabled(self) -> bool:
        """Return whether the learned ranker has enough signal to influence serving."""
        from backend.pipeline.recommender_core import learned_ranker_enabled

        return learned_ranker_enabled(self)

    def _behavior_boost(self, movie_id) -> tuple:
        """Return a bounded score nudge from recent product behavior."""
        from backend.pipeline.recommender_core import behavior_boost

        return behavior_boost(self, movie_id)

    @staticmethod
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

    def _genre_affinity_from_profile(self, profile) -> dict:
        """Build genre affinity weights from positive user events."""
        from backend.pipeline.recommender_core import genre_affinity_from_profile

        return genre_affinity_from_profile(self, profile)

    def recommend_for_user_profile(self, profile: dict[str, Any], n: int = 10) -> list[dict[str, Any]]:
        """Blend seed-item, search-intent, genre-affinity, and trending signals for a user."""
        result_limit = max(1, min(int(n), 50))
        if self._retrieval_pipeline is not None and self._ranking_pipeline is not None:
            liked_vecs = [
                self._vectors[idx]
                for event in (profile.get("recent_events") or [])
                if not event.get("negative")
                and event.get("movie_id") is not None
                and (idx := self._index_for_movie_id(int(event["movie_id"]))) is not None
                and self._vectors is not None
            ]
            if not liked_vecs:
                return self._metadata_recommend_by_index(0, n=n)
            query_vector = np.mean(liked_vecs, axis=0).astype(np.float32).reshape(1, -1)
            candidates = self._retrieval_pipeline.retrieve(query_vector, n=min(100, len(self._movies)))
            ranked = self._ranking_pipeline.rank(candidates, user_context={"profile": profile})
            final = self._reranking_pipeline.rerank(ranked, constraints={"profile": profile})
            results = [self._candidate_to_dict(item) for item in final[:result_limit]]
            for r in results:
                if "retrieval_stage" in r:
                    r["retrieval_stage"] = f"personalized_v2_{r['retrieval_stage']}"
            return results
        from backend.pipeline.recommender_core import user_profile_fallback

        return user_profile_fallback(self, profile, result_limit)

    def get_movie_by_id(self, movie_id: int) -> dict | None:
        """Get movie details by TMDB ID."""
        movie_idx = self._index_for_movie_id(movie_id)
        if movie_idx is None:
            return None
        return self.get_movie_by_index(movie_idx)

    def get_movie_by_index(self, idx: int) -> dict:
        """Get movie details by DataFrame index."""
        if self._movie_records and idx < len(self._movie_records):
            return self._clean_response_record(self._movie_records[idx])
        return self._clean_response_record(self._movies.iloc[idx].to_dict())

    def get_all_titles(self, limit: int = 100000) -> list[dict]:
        """Return lightweight movie ID + title list for autocomplete."""
        from backend.pipeline.recommender_core import get_all_titles

        return get_all_titles(self, limit)

    def search_movies(self, query: str, limit: int = 20) -> list[dict]:
        """Search movies by title, overview, and genres. Delegates to pipeline or sparse fallback."""
        if not query or self._movies is None:
            return []
        if self._retrieval_pipeline is not None and self._dense_query_enabled():
            try:
                encoder = self._get_query_encoder()
                query_embedding = encoder.encode([query], convert_to_numpy=True)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                # Query a wider pool to make sure exact matches aren't missed by vector approximations
                candidate_pool_size = max(limit * 3, 50)
                candidates = self._retrieval_pipeline.retrieve(
                    query_embedding.astype(np.float32), n=candidate_pool_size, query_text=query
                )
                results = [self._candidate_to_dict(item) for item in candidates]

                # Boost exact/prefix matches and format the final sorted search results
                q_lower = query.lower().strip()
                import re

                q_norm = re.sub(r"[^a-z0-9]+", " ", q_lower).strip()

                boosted_results = []
                for item in results:
                    title = str(item.get("title", "")).lower().strip()
                    title_norm = re.sub(r"[^a-z0-9]+", " ", title).strip()

                    # Filter out noise for dense search: if no keyword overlap and similarity is very low, skip.
                    overview = str(item.get("overview", "")).lower()
                    genres_val = item.get("genres", "")
                    if isinstance(genres_val, list):
                        genres_str = " ".join(str(g).lower() for g in genres_val)
                    else:
                        genres_str = str(genres_val).lower()

                    has_keyword = q_lower in title or q_lower in overview or q_lower in genres_str
                    if not has_keyword and q_norm:
                        has_keyword = (
                            q_norm in title_norm
                            or q_norm in re.sub(r"[^a-z0-9]+", " ", overview)
                            or q_norm in re.sub(r"[^a-z0-9]+", " ", genres_str)
                        )

                    similarity = float(item.get("similarity_score") or 0.0)
                    if not has_keyword and similarity < 0.22:
                        continue

                    boost = 0.0
                    if title == q_lower or (q_norm and title_norm == q_norm):
                        boost += 100.0  # Exact match priority
                    elif (
                        title.startswith(q_lower + " ")
                        or title.startswith(q_lower + ":")
                        or title.startswith(q_lower + "-")
                    ):
                        boost += 30.0  # Boundary match
                    elif q_norm and title_norm.startswith(q_norm + " "):
                        boost += 30.0
                    elif title.startswith(q_lower):
                        boost += 10.0  # Prefix match
                    elif q_norm and title_norm.startswith(q_norm):
                        boost += 10.0
                    elif q_lower in title:
                        boost += 5.0  # Substring match

                    popularity = float(item.get("popularity") or 0.0)
                    vote_count = float(item.get("vote_count") or 0.0)
                    quality_boost = np.log1p(popularity) * 2.0 + np.log1p(vote_count) * 0.8

                    item["_search_boosted_score"] = similarity + boost + quality_boost
                    boosted_results.append(item)

                boosted_results.sort(key=lambda x: x["_search_boosted_score"], reverse=True)
                for item in boosted_results:
                    item.pop("_search_boosted_score", None)
                return boosted_results[:limit]
            except Exception as exc:
                logger.warning("search_movies pipeline failed (%s); falling back.", type(exc).__name__)
        return self._sparse_search_movies(query, limit)

    def _sparse_search_movies(self, query: str, limit: int = 20) -> list[dict]:
        """TF-IDF + relevance-scoring fallback for search_movies."""
        from backend.pipeline.recommender_core import sparse_search_movies

        return sparse_search_movies(self, query, limit)

    def _metadata_recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
        """Content-based fallback — delegates to recommender_core."""
        from backend.pipeline.recommender_core import metadata_recommend_by_index

        return metadata_recommend_by_index(self, movie_idx, n)

    def kg_recommend(self, movie_id: int, n: int = 10) -> list[dict]:
        """Knowledge Graph multi-hop thematic recommendations."""
        if not self.kg_engine or not self.kg_engine.graph:
            return []
        kg_results = self.kg_engine.find_thematically_similar(movie_id, top_k=n * 2)
        if not kg_results or not self.get_movie_by_id(movie_id):
            return []
        results = []
        for sim_id, score in kg_results:
            movie = self.get_movie_by_id(sim_id)
            if not movie:
                continue
            movie = dict(movie)
            movie["similarity_score"] = float(score) / 10.0
            movie["retrieval_signals"] = {"kg_shared_semantics": score}
            movie["explanation"] = [f"Shares {int(score)} narrative themes/moods"]
            movie["explanation_text"] = movie["explanation"][0]
            results.append(movie)
            if len(results) >= n:
                break
        return results

    def visual_search(self, movie_id: int, n: int = 10) -> list:
        """Multi-Modal similarity search using Text + Visual (Poster) embeddings."""
        from backend.pipeline.recommender_core import visual_search

        return visual_search(self, movie_id, n)

    def _candidate_to_dict(self, item) -> dict:
        """Convert a FinalItem from the pipeline to the response dict shape."""
        from backend.pipeline.recommender_core import candidate_to_dict

        return candidate_to_dict(self, item)

    def recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
        """Get recommendations for a movie by its DataFrame index."""
        if (
            self._retrieval_pipeline is None
            or self._ranking_pipeline is None
            or self._vectors is None
            or self._index is None
        ):
            return self._metadata_recommend_by_index(movie_idx, n=n)
        try:
            import re

            query_vector = self._vectors[movie_idx].reshape(1, -1).astype(np.float32)
            candidates = self._retrieval_pipeline.retrieve(query_vector, n=min(100, len(self._movies)))
            if not candidates:
                return self._metadata_recommend_by_index(movie_idx, n=n)

            # Filter out the query movie itself
            query_movie_id = int(self._movies.iloc[movie_idx]["id"])
            candidates = [c for c in candidates if c.movie_id != query_movie_id]

            seed_row = self._movies.iloc[movie_idx]
            seed_genres_str = str(seed_row.get("genres", "") or "")
            seed_genres = {g.strip().lower() for g in seed_genres_str.split(",") if g.strip()}
            seed_title = str(seed_row.get("title", "") or "")
            seed_title_lower = seed_title.lower()
            seed_collection = seed_row.get("belongs_to_collection", None)
            seed_director = seed_row.get("director", None)

            # ── Franchise/Sequel Injection ──────────────────────────────
            from backend.pipeline.pipeline_types import CandidateItem

            existing_ids = {c.movie_id for c in candidates}
            existing_ids.add(query_movie_id)

            # Fallback initialization if lookup maps are missing or empty
            if not hasattr(self, "_franchise_to_indices") or not self._franchise_to_indices:
                self._rebuild_lookup_maps()

            def _extract_franchise(title: str) -> str:
                t = title.lower().strip()
                t = re.sub(r"\(\d{4}\)", "", t).strip()
                t = re.sub(r"\s*:\s+.*$", "", t)
                t = re.sub(r"\s+(part|chapter|vol\.?|volume|episode)\s+\S+$", "", t, flags=re.IGNORECASE)
                t = re.sub(r"\s+(i{1,3}|iv|v|vi{0,3}|[2-9]|10|11|12)$", "", t, flags=re.IGNORECASE)
                return t.strip()

            seed_franchise = _extract_franchise(seed_title_lower)
            max_retrieval_score = max((c.retrieval_score for c in candidates), default=1.0)

            injected_count = 0
            if seed_franchise and len(seed_franchise) >= 3 and self._movies is not None:
                # 1. Exact franchise match positions from precomputed dict
                exact_positions = (
                    self._franchise_to_indices.get(seed_franchise, []) if hasattr(self, "_franchise_to_indices") else []
                )

                # 2. Vectorized substring match positions
                substring_positions = []
                if len(seed_franchise) >= 4:
                    mask = self._movies["title"].str.lower().str.contains(seed_franchise, na=False)
                    substring_positions = self._movies[mask].index.tolist()

                # Combine: exact matches first
                matching_candidates = []
                seen_positions = set()
                for pos in exact_positions:
                    if pos != movie_idx:
                        matching_candidates.append((pos, True))
                        seen_positions.add(pos)

                for pos in substring_positions:
                    if pos != movie_idx and pos not in seen_positions:
                        matching_candidates.append((pos, False))

                for idx_row, exact_franchise in matching_candidates:
                    if injected_count >= 15:
                        break

                    try:
                        row_title_str = str(self._movies.iloc[idx_row]["title"] or "").lower()
                    except Exception:
                        continue

                    # For substring matches, require at least one shared genre
                    substring_franchise = not exact_franchise
                    if substring_franchise and seed_genres:
                        try:
                            cand_g_str = str(self._movies.iloc[idx_row].get("genres", "") or "")
                            cand_g = {g.strip().lower() for g in cand_g_str.split(",") if g.strip()}
                            if not (seed_genres & cand_g):
                                continue
                        except Exception:
                            continue

                    try:
                        mid = int(self._movies.iloc[idx_row]["id"])
                    except (ValueError, TypeError, KeyError):
                        continue

                    meta = {}
                    for col in ["title", "genres", "release_date", "vote_average", "vote_count", "director"]:
                        if col in self._movies.columns:
                            meta[col] = self._movies.iloc[idx_row].get(col)

                    boost = 1.1 if exact_franchise else 0.95
                    target_score = max_retrieval_score * boost

                    if mid in existing_ids:
                        for c in candidates:
                            if c.movie_id == mid:
                                if target_score > c.retrieval_score:
                                    c.retrieval_score = target_score
                                break
                        continue

                    candidates.append(
                        CandidateItem(
                            movie_id=mid,
                            retrieval_score=target_score,
                            retrieval_source="hybrid",
                            metadata=meta,
                        )
                    )
                    existing_ids.add(mid)
                    injected_count += 1

            if seed_collection and isinstance(seed_collection, str) and seed_collection.strip():
                collection_name = seed_collection.strip().lower()
                coll_positions = (
                    self._collection_to_indices.get(collection_name, [])
                    if hasattr(self, "_collection_to_indices")
                    else []
                )

                for idx_row in coll_positions:
                    if injected_count >= 25:
                        break
                    try:
                        mid = int(self._movies.iloc[idx_row]["id"])
                    except (ValueError, TypeError, KeyError):
                        continue

                    meta = {}
                    for col in ["title", "genres", "release_date", "vote_average", "vote_count", "director"]:
                        if col in self._movies.columns:
                            meta[col] = self._movies.iloc[idx_row].get(col)

                    target_score = max_retrieval_score * 1.15
                    if mid in existing_ids:
                        for c in candidates:
                            if c.movie_id == mid:
                                if target_score > c.retrieval_score:
                                    c.retrieval_score = target_score
                                break
                        continue

                    candidates.append(
                        CandidateItem(
                            movie_id=mid,
                            retrieval_score=target_score,
                            retrieval_source="hybrid",
                            metadata=meta,
                        )
                    )
                    existing_ids.add(mid)
                    injected_count += 1

            # ── Genre-aware score boosting ──────────────────────────────
            if seed_genres:
                for c in candidates:
                    meta = getattr(c, "metadata", {}) or {}
                    cand_genres_str = str(meta.get("genres", "") or "")
                    cand_genres = {g.strip().lower() for g in cand_genres_str.split(",") if g.strip()}
                    cand_title = str(meta.get("title", "") or "").lower()

                    if cand_genres and seed_genres:
                        overlap = len(seed_genres & cand_genres)
                        union = len(seed_genres | cand_genres)
                        jaccard = overlap / union if union > 0 else 0.0

                        if overlap == 0:
                            c.retrieval_score *= 0.5
                        else:
                            c.retrieval_score *= 1.0 + 0.4 * jaccard

                    if cand_title and seed_title_lower and cand_title != seed_title_lower:
                        seed_words = set(seed_title_lower.split()) - {
                            "the",
                            "a",
                            "an",
                            "of",
                            "in",
                            "and",
                            "or",
                            "to",
                            "is",
                            "at",
                        }
                        cand_words = set(cand_title.split()) - {
                            "the",
                            "a",
                            "an",
                            "of",
                            "in",
                            "and",
                            "or",
                            "to",
                            "is",
                            "at",
                        }
                        shared_title_words = seed_words & cand_words
                        if shared_title_words and cand_genres and not (seed_genres & cand_genres):
                            c.retrieval_score *= 0.6

                    # Director boost (e.g. Ridley Scott matching Prometheus for Blade Runner)
                    cand_director = meta.get("director")
                    if (
                        cand_director
                        and seed_director
                        and str(cand_director).strip().lower() == str(seed_director).strip().lower()
                    ):
                        c.retrieval_score *= 1.1

                    # Granular popularity/vote-count penalisation for copycats and niche titles
                    vote_count = float(meta.get("vote_count", 0) or 0)
                    if vote_count < 10:
                        c.retrieval_score *= 0.3
                    elif vote_count < 100:
                        c.retrieval_score *= 0.5
                    elif vote_count < 500:
                        c.retrieval_score *= 0.75
                    elif vote_count < 1000:
                        c.retrieval_score *= 0.90

            candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
            ranked = self._ranking_pipeline.rank(candidates, user_context={})
            final = self._reranking_pipeline.rerank(ranked, constraints={})
            return [self._candidate_to_dict(item) for item in final[:n]]
        except Exception as exc:
            logger.warning("Pipeline delegation failed (%s); falling back.", type(exc).__name__)
            return self._metadata_recommend_by_index(movie_idx, n=n)

    def recommend_by_id(self, movie_id: int, n: int = 10) -> list[dict]:
        """Get recommendations for a movie by its TMDB ID (with 5-min TTL cache)."""
        movie_idx = self._index_for_movie_id(movie_id)
        if movie_idx is None:
            return []
        import time as _time

        if not hasattr(self, "_rec_cache"):
            self._rec_cache: dict = {}
        cached = self._rec_cache.get((movie_id, n))
        if cached is not None and _time.time() - cached[0] < 300:
            return cached[1]
        result = self.recommend_by_index(movie_idx, n)
        if len(self._rec_cache) >= 500:
            with contextlib.suppress(StopIteration):
                self._rec_cache.pop(next(iter(self._rec_cache)))
        self._rec_cache[(movie_id, n)] = (_time.time(), result)
        return result

    def recommend_batch(self, movie_ids: list[int], n: int = 10) -> dict[int, list[dict]]:
        """Batch recommendations — delegates to recommender_core."""
        from backend.pipeline.recommender_core import recommend_batch

        return recommend_batch(self, movie_ids, n)

    def recommend_by_title(self, title: str, n: int = 10) -> list[dict]:
        """Get recommendations for a movie by its title (case-insensitive)."""
        title_lower = title.lower()
        matches = self._movies[self._movies["title"].str.lower() == title_lower].index
        if len(matches) == 0:
            matches = self._movies[self._movies["title"].str.lower().str.contains(title_lower, na=False)].index
        if len(matches) == 0:
            return []
        return self.recommend_by_index(matches[0], n)

    def ai_search(self, query: str, n: int = 10, fetch_k: int = 80) -> list[dict]:
        """Multi-stage AI search. Delegates to pipeline or legacy SBERT+TurboVec+MMR fallback (with 5-min TTL cache)."""
        if not query or self._movies is None:
            return []
        import time as _time

        if not hasattr(self, "_ai_search_cache"):
            self._ai_search_cache = {}
        cached = self._ai_search_cache.get((query, n, fetch_k))
        if cached is not None and _time.time() - cached[0] < 300:
            return cached[1]

        result = self._ai_search_uncached(query, n, fetch_k)
        if len(self._ai_search_cache) >= 500:
            import contextlib

            with contextlib.suppress(StopIteration):
                self._ai_search_cache.pop(next(iter(self._ai_search_cache)))
        self._ai_search_cache[(query, n, fetch_k)] = (_time.time(), result)
        return result

    def _ai_search_uncached(self, query: str, n: int = 10, fetch_k: int = 80) -> list[dict]:
        if self._retrieval_pipeline is not None and self._dense_query_enabled():
            try:
                encoder = self._get_query_encoder()
                query_embedding = encoder.encode([query], convert_to_numpy=True)
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
                candidates = self._retrieval_pipeline.retrieve(
                    query_embedding.astype(np.float32), n=n, query_text=query
                )
                return [self._candidate_to_dict(item) for item in candidates]
            except Exception as exc:
                logger.warning("ai_search pipeline failed (%s); falling back.", type(exc).__name__)
        return self._legacy_ai_search(query, n=n, fetch_k=fetch_k)

    def _legacy_ai_search(self, query: str, n: int = 10, fetch_k: int = 80) -> list[dict]:
        """Legacy AI search fallback — delegates to recommender_core."""
        from backend.pipeline.recommender_core import legacy_ai_search

        return legacy_ai_search(self, query, n, fetch_k)

    def semantic_search(self, query: str, n: int = 10) -> list[dict]:
        """Semantic search via SBERT + TurboVec. Delegates to ai_search."""
        return self.ai_search(query, n=n)


# Lazy-loaded SBERT model
_sbert_model = None


def _get_sbert_model():
    """Get or load the SBERT model (lazy singleton)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer

        _sbert_model = SentenceTransformer("all-mpnet-base-v2")
        logger.info("Loaded SBERT model for semantic search")
    return _sbert_model


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
