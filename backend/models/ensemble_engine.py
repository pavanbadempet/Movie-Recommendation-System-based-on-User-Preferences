import collections
from collections import defaultdict
import json
import logging
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from backend.events import iter_events
from backend.learning.adaptive_router_trainer import AdaptiveRouterTrainer
from backend.models.clifford_recommender import CliffordRecommender
from backend.models.contextual_router import ContextualRouter
from backend.models.diffusion_recommender import LatentDiffusionRecommender
from backend.models.hyperbolic_recommender import HyperbolicRecommender
from backend.models.kan_ranker import KANRanker
from backend.models.lightgcn import LightGCN

# Import all 4 Advanced Research Models
from backend.models.neural_ode_recommender import QuantumFluidRecommender
from backend.models.sasrec import SASRec
from backend.serving.model_health_monitor import ModelHealthMonitor

logger = logging.getLogger(__name__)

GOLD_DIR = Path("data/datalake/gold")
ARTIFACTS_DIR = Path("backend/artifacts")
MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Module-level user event index — built once, avoids full JSONL scan per request
# ---------------------------------------------------------------------------
_user_event_index: dict[str, list[tuple[str, int, str, float | None]]] | None = None
_user_event_index_lock = threading.Lock()
_user_event_index_built_at: float = 0.0
_USER_EVENT_INDEX_TTL = 300.0  # rebuild every 5 minutes


def _get_user_event_index() -> dict[str, list[tuple[str, int, str, float | None]]]:
    """Return a cached user→[(event_ts, movie_id, event_type, rating)] index, rebuilding every 5 minutes."""
    global _user_event_index, _user_event_index_built_at
    now = time.time()
    if _user_event_index is not None and (now - _user_event_index_built_at) < _USER_EVENT_INDEX_TTL:
        return _user_event_index
    with _user_event_index_lock:
        # Double-check after acquiring lock
        if _user_event_index is not None and (time.time() - _user_event_index_built_at) < _USER_EVENT_INDEX_TTL:
            return _user_event_index
        logger.info("Building user event index from Event Store...")
        index = defaultdict(list)
        INTERACTION_TYPES = {"click", "rating", "view"}
        try:
            for event in iter_events():
                uid = event.get("user_id")
                if uid is None:
                    continue
                et = str(event.get("event_type", "")).lower()
                if et not in INTERACTION_TYPES:
                    continue
                mid = event.get("movie_id")
                if mid is None:
                    continue
                try:
                    mid = int(mid)
                except (TypeError, ValueError):
                    continue
                ts = str(event.get("event_ts") or "")
                rating_val = event.get("rating")
                try:
                    rating = float(rating_val) if rating_val is not None else None
                except (TypeError, ValueError):
                    rating = None
                index[str(uid)].append((ts, mid, et, rating))
        except Exception as exc:
            logger.warning("Failed to build user event index: %s", exc)
            index = defaultdict(list)
        _user_event_index = dict(index)
        _user_event_index_built_at = time.time()
        logger.info("User event index built: %d users", len(_user_event_index))
        return _user_event_index


class ApexEnsembleEngine(nn.Module):
    """
    The True, Unified Hybrid Reranking Engine.
    This replaces roleplay code with actual mathematical execution.
    It ensembles the predictions of the 4 most advanced ML paradigms currently researched:
    1. Quantum-Fluid Neural ODEs (Continuous-time wave interference)
    2. Hyperbolic Poincaré Manifolds (Hierarchical tree graphs)
    3. Kolmogorov-Arnold Networks (B-Spline activation functions)
    4. Generative Latent Diffusion (Score-based generative steps)
    """

    def __init__(self, num_users: int = 1000, num_items: int = 10000, emb_dim: int = 16, device: str | None = None):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self._device = device or "cpu"
        self._compiled: dict[str, bool] = {}
        self.has_trained_weights = False
        self._item_id_to_index: dict[int, int] = {}
        # LRU session cache: OrderedDict gives O(1) move-to-end for recency tracking.
        # Capped at _SESSION_CACHE_MAX entries; oldest entry evicted when full.
        self._session_cache: collections.OrderedDict[str, tuple[float, list[int]]] = collections.OrderedDict()
        self._session_cache_lock = threading.Lock()
        self._SESSION_CACHE_MAX = 10_000
        self._weights_lock = threading.Lock()

        logger.info("Initializing the 4 Pillars of the Apex Ensemble...")
        # 1. Quantum Fluid
        self.quantum = QuantumFluidRecommender(max(num_users, 610), max(num_items, 9724), emb_dim)

        # 2. Hyperbolic
        self.hyperbolic = HyperbolicRecommender(max(num_users, 610), max(num_items, 9724), emb_dim)

        # 3. KAN (Kolmogorov-Arnold)
        # KAN expects input size: emb_dim * 2 (user + item concat)
        self.kan = KANRanker(input_dim=emb_dim * 2, hidden_dim=64)

        # 4. Latent Diffusion
        self.diffusion = LatentDiffusionRecommender(emb_dim=emb_dim, num_timesteps=20)

        # 5. SASRec (Transformers)
        # SASRec reserves index 0 for padding internally, so pass the content item count directly.
        self.sasrec = SASRec(num_items=max(num_items, 32660), hidden_dim=128, num_blocks=3, num_heads=4)

        # 6. LightGCN (Graph Networks)
        self.lightgcn = LightGCN(num_users=max(num_users, 1110), num_items=max(num_items, 12966), embedding_dim=emb_dim)

        # 7. Clifford Geometric Algebra (Multivectors)
        self.clifford = CliffordRecommender(
            num_users=max(num_users, 610), num_items=max(num_items, 9724), emb_dim=emb_dim
        )

        # 7. Contextual Router (MoE)
        self.router = ContextualRouter(emb_dim=emb_dim)
        router_path = MODELS_DIR / "contextual_router.pth"
        if router_path.exists():
            try:
                self.router.load_state_dict(torch.load(router_path, map_location=self._device, weights_only=True))
                logger.info("Loaded Contextual Router weights from %s", router_path.name)
            except Exception as e:
                logger.error("Failed to load Contextual Router weights: %s", e)
        else:
            logger.info("No Contextual Router weights found. Router will use random initialization.")

        # Try to load PySpark Gold Embeddings to anchor the models in reality
        self._inject_pyspark_priors()

        # Load trained weights if they exist
        self._load_trained_weights()

        # Load ensemble blend weights (from file or hard-coded defaults)
        self._weights = self._load_weights()

        # Move to device and compile if GPU tier
        if self._device != "cpu":
            self._move_to_device()
        if self._device == "cuda":
            self._try_compile_all()
        else:
            # Apply dynamic quantization on CPU for faster inference
            self.apply_dynamic_quantization()

        self.eval()  # Ensure all models are in inference mode

        # Initialize Rényi Differential Privacy (RDP) Privacy Budget Accountant
        try:
            from backend.privacy.privacy_preserving_ml import PrivacyBudgetAccountant

            self.privacy_accountant = PrivacyBudgetAccountant()
        except Exception as exc:
            logger.warning("Failed to initialize PrivacyBudgetAccountant: %s", exc)
            self.privacy_accountant = None

        # Initialize Adaptive Online Router Trainer
        try:
            self.router_trainer = AdaptiveRouterTrainer(router=self.router)
            logger.info("Adaptive Router Trainer initialized (buffer_capacity=%d)", self.router_trainer.buffer_capacity)
        except Exception as exc:
            logger.warning("Failed to initialize AdaptiveRouterTrainer: %s", exc)
            self.router_trainer = None

        # Initialize Model Health Monitor
        try:
            self.health_monitor = ModelHealthMonitor()
            logger.info("Model Health Monitor initialized")
        except Exception as exc:
            logger.warning("Failed to initialize ModelHealthMonitor: %s", exc)
            self.health_monitor = None

    # ------------------------------------------------------------------ #
    # Ensemble weight loading                                              #
    # ------------------------------------------------------------------ #

    _REQUIRED_WEIGHT_KEYS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion", "clifford")
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "lightgcn": 0.60,
        "quantum": 0.20,
        "sasrec": 0.10,
        "clifford": 0.05,
        "kan": 0.00,
        "hyperbolic": 0.05,
        "diffusion": 0.00,
    }

    def _load_weights(self) -> dict[str, float]:
        """Load ensemble blend weights from ``models/ensemble_weights.json``.

        Returns the hard-coded defaults on any failure (missing file, JSON
        parse error, or missing required key).  If the loaded weights do not
        sum to 1.0 (within 1e-6 tolerance) they are re-normalised before
        being returned.

        This method is called from ``__init__`` and from ``reload_weights``.
        The caller is responsible for holding ``_weights_lock`` when swapping
        ``self._weights`` so that concurrent requests see a consistent dict.
        """
        weights_path = MODELS_DIR / "ensemble_weights.json"

        # --- attempt to load from file ---
        try:
            with open(weights_path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except FileNotFoundError:
            logger.warning(
                "ensemble_weights.json not found at %s; using hard-coded defaults.",
                weights_path,
            )
            return dict(self._DEFAULT_WEIGHTS)
        except json.JSONDecodeError as exc:
            logger.warning(
                "Failed to parse ensemble_weights.json (%s); using hard-coded defaults.",
                exc,
            )
            return dict(self._DEFAULT_WEIGHTS)
        except Exception as exc:
            logger.warning(
                "Could not read ensemble_weights.json (%s); using hard-coded defaults.",
                exc,
            )
            return dict(self._DEFAULT_WEIGHTS)

        # --- validate required keys ---
        missing = [k for k in self._REQUIRED_WEIGHT_KEYS if k not in raw]
        if missing:
            logger.warning(
                "ensemble_weights.json is missing required keys %s; using hard-coded defaults.",
                missing,
            )
            return dict(self._DEFAULT_WEIGHTS)

        weights: dict[str, float] = {k: float(raw[k]) for k in self._REQUIRED_WEIGHT_KEYS}

        # --- log experimental-model warnings ---
        _EXPERIMENTAL_THRESHOLD = 0.01
        for model_name, w in weights.items():
            if 0 < w < _EXPERIMENTAL_THRESHOLD:
                logger.info(
                    "Ensemble model '%s' has weight %.4f (< %.2f) — classified as experimental contribution.",
                    model_name,
                    w,
                    _EXPERIMENTAL_THRESHOLD,
                )

        # --- normalise if sum != 1.0 ---
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Loaded ensemble weights sum to %.8f (expected 1.0); re-normalising.",
                total,
            )
            if total == 0.0:
                # Degenerate case: all zeros — fall back to defaults
                logger.warning("All loaded weights are zero; using hard-coded defaults.")
                return dict(self._DEFAULT_WEIGHTS)
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def reload_weights(self) -> dict[str, float]:
        """Re-read ``models/ensemble_weights.json`` without restarting.

        Thread-safe: acquires ``_weights_lock`` before swapping ``self._weights``.
        Returns the newly loaded (or default) weights dict.
        """
        new_weights = self._load_weights()
        with self._weights_lock:
            self._weights = new_weights
        return dict(self._weights)

    def model_classifications(self) -> dict[str, dict[str, object]]:
        """Return a transparency report for each ensemble model.

        Models with weight >= 0.01 are classified as ``core``; those below
        are ``experimental`` — they contribute to the ensemble but their
        marginal impact is minimal.  This classification is informational
        only and does not affect scoring.
        """
        _EXPERIMENTAL_THRESHOLD = 0.01
        with self._weights_lock:
            weights = dict(self._weights)
        classifications: dict[str, dict[str, object]] = {}
        for name, weight in weights.items():
            tier = "core" if weight >= _EXPERIMENTAL_THRESHOLD else "experimental"
            classifications[name] = {
                "weight": weight,
                "tier": tier,
                "weight_pct": round(weight * 100, 2),
            }
        return classifications

    def _move_to_device(self) -> None:
        """Move all sub-models to self._device."""
        for name in ("quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn", "router"):
            try:
                getattr(self, name).to(self._device)
            except Exception as exc:
                logger.warning("Failed to move %s to %s: %s", name, self._device, exc)

    def _try_compile(self, name: str) -> None:
        """Attempt torch.compile on a single model. Logs warning on failure."""
        try:
            model = getattr(self, name)
            setattr(self, name, torch.compile(model))
            self._compiled[name] = True
            logger.info("torch.compile applied to %s", name)
        except Exception as exc:
            logger.warning("torch.compile failed for %s (%s); running uncompiled", name, exc)
            self._compiled[name] = False

    def _try_compile_all(self) -> None:
        """Attempt torch.compile on all models not yet compiled."""
        for name in ("quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn"):
            if not self._compiled.get(name, False):
                self._try_compile(name)

    def apply_dynamic_quantization(self) -> None:
        """
        Apply dynamic INT8 quantization to linear layers in all ensemble models.
        Reduces model size by ~4x and speeds up CPU inference by 2-3x with ~1% accuracy loss.
        Only applied on CPU (quantization is not beneficial on GPU).
        """
        if self._device != "cpu":
            logger.info("Skipping quantization — device is %s (quantization is CPU-only)", self._device)
            return
        try:
            import torch.quantization as tq

            quantizable = ("kan", "diffusion")  # Models with Linear layers that benefit most
            for name in quantizable:
                try:
                    model = getattr(self, name)
                    quantized = tq.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
                    setattr(self, name, quantized)
                    logger.info("Dynamic INT8 quantization applied to %s", name)
                except Exception as exc:
                    logger.warning("Quantization failed for %s: %s", name, exc)
        except Exception as exc:
            logger.warning("Dynamic quantization unavailable: %s", exc)

    def _get_item_embedding_cache(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return pre-computed LightGCN and Hyperbolic item embeddings (cached)."""
        if not hasattr(self, "_item_emb_cache") or self._item_emb_cache is None:
            with torch.no_grad():
                self._item_emb_cache = (
                    self.lightgcn.item_embedding.weight.detach(),  # [num_items, emb_dim]
                    self.hyperbolic.item_embedding.weight.detach(),  # [num_items, emb_dim]
                )
            logger.info("Pre-computed item embedding cache: %d items", self.num_items)
        return self._item_emb_cache

    def get_item_embedding(self, movie_id: int) -> torch.Tensor | None:
        """Return the exact serving embedding for a catalog movie ID."""
        try:
            item_index = self._item_id_to_index.get(int(movie_id))
        except (TypeError, ValueError):
            return None
        if item_index is None:
            return None

        weight = self.lightgcn.item_embedding.weight
        if item_index < 0 or item_index >= weight.shape[0]:
            return None
        with torch.no_grad():
            index = torch.tensor([item_index], dtype=torch.long, device=weight.device)
            return self.lightgcn.item_embedding(index).detach().clone()

    def _inject_pyspark_priors(self):
        """Loads real PySpark ALS embeddings from the Delta Lake if available."""
        user_emb_path = GOLD_DIR / "model_user_embeddings"
        item_emb_path = GOLD_DIR / "model_item_embeddings"

        if user_emb_path.exists() and item_emb_path.exists():
            try:
                import polars as pl

                def load_embeddings_by_dim(emb_path, target_dim):
                    files = list(emb_path.glob("*.parquet")) if emb_path.is_dir() else [emb_path]
                    dfs = []
                    for f in files:
                        df = pl.read_parquet(f)
                        if "features" in df.columns and len(df) > 0:
                            first_val = df["features"][0]
                            if first_val is not None and len(first_val) == target_dim:
                                if "id" in df.columns:
                                    df = df.with_columns(pl.col("id").cast(pl.Int64))
                                df = df.with_columns(pl.col("features").cast(pl.List(pl.Float32)))
                                dfs.append(df)
                    if not dfs:
                        raise ValueError(f"No embeddings found with dimension {target_dim} in {emb_path}")
                    return pl.concat(dfs, how="vertical").sort("id")

                users_df = load_embeddings_by_dim(user_emb_path, self.emb_dim)
                items_df = load_embeddings_by_dim(item_emb_path, self.emb_dim)
                self._item_id_to_index = {
                    int(movie_id): index for index, movie_id in enumerate(items_df["id"].to_list())
                }

                user_tensor = torch.tensor(np.vstack(users_df["features"].to_list()), dtype=torch.float32)
                item_tensor = torch.tensor(np.vstack(items_df["features"].to_list()), dtype=torch.float32)

                # Update dimensions to match the actual data
                self.num_users = user_tensor.shape[0]
                self.num_items = item_tensor.shape[0]

                # Inject into Quantum
                self.quantum.user_embedding.amplitude.weight.data[: self.num_users] = user_tensor
                self.quantum.item_embedding.amplitude.weight.data[: self.num_items] = item_tensor

                # Inject into Hyperbolic
                self.hyperbolic.user_embedding.weight.data[: self.num_users] = user_tensor
                self.hyperbolic.item_embedding.weight.data[: self.num_items] = item_tensor

                # Inject into LightGCN
                self.lightgcn.user_embedding.weight.data[: self.num_users] = user_tensor
                self.lightgcn.item_embedding.weight.data[: self.num_items] = item_tensor

                # Inject into SASRec
                if self.sasrec.item_emb.weight.data.shape[1] == item_tensor.shape[1]:
                    limit = min(self.num_items, self.sasrec.item_emb.weight.data.shape[0] - 1)
                    self.sasrec.item_emb.weight.data[1 : limit + 1] = item_tensor[:limit]

                # Diffusion and KAN do not maintain separate embeddings; they dynamically
                # route through the Hyperbolic/Quantum priors during the forward pass.

                self.has_trained_weights = True
                logger.info(
                    "Successfully injected PySpark embeddings (%d users, %d items) into all models.",
                    self.num_users,
                    self.num_items,
                )
            except Exception as e:
                logger.error("Failed to inject PySpark priors: %s", e)
        else:
            logger.warning("PySpark Gold embeddings not found. Models will use their initialized mathematical priors.")

    def _load_trained_weights(self):
        """Loads trained weights for the neural models if available."""
        models_to_load = {
            "quantum": MODELS_DIR / "quantum_fluid.pth",
            "hyperbolic": MODELS_DIR / "hyperbolic.pth",
            "kan": MODELS_DIR / "kan_ranker.pth",
            "sasrec": MODELS_DIR / "sasrec.pth",
            "clifford": MODELS_DIR / "clifford.pth",
            # Prefer IPS-debiased weights when available; fall back to standard weights
            "lightgcn": MODELS_DIR / "lightgcn_ips.pth"
            if (MODELS_DIR / "lightgcn_ips.pth").exists()
            else MODELS_DIR / "lightgcn.pth",
        }

        loaded = 0
        for name, path in models_to_load.items():
            if path.exists():
                try:
                    model_attr = getattr(self, name)
                    model_attr.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                    ips_tag = " (IPS-debiased)" if name == "lightgcn" and "ips" in path.name else ""
                    logger.info("Loaded %s weights from %s%s", name, path.name, ips_tag)
                    loaded += 1
                except Exception as e:
                    logger.error("Failed to load %s weights: %s", name, e)

        if loaded > 0:
            logger.info("Successfully loaded trained weights for %d models.", loaded)
            self.has_trained_weights = True
        else:
            logger.warning("No trained weights found. Models will use random initialization.")

    def _get_session_sequence(
        self,
        user_id: int,
        override: list[int] | None = None,
    ) -> torch.LongTensor:
        """Return a [1, 50] LongTensor of item indices for SASRec.

        Priority chain:
          1. ``override`` — if provided, use directly (skip all lookups).
          2. Feature Store cache — if ``_session_cache`` has a fresh entry (< 60 s), use it.
          3. Event Store query — iterate ``iter_events()``, filter by user/type, sort, take 50.
          4. Zero fallback — cold-start users or any Event Store exception.
        """
        SEQ_LEN = 50
        CACHE_TTL = 60.0
        INTERACTION_TYPES = {"click", "rating", "view"}  # noqa: F841 — used by _get_user_event_index

        # --- 1. Override ---
        if override is not None:
            safe_ids = [item_id % self.num_items for item_id in override[-SEQ_LEN:]]
            padded = [0] * (SEQ_LEN - len(safe_ids)) + safe_ids
            return torch.tensor([padded], dtype=torch.long)

        cache_key = str(user_id)

        # --- 2. Feature Store cache ---
        with self._session_cache_lock:
            cached = self._session_cache.get(cache_key)
            if cached is not None:
                self._session_cache.move_to_end(cache_key)
        if cached is not None:
            cached_ts, cached_seq = cached
            if time.time() - cached_ts < CACHE_TTL:
                padded = [0] * (SEQ_LEN - len(cached_seq)) + cached_seq
                return torch.tensor([padded], dtype=torch.long)

        # --- 3. Event Store query — O(1) lookup via pre-built index ---
        try:
            # First try the real-time in-memory index (millisecond latency)
            try:
                from backend.serving.realtime_feature_updater import get_user_session_sequence

                rt_seq = get_user_session_sequence(user_id, max_len=SEQ_LEN)
                if rt_seq:
                    safe_ids = [item_id % self.num_items for item_id in rt_seq]
                    with self._session_cache_lock:
                        self._session_cache[cache_key] = (time.time(), safe_ids)
                        self._session_cache.move_to_end(cache_key)
                        if len(self._session_cache) > self._SESSION_CACHE_MAX:
                            self._session_cache.popitem(last=False)
                    padded = [0] * (SEQ_LEN - len(safe_ids)) + safe_ids
                    return torch.tensor([padded], dtype=torch.long)
            except Exception as exc:
                logger.debug(
                    "Real-time session lookup failed for user %s; using background index: %s",
                    user_id,
                    exc,
                )

            index = _get_user_event_index()
            interactions = index.get(cache_key, [])

            # Sort ascending by timestamp, take the 50 most recent
            recent = sorted(interactions, key=lambda x: x[0])[-SEQ_LEN:]

            if not recent:
                return torch.zeros((1, SEQ_LEN), dtype=torch.long)

            safe_ids = [x[1] % self.num_items for x in recent]

            # Cache the result with proper LRU eviction
            with self._session_cache_lock:
                self._session_cache[cache_key] = (time.time(), safe_ids)
                self._session_cache.move_to_end(cache_key)
                if len(self._session_cache) > self._SESSION_CACHE_MAX:
                    self._session_cache.popitem(last=False)

            padded = [0] * (SEQ_LEN - len(safe_ids)) + safe_ids
            return torch.tensor([padded], dtype=torch.long)

        except Exception as exc:
            logger.warning(
                "Event Store query failed for user %s; falling back to zero sequence: %s",
                user_id,
                exc,
            )
            return torch.zeros((1, SEQ_LEN), dtype=torch.long)

    def predict_ensemble(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        session_sequence: list[int] | None = None,
        user_emb_override: "torch.Tensor | None" = None,
        use_router: bool = True,
        router_k: int = 2,
    ) -> dict[int, float]:
        """
        Executes a real forward pass across the architectures and returns
        the fused ensemble score for each candidate.
        Uses Contextual Router (Mixture of Experts) to run only top-k models when enabled.
        Uses ONNX Runtime when available (Tier 2) for 2-5x faster CPU inference.

        Parameters
        ----------
        user_id:
            Integer user identifier.
        candidate_item_ids:
            List of item IDs to score.
        session_sequence:
            Optional pre-built session sequence override (bypasses event-store lookup).
        user_emb_override:
            Optional pre-computed (e.g. DP-noised) user embedding tensor ``[emb_dim]``.
            When provided, the LightGCN user embedding table is NOT read — this tensor
            is used instead, ensuring the shared embedding table is never mutated.
            Ignored by the ONNX path (which reads embeddings internally).
        use_router:
            Whether to use the Contextual Router (Mixture of Experts) to prune model execution.
        router_k:
            Number of top models to dynamically execute.
        """
        if not candidate_item_ids:
            return {}

        # Check for cold start user context
        is_cold_start = user_id % self.num_users == 0
        if not is_cold_start:
            try:
                events_list = _get_user_event_index().get(str(user_id), [])
                if not events_list:
                    if session_sequence is None or len(session_sequence) == 0 or all(s == 0 for s in session_sequence):
                        is_cold_start = True
            except Exception:
                is_cold_start = True

        if is_cold_start:
            use_router = False

        # Try ONNX-accelerated path first (Tier 2) when no user embedding override is requested and router is disabled.
        # This ensures DP overrides (which are processed in PyTorch) and MoE routing are respected.
        if user_emb_override is None and not use_router:
            try:
                from backend.serving.onnx_engine import get_onnx_engine

                onnx = get_onnx_engine()
                if onnx.has_any_onnx_models():
                    return self._predict_ensemble_onnx(
                        user_id, candidate_item_ids, onnx, session_sequence, is_cold_start=is_cold_start
                    )
            except Exception as exc:
                logger.debug("ONNX ensemble path unavailable; falling back to PyTorch: %s", exc)

        return self._predict_ensemble_pytorch(
            user_id,
            candidate_item_ids,
            session_sequence,
            user_emb_override,
            use_router=use_router,
            router_k=router_k,
            is_cold_start=is_cold_start,
        )

    def _predict_ensemble_onnx(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        onnx,
        session_sequence: list[int] | None = None,
        is_cold_start: bool = False,
    ) -> dict[int, float]:
        """ONNX Runtime inference path — bypasses Python GIL for 2-5x speedup."""
        import numpy as _np

        safe_user_id = user_id % self.num_users
        safe_item_ids = [item_id % self.num_items for item_id in candidate_item_ids]
        u_arr = _np.array([safe_user_id], dtype=_np.int64)
        i_arr = _np.array(safe_item_ids, dtype=_np.int64)

        scores_list = []

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-6:
                return _np.full_like(arr, 0.5)
            return (arr - mn) / (mx - mn)

        if is_cold_start:
            w = {
                "quantum": 0.40,
                "hyperbolic": 0.40,
                "diffusion": 0.20,
                "lightgcn": 0.00,
                "sasrec": 0.00,
                "kan": 0.00,
                "clifford": 0.00,
            }
        else:
            with self._weights_lock:
                w = dict(self._weights)

            # Use contextual weights if available (context-dependent ensemble blending)
            try:
                from backend.models.neural_weight_optimizer import get_contextual_weights

                contextual_w = get_contextual_weights({})
                if contextual_w:
                    w = contextual_w
            except Exception as exc:
                logger.debug("Contextual weights unavailable; using static ensemble weights: %s", exc)

        # LightGCN via ONNX
        try:
            lgcn_s = onnx.predict_lightgcn(u_arr, i_arr)
            scores_list.append(("lightgcn", _norm(lgcn_s.flatten())))
        except Exception:
            scores_list.append(("lightgcn", _np.full(len(safe_item_ids), 0.5)))

        # Quantum via ONNX
        try:
            q_s = onnx.predict_quantum(u_arr, i_arr)
            scores_list.append(("quantum", _norm(q_s.flatten())))
        except Exception:
            scores_list.append(("quantum", _np.full(len(safe_item_ids), 0.5)))

        # Hyperbolic via ONNX
        try:
            h_s = onnx.predict_hyperbolic(u_arr, i_arr)
            scores_list.append(("hyperbolic", _norm(h_s.flatten())))
        except Exception:
            scores_list.append(("hyperbolic", _np.full(len(safe_item_ids), 0.5)))

        # SASRec via ONNX
        try:
            seq = self._get_session_sequence(user_id, override=session_sequence).numpy()
            cand = _np.array([safe_item_ids], dtype=_np.int64)
            sar_s = onnx.predict_sasrec(seq, cand)
            scores_list.append(("sasrec", _norm(sar_s.flatten())))
        except Exception:
            scores_list.append(("sasrec", _np.full(len(safe_item_ids), 0.5)))

        # KAN via ONNX (needs embeddings — use LightGCN embeddings as proxy)
        try:
            with torch.no_grad():
                u_emb = (
                    self.lightgcn.user_embedding(torch.tensor([safe_user_id])).expand(len(safe_item_ids), -1).numpy()
                )
                i_emb = self.lightgcn.item_embedding(torch.tensor(safe_item_ids)).numpy()
            k_s = onnx.predict_kan(u_emb, i_emb)
            scores_list.append(("kan", _norm(k_s.flatten())))
        except Exception:
            scores_list.append(("kan", _np.full(len(safe_item_ids), 0.5)))

        # Diffusion via ONNX
        try:
            with torch.no_grad():
                u_emb_d = (
                    self.lightgcn.user_embedding(torch.tensor([safe_user_id])).expand(len(safe_item_ids), -1).numpy()
                )
                i_emb_d = self.lightgcn.item_embedding(torch.tensor(safe_item_ids)).numpy()
            t_arr = _np.full((len(safe_item_ids), 1), 0.5, dtype=_np.float32)
            d_noise = onnx.predict_diffusion(i_emb_d, t_arr, u_emb_d)
            d_s = 1.0 / (1.0 + _np.linalg.norm(d_noise, axis=1))
            scores_list.append(("diffusion", _norm(d_s)))
        except Exception:
            scores_list.append(("diffusion", _np.full(len(safe_item_ids), 0.5)))

        # Clifford via ONNX (fallback since ONNX isn't exported yet)
        scores_list.append(("clifford", _np.full(len(safe_item_ids), 0.5)))

        # Blend
        key_map = {
            "lightgcn": "lightgcn",
            "quantum": "quantum",
            "sasrec": "sasrec",
            "kan": "kan",
            "hyperbolic": "hyperbolic",
            "diffusion": "diffusion",
            "clifford": "clifford",
        }
        final = _np.zeros(len(safe_item_ids), dtype=_np.float32)
        for name, s in scores_list:
            final += s * w.get(key_map.get(name, name), 0.0)

        return {orig_id: float(final[idx]) for idx, orig_id in enumerate(candidate_item_ids)}

    def _predict_ensemble_pytorch(
        self,
        user_id: int,
        candidate_item_ids: list[int],
        session_sequence: list[int] | None = None,
        user_emb_override: "torch.Tensor | None" = None,
        use_router: bool = True,
        router_k: int = 2,
        is_cold_start: bool = False,
    ) -> dict[int, float]:
        if not candidate_item_ids:
            return {}

        # Check and deduct privacy budget
        budget_allowed = True
        if self.privacy_accountant is not None:
            import os

            # If user_emb_override is provided, we deduct the budget.
            if user_emb_override is not None:
                dp_epsilon = float(os.getenv("APEX_DP_EPSILON", "1.0"))
                budget_allowed, remaining = self.privacy_accountant.check_and_deduct_budget(
                    user_id=user_id, request_epsilon=dp_epsilon, request_delta=1e-5, mechanism="gaussian"
                )
                if not budget_allowed:
                    logger.warning(
                        "Privacy budget exhausted for user %d (remaining budget: %.4f). Falling back to safe zero/dummy representations.",
                        user_id,
                        remaining,
                    )

        if not budget_allowed:
            # Fallback user ID to a generic user (e.g. 0)
            safe_user_id = 0
            if user_emb_override is not None:
                user_emb_override = torch.zeros((self.emb_dim,), dtype=torch.float32).to(self._device)
            session_sequence = [0] * 50
        else:
            safe_user_id = user_id % self.num_users

        # Ensure ID bounds are respected (hash to max size if unknown)
        safe_item_ids = [item_id % self.num_items for item_id in candidate_item_ids]

        u_tensor = torch.tensor([safe_user_id], dtype=torch.long)
        i_tensor = torch.tensor(safe_item_ids, dtype=torch.long)

        # Pre-compute shared embeddings (used by KAN, Diffusion, LightGCN)
        # Uses cached item embeddings to avoid recomputing on every request
        lgcn_all_items, hyp_all_items = self._get_item_embedding_cache()
        with torch.no_grad():
            u_emb = self.hyperbolic.user_embedding(u_tensor).expand(len(i_tensor), -1)
            i_emb = hyp_all_items[i_tensor]  # lookup from cache
            # If a DP-noised override is provided, use it instead of reading the shared table.
            # This ensures the shared embedding table is never mutated under concurrent requests.
            if user_emb_override is not None:
                lgcn_u_emb = user_emb_override.unsqueeze(0).to(self._device).expand(len(i_tensor), -1)
            else:
                lgcn_u_emb = self.lightgcn.user_embedding(u_tensor).expand(len(i_tensor), -1)
            lgcn_i_emb = lgcn_all_items[i_tensor]  # lookup from cache
            simulated_seq = self._get_session_sequence(safe_user_id, override=session_sequence)

        # Enrich user embedding with attention over session history
        if budget_allowed:
            try:
                from backend.models.attention_user_model import (
                    build_attended_user_embedding,
                    get_user_attention_encoder,
                )

                seq_list = simulated_seq.squeeze().tolist()
                if isinstance(seq_list, int):
                    seq_list = [seq_list]
                seq_list = [s for s in seq_list if s > 0]  # Remove padding
                if seq_list:
                    encoder = get_user_attention_encoder(emb_dim=self.emb_dim)
                    attended = build_attended_user_embedding(user_id, lgcn_all_items, seq_list, encoder)
                    if attended is not None:
                        # Blend attended embedding with base user embedding
                        attended_expanded = attended.expand(len(i_tensor), -1)
                        lgcn_u_emb = 0.7 * lgcn_u_emb + 0.3 * attended_expanded
            except Exception as exc:
                logger.debug("Attention user embedding unavailable; using base embedding: %s", exc)

        def _norm(t):
            t_min, t_max = t.min(), t.max()
            if t_max - t_min < 1e-6:
                return torch.ones_like(t) * 0.5
            return (t - t_min) / (t_max - t_min)

        # Define per-model scoring functions for parallel execution
        def score_quantum():
            with torch.no_grad():
                s = self.quantum.predict(u_tensor, i_tensor, time_delta=1.0).squeeze()
                return _norm(s.unsqueeze(0) if s.dim() == 0 else s)

        def score_hyperbolic():
            with torch.no_grad():
                s = -self.hyperbolic.predict(u_tensor.expand_as(i_tensor), i_tensor)
                return _norm(s)

        def score_kan():
            with torch.no_grad():
                s = self.kan(u_emb, i_emb).squeeze()
                return _norm(s.unsqueeze(0) if s.dim() == 0 else s)

        def score_diffusion():
            with torch.no_grad():
                t_val = torch.ones(len(i_tensor), 1) * 0.5
                noise = self.diffusion.denoiser(i_emb, t_val, u_emb)
                return _norm(1.0 / (1.0 + torch.norm(noise, dim=-1)))

        def score_sasrec():
            with torch.no_grad():
                s = self.sasrec.predict(simulated_seq, i_tensor.unsqueeze(0)).squeeze()
                return _norm(s.unsqueeze(0) if s.dim() == 0 else s)

        def score_lightgcn():
            with torch.no_grad():
                return _norm((lgcn_u_emb * lgcn_i_emb).sum(dim=1))

        def score_clifford():
            with torch.no_grad():
                s = self.clifford.predict(u_tensor, i_tensor).squeeze()
                return _norm(s.unsqueeze(0) if s.dim() == 0 else s)

        scores = {}
        try:
            from concurrent.futures import as_completed

            model_fns = {
                "quantum": score_quantum,
                "hyperbolic": score_hyperbolic,
                "kan": score_kan,
                "diffusion": score_diffusion,
                "sasrec": score_sasrec,
                "lightgcn": score_lightgcn,
                "clifford": score_clifford,
            }

            # Run contextual router (Mixture of Experts) to determine active models
            selected_models = None
            routing_weights = None
            _router_user_state = None  # Sentinel for router trainer feedback
            if use_router:
                try:
                    import os

                    from backend.models.contextual_router import build_user_state

                    # 1. Get interaction count for user
                    try:
                        interaction_count = len(_get_user_event_index().get(str(safe_user_id), []))
                    except Exception:
                        interaction_count = 0

                    # 2. Get base user embedding
                    base_u_emb = lgcn_u_emb[0].detach()

                    # 3. Build user state vector [emb_dim + 4]
                    user_state = build_user_state(
                        user_id=safe_user_id,
                        user_emb=base_u_emb,
                        session_seq=simulated_seq,
                        item_embeddings=lgcn_all_items,
                        interaction_count=interaction_count,
                        inference_energy=float(os.getenv("APEX_INFERENCE_ENERGY", "0.5")),
                    )

                    # 4. Query router
                    selected_models, routing_weights = self.router.route(user_state.to(self._device), k=router_k)
                    _router_user_state = user_state  # Save for router trainer feedback
                except Exception as router_exc:
                    logger.warning(
                        "Dynamic router execution failed; falling back to full ensemble. Error: %s", router_exc
                    )
                    selected_models = None
                    routing_weights = None

            if is_cold_start:
                w = {
                    "quantum": 0.40,
                    "hyperbolic": 0.40,
                    "diffusion": 0.20,
                    "lightgcn": 0.00,
                    "sasrec": 0.00,
                    "kan": 0.00,
                    "clifford": 0.00,
                }
            elif selected_models is not None and routing_weights is not None:
                # Use normalized routing weights from router
                w = {name: routing_weights[idx].item() for idx, name in enumerate(selected_models)}
            else:
                # Fallback to static or contextual weights
                with self._weights_lock:
                    w = dict(self._weights)

                # Use contextual weights when available (context-dependent ensemble blending)
                try:
                    from backend.models.neural_weight_optimizer import get_contextual_weights

                    # Build real behavior profile from event index cache
                    events = _get_user_event_index().get(str(safe_user_id), [])
                    total_ratings = sum(1 for e in events if e[2] == "rating")
                    ratings_list = [float(e[3]) for e in events if e[2] == "rating" and e[3] is not None]
                    avg_rating = sum(ratings_list) / len(ratings_list) if ratings_list else 3.5
                    click_count = sum(1 for e in events if e[2] == "click")
                    view_count = sum(1 for e in events if e[2] == "view")

                    profile = {
                        "total_ratings": total_ratings,
                        "avg_rating": avg_rating,
                        "click_count": click_count,
                        "view_count": view_count,
                    }

                    contextual_w = get_contextual_weights(
                        behavior_profile=profile,
                        als_user_embedding=base_u_emb.cpu().numpy() if base_u_emb is not None else None,
                    )
                    if contextual_w:
                        w = contextual_w
                except Exception as exc:
                    logger.debug("Contextual weights unavailable; using static weights: %s", exc)

            # Filter functions based on router selections or non-zero weights
            if selected_models is not None:
                active_fns = {m: model_fns[m] for m in selected_models if m in model_fns}
            else:
                active_fns = {m: fn for m, fn in model_fns.items() if w.get(m, 0.0) > 0.0}

            # Apply health monitor filtering — exclude degraded models
            if self.health_monitor is not None:
                healthy_models = set(self.health_monitor.get_active_models())
                active_fns = {m: fn for m, fn in active_fns.items() if m in healthy_models}
                if not active_fns:
                    # Safety: if health filter removed everything, fall back to non-zero weight active ones
                    active_fns = {m: fn for m, fn in model_fns.items() if w.get(m, 0.0) > 0.0}

            # Run active models in parallel using the module-level thread pool
            executor = _get_model_thread_pool()
            results: dict[str, torch.Tensor] = {}
            model_latencies: dict[str, float] = {}
            model_successes: dict[str, bool] = {}
            futures = {executor.submit(fn): (name, time.perf_counter()) for name, fn in active_fns.items()}
            for future in as_completed(futures):
                name, start_t = futures[future]
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                model_latencies[name] = elapsed_ms
                try:
                    results[name] = future.result()
                    model_successes[name] = True
                except Exception as exc:
                    logger.warning("Model %s failed: %s; using 0.5 fallback", name, exc)
                    results[name] = torch.ones(len(safe_item_ids), device=self._device) * 0.5
                    model_successes[name] = False

            # -----------------------------------------------------------------
            # Uncertainty-gated ensemble blending
            # When models strongly disagree on an item, reduce its overall
            # score proportionally — high disagreement signals low confidence.
            # -----------------------------------------------------------------
            try:
                active_names = list(results.keys())
                stacked = torch.stack(
                    [results[m] * w.get(m, 0.0) for m in active_names], dim=0
                )  # [num_active, num_items]

                # Weighted mean and variance across active models
                w_tensor = torch.tensor([w.get(m, 0.0) for m in active_names], dtype=torch.float32, device=self._device)
                w_tensor = w_tensor / (w_tensor.sum() + 1e-8)
                w_expanded = w_tensor.unsqueeze(1)
                weighted_mean = (stacked * w_expanded).sum(dim=0, keepdim=True)

                per_item_uncertainty = ((stacked - weighted_mean) ** 2 * w_expanded).sum(
                    dim=0
                )  # [num_items] — weighted variance per item

                # Confidence gate: items with high model disagreement get penalised
                max_var = per_item_uncertainty.max()
                if max_var > 1e-6:
                    normalised_unc = per_item_uncertainty / max_var
                    confidence_gate = 1.0 - 0.5 * normalised_unc  # [0.5, 1.0]
                else:
                    confidence_gate = torch.ones(len(safe_item_ids), device=self._device)
            except Exception as exc:
                logger.debug("Uncertainty gating failed; skipping: %s", exc)
                confidence_gate = torch.ones(len(safe_item_ids), device=self._device)

            blend_mode = os.getenv("APEX_ENSEMBLE_BLEND_MODE", "linear").lower()
            if blend_mode == "geometric":
                # Normalize active weights to sum to 1
                active_weights = torch.tensor([w.get(m, 0.0) for m in active_names], device=self._device)
                w_sum = active_weights.sum()
                if w_sum > 1e-8:
                    active_weights = active_weights / w_sum
                else:
                    active_weights = torch.ones_like(active_weights) / len(active_names)

                log_scores = torch.zeros(len(safe_item_ids), device=self._device)
                eps = 1e-6
                for idx, m in enumerate(active_names):
                    log_scores += active_weights[idx] * torch.log(results[m] + eps)
                final_scores = torch.exp(log_scores)
            else:
                final_scores = torch.zeros(len(safe_item_ids), device=self._device)
                for m in active_names:
                    final_scores += results[m] * w.get(m, 0.0)

            final_scores = final_scores * confidence_gate  # Apply uncertainty gate

            for idx, original_item_id in enumerate(candidate_item_ids):
                scores[original_item_id] = final_scores[idx].item()

            # --- Record feedback into Health Monitor and Router Trainer ---
            try:
                if self.health_monitor is not None:
                    # Compute per-model error as deviation from ensemble mean
                    ensemble_mean = final_scores.mean().item()
                    for m in active_names:
                        model_mean = results[m].mean().item()
                        error = abs(model_mean - ensemble_mean)
                        self.health_monitor.record_prediction(
                            model_name=m,
                            error=error,
                            latency_ms=model_latencies.get(m, 0.0),
                            success=model_successes.get(m, True),
                        )
            except Exception as health_exc:
                logger.debug("Health monitor recording failed: %s", health_exc)

            try:
                if self.router_trainer is not None and use_router and _router_user_state is not None:
                    # Record model scores for online router training
                    per_model_scores = {m: results[m].mean().item() for m in active_names}
                    self.router_trainer.record(
                        user_state=_router_user_state,
                        model_scores=per_model_scores,
                        selected_models=selected_models,
                    )
                    # Trigger async training step if buffer is ready
                    if self.router_trainer.is_ready:
                        try:
                            executor.submit(self.router_trainer.train_step)
                        except Exception:
                            pass  # Non-critical: training is best-effort
            except Exception as trainer_exc:
                logger.debug("Router trainer recording failed: %s", trainer_exc)

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            for idx in candidate_item_ids:
                scores[idx] = 0.0

        return scores

    def get_system_health(self) -> dict:
        """Return a comprehensive health report for the entire ensemble system."""
        report = {"engine": "ApexEnsembleEngine"}

        # Model health
        if self.health_monitor is not None:
            report["model_health"] = self.health_monitor.get_health_report()
        else:
            report["model_health"] = {"status": "unavailable"}

        # Router trainer stats
        if self.router_trainer is not None:
            report["router_trainer"] = self.router_trainer.get_stats()
        else:
            report["router_trainer"] = {"status": "unavailable"}

        # Privacy budget
        if self.privacy_accountant is not None:
            report["privacy_budget"] = {"status": "active"}
        else:
            report["privacy_budget"] = {"status": "unavailable"}

        return report

    def explain_routing(self, user_id: int, k: int = 2) -> dict:
        """
        Explain why the router selected specific models for a user.

        Returns feature attributions and human-readable explanations.
        """
        try:
            from backend.intelligence.router_explainer import RouterExplainer
            from backend.models.contextual_router import build_user_state

            safe_user_id = user_id % self.num_users
            u_tensor = torch.tensor([safe_user_id], dtype=torch.long)

            with torch.no_grad():
                lgcn_all_items, _ = self._get_item_embedding_cache()
                base_u_emb = self.lightgcn.user_embedding(u_tensor).squeeze()
                simulated_seq = self._get_session_sequence(safe_user_id)

            try:
                interaction_count = len(_get_user_event_index().get(str(safe_user_id), []))
            except Exception:
                interaction_count = 0

            user_state = build_user_state(
                user_id=safe_user_id,
                user_emb=base_u_emb,
                session_seq=simulated_seq,
                item_embeddings=lgcn_all_items,
                interaction_count=interaction_count,
            )

            explainer = RouterExplainer(router=self.router, emb_dim=self.emb_dim)
            explanation = explainer.explain(user_state.to(self._device), k=k)

            return {
                "selected_models": explanation.selected_models,
                "routing_weights": explanation.routing_weights,
                "all_model_probabilities": explanation.all_model_probabilities,
                "feature_attributions": explanation.feature_attributions,
                "top_positive_features": explanation.top_positive_features,
                "top_negative_features": explanation.top_negative_features,
                "explanation_text": explanation.explanation_text,
                "user_state_summary": explanation.user_state_summary,
            }
        except Exception as exc:
            logger.warning("Routing explanation failed: %s", exc)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level thread pool — created once, reused across all requests.
# Creating a ThreadPoolExecutor per request costs ~1 ms per call under load.
# ---------------------------------------------------------------------------
_MODEL_THREAD_POOL = None
_MODEL_THREAD_POOL_LOCK = threading.Lock()


def _get_model_thread_pool():
    """Return the module-level 6-worker thread pool, creating it on first call."""
    global _MODEL_THREAD_POOL
    if _MODEL_THREAD_POOL is None:
        with _MODEL_THREAD_POOL_LOCK:
            if _MODEL_THREAD_POOL is None:
                from concurrent.futures import ThreadPoolExecutor

                _MODEL_THREAD_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="apex-model")
                logger.info("Created module-level model thread pool (max_workers=6)")
    return _MODEL_THREAD_POOL


# ---------------------------------------------------------------------------
# Singleton — thread-safe double-checked locking
# ---------------------------------------------------------------------------
_apex_engine: "ApexEnsembleEngine | None" = None
_apex_engine_lock = threading.Lock()


def get_apex_engine(num_users: int = 1000, num_items: int = 10000, device: str | None = None) -> "ApexEnsembleEngine":
    """Return the module-level ApexEnsembleEngine singleton.

    Thread-safe: uses double-checked locking so concurrent FastAPI startup
    requests cannot construct two engines simultaneously.
    """
    global _apex_engine
    if _apex_engine is None:
        with _apex_engine_lock:
            if _apex_engine is None:
                _apex_engine = ApexEnsembleEngine(num_users=num_users, num_items=num_items, device=device)
    return _apex_engine
