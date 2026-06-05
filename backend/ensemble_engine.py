import collections
from collections import defaultdict
import json
import logging
from pathlib import Path
import threading
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from backend.diffusion_recommender import LatentDiffusionRecommender
from backend.events import iter_events
from backend.hyperbolic_recommender import HyperbolicRecommender
from backend.kan_ranker import KANRanker
from backend.lightgcn import LightGCN

# Import all 4 Advanced Research Models
from backend.neural_ode_recommender import QuantumFluidRecommender
from backend.sasrec import SASRec

logger = logging.getLogger(__name__)

GOLD_DIR = Path("data/datalake/gold")
ARTIFACTS_DIR = Path("backend/artifacts")
MODELS_DIR = Path("models")

# ---------------------------------------------------------------------------
# Module-level user event index — built once, avoids full JSONL scan per request
# ---------------------------------------------------------------------------
_user_event_index: dict[str, list[tuple[str, int]]] | None = None
_user_event_index_lock = threading.Lock()
_user_event_index_built_at: float = 0.0
_USER_EVENT_INDEX_TTL = 300.0  # rebuild every 5 minutes


def _get_user_event_index() -> dict[str, list[tuple[str, int]]]:
    """Return a cached user→[(event_ts, movie_id)] index, rebuilding every 5 minutes."""
    global _user_event_index, _user_event_index_built_at
    now = time.time()
    if _user_event_index is not None and (now - _user_event_index_built_at) < _USER_EVENT_INDEX_TTL:
        return _user_event_index
    with _user_event_index_lock:
        # Double-check after acquiring lock
        if _user_event_index is not None and (time.time() - _user_event_index_built_at) < _USER_EVENT_INDEX_TTL:
            return _user_event_index
        logger.info("Building user event index from Event Store...")
        index: dict[str, list[tuple[str, int]]] = defaultdict(list)
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
                index[str(uid)].append((ts, mid))
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
        # LRU session cache: OrderedDict gives O(1) move-to-end for recency tracking.
        # Capped at _SESSION_CACHE_MAX entries; oldest entry evicted when full.
        self._session_cache: collections.OrderedDict[str, tuple[float, list[int]]] = collections.OrderedDict()
        self._session_cache_lock = threading.Lock()
        self._SESSION_CACHE_MAX = 10_000
        self._weights_lock = threading.Lock()

        logger.info("Initializing the 4 Pillars of the Apex Ensemble...")
        # 1. Quantum Fluid
        self.quantum = QuantumFluidRecommender(num_users, num_items, emb_dim)

        # 2. Hyperbolic
        self.hyperbolic = HyperbolicRecommender(num_users, num_items, emb_dim)

        # 3. KAN (Kolmogorov-Arnold)
        # KAN expects input size: emb_dim * 2 (user + item concat)
        self.kan = KANRanker(input_dim=emb_dim * 2, hidden_dim=64)

        # 4. Latent Diffusion
        self.diffusion = LatentDiffusionRecommender(emb_dim=emb_dim, num_timesteps=20)

        # 5. SASRec (Transformers)
        # SASRec reserves index 0 for padding internally, so pass the content item count directly.
        self.sasrec = SASRec(num_items=num_items, hidden_dim=emb_dim)

        # 6. LightGCN (Graph Networks)
        self.lightgcn = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=emb_dim)

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

    # ------------------------------------------------------------------ #
    # Ensemble weight loading                                              #
    # ------------------------------------------------------------------ #

    _REQUIRED_WEIGHT_KEYS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion")
    _DEFAULT_WEIGHTS: dict[str, float] = {
        "lightgcn": 0.65,
        "quantum": 0.25,
        "sasrec": 0.10,
        "kan": 0.00,
        "hyperbolic": 0.00,
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
        except Exception as exc:  # noqa: BLE001
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

    def _move_to_device(self) -> None:
        """Move all sub-models to self._device."""
        for name in ("quantum", "hyperbolic", "kan", "diffusion", "sasrec", "lightgcn"):
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

    def _inject_pyspark_priors(self):
        """Loads real PySpark ALS embeddings from the Delta Lake if available."""
        user_emb_path = GOLD_DIR / "model_user_embeddings"
        item_emb_path = GOLD_DIR / "model_item_embeddings"

        if user_emb_path.exists() and item_emb_path.exists():
            try:
                users_df = pd.read_parquet(user_emb_path, engine="pyarrow").sort_values("id")
                items_df = pd.read_parquet(item_emb_path, engine="pyarrow").sort_values("id")

                user_tensor = torch.tensor(np.vstack(users_df["features"].values), dtype=torch.float32)
                item_tensor = torch.tensor(np.vstack(items_df["features"].values), dtype=torch.float32)

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
                self.sasrec.item_emb.weight.data[: self.num_items] = item_tensor

                # Diffusion and KAN do not maintain separate embeddings; they dynamically
                # route through the Hyperbolic/Quantum priors during the forward pass.

                logger.info(
                    f"✅ Successfully injected Real PySpark Embeddings ({self.num_users} users, {self.num_items} items) into all 4 models."
                )
            except Exception as e:
                logger.error(f"Failed to inject PySpark priors: {e}")
        else:
            logger.warning("PySpark Gold embeddings not found. Models will use their initialized mathematical priors.")

    def _load_trained_weights(self):
        """Loads trained weights for the neural models if available."""
        models_to_load = {
            "quantum": MODELS_DIR / "quantum_fluid.pth",
            "hyperbolic": MODELS_DIR / "hyperbolic.pth",
            "kan": MODELS_DIR / "kan_ranker.pth",
            "sasrec": MODELS_DIR / "sasrec.pth",
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
                    model_attr.load_state_dict(torch.load(path, weights_only=True))
                    ips_tag = " (IPS-debiased)" if name == "lightgcn" and "ips" in path.name else ""
                    logger.info("Loaded %s weights from %s%s", name, path.name, ips_tag)
                    loaded += 1
                except Exception as e:
                    logger.error(f"Failed to load {name} weights: {e}")

        if loaded > 0:
            logger.info(f"✅ Successfully loaded trained weights for {loaded} models.")
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
                from backend.realtime_feature_updater import get_user_session_sequence

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

            safe_ids = [item_id % self.num_items for _, item_id in recent]

            # Cache the result with proper LRU eviction
            with self._session_cache_lock:
                self._session_cache[cache_key] = (time.time(), safe_ids)
                self._session_cache.move_to_end(cache_key)
                if len(self._session_cache) > self._SESSION_CACHE_MAX:
                    self._session_cache.popitem(last=False)

            padded = [0] * (SEQ_LEN - len(safe_ids)) + safe_ids
            return torch.tensor([padded], dtype=torch.long)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Event Store query failed for user %s; falling back to zero sequence: %s",
                user_id,
                exc,
            )
            return torch.zeros((1, SEQ_LEN), dtype=torch.long)

    def predict_ensemble(
        self, user_id: int, candidate_item_ids: list[int], session_sequence: list[int] | None = None
    ) -> dict[int, float]:
        """
        Executes a real forward pass across all 4 architectures and returns
        the fused ensemble score for each candidate.
        Uses ONNX Runtime when available (Tier 2) for 2-5x faster CPU inference.
        """
        if not candidate_item_ids:
            return {}

        # Try ONNX-accelerated path first (Tier 2)
        try:
            from backend.onnx_engine import get_onnx_engine

            onnx = get_onnx_engine()
            if onnx.has_any_onnx_models():
                return self._predict_ensemble_onnx(user_id, candidate_item_ids, onnx, session_sequence)
        except Exception as exc:
            logger.debug("ONNX ensemble path unavailable; falling back to PyTorch: %s", exc)

        return self._predict_ensemble_pytorch(user_id, candidate_item_ids, session_sequence)

    def _predict_ensemble_onnx(
        self, user_id: int, candidate_item_ids: list[int], onnx, session_sequence: list[int] | None = None
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

        with self._weights_lock:
            w = dict(self._weights)

        # Use contextual weights if available (context-dependent ensemble blending)
        try:
            from backend.neural_weight_optimizer import get_contextual_weights

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

        # Blend
        key_map = {
            "lightgcn": "lightgcn",
            "quantum": "quantum",
            "sasrec": "sasrec",
            "kan": "kan",
            "hyperbolic": "hyperbolic",
            "diffusion": "diffusion",
        }
        final = _np.zeros(len(safe_item_ids), dtype=_np.float32)
        for name, s in scores_list:
            final += s * w.get(key_map.get(name, name), 0.0)

        return {orig_id: float(final[idx]) for idx, orig_id in enumerate(candidate_item_ids)}

    def _predict_ensemble_pytorch(
        self, user_id: int, candidate_item_ids: list[int], session_sequence: list[int] | None = None
    ) -> dict[int, float]:
        if not candidate_item_ids:
            return {}

        # Ensure ID bounds are respected (hash to max size if unknown)
        safe_user_id = user_id % self.num_users
        safe_item_ids = [item_id % self.num_items for item_id in candidate_item_ids]

        u_tensor = torch.tensor([safe_user_id], dtype=torch.long)
        i_tensor = torch.tensor(safe_item_ids, dtype=torch.long)

        # Pre-compute shared embeddings (used by KAN, Diffusion, LightGCN)
        # Uses cached item embeddings to avoid recomputing on every request
        lgcn_all_items, hyp_all_items = self._get_item_embedding_cache()
        with torch.no_grad():
            u_emb = self.hyperbolic.user_embedding(u_tensor).expand(len(i_tensor), -1)
            i_emb = hyp_all_items[i_tensor]  # lookup from cache
            lgcn_u_emb = self.lightgcn.user_embedding(u_tensor).expand(len(i_tensor), -1)
            lgcn_i_emb = lgcn_all_items[i_tensor]  # lookup from cache
            simulated_seq = self._get_session_sequence(user_id, override=session_sequence)

        # Enrich user embedding with attention over session history
        try:
            from backend.attention_user_model import build_attended_user_embedding, get_user_attention_encoder

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
            }

            with self._weights_lock:
                w = dict(self._weights)

            # Run all 6 models in parallel using the module-level thread pool.
            # Reusing the pool avoids the ~1 ms thread-creation overhead per request.
            executor = _get_model_thread_pool()
            results: dict[str, torch.Tensor] = {}
            futures = {executor.submit(fn): name for name, fn in model_fns.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                except Exception as exc:
                    logger.warning("Model %s failed: %s; using 0.5 fallback", name, exc)
                    results[name] = torch.ones(len(safe_item_ids)) * 0.5

            final_scores = (
                results["lightgcn"] * w["lightgcn"]
                + results["quantum"] * w["quantum"]
                + results["sasrec"] * w["sasrec"]
                + results["kan"] * w["kan"]
                + results["hyperbolic"] * w["hyperbolic"]
                + results["diffusion"] * w["diffusion"]
            )

            for idx, original_item_id in enumerate(candidate_item_ids):
                scores[original_item_id] = final_scores[idx].item()

        except Exception as e:
            logger.error(f"Ensemble prediction failed: {e}")
            for idx in candidate_item_ids:
                scores[idx] = 0.0

        return scores


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
