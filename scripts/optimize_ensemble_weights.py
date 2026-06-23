"""
Offline Dirichlet grid-search for ensemble blend weights.

Usage:
    python scripts/optimize_ensemble_weights.py [--num-candidates N] [--k K] [--output-path PATH]

This script:
1. Loads interaction data from the Event Store (rating + click events).
2. Splits each user's events into train (80%) / validation (last 20%) by event_ts.
3. Pre-computes per-model raw scores ONCE for a sample of validation users.
4. Samples ``num_candidates`` weight vectors from Dirichlet(alpha=[1]*6).
5. For each vector, blends the pre-computed scores analytically (fast).
6. Persists the best vector to ``output_path`` as JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import random
import sys
from typing import TYPE_CHECKING

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend import events as event_module

if TYPE_CHECKING:
    from backend.models.ensemble_engine import ApexEnsembleEngine

logger = logging.getLogger(__name__)

MODELS_DIR = _REPO_ROOT / "models"
WEIGHT_KEYS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion", "clifford")

_DEFAULT_WEIGHTS: dict[str, float] = {
    "lightgcn": 0.60,
    "quantum": 0.20,
    "sasrec": 0.10,
    "clifford": 0.05,
    "kan": 0.00,
    "hyperbolic": 0.05,
    "diffusion": 0.00,
}


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _ideal_dcg(num_hits: int, k: int) -> float:
    return sum(1.0 / math.log2(rank + 2) for rank in range(min(num_hits, k)))


def _ndcg_at_k(ranked_items: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    top_k = ranked_items[:k]
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(top_k) if item in ground_truth)
    idcg = _ideal_dcg(len(ground_truth), k)
    return dcg / idcg if idcg > 0 else 0.0


def _hit_rate_at_k(ranked_items: list[int], ground_truth: set[int], k: int) -> float:
    if not ground_truth:
        return 0.0
    return 1.0 if any(item in ground_truth for item in ranked_items[:k]) else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_interaction_data() -> dict[str, list[dict]]:
    user_events: dict[str, list[dict]] = defaultdict(list)
    for event in event_module.iter_events():
        et = str(event.get("event_type", "")).lower()
        if et not in {"rating", "click"}:
            continue
        movie_id = event.get("movie_id")
        user_id = event.get("user_id")
        if movie_id is None or user_id is None:
            continue
        try:
            movie_id = int(movie_id)
        except (TypeError, ValueError):
            continue
        user_events[str(user_id)].append(
            {
                "event_ts": str(event.get("event_ts") or ""),
                "movie_id": movie_id,
            }
        )
    return dict(user_events)


def _build_validation_split(
    user_events: dict[str, list[dict]],
) -> tuple[dict[str, list[int]], dict[str, set[int]]]:
    train_history: dict[str, list[int]] = {}
    val_ground_truth: dict[str, set[int]] = {}
    for user_id, events in user_events.items():
        sorted_events = sorted(events, key=lambda e: e["event_ts"])
        n = len(sorted_events)
        split_idx = max(1, math.ceil(n * 0.8))
        train_history[user_id] = [e["movie_id"] for e in sorted_events[:split_idx]]
        val_ground_truth[user_id] = {e["movie_id"] for e in sorted_events[split_idx:]}
    return train_history, val_ground_truth


# ---------------------------------------------------------------------------
# Fast pre-compute: run all 6 models ONCE per user, then blend analytically
# ---------------------------------------------------------------------------


def _precompute_per_model_scores(
    engine: ApexEnsembleEngine,
    train_history: dict[str, list[int]],
    val_ground_truth: dict[str, set[int]],
    rng: random.Random,
    max_users: int = 80,
) -> dict[str, dict[int, list[float]]]:
    """
    For each sampled user, run all 6 models once and store normalised scores.
    Returns: user_id -> {item_id: [lgcn, quantum, sasrec, kan, hyp, diff]}
    """
    import torch

    all_item_ids: list[int] = []
    seen: set[int] = set()
    for items in train_history.values():
        for item in items:
            if item not in seen:
                seen.add(item)
                all_item_ids.append(item)

    valid_users = [uid for uid, gt in val_ground_truth.items() if gt and train_history.get(uid)]
    if len(valid_users) > max_users:
        valid_users = rng.sample(valid_users, max_users)

    per_model_scores: dict[str, dict[int, list[float]]] = {}

    for user_id in valid_users:
        user_train = train_history.get(user_id, [])
        neg_pool = [x for x in all_item_ids if x not in set(user_train)]
        num_neg = min(len(user_train) * 9, len(neg_pool))
        sampled_neg = rng.sample(neg_pool, min(num_neg, len(neg_pool))) if neg_pool else []
        candidate_ids = list(set(user_train) | set(sampled_neg))
        if not candidate_ids:
            continue

        try:
            uid_int = int(user_id)
        except (ValueError, TypeError):
            uid_int = abs(hash(user_id)) % max(engine.num_users, 1)

        safe_uid = uid_int % engine.num_users
        safe_items = [item % engine.num_items for item in candidate_ids]
        # Also remap candidate_ids for score storage using the safe indices
        {(item % engine.num_items): item for item in candidate_ids}
        u_t = torch.tensor([safe_uid], dtype=torch.long)
        i_t = torch.tensor(safe_items, dtype=torch.long)

        try:
            with torch.no_grad():
                # LightGCN
                lu = engine.lightgcn.user_embedding(u_t).expand(len(i_t), -1)
                li = engine.lightgcn.item_embedding(i_t)
                lgcn_s = (lu * li).sum(dim=1).numpy()

                # Quantum
                qs = engine.quantum.predict(u_t, i_t, time_delta=1.0).squeeze()
                if qs.dim() == 0:
                    qs = qs.unsqueeze(0)
                q_s = qs.numpy()

                # Hyperbolic
                hs = -engine.hyperbolic.predict(u_t.expand_as(i_t), i_t)
                h_s = hs.numpy()

                # KAN + Diffusion (share embeddings)
                u_emb = engine.hyperbolic.user_embedding(u_t).expand(len(i_t), -1)
                i_emb = engine.hyperbolic.item_embedding(i_t)
                ks = engine.kan(u_emb, i_emb).squeeze()
                if ks.dim() == 0:
                    ks = ks.unsqueeze(0)
                k_s = ks.numpy()

                t_val = torch.ones(len(i_t), 1) * 0.5
                d_noise = engine.diffusion.denoiser(i_emb, t_val, u_emb)
                d_s = (1.0 / (1.0 + torch.norm(d_noise, dim=-1))).numpy()

                # SASRec — use training history directly instead of scanning event store
                history_ids = train_history.get(user_id, [])[-50:]
                safe_hist = [h % engine.num_items for h in history_ids]
                padded = [0] * (50 - len(safe_hist)) + safe_hist
                seq = torch.tensor([padded], dtype=torch.long)
                ss = engine.sasrec.predict(seq, i_t.unsqueeze(0)).squeeze()
                if ss.dim() == 0:
                    ss = ss.unsqueeze(0)
                sar_s = ss.numpy()

                # Clifford
                cliffs = engine.clifford.predict(u_t, i_t).squeeze()
                if cliffs.dim() == 0:
                    cliffs = cliffs.unsqueeze(0)
                cliff_s = cliffs.numpy()

            def _norm(arr):
                mn, mx = arr.min(), arr.max()
                if mx - mn < 1e-6:
                    return np.full_like(arr, 0.5)
                return (arr - mn) / (mx - mn)

            scores_matrix = np.stack(
                [_norm(lgcn_s), _norm(q_s), _norm(sar_s), _norm(k_s), _norm(h_s), _norm(d_s), _norm(cliff_s)], axis=1
            )  # [N_items, 7]

            per_model_scores[user_id] = {
                orig_id: scores_matrix[idx].tolist() for idx, orig_id in enumerate(candidate_ids)
            }

        except Exception as exc:
            logger.warning("Pre-compute failed for user %s: %s", user_id, exc)

    logger.info("Pre-computed scores for %d users.", len(per_model_scores))
    return per_model_scores


def _evaluate_weights_fast(
    weight_vector: np.ndarray,
    per_model_scores: dict[str, dict[int, list[float]]],
    val_ground_truth: dict[str, set[int]],
    k: int,
) -> tuple[float, float]:
    """Blend pre-computed scores analytically — no neural forward pass."""
    ndcg_scores: list[float] = []
    hit_scores: list[float] = []
    for user_id, item_scores in per_model_scores.items():
        gt = val_ground_truth.get(user_id, set())
        if not gt or not item_scores:
            continue
        blended = {item_id: float(np.dot(weight_vector, np.array(ms))) for item_id, ms in item_scores.items()}
        ranked = sorted(item_scores.keys(), key=lambda x: blended.get(x, 0.0), reverse=True)
        ndcg_scores.append(_ndcg_at_k(ranked, gt, k))
        hit_scores.append(_hit_rate_at_k(ranked, gt, k))
    if not ndcg_scores:
        return 0.0, 0.0
    return float(np.mean(ndcg_scores)), float(np.mean(hit_scores))


# ---------------------------------------------------------------------------
# Main grid-search
# ---------------------------------------------------------------------------


def run_dirichlet_grid_search(
    num_candidates: int = 500,
    k: int = 10,
    output_path: Path = Path("models/ensemble_weights.json"),
    engine: ApexEnsembleEngine | None = None,
    dirichlet_alpha: float | None = None,
) -> dict[str, float]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    import os

    if dirichlet_alpha is None:
        try:
            dirichlet_alpha = float(os.getenv("APEX_DIRICHLET_ALPHA", "1.0"))
        except ValueError:
            dirichlet_alpha = 1.0

    logger.info("Loading interaction data from Event Store …")
    user_events = _load_interaction_data()
    logger.info("Found %d users with rating/click events.", len(user_events))

    train_history, val_ground_truth = _build_validation_split(user_events)
    valid_users = {uid for uid, gt in val_ground_truth.items() if gt}

    if len(valid_users) < 10:
        print(f"WARNING: Only {len(valid_users)} users have validation interactions. Returning defaults.", flush=True)
        return dict(_DEFAULT_WEIGHTS)

    val_ground_truth = {uid: val_ground_truth[uid] for uid in valid_users}
    train_history = {uid: train_history[uid] for uid in valid_users if uid in train_history}
    logger.info("Validation split: %d users, NDCG@%d.", len(valid_users), k)

    if engine is None:
        logger.info("Initialising ApexEnsembleEngine …")
        from backend.models.ensemble_engine import ApexEnsembleEngine

        # Use catalog size to ensure embedding tables cover all item IDs
        try:
            from backend.pipeline.recommender import get_recommender

            rec = get_recommender()
            num_items = len(rec.movies) if rec._movies is not None else 50000
        except Exception:
            num_items = 50000
        engine = ApexEnsembleEngine(num_users=1000, num_items=num_items)

    rng = random.Random(42)

    # Pre-compute per-model scores ONCE
    logger.info("Pre-computing per-model scores (runs neural models once) …")
    per_model_scores = _precompute_per_model_scores(engine, train_history, val_ground_truth, rng)

    if not per_model_scores:
        logger.warning("No per-model scores computed; returning defaults.")
        return dict(_DEFAULT_WEIGHTS)

    logger.info("Starting Dirichlet grid-search with %d candidates (alpha=%.2f) …", num_candidates, dirichlet_alpha)
    results: list[tuple[float, float, np.ndarray]] = []

    for i in range(num_candidates):
        wv = np.random.dirichlet([dirichlet_alpha] * len(WEIGHT_KEYS))
        ndcg, hit_rate = _evaluate_weights_fast(wv, per_model_scores, val_ground_truth, k)
        results.append((ndcg, hit_rate, wv))
        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d | Best NDCG@%d: %.4f", i + 1, num_candidates, k, max(r[0] for r in results))

    results.sort(key=lambda r: r[0], reverse=True)

    print(f"\nTop-5 candidates by NDCG@{k}:", flush=True)
    for rank, (ndcg, hit_rate, wv) in enumerate(results[:5], start=1):
        ws = ", ".join(f"{key}={wv[i]:.4f}" for i, key in enumerate(WEIGHT_KEYS))
        print(f"  #{rank}: NDCG@{k}={ndcg:.4f}, Hit_Rate@{k}={hit_rate:.4f} | {ws}", flush=True)

    best_ndcg, best_hit_rate, best_vector = results[0]
    best_vector = np.maximum(best_vector, 0.0)
    total = best_vector.sum()
    if total == 0.0:
        return dict(_DEFAULT_WEIGHTS)
    if abs(total - 1.0) > 1e-6:
        best_vector = best_vector / total

    best_weights = {key: float(best_vector[i]) for i, key in enumerate(WEIGHT_KEYS)}
    assert all(v >= 0.0 for v in best_weights.values())
    assert abs(sum(best_weights.values()) - 1.0) < 1e-6

    output_record = {
        **best_weights,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "ndcg_at_10": round(best_ndcg, 6),
        "hit_rate_at_10": round(best_hit_rate, 6),
        "num_candidates_evaluated": num_candidates,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output_record, fh, indent=2)

    logger.info("Best weights written to %s (NDCG@%d=%.4f).", output_path, k, best_ndcg)
    print(f"\nBest weights saved to {output_path}", flush=True)
    return best_weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Offline Dirichlet grid-search for ensemble blend weights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num-candidates", type=int, default=500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output-path", type=Path, default=MODELS_DIR / "ensemble_weights.json")
    parser.add_argument("--dirichlet-alpha", type=float, default=None, help="Dirichlet concentration parameter (alpha)")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    best = run_dirichlet_grid_search(
        num_candidates=args.num_candidates,
        k=args.k,
        output_path=args.output_path,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    print("\nReturned weights:", best, flush=True)
