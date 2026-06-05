"""
Causal Debiasing for APEX Recommendation Training.

This module implements Inverse Propensity Scoring (IPS) and Doubly Robust (DR)
estimation to correct for popularity bias and position bias during model training.

Why this matters:
  Standard recommendation training optimizes for what users clicked — but users
  only click what they were shown. Popular items get shown more, so they get more
  clicks, so models learn to recommend popular items more, creating a feedback loop.

  IPS breaks this loop by reweighting each training sample by the inverse probability
  that the logging policy showed that item. Items that were rarely shown but clicked
  get high weight; items that were always shown get low weight.

  This is the same technique used by Netflix (published in RecSys 2020) and Spotify.
  No open-source recommendation system has this properly wired into training.

References:
  - Schnabel et al. "Recommendations as Treatments" (ICML 2016)
  - Saito et al. "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback" (WSDM 2020)
  - Dudík et al. "Doubly Robust Policy Evaluation and Learning" (ICML 2011)

Model cards: docs/MODEL_CARDS.md (documents IPS training details per model)

Usage:
    python scripts/causal_debias_training.py [--epochs N] [--clip-val N]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.events import iter_events
from backend.lightgcn import LightGCN

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# ---------------------------------------------------------------------------
# Propensity estimation
# ---------------------------------------------------------------------------


def estimate_item_propensities(
    events: list[dict],
    smoothing: float = 0.1,
) -> dict[int, float]:
    """
    Estimate the propensity (exposure probability) of each item using
    the empirical frequency of impressions.

    Items that appear frequently in recommendations have high propensity.
    Items that rarely appear have low propensity (and thus high IPS weight).

    Uses Laplace smoothing to avoid zero propensities.
    """
    impression_counts: dict[int, int] = defaultdict(int)
    total_impressions = 0

    for event in events:
        et = str(event.get("event_type", "")).lower()
        if et in {"recommendation_impression", "click", "rating", "view"}:
            mid = event.get("movie_id")
            if mid is not None:
                try:
                    impression_counts[int(mid)] += 1
                    total_impressions += 1
                except (TypeError, ValueError):
                    pass

    if total_impressions == 0:
        logger.warning("No impression events found; using uniform propensities")
        return {}

    # Normalize to probabilities with Laplace smoothing
    n_items = len(impression_counts)
    propensities: dict[int, float] = {}
    for item_id, count in impression_counts.items():
        propensities[item_id] = (count + smoothing) / (total_impressions + smoothing * n_items)

    # Global mean for unseen items
    mean_propensity = float(np.mean(list(propensities.values())))
    propensities["__default__"] = mean_propensity  # type: ignore[assignment]

    logger.info(
        "Estimated propensities for %d items (mean=%.6f, min=%.6f, max=%.6f)",
        n_items,
        mean_propensity,
        min(propensities.values()),
        max(propensities.values()),
    )
    return propensities


def get_ips_weight(
    item_id: int,
    propensities: dict[int, float],
    clip_val: float = 10.0,
) -> float:
    """Return the IPS weight for an item: 1 / propensity, clipped."""
    p = propensities.get(item_id, propensities.get("__default__", 0.01))  # type: ignore[arg-type]
    p = max(p, 1e-6)  # floor to prevent division by zero
    return min(1.0 / p, clip_val)


# ---------------------------------------------------------------------------
# IPS-weighted BPR training for LightGCN
# ---------------------------------------------------------------------------


def train_lightgcn_ips(
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 4096,
    clip_val: float = 10.0,
) -> None:
    """
    Train LightGCN with IPS-weighted BPR loss to correct for popularity bias.

    Standard BPR treats all positive interactions equally.
    IPS-BPR upweights rare positive interactions (items that were rarely shown
    but still clicked) and downweights popular items (shown constantly).

    This produces embeddings that reflect true user preference rather than
    exposure frequency.
    """
    logger.info("Loading events for propensity estimation...")
    all_events = list(iter_events())
    propensities = estimate_item_propensities(all_events)

    # Build interaction data
    positives: list[tuple[int, int, float]] = []  # (user_id, item_id, ips_weight)
    user_map: dict[str, int] = {}
    item_map: dict[int, int] = {}

    for event in all_events:
        et = str(event.get("event_type", "")).lower()
        if et not in {"rating", "click"}:
            continue
        if et == "rating":
            r = event.get("rating", 0)
            try:
                if float(r) < 3.5:
                    continue
            except (TypeError, ValueError):
                continue

        uid = event.get("user_id")
        mid = event.get("movie_id")
        if uid is None or mid is None:
            continue
        try:
            mid = int(mid)
        except (TypeError, ValueError):
            continue

        uid_str = str(uid)
        if uid_str not in user_map:
            user_map[uid_str] = len(user_map)
        if mid not in item_map:
            item_map[mid] = len(item_map)

        ips_w = get_ips_weight(mid, propensities, clip_val=clip_val)
        positives.append((user_map[uid_str], item_map[mid], ips_w))

    if len(positives) < 100:
        logger.warning("Too few positive interactions (%d); skipping IPS training", len(positives))
        return

    num_users = len(user_map)
    num_items = len(item_map)
    logger.info(
        "IPS training: %d positives, %d users, %d items",
        len(positives),
        num_users,
        num_items,
    )

    # Load existing LightGCN weights as starting point
    model = LightGCN(num_users=num_users, num_items=num_items, embedding_dim=16)
    lgcn_path = MODELS_DIR / "lightgcn.pth"
    if lgcn_path.exists():
        try:
            ckpt = torch.load(lgcn_path, map_location="cpu", weights_only=True)
            # Only load if dimensions match
            if (
                ckpt["user_embedding.weight"].shape[0] == num_users
                and ckpt["item_embedding.weight"].shape[0] == num_items
            ):
                model.load_state_dict(ckpt)
                logger.info("Loaded existing LightGCN weights as starting point")
            else:
                logger.info("Dimension mismatch; starting from scratch")
        except Exception as exc:
            logger.warning("Could not load LightGCN weights: %s", exc)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    rng = np.random.default_rng(42)
    u_arr = np.array([p[0] for p in positives], dtype=np.int64)
    p_arr = np.array([p[1] for p in positives], dtype=np.int64)
    w_arr = np.array([p[2] for p in positives], dtype=np.float32)

    logger.info("Training IPS-weighted LightGCN for %d epochs...", epochs)
    for epoch in range(epochs):
        model.train()
        idx = rng.choice(len(u_arr), size=min(200000, len(u_arr)), replace=False)
        n_arr = rng.integers(0, num_items, size=len(idx)).astype(np.int64)
        perm = rng.permutation(len(idx))

        total_loss, nb = 0.0, 0
        for start in range(0, len(perm), batch_size):
            bi = perm[start : start + batch_size]
            u = torch.tensor(u_arr[idx[bi]], dtype=torch.long)
            p = torch.tensor(p_arr[idx[bi]], dtype=torch.long)
            n = torch.tensor(n_arr[bi], dtype=torch.long)
            w = torch.tensor(w_arr[idx[bi]], dtype=torch.float32)

            ue = model.user_embedding(u)
            pe = model.item_embedding(p)
            ne = model.item_embedding(n)

            # IPS-weighted BPR loss: weight each sample by its propensity weight
            bpr_loss = F.softplus((ue * ne).sum(1) - (ue * pe).sum(1))
            loss = (bpr_loss * w).mean()  # IPS weighting here
            loss += 1e-4 * (ue.norm(2).pow(2) + pe.norm(2).pow(2) + ne.norm(2).pow(2)) / len(bi)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            nb += 1

        scheduler.step()
        if (epoch + 1) % 20 == 0:
            logger.info("  Epoch %d/%d | IPS-BPR Loss: %.4f", epoch + 1, epochs, total_loss / max(nb, 1))

    # Save IPS-debiased model
    save_path = MODELS_DIR / "lightgcn_ips.pth"
    torch.save(model.state_dict(), save_path)
    logger.info("IPS-debiased LightGCN saved to %s", save_path)

    # Also update the main lightgcn.pth
    torch.save(model.state_dict(), MODELS_DIR / "lightgcn.pth")
    logger.info("Updated lightgcn.pth with IPS-debiased weights")


# ---------------------------------------------------------------------------
# Doubly Robust ensemble weight selection
# ---------------------------------------------------------------------------


def select_weights_doubly_robust(
    num_candidates: int = 200,
    clip_val: float = 10.0,
) -> dict[str, float]:
    """
    Select ensemble blend weights using Doubly Robust (DR) estimation.

    DR combines:
    1. Direct reward imputation (what the model predicts)
    2. IPS correction (correcting for what was actually shown)

    This gives unbiased weight selection even when the historical logging
    policy was biased toward popular items.
    """
    from datetime import UTC, datetime
    import json
    import random

    from scripts.optimize_ensemble_weights import (
        _DEFAULT_WEIGHTS,
        WEIGHT_KEYS,
        _build_validation_split,
        _load_interaction_data,
        _precompute_per_model_scores,
    )

    logger.info("Loading interaction data for DR weight selection...")
    user_events = _load_interaction_data()
    if len(user_events) < 10:
        logger.warning("Insufficient data for DR weight selection; using defaults")
        return dict(_DEFAULT_WEIGHTS)

    train_history, val_ground_truth = _build_validation_split(user_events)
    valid_users = {uid for uid, gt in val_ground_truth.items() if gt}
    val_ground_truth = {uid: val_ground_truth[uid] for uid in valid_users}
    train_history = {uid: train_history[uid] for uid in valid_users if uid in train_history}

    # Estimate propensities from all events
    all_events = list(iter_events())
    propensities = estimate_item_propensities(all_events)

    # Load ensemble engine
    from backend.ensemble_engine import ApexEnsembleEngine

    engine = ApexEnsembleEngine()

    rng_obj = random.Random(42)
    per_model_scores = _precompute_per_model_scores(engine, train_history, val_ground_truth, rng_obj)

    if not per_model_scores:
        logger.warning("No per-model scores; using defaults")
        return dict(_DEFAULT_WEIGHTS)

    def _dr_score(weight_vector: np.ndarray) -> float:
        """Compute Doubly Robust score for a weight vector."""
        dr_total = 0.0
        n_users = 0

        for user_id, item_scores in per_model_scores.items():
            gt = val_ground_truth.get(user_id, set())
            if not gt or not item_scores:
                continue

            # Blend scores
            blended = {iid: float(np.dot(weight_vector, np.array(ms))) for iid, ms in item_scores.items()}
            ranked = sorted(item_scores.keys(), key=lambda x: blended.get(x, 0.0), reverse=True)

            # Direct imputation: predicted reward for top-10
            imputed = 0.0
            for rank, item_id in enumerate(ranked[:10]):
                p_observe = 1.0 / math.log2(rank + 2)
                # Predict click probability as normalized blend score
                pred_click = min(blended.get(item_id, 0.0), 1.0)
                imputed += p_observe * pred_click

            # IPS correction for ground truth items
            correction = 0.0
            for gt_item in gt:
                if gt_item in item_scores:
                    ips_w = get_ips_weight(gt_item, propensities, clip_val=clip_val)
                    new_rank = ranked.index(gt_item) if gt_item in ranked else len(ranked)
                    p_new = 1.0 / math.log2(new_rank + 2) if new_rank < 10 else 0.0
                    pred_gt = min(blended.get(gt_item, 0.0), 1.0)
                    correction += ips_w * (1.0 - pred_gt) * p_new

            dr_total += imputed + correction
            n_users += 1

        return dr_total / max(n_users, 1)

    logger.info("Running DR weight selection with %d candidates...", num_candidates)
    best_score, best_wv = 0.0, None
    results = []

    for _ in range(num_candidates):
        wv = np.random.dirichlet([1.0] * len(WEIGHT_KEYS))
        score = _dr_score(wv)
        results.append((score, wv))
        if score > best_score:
            best_score, best_wv = score, wv

    results.sort(key=lambda x: x[0], reverse=True)
    logger.info("Top-3 DR weight vectors:")
    for i, (score, wv) in enumerate(results[:3]):
        ws = ", ".join(f"{k}={wv[j]:.3f}" for j, k in enumerate(WEIGHT_KEYS))
        logger.info("  #%d: DR=%.4f | %s", i + 1, score, ws)

    if best_wv is None:
        return dict(_DEFAULT_WEIGHTS)

    best_wv = np.maximum(best_wv, 0.0)
    best_wv /= best_wv.sum()
    best_weights = {k: float(best_wv[i]) for i, k in enumerate(WEIGHT_KEYS)}

    output = {
        **best_weights,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "dr_score": round(best_score, 6),
        "method": "doubly_robust_ips",
        "num_candidates_evaluated": num_candidates,
    }
    out_path = MODELS_DIR / "ensemble_weights.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info("DR-optimized weights saved to %s (DR=%.4f)", out_path, best_score)
    return best_weights


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal debiasing: IPS-weighted LightGCN + DR ensemble weight selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--epochs", type=int, default=100, help="IPS-BPR training epochs")
    p.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    p.add_argument("--clip-val", type=float, default=10.0, help="IPS weight clip value")
    p.add_argument("--dr-candidates", type=int, default=200, help="DR weight search candidates")
    p.add_argument("--skip-lgcn", action="store_true", help="Skip LightGCN IPS training")
    p.add_argument("--skip-dr", action="store_true", help="Skip DR weight selection")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if not args.skip_lgcn:
        logger.info("=" * 60)
        logger.info("Step 1: IPS-weighted LightGCN training")
        logger.info("=" * 60)
        train_lightgcn_ips(epochs=args.epochs, lr=args.lr, clip_val=args.clip_val)

    if not args.skip_dr:
        logger.info("=" * 60)
        logger.info("Step 2: Doubly Robust ensemble weight selection")
        logger.info("=" * 60)
        weights = select_weights_doubly_robust(
            num_candidates=args.dr_candidates,
            clip_val=args.clip_val,
        )
        logger.info("Final DR weights: %s", {k: round(v, 3) for k, v in weights.items()})

    logger.info("Causal debiasing complete.")
