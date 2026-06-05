"""
Benchmark the 6-way Ensemble against individual models.

Metrics:
  - NDCG@10 (Normalized Discounted Cumulative Gain)
  - HitRatio@10
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from backend.ensemble_engine import get_apex_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def calculate_metrics(ranked_list, true_items):
    """Calculate HitRatio and NDCG at top 10."""
    hr = 0.0
    ndcg = 0.0

    true_items_set = set(true_items)
    for i, item in enumerate(ranked_list[:10]):
        if item in true_items_set:
            hr = 1.0
            ndcg = 1.0 / np.log2(i + 2)
            break  # Since it's usually leave-one-out or small target set

    return hr, ndcg


def main():
    logger.info("=" * 60)
    logger.info("BENCHMARKING: APEX Ensemble vs Individual Models")
    logger.info("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Load data
    ratings = pd.read_parquet(DATA_DIR / "ratings_transformed.parquet")
    ratings_sorted = ratings.sort_values("timestamp")

    # Simulate a leave-one-out evaluation protocol
    # For each user, the last interaction is the target.
    # The pool of candidates is the target item + 99 random negative items.

    user_histories = ratings_sorted.groupby("userId")["movieId"].apply(list).to_dict()
    all_items = ratings["movieId"].unique()

    # 2. Load the Ensemble Engine
    # It automatically loads all 5 weights from models/ + PySpark Gold Embeddings
    # Match the sizes from the trained models!
    engine = get_apex_engine(num_users=610, num_items=9724)
    engine.eval()

    num_eval_users = 200  # Sample for speed
    eval_users = np.random.choice(list(user_histories.keys()), size=num_eval_users, replace=False)

    metrics = {
        "Ensemble": {"hr": [], "ndcg": []},
        "SASRec": {"hr": [], "ndcg": []},
        "LightGCN": {"hr": [], "ndcg": []},
        "Quantum": {"hr": [], "ndcg": []},
        "Hyperbolic": {"hr": [], "ndcg": []},
        "KAN": {"hr": [], "ndcg": []},
    }

    logger.info(f"Evaluating on {num_eval_users} users with 100 candidates each...")

    with torch.no_grad():
        for user_id in tqdm(eval_users, desc="Benchmarking"):
            history = user_histories[user_id]
            if len(history) < 5:
                continue

            target_item = history[-1]
            train_items = set(history[:-1])

            # 99 negatives
            negatives = set()
            while len(negatives) < 99:
                neg = np.random.choice(all_items)
                if neg not in train_items and neg != target_item:
                    negatives.add(neg)

            candidates = list(negatives) + [target_item]
            np.random.shuffle(candidates)

            # Map IDs to safe internal indices
            safe_user_id = user_id % engine.num_users
            safe_item_ids = [item_id % engine.num_items for item_id in candidates]

            # Use the actual engine!
            ensemble_score_dict = engine.predict_ensemble(safe_user_id, safe_item_ids)

            # Get individual scores for comparison
            u_tensor = torch.tensor([safe_user_id], dtype=torch.long)
            i_tensor = torch.tensor(safe_item_ids, dtype=torch.long)

            q_scores = engine.quantum.predict(u_tensor, i_tensor).squeeze()
            if q_scores.dim() == 0:
                q_scores = q_scores.unsqueeze(0)

            h_scores = -engine.hyperbolic.predict(u_tensor.expand_as(i_tensor), i_tensor)

            u_emb = engine.hyperbolic.user_embedding(u_tensor).expand(len(i_tensor), -1)
            i_emb = engine.hyperbolic.item_embedding(i_tensor)
            k_scores = engine.kan(u_emb, i_emb).squeeze()
            if k_scores.dim() == 0:
                k_scores = k_scores.unsqueeze(0)

            simulated_seq = torch.zeros((1, 50), dtype=torch.long)
            s_scores = engine.sasrec.predict(simulated_seq, i_tensor.unsqueeze(0)).squeeze()
            if s_scores.dim() == 0:
                s_scores = s_scores.unsqueeze(0)

            lgcn_u_emb = engine.lightgcn.user_embedding(u_tensor).expand(len(i_tensor), -1)
            lgcn_i_emb = engine.lightgcn.item_embedding(i_tensor)
            l_scores = (lgcn_u_emb * lgcn_i_emb).sum(dim=1)

            # Convert ensemble dict back to tensor for unified ranking
            e_scores = torch.tensor([ensemble_score_dict[i] for i in safe_item_ids], dtype=torch.float32)

            # Rank candidates
            def get_ranked_items(scores_tensor, _candidates=candidates):
                _, indices = torch.sort(scores_tensor, descending=True)
                return [_candidates[idx.item()] for idx in indices]

            ranks = {
                "Ensemble": get_ranked_items(e_scores),
                "SASRec": get_ranked_items(s_scores),
                "LightGCN": get_ranked_items(l_scores),
                "Quantum": get_ranked_items(q_scores),
                "Hyperbolic": get_ranked_items(h_scores),
                "KAN": get_ranked_items(k_scores),
            }

            for model_name, ranked_list in ranks.items():
                hr, ndcg = calculate_metrics(ranked_list, [target_item])
                metrics[model_name]["hr"].append(hr)
                metrics[model_name]["ndcg"].append(ndcg)

    logger.info("=" * 60)
    logger.info("RESULTS (Leave-One-Out, 100 candidates)")
    logger.info("=" * 60)
    logger.info(f"{'Model':<15} | {'HR@10':<10} | {'NDCG@10':<10}")
    logger.info("-" * 40)

    sorted_models = sorted(metrics.keys(), key=lambda k: np.mean(metrics[k]["ndcg"]), reverse=True)

    ensemble_ndcg = np.mean(metrics["Ensemble"]["ndcg"])
    best_individual_name = [m for m in sorted_models if m != "Ensemble"][0]
    best_individual_ndcg = np.mean(metrics[best_individual_name]["ndcg"])

    for model_name in sorted_models:
        hr = np.mean(metrics[model_name]["hr"])
        ndcg = np.mean(metrics[model_name]["ndcg"])
        prefix = "🌟 " if model_name == "Ensemble" else "   "
        logger.info(f"{prefix}{model_name:<12} | {hr:.4f}     | {ndcg:.4f}")

    logger.info("=" * 60)

    if ensemble_ndcg > best_individual_ndcg:
        lift = ((ensemble_ndcg / best_individual_ndcg) - 1) * 100
        logger.info(f"✅ Ensemble beats best individual model ({best_individual_name}) by +{lift:.1f}% NDCG@10")
    else:
        gap = ((best_individual_ndcg / ensemble_ndcg) - 1) * 100 if ensemble_ndcg > 0 else float("inf")
        logger.warning(
            f"⚠️  Ensemble does not beat best individual model ({best_individual_name}). "
            f"Gap: {gap:.1f}%. Consider re-running optimize_ensemble_weights.py."
        )

    # Persist results to reports/
    from datetime import UTC, datetime
    import json

    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "protocol": "leave_one_out_100_candidates",
        "num_eval_users": num_eval_users,
        "results": {
            model: {
                "hr_at_10": float(np.mean(metrics[model]["hr"])),
                "ndcg_at_10": float(np.mean(metrics[model]["ndcg"])),
            }
            for model in metrics
        },
        "ensemble_beats_best_individual": bool(ensemble_ndcg > best_individual_ndcg),
        "best_individual_model": best_individual_name,
        "ensemble_lift_pct": float(((ensemble_ndcg / best_individual_ndcg) - 1) * 100)
        if best_individual_ndcg > 0
        else None,
    }

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / "model_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
