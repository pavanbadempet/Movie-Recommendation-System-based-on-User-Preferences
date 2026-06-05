import logging
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVENTS_DIR = PROJECT_ROOT / "data" / "events"


class CounterfactualReplayEngine:
    """
    Simulates how a new recommendation model would have performed historically
    without deploying it to production. Uses Inverse Propensity Scoring (IPS)
    and Doubly Robust (DR) estimation to correct for the "position bias" of the
    historical logging policy.
    """

    def __init__(self, clip_val: float = 100.0):
        self.clip_val = clip_val  # Max weight to prevent infinite variance

    def _compute_propensity(self, rank: int, max_rank: int = 10) -> float:
        """
        Estimates the probability (propensity) that a user observed an item at `rank`.
        Users are much more likely to see and click rank 1 than rank 10.
        Uses a standard Position Bias Model (PBM): P(observe|rank) ~ 1/log2(rank+1)
        """
        if rank < 0 or rank >= max_rank:
            return 0.01  # Floor propensity
        return 1.0 / np.log2(rank + 2)

    def evaluate_ips(self, historical_logs: list[dict[str, Any]], new_policy_func) -> float:
        """
        Evaluates a new model using Inverse Propensity Scoring (IPS).

        Args:
            historical_logs: List of dicts {user_id, recommended_items, clicked_item_id, positions}
            new_policy_func: Function f(user_id) -> list of predicted item IDs

        Returns:
            Estimated CTR (Click-Through Rate) of the new model
        """
        ips_estimator = 0.0
        total_sessions = len(historical_logs)

        if total_sessions == 0:
            return 0.0

        for log in historical_logs:
            user_id = log["user_id"]
            clicked_item = log.get("clicked_item_id")

            if not clicked_item:
                continue  # No reward in this session

            # Get historical propensity (probability logging policy showed this item)
            hist_rank = log["recommended_items"].index(clicked_item) if clicked_item in log["recommended_items"] else -1
            if hist_rank == -1:
                continue

            p_log = self._compute_propensity(hist_rank)

            # Simulate what the NEW model would have done
            new_slate = new_policy_func(user_id)

            # Check if the new model would have recommended the clicked item
            new_rank = new_slate.index(clicked_item) if clicked_item in new_slate else -1

            if new_rank != -1:
                p_new = self._compute_propensity(new_rank)

                # Importance weight: P(new) / P(log)
                weight = p_new / p_log
                weight = min(weight, self.clip_val)  # Clipping to reduce variance

                # Reward is 1.0 for a click
                ips_estimator += weight * 1.0

        return ips_estimator / total_sessions

    def evaluate_doubly_robust(
        self, historical_logs: list[dict[str, Any]], new_policy_func, reward_predictor_func
    ) -> float:
        """
        Doubly Robust (DR) Estimator: Combines IPS with a direct reward imputation model
        to lower variance while remaining unbiased.
        """
        dr_estimator = 0.0
        total_sessions = len(historical_logs)

        if total_sessions == 0:
            return 0.0

        for log in historical_logs:
            user_id = log["user_id"]
            clicked_item = log.get("clicked_item_id")
            new_slate = new_policy_func(user_id)

            # 1. Direct Imputation (What does our ML model think the reward will be?)
            imputed_reward = 0.0
            for rank, item in enumerate(new_slate):
                p_observe = self._compute_propensity(rank)
                # Predict probability of click
                pred_click_prob = reward_predictor_func(user_id, item)
                imputed_reward += p_observe * pred_click_prob

            # 2. IPS Correction
            correction = 0.0
            if clicked_item and clicked_item in log["recommended_items"]:
                hist_rank = log["recommended_items"].index(clicked_item)
                p_log = self._compute_propensity(hist_rank)

                new_rank = new_slate.index(clicked_item) if clicked_item in new_slate else -1
                p_new = self._compute_propensity(new_rank) if new_rank != -1 else 0.0

                weight = min(p_new / p_log, self.clip_val)

                # Predict reward for the historical item
                pred_hist_click = reward_predictor_func(user_id, clicked_item)

                # Correction: Add weight * (Actual Reward - Predicted Reward)
                actual_reward = 1.0
                correction = weight * (actual_reward - pred_hist_click)

            dr_estimator += imputed_reward + correction

        return dr_estimator / total_sessions


def generate_comparison_report(model_a_name: str, model_b_name: str, logs: list[dict], policy_a, policy_b) -> str:
    """Runs a counterfactual simulation and generates a deployment report."""
    engine = CounterfactualReplayEngine()

    logger.info(f"Replaying {len(logs)} historical sessions for {model_a_name}...")
    ips_a = engine.evaluate_ips(logs, policy_a)

    logger.info(f"Replaying {len(logs)} historical sessions for {model_b_name}...")
    ips_b = engine.evaluate_ips(logs, policy_b)

    improvement = ((ips_b - ips_a) / max(ips_a, 1e-5)) * 100

    report = [
        "## Offline Counterfactual Replay Report",
        f"- **Historical Sessions Replayed:** {len(logs):,}",
        "",
        "### Estimated Click-Through Rate (CTR)",
        f"- **{model_a_name} (Control):** {ips_a:.4f}",
        f"- **{model_b_name} (Treatment):** {ips_b:.4f}",
        "",
        "### Deployment Decision",
        f"- **Relative Improvement:** {improvement:+.2f}%",
    ]

    if improvement > 1.0:
        report.append("- **Gate Status:** ✅ APPROVED FOR DEPLOYMENT (Estimated Lift > 1%)")
    else:
        report.append("- **Gate Status:** ❌ REJECTED (Failed to beat baseline in counterfactual simulation)")

    return "\n".join(report)
