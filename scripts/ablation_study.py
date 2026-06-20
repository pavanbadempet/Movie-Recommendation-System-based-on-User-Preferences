"""
Ablation study for the APEX Movie Recommendation System ensemble.

Measures the marginal contribution of each of the 6 ensemble models
(LightGCN, Quantum, SASRec, KAN, Hyperbolic, Diffusion) by running
leave-one-out NDCG@10 evaluations.

Usage:
    python scripts/ablation_study.py [--sample-size N] [--output PATH]

Defaults:
    --sample-size  1000
    --output       reports/ablation_report.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import logging
import math
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np

# Ensure repo root is on sys.path when run as a script
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model names that participate in the ablation
# ---------------------------------------------------------------------------
ABLATION_MODELS = ("lightgcn", "quantum", "sasrec", "kan", "hyperbolic", "diffusion", "clifford")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ModelAblationResult:
    """Per-model result from a leave-one-out ablation run.

    Attributes
    ----------
    model:
        One of ``"lightgcn"``, ``"quantum"``, ``"sasrec"``, ``"kan"``,
        ``"hyperbolic"``, ``"diffusion"``.
    ndcg_without:
        NDCG@10 of the ensemble when this model is removed.
        ``None`` when the model failed to load or its weight could not be
        temporarily zeroed.
    delta:
        ``full_ensemble_ndcg - ndcg_without``.  Positive means the model
        helps; negative means it hurts.  ``None`` when ``ndcg_without`` is
        ``None``.
    marginal_contribution_pct:
        ``delta / full_ensemble_ndcg * 100``.  ``None`` when ``delta`` is
        ``None`` or ``full_ensemble_ndcg`` is zero.
    """

    model: str
    ndcg_without: float | None
    delta: float | None
    marginal_contribution_pct: float | None


@dataclass
class AblationReport:
    """Full ablation study report.

    Attributes
    ----------
    run_timestamp:
        ISO 8601 UTC timestamp of when the study was executed.
    full_ensemble_ndcg:
        NDCG@10 with all 6 models active.
    models:
        One ``ModelAblationResult`` per ablated model.
    """

    run_timestamp: str  # ISO 8601
    full_ensemble_ndcg: float
    models: list[ModelAblationResult]


# ---------------------------------------------------------------------------
# NDCG@10 helper
# ---------------------------------------------------------------------------


def _ndcg_at_10(rank: int | None) -> float:
    """Return NDCG@10 gain for a test item found at *rank* (0-indexed).

    Uses the formula ``1.0 / log2(rank + 2)`` when the item is in the top-10,
    otherwise returns 0.0.

    Parameters
    ----------
    rank:
        0-indexed position of the test item in the recommendation list, or
        ``None`` / any value >= 10 when the item is not in the top-10.
    """
    if rank is None or rank >= 10:
        return 0.0
    return 1.0 / math.log2(rank + 2)


# ---------------------------------------------------------------------------
# Synthetic evaluation data (fallback when event store is empty)
# ---------------------------------------------------------------------------


def _build_synthetic_eval_data(sample_size: int, num_items: int = 10_000) -> list[dict[str, Any]]:
    """Generate synthetic (user_id, seed_item_id, test_item_id) pairs for evaluation.

    Used when the event store contains no interaction data.  Each synthetic
    user is assigned a random seed item and a different test item.

    Parameters
    ----------
    sample_size:
        Number of synthetic users to generate.
    num_items:
        Upper bound (inclusive) for synthetic movie IDs.

    Returns
    -------
    list of dicts with keys ``user_id`` (int), ``seed_item_id`` (int), and ``test_item_id`` (int).
    """
    rng = random.Random(42)
    eval_data = []
    for i in range(sample_size):
        seed_item = rng.randint(1, num_items)
        test_item = rng.randint(1, num_items)
        while test_item == seed_item:
            test_item = rng.randint(1, num_items)
        eval_data.append({
            "user_id": i,
            "seed_item_id": seed_item,
            "test_item_id": test_item,
        })
    return eval_data



# ---------------------------------------------------------------------------
# Event-store sampling
# ---------------------------------------------------------------------------


def _sample_eval_data_from_events(sample_size: int) -> list[dict[str, Any]]:
    """Sample up to *sample_size* (user_id, seed_item_id, test_item_id) pairs from the event store.

    Reads all interaction events (click / rating / view), groups by user, and
    for each user with >= 2 interactions uses the second-most-recent as seed and
    the most-recent interaction as the held-out test item.
    Returns an empty list when the event store is unavailable or empty.

    Parameters
    ----------
    sample_size:
        Maximum number of users to include.

    Returns
    -------
    list of dicts with keys ``user_id``, ``seed_item_id``, and ``test_item_id``.
    """
    try:
        from backend.events import iter_events
    except ImportError as exc:
        logger.warning("Could not import event store: %s", exc)
        return []

    INTERACTION_TYPES = {"click", "rating", "view"}
    # user_id -> list of (event_ts, movie_id)
    user_interactions: dict[str, list[tuple[str, int]]] = {}

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
            uid_str = str(uid)
            if uid_str not in user_interactions:
                user_interactions[uid_str] = []
            user_interactions[uid_str].append((ts, mid))
    except Exception as exc:
        logger.warning("Failed to read event store: %s", exc)
        return []

    if not user_interactions:
        return []

    # For each user with >= 2 interactions, pick the second-most-recent as seed and most-recent as test
    eval_data: list[dict[str, Any]] = []
    for uid_str, interactions in user_interactions.items():
        interactions.sort(key=lambda x: x[0])  # sort ascending by timestamp
        if len(interactions) < 2:
            continue
        seed_item_id = interactions[-2][1]
        test_item_id = interactions[-1][1]
        try:
            eval_data.append({
                "user_id": int(uid_str),
                "seed_item_id": seed_item_id,
                "test_item_id": test_item_id,
            })
        except ValueError:
            eval_data.append({
                "user_id": uid_str,
                "seed_item_id": seed_item_id,
                "test_item_id": test_item_id,
            })

    # Shuffle deterministically and cap at sample_size
    rng = random.Random(42)
    rng.shuffle(eval_data)
    return eval_data[:sample_size]


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


def _evaluate_ndcg(
    recommender: Any,
    eval_data: list[dict[str, Any]],
) -> float:
    """Compute mean NDCG@10 over *eval_data* using *recommender*.

    For each pair, calls ``recommender.recommend_by_id(seed_item_id, n=10)``
    and checks whether ``test_item_id`` appears in the top-10 results.

    Parameters
    ----------
    recommender:
        A loaded ``Recommender`` instance (or any object with a
        ``recommend_by_id(movie_id, n)`` method).
    eval_data:
        List of dicts with ``user_id``, ``seed_item_id``, and ``test_item_id`` keys.

    Returns
    -------
    Mean NDCG@10 across all evaluated users.  Returns 0.0 when *eval_data*
    is empty or all recommendations fail.
    """
    if not eval_data:
        return 0.0

    ndcg_scores: list[float] = []

    for i, sample in enumerate(eval_data):
        seed_item_id = sample.get("seed_item_id")
        test_item_id = sample["test_item_id"]
        query_item_id = seed_item_id if seed_item_id is not None else test_item_id
        try:
            recs = recommender.recommend_by_id(int(query_item_id), n=10)
        except Exception as exc:
            logger.debug("recommend_by_id failed for test_item=%s: %s", query_item_id, exc)
            ndcg_scores.append(0.0)
            continue

        rec_ids = [r.get("id") for r in recs if r.get("id") is not None]

        if test_item_id in rec_ids:
            rank = rec_ids.index(test_item_id)
            ndcg_scores.append(_ndcg_at_10(rank))
        else:
            ndcg_scores.append(0.0)

        if (i + 1) % 100 == 0:
            logger.info(
                "Evaluation progress: %d/%d users (running NDCG@10=%.4f)",
                i + 1,
                len(eval_data),
                float(np.mean(ndcg_scores)),
            )

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


# ---------------------------------------------------------------------------
# AblationStudy class
# ---------------------------------------------------------------------------


class AblationStudy:
    """Measure the marginal contribution of each ensemble model via leave-one-out.

    Parameters
    ----------
    recommender:
        A loaded ``Recommender`` instance.  The recommender must expose
        ``recommend_by_id(movie_id, n)`` and, optionally, an
        ``ensemble_engine`` attribute that is an ``ApexEnsembleEngine``
        instance (used for weight manipulation).
    sample_size:
        Number of users to evaluate per run.  Defaults to 1000.
    """

    def __init__(self, recommender: Any, sample_size: int = 1000) -> None:
        self.recommender = recommender
        self.sample_size = sample_size
        self._eval_data: list[dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Lazy evaluation data — built once, reused across all runs
    # ------------------------------------------------------------------

    def _get_eval_data(self) -> list[dict[str, Any]]:
        """Return (and cache) the evaluation dataset.

        Tries the event store first; falls back to synthetic data when the
        event store is empty or unavailable.
        """
        if self._eval_data is not None:
            return self._eval_data

        logger.info("Sampling up to %d users from event store …", self.sample_size)
        data = _sample_eval_data_from_events(self.sample_size)

        if not data:
            logger.warning(
                "Event store is empty or unavailable — using synthetic evaluation data (sample_size=%d).",
                self.sample_size,
            )
            data = _build_synthetic_eval_data(self.sample_size)

        logger.info("Evaluation dataset ready: %d samples.", len(data))
        self._eval_data = data
        return self._eval_data

    # ------------------------------------------------------------------
    # Ensemble engine access
    # ------------------------------------------------------------------

    def _get_ensemble_engine(self) -> Any | None:
        """Return the ``ApexEnsembleEngine`` attached to the recommender, or None."""
        # The recommender may expose the engine directly or via a lazy getter
        for attr in ("ensemble_engine", "_ensemble_engine", "_apex_engine"):
            engine = getattr(self.recommender, attr, None)
            if engine is not None:
                return engine

        # Try the module-level singleton as a last resort
        try:
            from backend.models.ensemble_engine import get_apex_engine

            return get_apex_engine()
        except Exception as exc:
            logger.warning("Could not access ApexEnsembleEngine: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Public evaluation methods
    # ------------------------------------------------------------------

    def run_full_ensemble(self) -> float:
        """Evaluate NDCG@10 with all 6 models active.

        Returns
        -------
        Mean NDCG@10 over the evaluation dataset.
        """
        logger.info("Running full-ensemble evaluation (all 6 models active) …")
        if hasattr(self.recommender, "_rec_cache"):
            self.recommender._rec_cache.clear()
        ndcg = _evaluate_ndcg(self.recommender, self._get_eval_data())
        logger.info("Full-ensemble NDCG@10 = %.4f", ndcg)
        return ndcg

    def run_leave_one_out(self, model_name: str) -> float | None:
        """Evaluate NDCG@10 with *model_name* temporarily removed from the ensemble.

        The model is removed by setting its blend weight to 0.0 and
        re-normalising the remaining weights so they sum to 1.0.  The
        original weights are restored after evaluation regardless of whether
        the evaluation succeeds or fails.

        Parameters
        ----------
        model_name:
            One of the six model names: ``"lightgcn"``, ``"quantum"``,
            ``"sasrec"``, ``"kan"``, ``"hyperbolic"``, ``"diffusion"``.

        Returns
        -------
        Mean NDCG@10 without *model_name*, or ``None`` if the model's weight
        could not be accessed (e.g. the engine failed to load).
        """
        logger.info("Leave-one-out: removing '%s' …", model_name)

        engine = self._get_ensemble_engine()
        if engine is None:
            logger.warning(
                "Cannot access ensemble engine — recording ndcg_without=None for '%s'.",
                model_name,
            )
            return None

        # Snapshot current weights (thread-safe read)
        try:
            with engine._weights_lock:
                original_weights: dict[str, float] = dict(engine._weights)
        except Exception as exc:
            logger.warning(
                "Failed to read ensemble weights for '%s': %s — recording None.",
                model_name,
                exc,
            )
            return None

        if model_name not in original_weights:
            logger.warning(
                "Model '%s' not found in ensemble weights — recording None.",
                model_name,
            )
            return None

        # Build modified weights: zero out the target model, re-normalise
        modified_weights = dict(original_weights)
        modified_weights[model_name] = 0.0
        remaining_sum = sum(modified_weights.values())

        if remaining_sum > 1e-9:
            modified_weights = {k: v / remaining_sum for k, v in modified_weights.items()}
        else:
            # All other weights are also zero — degenerate ensemble
            logger.warning(
                "All remaining weights are zero after removing '%s'; evaluation will use uniform fallback.",
                model_name,
            )
            n_remaining = len(modified_weights) - 1
            if n_remaining > 0:
                uniform = 1.0 / n_remaining
                modified_weights = {k: (0.0 if k == model_name else uniform) for k in modified_weights}

        # Apply modified weights
        try:
            with engine._weights_lock:
                engine._weights = modified_weights
        except Exception as exc:
            logger.warning(
                "Failed to apply modified weights for '%s': %s — recording None.",
                model_name,
                exc,
            )
            return None

        # Run evaluation with the modified ensemble
        ndcg: float | None
        try:
            if hasattr(self.recommender, "_rec_cache"):
                self.recommender._rec_cache.clear()
            ndcg = _evaluate_ndcg(self.recommender, self._get_eval_data())
            logger.info("Leave-one-out NDCG@10 without '%s' = %.4f", model_name, ndcg)
        except Exception as exc:
            logger.warning(
                "Evaluation failed for leave-one-out '%s': %s — recording None.",
                model_name,
                exc,
            )
            ndcg = None
        finally:
            # Always restore original weights
            try:
                with engine._weights_lock:
                    engine._weights = original_weights
                logger.debug("Restored original weights after removing '%s'.", model_name)
            except Exception as exc:
                logger.error(
                    "CRITICAL: Failed to restore ensemble weights after removing '%s': %s. "
                    "The ensemble may be in an inconsistent state.",
                    model_name,
                    exc,
                )

        return ndcg

    def run_all(self) -> AblationReport:
        """Run the full ablation study.

        Executes:
        1. Full-ensemble NDCG@10 evaluation.
        2. Six leave-one-out evaluations (one per model).

        Returns
        -------
        ``AblationReport`` with all results populated.
        """
        logger.info(
            "Starting ablation study (sample_size=%d, models=%s) …",
            self.sample_size,
            ABLATION_MODELS,
        )

        # Warm up the evaluation dataset once
        _ = self._get_eval_data()

        full_ndcg = self.run_full_ensemble()

        model_results: list[ModelAblationResult] = []
        for model_name in ABLATION_MODELS:
            ndcg_without = self.run_leave_one_out(model_name)

            if ndcg_without is not None:
                delta: float | None = full_ndcg - ndcg_without
                if full_ndcg > 1e-9:
                    marginal_pct: float | None = delta / full_ndcg * 100.0
                else:
                    marginal_pct = None
            else:
                delta = None
                marginal_pct = None

            model_results.append(
                ModelAblationResult(
                    model=model_name,
                    ndcg_without=ndcg_without,
                    delta=delta,
                    marginal_contribution_pct=marginal_pct,
                )
            )

        report = AblationReport(
            run_timestamp=datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            full_ensemble_ndcg=full_ndcg,
            models=model_results,
        )

        logger.info("Ablation study complete.")
        return report

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def print_table(self, report: AblationReport) -> None:
        """Print a formatted ablation table to stdout.

        Columns: Model | NDCG@10 Without | Delta | Marginal Contribution %

        Parameters
        ----------
        report:
            The ``AblationReport`` to display.
        """
        header = f"{'Model':<14} {'NDCG@10 Without':>17} {'Delta':>10} {'Marginal Contribution %':>24}"
        separator = "-" * len(header)

        print()
        print(f"Ablation Study Report — {report.run_timestamp}")
        print(f"Full Ensemble NDCG@10: {report.full_ensemble_ndcg:.4f}")
        print(separator)
        print(header)
        print(separator)

        for result in report.models:
            ndcg_str = f"{result.ndcg_without:.4f}" if result.ndcg_without is not None else "N/A (load failure)"
            delta_str = f"{result.delta:+.4f}" if result.delta is not None else "N/A"
            pct_str = (
                f"{result.marginal_contribution_pct:+.2f}%" if result.marginal_contribution_pct is not None else "N/A"
            )
            print(f"{result.model:<14} {ndcg_str:>17} {delta_str:>10} {pct_str:>24}")

        print(separator)
        print()

    def save_report(self, report: AblationReport, output_path: Path) -> None:
        """Serialize *report* to JSON at *output_path*.

        Creates the parent directory if it does not exist.

        Parameters
        ----------
        report:
            The ``AblationReport`` to serialize.
        output_path:
            Destination file path.  The file is written with UTF-8 encoding
            and 2-space indentation.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert dataclasses to plain dicts for JSON serialization
        report_dict = asdict(report)

        output_path.write_text(
            json.dumps(report_dict, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Ablation report written to '%s'.", output_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an ablation study on the APEX ensemble to measure the "
            "marginal contribution of each model via leave-one-out NDCG@10."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        metavar="N",
        help="Number of users to evaluate per run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/ablation_report.json"),
        metavar="PATH",
        help="Path to write the ablation report JSON.",
    )
    return parser


def main() -> None:
    """CLI entry point for the ablation study."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    logger.info(
        "Ablation study starting (sample_size=%d, output=%s) …",
        args.sample_size,
        args.output,
    )

    # Load the recommender
    try:
        from backend.pipeline.recommender import get_recommender

        recommender = get_recommender()
        logger.info("Recommender loaded successfully.")

        # Monkey-patch get_contextual_weights to prevent weights from being overridden during leave-one-out runs
        try:
            import backend.models.neural_weight_optimizer
            from backend.models.ensemble_engine import get_apex_engine
            engine = get_apex_engine()
            if engine is not None:
                backend.models.neural_weight_optimizer.get_contextual_weights = lambda behavior_profile, als_user_embedding=None: engine._weights
                logger.info("Successfully monkey-patched get_contextual_weights to respect in-memory weight changes.")
        except Exception as e:
            logger.warning("Could not monkey-patch get_contextual_weights: %s", e)
    except Exception as exc:
        logger.error("Failed to load recommender: %s", exc)
        logger.warning("Proceeding with a stub recommender — results will reflect synthetic evaluation data only.")
        recommender = _StubRecommender()

    # Run the study
    study = AblationStudy(recommender=recommender, sample_size=args.sample_size)
    report = study.run_all()

    # Display and save
    study.print_table(report)
    study.save_report(report, args.output)

    logger.info("Done. Report saved to '%s'.", args.output)


# ---------------------------------------------------------------------------
# Stub recommender — used when the real recommender cannot be loaded
# ---------------------------------------------------------------------------


class _StubRecommender:
    """Minimal recommender stub for CI / offline testing.

    Returns a deterministic list of 10 movie IDs derived from the seed ID.
    This allows the ablation script to run end-to-end without a loaded model.
    """

    def recommend_by_id(self, movie_id: int, n: int = 10) -> list[dict[str, Any]]:
        """Return *n* deterministic recommendations seeded by *movie_id*."""
        rng = random.Random(movie_id)
        ids = random.sample(range(1, 10_001), min(n, 10_000))
        # Use a seeded RNG for determinism
        rng.shuffle(ids)
        return [{"id": mid, "title": f"Movie {mid}"} for mid in ids[:n]]


if __name__ == "__main__":
    main()
