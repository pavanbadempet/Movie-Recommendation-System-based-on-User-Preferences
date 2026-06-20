"""Generate an evidence-backed internal fairness diagnostic."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"


class FairnessAuditor:
    """Compute bounded fairness diagnostics from supplied production evidence."""

    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self.movies = self.load_data()

    def load_data(self) -> pd.DataFrame:
        movies_path = self.data_dir / "movies_transformed.parquet"
        if not movies_path.is_file():
            raise FileNotFoundError(
                f"Required catalog evidence not found: {movies_path}. "
                "Run the production data pipeline before auditing."
            )
        movies = pd.read_parquet(movies_path)
        if "id" not in movies.columns:
            raise ValueError(f"Catalog evidence has no id column: {movies_path}")
        return movies

    def measure_popularity_bias(self, recommended_items: list[list[int]]) -> float:
        """Calculate the Gini coefficient of observed recommendation slates."""
        item_counts: dict[int, int] = {}
        total_recs = 0
        for rec_list in recommended_items:
            for item in rec_list:
                item_counts[int(item)] = item_counts.get(int(item), 0) + 1
                total_recs += 1
        if total_recs == 0:
            raise ValueError("At least one observed recommendation is required")

        counts = np.sort(np.array(list(item_counts.values()), dtype=np.float64))
        n = len(counts)
        index = np.arange(1, n + 1)
        return float(np.sum((2 * index - n - 1) * counts) / (n * np.sum(counts)))

    def measure_calibration(self, user_history_genres: dict, recommended_genres: dict) -> float:
        """Calculate KL divergence between observed history and recommendation genres."""

        def normalize(values: dict) -> dict:
            if not values:
                raise ValueError("Genre evidence must not be empty")
            total = sum(float(value) for value in values.values())
            if total <= 0:
                raise ValueError("Genre evidence totals must be positive")
            return {key: float(value) / total for key, value in values.items()}

        p = normalize(user_history_genres)
        q = normalize(recommended_genres)
        kl_div = 0.0
        for genre in set(p) | set(q):
            px = p.get(genre, 1e-10)
            qx = q.get(genre, 1e-10)
            kl_div += px * np.log(px / qx)
        return float(kl_div)

    def generate_report(
        self,
        recommendation_slates: list[list[int]],
        user_history_genres: dict,
        recommended_genres: dict,
        privacy_evidence: dict | None = None,
    ) -> str:
        """Generate an internal diagnostic without making legal compliance claims."""
        logger.info("Generating evidence-backed fairness diagnostic...")
        gini = self.measure_popularity_bias(recommendation_slates)
        kl_div = self.measure_calibration(user_history_genres, recommended_genres)

        privacy_lines = ["- **Status:** NOT EVALUATED", "- No privacy-runtime evidence was supplied."]
        if privacy_evidence:
            mechanism = str(privacy_evidence.get("mechanism") or "").strip()
            epsilon = privacy_evidence.get("epsilon")
            source = str(privacy_evidence.get("source") or "").strip()
            if mechanism and epsilon is not None and source:
                privacy_lines = [
                    "- **Status:** EVIDENCE PROVIDED (manual verification required)",
                    f"- **Mechanism:** {mechanism}",
                    f"- **Epsilon:** {epsilon}",
                    f"- **Evidence source:** {source}",
                ]

        report = [
            "# Fairness Diagnostic Report",
            "",
            "> **Scope:** Internal engineering diagnostic only; this is not a compliance certification.",
            "> Results describe only the supplied recommendation and genre evidence.",
            "",
            "## Evidence Summary",
            f"- Observed slates: {len(recommendation_slates)}",
            f"- Observed recommendations: {sum(len(slate) for slate in recommendation_slates)}",
            "",
            "## Popularity Bias (Gini Coefficient)",
            f"- **Score:** {gini:.4f}",
            "- **Internal target:** < 0.70",
            "- **Threshold result:** " + ("WITHIN TARGET" if gini < 0.7 else "OUTSIDE TARGET"),
            "",
            "## Recommendation Calibration (KL Divergence)",
            f"- **Score:** {kl_div:.4f}",
            "- **Internal target:** < 0.50",
            "- **Threshold result:** " + ("WITHIN TARGET" if kl_div < 0.5 else "OUTSIDE TARGET"),
            "",
            "## Differential Privacy Evidence",
            *privacy_lines,
        ]
        return "\n".join(report)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="JSON file containing recommendation_slates, user_history_genres, and recommended_genres.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "FAIRNESS_AUDIT_REPORT.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    auditor = FairnessAuditor()
    report_md = auditor.generate_report(
        recommendation_slates=evidence["recommendation_slates"],
        user_history_genres=evidence["user_history_genres"],
        recommended_genres=evidence["recommended_genres"],
        privacy_evidence=evidence.get("privacy_evidence"),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report_md, encoding="utf-8")
    logger.info("Diagnostic written to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
