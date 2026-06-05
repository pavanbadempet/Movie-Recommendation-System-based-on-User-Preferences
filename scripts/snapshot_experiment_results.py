"""
Snapshot current A/B experiment metrics to:
  1. reports/experiment_results.json  (human-readable, committed to repo)
  2. experiment_results_snapshot table in Postgres (if DATABASE_URL is set)

Run manually or from the daily data-refresh workflow:
  python scripts/snapshot_experiment_results.py
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import logging
import os
from pathlib import Path
import sys

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def run() -> dict:
    from backend.experiments import summarize_experiment_metrics

    logger.info("Summarising experiment metrics from event store...")
    summary = summarize_experiment_metrics()
    experiments = summary.get("experiments", [])

    if not experiments:
        logger.warning("No experiment events found. Is the event store populated?")
        return summary

    # Log a readable table
    logger.info(
        "%-30s %-20s %8s %8s %8s %8s %10s %10s",
        "experiment",
        "variant",
        "events",
        "clicks",
        "ratings",
        "ctr",
        "avg_rating",
        "p_value",
    )
    for row in experiments:
        logger.info(
            "%-30s %-20s %8d %8d %8d %8.4f %10s %10s",
            row.get("experiment", ""),
            row.get("variant", ""),
            row.get("events", 0),
            row.get("clicks", 0),
            row.get("ratings", 0),
            row.get("ctr", 0.0),
            str(row.get("avg_rating", "N/A")),
            str(row.get("p_value", "N/A")),
        )

    # Identify winners
    for exp_name in {r["experiment"] for r in experiments}:
        rows = [r for r in experiments if r["experiment"] == exp_name]
        control = next((r for r in rows if r["variant"] == "control"), None)
        treatments = [r for r in rows if r["variant"] != "control"]
        for t in treatments:
            if control and t.get("significant"):
                ctrl_ctr = control.get("ctr", 0) or 0
                t_ctr = t.get("ctr", 0) or 0
                lift = ((t_ctr / ctrl_ctr) - 1) * 100 if ctrl_ctr > 0 else 0
                logger.info(
                    "✅ %s/%s beats control: CTR lift +%.1f%%, p=%.4f",
                    exp_name,
                    t["variant"],
                    lift,
                    t.get("p_value", 0),
                )
            elif control and not t.get("significant"):
                logger.info(
                    "⏳ %s/%s: not yet significant (p=%s, need more traffic)",
                    exp_name,
                    t["variant"],
                    t.get("p_value", "N/A"),
                )

    # Write JSON report
    REPORTS_DIR.mkdir(exist_ok=True)
    report_path = REPORTS_DIR / "experiment_results.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Report written to %s", report_path)

    # Persist to Postgres if available
    db_url = os.getenv("NOVA_EVENT_DATABASE_URL") or os.getenv("DATABASE_URL")
    if db_url:
        _persist_to_postgres(experiments, db_url)

    return summary


def _persist_to_postgres(experiments: list[dict], db_url: str) -> None:
    try:
        import psycopg

        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                for row in experiments:
                    cur.execute(
                        """
                        INSERT INTO experiment_results_snapshot
                            (experiment, variant, events, impressions, clicks, ratings,
                             avg_rating, ctr, p_value, significant, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            row.get("experiment"),
                            row.get("variant"),
                            row.get("events", 0),
                            row.get("impressions", 0),
                            row.get("clicks", 0),
                            row.get("ratings", 0),
                            row.get("avg_rating"),
                            row.get("ctr"),
                            row.get("p_value") if isinstance(row.get("p_value"), float) else None,
                            bool(row.get("significant", False)),
                            json.dumps({"notes": row.get("notes")}),
                        ),
                    )
            conn.commit()
        logger.info("Experiment snapshot persisted to Postgres.")
    except Exception as exc:
        logger.warning("Could not persist to Postgres (non-fatal): %s", exc)


if __name__ == "__main__":
    run()
