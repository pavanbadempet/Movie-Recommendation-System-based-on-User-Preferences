"""
Deterministic experiment assignment and metrics for Nova.

This stays intentionally lightweight: variants are assigned by stable hashing,
and outcome metrics are derived from the same behavior events customers send to
the product API.
"""

from __future__ import annotations

import hashlib
import os
from collections import defaultdict
from typing import Any, Iterable

from backend.events import iter_events, utc_now

DEFAULT_EXPERIMENT_NAME = "nova_personalization_v2"
DEFAULT_VARIANTS = {"control": 50, "personalized_v2": 50}


def configured_variants() -> dict[str, int]:
    """Parse NOVA_EXPERIMENT_VARIANTS as `name:weight,name:weight`."""
    configured = os.getenv("NOVA_EXPERIMENT_VARIANTS", "").strip()
    if not configured:
        return DEFAULT_VARIANTS.copy()

    variants: dict[str, int] = {}
    for part in configured.split(","):
        if ":" not in part:
            continue
        name, weight = part.split(":", 1)
        name = name.strip()
        try:
            parsed_weight = int(weight.strip())
        except ValueError:
            continue
        if name and parsed_weight > 0:
            variants[name] = parsed_weight
    return variants or DEFAULT_VARIANTS.copy()


def assign_experiment(
    subject_id: str | None,
    experiment_name: str | None = None,
    variants: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Assign a subject to a stable experiment variant."""
    experiment_name = experiment_name or os.getenv("NOVA_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    variants = variants or configured_variants()
    subject_id = str(subject_id or "anonymous")
    salt = os.getenv("NOVA_EXPERIMENT_SALT", "nova")
    total_weight = sum(variants.values())
    digest = hashlib.sha256(f"{salt}:{experiment_name}:{subject_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:12], 16) % total_weight

    cursor = 0
    selected = next(iter(variants))
    for variant, weight in variants.items():
        cursor += weight
        if bucket < cursor:
            selected = variant
            break

    return {
        "experiment": experiment_name,
        "variant": selected,
        "subject_id": subject_id,
        "bucket": bucket,
        "variants": variants,
    }


def attach_experiment(candidates: list[dict[str, Any]], assignment: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach experiment metadata to recommendation candidates."""
    enriched = []
    for candidate in candidates:
        item = dict(candidate)
        signals = dict(item.get("retrieval_signals") or {})
        signals["experiment"] = assignment["experiment"]
        signals["variant"] = assignment["variant"]
        item["retrieval_signals"] = signals
        enriched.append(item)
    return enriched


def summarize_experiment_metrics(events: Iterable[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Summarize variant-level behavior outcomes from product events."""
    events = events if events is not None else iter_events()
    metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "events": 0,
            "impressions": 0,
            "views": 0,
            "clicks": 0,
            "ratings": 0,
            "rating_sum": 0.0,
        }
    )

    for event in events:
        metadata = event.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        experiment = metadata.get("experiment") or event.get("experiment")
        variant = metadata.get("variant") or event.get("variant")
        if not experiment or not variant:
            continue

        key = f"{experiment}:{variant}"
        row = metrics[key]
        row["experiment"] = str(experiment)
        row["variant"] = str(variant)
        row["events"] += 1

        event_type = str(event.get("event_type") or "").lower()
        if event_type == "recommendation_impression":
            row["impressions"] += 1
        elif event_type == "view":
            row["views"] += 1
        elif event_type == "click":
            row["clicks"] += 1
        elif event_type == "rating":
            row["ratings"] += 1
            try:
                row["rating_sum"] += float(event.get("rating") or 0.0)
            except (TypeError, ValueError):
                pass

    rows = []
    for row in metrics.values():
        impressions = int(row["impressions"])
        clicks = int(row["clicks"])
        ratings = int(row["ratings"])
        row = dict(row)
        row["ctr"] = round(clicks / impressions, 6) if impressions else 0.0
        row["avg_rating"] = round(float(row.pop("rating_sum")) / ratings, 4) if ratings else None
        rows.append(row)

    rows.sort(key=lambda item: (item["experiment"], item["variant"]))
    return {
        "generated_at": utc_now(),
        "experiments": rows,
    }

