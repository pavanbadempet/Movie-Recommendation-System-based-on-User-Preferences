"""
Free-tier-safe API usage logging.

This is intentionally JSONL-backed so the public demo does not need Postgres,
Redis, or a paid observability vendor. It can later be streamed into Kafka or
loaded into the Delta event fact table.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Fast JSON for usage log writes
try:
    import orjson as _orjson
    def _usage_dumps(obj) -> str: return _orjson.dumps(obj, option=_orjson.OPT_SORT_KEYS).decode()
    def _usage_loads(s): return _orjson.loads(s)
except ImportError:
    def _usage_dumps(obj) -> str: return json.dumps(obj, sort_keys=True, ensure_ascii=True)
    def _usage_loads(s): return json.loads(s)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_USAGE_PATH = REPO_ROOT / "data" / "events" / "api_usage.jsonl"


def get_usage_path() -> Path:
    configured = os.getenv("NOVA_USAGE_PATH")
    return Path(configured) if configured else DEFAULT_USAGE_PATH


def record_usage(
    operation: str,
    tenant_id: str,
    catalog_id: str,
    plan: str = "demo",
    authenticated: bool = False,
    status: str = "ok",
) -> dict[str, Any]:
    """Append one API usage event and return the persisted record."""
    record = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "operation": operation,
        "tenant_id": tenant_id,
        "catalog_id": catalog_id,
        "plan": plan,
        "authenticated": authenticated,
        "status": status,
    }
    path = get_usage_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(_usage_dumps(record))
        fh.write("\n")
    return record


def summarize_usage(limit: int = 20) -> dict[str, Any]:
    """Aggregate lightweight usage metrics for the console/admin API."""
    path = get_usage_path()
    operation_counts: Counter[str] = Counter()
    tenant_counts: Counter[str] = Counter()
    total = 0
    last_seen = None

    if path.exists():
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = _usage_loads(line)
                except (json.JSONDecodeError, Exception):
                    continue
                total += 1
                operation_counts[str(record.get("operation") or "unknown")] += 1
                tenant_key = f"{record.get('tenant_id', 'unknown')}:{record.get('catalog_id', 'unknown')}"
                tenant_counts[tenant_key] += 1
                last_seen = max(last_seen or "", str(record.get("ts") or "")) or None

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "usage_log_path": str(path),
        "total_requests": total,
        "last_seen": last_seen,
        "operation_counts": dict(operation_counts.most_common(max(limit, 0))),
        "tenant_counts": dict(tenant_counts.most_common(max(limit, 0))),
    }
