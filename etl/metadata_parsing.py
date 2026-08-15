"""Metadata parsing helpers shared by ETL implementations."""

from __future__ import annotations

import ast


def parse_metadata_name_list(value) -> str:
    """Normalize Kaggle list/dict metadata strings into comma-separated names."""
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return ", ".join(part.strip() for part in text.split(",") if part.strip())

    if isinstance(parsed, list):
        names = []
        for item in parsed:
            if isinstance(item, dict):
                name = str(item.get("name") or "").strip()
                if name:
                    names.append(name)
            elif item:
                names.append(str(item).strip())
        return ", ".join(names)
    if isinstance(parsed, dict):
        return str(parsed.get("name") or "").strip()
    return str(parsed).strip()
