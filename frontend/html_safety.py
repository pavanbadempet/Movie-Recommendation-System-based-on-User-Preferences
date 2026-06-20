"""Safety helpers for values interpolated into Streamlit HTML fragments."""

from html import escape
import re
from urllib.parse import urlsplit

_YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_CSS_CLASS_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def escape_html(value: object) -> str:
    """Escape untrusted text for HTML text and quoted attribute contexts."""
    return escape(str(value if value is not None else ""), quote=True)


def valid_youtube_id(value: object) -> str | None:
    """Return a YouTube video ID only when it is safe for URL and JS contexts."""
    candidate = str(value or "")
    return candidate if _YOUTUBE_ID_RE.fullmatch(candidate) else None


def safe_https_url(value: object) -> str:
    """Return an escaped HTTPS URL, rejecting script and other active schemes."""
    candidate = str(value or "").strip()
    parsed = urlsplit(candidate)
    if parsed.scheme != "https" or not parsed.netloc:
        return ""
    return escape_html(candidate)


def safe_css_class(value: object) -> str:
    """Return a bounded CSS class token or a neutral fallback."""
    candidate = str(value or "")
    return candidate if _CSS_CLASS_RE.fullmatch(candidate) else "unknown"
