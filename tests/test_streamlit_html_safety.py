import importlib.util
from pathlib import Path


def test_streamlit_html_helpers_escape_text_and_reject_unsafe_attributes():
    spec = importlib.util.find_spec("frontend.html_safety")
    assert spec is not None, "Streamlit HTML rendering needs a separately testable safety boundary"

    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.escape_html('<img src=x onerror="alert(1)">') == (
        "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;"
    )
    assert module.valid_youtube_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"
    assert module.valid_youtube_id('bad"><script') is None
    assert module.safe_https_url("javascript:alert(1)") == ""
    assert module.safe_https_url("https://image.tmdb.org/t/p/original/poster.jpg").startswith("https://")
    assert module.safe_css_class('rating" onclick="alert(1)') == "unknown"


def test_streamlit_app_uses_html_safety_boundary_for_dynamic_markup():
    source = Path("frontend/streamlit_app.py").read_text(encoding="utf-8")

    assert "from frontend.html_safety import" in source
    assert "valid_youtube_id(youtube_key)" in source
    assert "valid_youtube_id(trailer_key)" in source
    assert "safe_event_type = safe_css_class(event_type)" in source
    assert 'event.get("title", "Unknown Movie")}</strong>' not in source
