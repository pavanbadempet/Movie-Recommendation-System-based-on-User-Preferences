# Movie Recommendation System - Premium UI
# Run: streamlit run app.py

import streamlit as st
import streamlit.components.v1 as components
import requests
import time
import os
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
from st_clickable_images import clickable_images

st.set_page_config(
    page_title="Nova Recommendation Console",
    page_icon="N",
    layout="wide"
)

# New Streamlit 1.54.0 Feature: Native Logo Support
st.logo("https://upload.wikimedia.org/wikipedia/commons/e/e4/Movie-icon.svg", icon_image=":material/movie:")

# Premium CSS - Hide branding + full-screen dark theme + WHITE TEXT
st.markdown("""
<style>
/* Hide Streamlit branding */
#MainMenu, footer, header {visibility: hidden;}
.stDeployButton {display: none;}

/* Dark theme base */
.stApp {
    background: #0a0a0f;
}

/* MAKE ALL TEXT WHITE */
.stApp, .stMarkdown, .stText, p, span, label, .stCaption, h1, h2, h3, h4, h5, h6 {
    color: #ffffff !important;
}

/* Streamlit titles and headers */
.stTitle, .stHeader, [data-testid="stHeader"] {
    color: #ffffff !important;
}

/* Captions and small text */
.stCaption, .caption, small {
    color: #ffffff !important;
}

/* Metric labels and values */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #ffffff !important;
}

/* Remove white header padding */
.block-container {
    padding-top: 1rem !important;
    padding-bottom: 0 !important;
}

/* Style tabs - scrollable with full titles */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,0,0,0.6);
    border-radius: 10px;
    padding: 5px;
    overflow-x: auto;
    flex-wrap: nowrap !important;
    gap: 5px;
}
.stTabs [data-baseweb="tab"] {
    color: #fff !important;
    font-weight: 500;
    white-space: nowrap !important;
    min-width: fit-content !important;
}

/* Style selectbox */
.stSelectbox > div > div {
    background: #1a1a2e;
    color: #fff !important;
}

/* Red primary button */
.stButton > button {
    background: linear-gradient(135deg, #e50914 0%, #b81d24 100%);
    color: white !important;
    border: none;
    font-weight: bold;
    padding: 0.5rem 2rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #ff1a1a 0%, #d32f2f 100%);
}
</style>
""", unsafe_allow_html=True)

# Config
# Support multiple backends for high availability (failover)
# Add your HF Space URL first, keep Render as fallback
BACKEND_URLS = [
    os.getenv("API_URL", ""),  # From Hugging Face Secrets or local env
    "https://pavanbadempet-movie-rec-api.hf.space", # Primary (Hugging Face)
    "https://movie-recs-api-5qvy.onrender.com" # Fallback (Render)
]
# Clean up empty strings
BACKEND_URLS = [url.strip("/") for url in BACKEND_URLS if url]

TMDB_KEY = os.getenv("TMDB_API_KEY")
NOVA_API_KEY = os.getenv("NOVA_API_KEY", "")
NOVA_TENANT_ID = os.getenv("NOVA_TENANT_ID", "demo-media-co")
NOVA_CATALOG_ID = os.getenv("NOVA_CATALOG_ID", "tmdb-movies")

# Validate TMDB Key
if not TMDB_KEY:
    st.warning("⚠️ TMDB_API_KEY not set. Posters and trailers will not load. Set it in your environment or Streamlit secrets.")


def api_headers(extra_headers=None):
    """Return optional Nova product headers without requiring paid auth infra."""
    headers = dict(extra_headers or {})
    if NOVA_API_KEY:
        headers["X-Nova-API-Key"] = NOVA_API_KEY
    else:
        headers["X-Tenant-ID"] = NOVA_TENANT_ID
        headers["X-Catalog-ID"] = NOVA_CATALOG_ID
    return headers


def api_get(path, params=None, timeout=15):
    """Call the active backend with the current tenant/API-key context."""
    if not st.session_state.get("backend_ready"):
        return None
    api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
    try:
        response = requests.get(
            f"{api_url}{path}",
            params=params,
            headers=api_headers(),
            timeout=timeout,
        )
        if response.ok:
            return response.json()
    except requests.RequestException:
        return None
    return None


def api_post(path, payload, timeout=15):
    """Post to the active backend with the current tenant/API-key context."""
    if not st.session_state.get("backend_ready"):
        return None
    api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
    try:
        response = requests.post(
            f"{api_url}{path}",
            json=payload,
            headers=api_headers(),
            timeout=timeout,
        )
        if response.ok:
            return response.json()
        return {"error": response.text, "status_code": response.status_code}
    except requests.RequestException as exc:
        return {"error": str(exc)}


@st.cache_data(ttl=600)
def fetch_trailer(movie_id):
    """Get YouTube trailer key - cached."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/videos",
            params={"api_key": TMDB_KEY, "language": "en-US"},
            timeout=3
        )
        data = r.json()
        for v in data.get("results", []):
            if v.get("type") == "Trailer":
                return v.get("key")
        if data.get("results"):
            return data["results"][0].get("key")
    except (requests.RequestException, KeyError, IndexError):
        pass
    return None


@st.cache_data(ttl=600)
def fetch_poster(poster_path):
    """Get full poster URL - cached."""
    if poster_path and not poster_path.startswith("http"):
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return "https://via.placeholder.com/500x750?text=No+Poster"


@st.cache_data(ttl=600)
def fetch_tmdb_details(movie_id):
    """Fetch movie details from TMDB - cached."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_KEY},
            timeout=3
        )
        return r.json()
    except (requests.RequestException, ValueError):
        return {}


@st.cache_data(ttl=600)
def fetch_credits(movie_id):
    """Fetch cast and crew from TMDB - cached."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/credits",
            params={"api_key": TMDB_KEY},
            timeout=3
        )
        data = r.json()
        cast = [c["name"] for c in data.get("cast", [])[:3]]
        director = next((c["name"] for c in data.get("crew", []) if c.get("job") == "Director"), "Unknown")
        return {"cast": ", ".join(cast), "director": director}
    except (requests.RequestException, KeyError, TypeError):
        return {"cast": "N/A", "director": "N/A"}


@st.cache_data(ttl=600)
def fetch_watch_providers(movie_id):
    """Fetch watch providers (streaming) from TMDB - cached."""
    try:
        r = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}/watch/providers",
            params={"api_key": TMDB_KEY},
            timeout=3
        )
        data = r.json()
        results = data.get("results", {})
        
        # Priority: IN (India) -> US -> First available
        providers = results.get("IN",results.get("US", {}))
        
        # We only care about "flatrate" (subscription) for now
        flatrate = providers.get("flatrate", [])
        return flatrate
    except (requests.RequestException, KeyError, TypeError):
        return []



def wake_up_backend():
    """
    Wake up backend and find an active server from the failover list.
    """
    # 1. Fast check existing known-good URL
    if "API_URL" in st.session_state:
        try:
            r = requests.get(f"{st.session_state.API_URL}/health", timeout=2)
            if r.ok:
                return True
        except requests.RequestException:
            pass # Move to full scan

    # 2. Sequential ping of all configured backends
    with st.spinner("🚀 Booting up the recommendation engine... (This can take ~45s if waking from sleep)"):
        # We try all backends. Max 30 loops = ~60-90 seconds total wait.
        for _ in range(30):
            for url in BACKEND_URLS:
                try:
                    r = requests.get(f"{url}/health", timeout=3)
                    if r.ok:
                        st.session_state.API_URL = url
                        domain = url.split("//")[-1].split(".")[0]
                        st.toast(f"✅ Connected to {domain}!", icon="⚡")
                        return True
                except requests.RequestException:
                    pass
            time.sleep(2)
        
    return False

# Initialize connection on app load
if "backend_ready" not in st.session_state or "API_URL" not in st.session_state:
    st.session_state.backend_ready = wake_up_backend()

def search_movies(query):
    """Search movies via API."""
    if not st.session_state.backend_ready:
        st.error("⚠️ The engine is still waking up. Give it a moment to stretch its legs.")
        return []
        
    try:
        api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
        r = requests.get(
            f"{api_url}/v1/search",
            params={"q": query, "limit": 100},
            headers=api_headers(),
            timeout=10,
        )
        if r.ok:
            return r.json()
    except requests.RequestException:
        st.error("⚠️ Lost connection to the brain. It might be restarting.")
    return []


def ai_search_movies(query):
    """Hybrid AI search via sparse+dense/rerank endpoint."""
    if not st.session_state.backend_ready:
        st.error("⚠️ The engine is still waking up. Give it a moment.")
        return []

    try:
        api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
        r = requests.get(
            f"{api_url}/v1/search/ai",
            params={"q": query, "limit": 100},
            headers=api_headers(),
            timeout=30,
        )
        if r.ok:
            return r.json()
    except requests.RequestException:
        st.error("⚠️ AI search is unavailable right now.")
    return []


@st.cache_data(ttl=3600)
def fetch_all_movie_titles(version=3):
    """Fetch all movie titles for the autocomplete dropdown."""
    if not st.session_state.backend_ready:
        return []
        
    try:
        api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
        r = requests.get(f"{api_url}/movies/titles", timeout=20)
        if r.ok:
            data = r.json()
            if data:
                return data  # Returns list of {"id": X, "title": "Y"}
    except Exception:
        pass
        
    # If we get here, it failed to load. Clear the cache so it retries later instead of serving [] for an hour.
    fetch_all_movie_titles.clear()
    return []


def get_recommendations(movie_id, n=10):
    """Get recommendations via API."""
    try:
        api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
        r = requests.get(
            f"{api_url}/v1/recommendations/id/{movie_id}",
            params={"n": n},
            headers=api_headers(),
            timeout=30,
        )
        if r.ok:
            return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
    return {}


def display_fullscreen_video(youtube_key):
    """Display YouTube video as dimmed background."""
    if not youtube_key:
        return
    
    # Simple dimmed video background - NO overlay affecting top UI
    video_html = """
    <style>
    .video-container {
        width: 60vw;
        height: 100vh;
        position: absolute;
        min-width: 80%; 
        filter: brightness(35%);
        pointer-events: none;
    }
    
    .video-container iframe {
        position: absolute;
        top: 52.5%;
        left: 60%;
        width: 100vw;
        height: 100vh;
        transform: translate(-50%, -50%);
        pointer-events: none;
    }
    </style>
""" + f"""
    <div class="video-container">
        <iframe src="https://www.youtube.com/embed/{youtube_key}?controls=0&autoplay=1&mute=1&loop=1&playlist={youtube_key}&modestbranding=1&showinfo=0&rel=0&iv_load_policy=3&disablekb=1" frameborder="0" allow="autoplay"></iframe>
    </div>
    """
    st.markdown(video_html, unsafe_allow_html=True)


def display_movie_card(rec, tmdb, credits, similarity):
    """Premium movie detail card with full details."""
    title = rec.get("title", "Unknown")
    year = tmdb.get("release_date", "")[:4] if tmdb.get("release_date") else "N/A"
    rating = rec.get("vote_average", 0)
    votes = int(rec.get("vote_count", 0))
    genres = rec.get("genres", "N/A")
    overview = rec.get("overview", "No overview available.")
    runtime = tmdb.get("runtime", 0)
    budget = tmdb.get("budget", 0)
    revenue = tmdb.get("revenue", 0)
    popularity = rec.get("popularity", 0)
    cast = credits.get("cast", "N/A")
    director = credits.get("director", "N/A")
    
    # Format budget/revenue in millions
    budget_m = f"${budget // 1000000}M" if budget else "N/A"
    revenue_m = f"${revenue // 1000000}M" if revenue else "N/A"
    
    # Card container styling - pure white text for maximum visibility
    st.markdown("""
    <style>
    .card-container {
        background: rgba(0,0,0,0.95);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.2);
        box-shadow: 0 8px 32px rgba(0,0,0,0.8);
    }
    .movie-title-main {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff !important;
        margin-bottom: 5px;
        text-shadow: 2px 2px 6px rgba(0,0,0,1);
    }
    .movie-subtitle {
        color: #ffffff !important;
        font-size: 1rem;
        margin-bottom: 10px;
        text-shadow: 1px 1px 4px rgba(0,0,0,1);
    }
    .match-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        padding: 6px 16px;
        border-radius: 15px;
        font-weight: 700;
        font-size: 0.9rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    .genre-pill {
        display: inline-block;
        background: rgba(229,9,20,0.5);
        color: #ffffff !important;
        padding: 4px 12px;
        border-radius: 10px;
        font-size: 0.8rem;
        margin: 2px 2px;
        font-weight: 500;
    }
    .detail-label {
        color: #ffffff !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        font-weight: 600;
        text-shadow: 1px 1px 2px rgba(0,0,0,1);
    }
    .detail-value {
        color: #ffffff !important;
        font-size: 1rem;
        font-weight: 600;
        text-shadow: 1px 1px 3px rgba(0,0,0,1);
    }
    /* Make Streamlit metrics white */
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #ffffff !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="card-container">
        <div class="movie-title-main">{title}</div>
        <div class="movie-subtitle">{year} {'• ' + str(runtime) + ' min' if runtime else ''} {'• ⭐ ' + str(round(rating, 1)) + '/10' if rating else ''}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Genre tags
    genre_html = "".join([f'<span class="genre-pill">{g.strip()}</span>' for g in str(genres).split(',')[:4]])
    st.markdown(f'<div style="margin: 8px 0;">{genre_html}</div>', unsafe_allow_html=True)
    
    # Match badge
    match_pct = int(similarity * 100)
    st.markdown(f'<div class="match-badge">🎯 {match_pct}% Match</div>', unsafe_allow_html=True)
    
    # Details grid - Cast, Director, Popularity
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.8); padding: 12px; border-radius: 10px; margin-bottom: 10px;">
        <div class="detail-label">🎬 Director</div>
        <div class="detail-value">{director}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.8); padding: 12px; border-radius: 10px; margin-bottom: 10px;">
        <div class="detail-label">🎭 Cast</div>
        <div class="detail-value">{cast}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Votes", f"{votes:,}")
    c2.metric("Budget", budget_m)
    c3.metric("Revenue", revenue_m)
    c4.metric("Popularity", f"{popularity:.0f}")
    
    # Overview
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background: rgba(0,0,0,0.8); padding: 12px; border-radius: 10px;">
        <div class="detail-label">📝 Overview</div>
        <div style="color: #ffffff !important; line-height: 1.6; font-size: 0.9rem; margin-top: 5px; text-shadow: 1px 1px 2px rgba(0,0,0,1);">{overview}</div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog(" ", width="large")
def show_movie_dialog(rec):
    """Show modal dialog with full movie details."""
    
    # --- CSS: Style Dialog (No Scrollbar, Full Height) ---
    st.markdown("""
        <style>
            /* TARGET: The main dialog container */
            div[data-testid="stDialog"], div[role="dialog"] {
                padding: 0 !important;
                margin: 0 !important;
                border: none !important;
                background-color: #000 !important;
                box-shadow: none !important;
            }
            /* Make dialog as tall as viewport */
            div[role="dialog"] > div {
                max-height: 95vh !important;
                height: auto !important;
            }
            /* Remove padding from content container */
            div[role="dialog"] > div > div {
                padding: 0 !important;
                border: none !important;
                background-color: #000 !important;
            }
            /* HIDE SCROLLBAR but allow scrolling if needed */
            div[role="dialog"] section[tabindex="0"] {
                scrollbar-width: none !important; /* Firefox */
                -ms-overflow-style: none !important; /* IE/Edge */
            }
            div[role="dialog"] section[tabindex="0"]::-webkit-scrollbar {
                display: none !important; /* Chrome/Safari */
                width: 0 !important;
            }
            /* Remove gap from vertical block */
            div[data-testid="stVerticalBlock"] {
                gap: 0 !important;
                padding: 0 !important;
                background-color: #000 !important;
            }
            /* HIDE THE HEADER */
            div[data-testid="stDialog"] header {
                display: none;
            }
            /* SCALE UP CLOSE BUTTON */
            div[data-testid="stDialog"] button[aria-label="Close"] {
                transform: scale(1.5);
                background-color: rgba(0,0,0,0.5);
                border-radius: 50%;
                color: white;
                z-index: 9999;
            }
            /* Full width content */
            section[tabindex="0"], section[tabindex="0"] > div, section[tabindex="0"] > div > div {
                 padding: 0 !important;
                 background-color: #000 !important;
            }
        </style>
    """, unsafe_allow_html=True)

    with st.spinner("Fetching details..."):
        movie_id = rec.get("id")
        tmdb = fetch_tmdb_details(movie_id)
        credits = fetch_credits(movie_id)
        trailer_key = fetch_trailer(movie_id)
        providers = fetch_watch_providers(movie_id)
        
    # === CINEMATIC BILLBOARD HEADER ===
    # Use the same style as the home page hero
    # Logic moved to components.html section below

    # Extract Metadata
    genres = ", ".join([g["name"] for g in tmdb.get("genres", [])[:3]]) if tmdb.get("genres") else rec.get("genres", "")
    runtime = f"{tmdb.get('runtime', 0)} min" if tmdb.get('runtime') else ""
    rating = tmdb.get("vote_average", rec.get("vote_average", 0))
    year = tmdb.get("release_date", "")[:4] or "N/A"
    
    # Prepare Provider HTML (Pre-computation for embedding)
    provider_html = ""
    if providers:
        cards = ""
        for p in providers[:4]: # Limit to 4 to save space
            logo = f"https://image.tmdb.org/t/p/original{p.get('logo_path')}"
            name = p.get('provider_name')
            # Create a Google Search link for the movie on this provider
            query = f"watch {rec.get('title')} on {name}"
            url = f"https://www.google.com/search?q={query}"
            
            cards += f'<a href="{url}" target="_blank" style="text-decoration:none; cursor:pointer;"><div style="display:inline-block; margin-right:10px; text-align:center; transition: transform 0.2s;"><img src="{logo}" style="width:40px; border-radius:8px; box-shadow:0 2px 5px rgba(0,0,0,0.5);" title="Watch on {name}"></div></a>'
        
        provider_html = f'<div class="db-providers"><div style="font-size:0.7rem; color:#aaa; margin-bottom:5px; text-transform:uppercase; letter-spacing:1px; font-weight:bold;">Watch Now</div>{cards}</div>'

    # Truncate overview for compact display
    overview_text = rec.get('overview', '')
    if len(overview_text) > 200:
        overview_text = overview_text[:200].rsplit(' ', 1)[0] + '...'

    # Build clickable Google search links for director and cast
    import urllib.parse
    director_name = credits.get('director', 'Unknown')
    director_link = f'<a href="https://www.google.com/search?q={urllib.parse.quote(director_name)}" target="_blank" class="credit-link">{director_name}</a>'
    
    cast_names = credits.get('cast', '').split(', ')
    cast_links = ', '.join([
        f'<a href="https://www.google.com/search?q={urllib.parse.quote(name.strip())}" target="_blank" class="credit-link">{name.strip()}</a>'
        for name in cast_names if name.strip()
    ])

    # Calculate rating percentage for radial progress bar
    rating_pct = (rating / 10) * 100
    rating_color = "#21d07a" if rating >= 7 else "#d2d531" if rating >= 5 else "#db2360"
    
    # === RENDER BILLBOARD WITH COMPONENTS.HTML FOR JS SUPPORT ===
    # Using components.html allows JavaScript execution for click-to-mute
    
    player_id = f"player_{movie_id}"
    
    # YouTube Player API for reliable mute control
    if trailer_key:
        player_id = f"ytplayer_{movie_id}"
        video_html = f'<div id="{player_id}" class="db-video-layer"></div>'
        youtube_js = f'''
        <script>
            var tag = document.createElement('script');
            tag.src = "https://www.youtube.com/iframe_api";
            var firstScriptTag = document.getElementsByTagName('script')[0];
            firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
            
            var player;
            var isMuted = true;
            var muteBtn = document.getElementById('muteBtn');
            
            function onYouTubeIframeAPIReady() {{
                player = new YT.Player('{player_id}', {{
                    videoId: '{trailer_key}',
                    playerVars: {{
                        'autoplay': 1,
                        'mute': 1,
                        'controls': 0,
                        'disablekb': 1,
                        'modestbranding': 1,
                        'loop': 1,
                        'playlist': '{trailer_key}',
                        'playsinline': 1,
                        'rel': 0,
                        'showinfo': 0,
                        'iv_load_policy': 3,
                        'fs': 0
                    }},
                    events: {{
                        'onReady': onPlayerReady
                    }}
                }});
            }}
            
            function onPlayerReady(event) {{
                event.target.playVideo();
                // Apply pointer-events:none to the generated iframe
                var playerDiv = document.getElementById('{player_id}');
                if (playerDiv) {{
                    var iframe = playerDiv.querySelector('iframe');
                    if (iframe) {{
                        iframe.style.pointerEvents = 'none';
                    }}
                }}
            }}
            
            function toggleMute() {{
                if (player && typeof player.isMuted === 'function') {{
                    if (player.isMuted()) {{
                        player.unMute();
                        isMuted = false;
                        muteBtn.innerHTML = '🔊';
                    }} else {{
                        player.mute();
                        isMuted = true;
                        muteBtn.innerHTML = '🔇';
                    }}
                }}
            }}
            
            muteBtn.addEventListener('click', toggleMute);
        </script>
        '''
    else:
        poster_url = fetch_poster(tmdb.get("backdrop_path") or rec.get("poster_path"))
        video_html = f'<div class="db-video-layer"><img src="{poster_url}" alt="backdrop"></div>'
        youtube_js = ""
    
    explanation_html = ""
    if rec.get("explanation_text"):
        explanation_html = f'<div style="font-size: 0.85rem; color: #fff; background: rgba(229,9,20,0.2); border-left: 3px solid #e50914; padding: 8px 12px; margin: 10px 0; border-radius: 4px; animation: fadeInUp 0.6s ease-out 0.45s both;"><strong>🤖 CineBot Vibe Check:</strong> {rec.get("explanation_text")}</div>'

    billboard_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ background: #000; font-family: 'Montserrat', sans-serif; overflow: hidden; }}
            
            @keyframes fadeInUp {{
                from {{ opacity: 0; transform: translateY(20px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; }}
                to {{ opacity: 1; }}
            }}
            
            .dialog-billboard {{
                background: #000;
                position: relative;
                width: 100%;
                height: 100vh;
                overflow: hidden;
                cursor: pointer;
                animation: fadeIn 0.5s ease-out;
            }}
            .db-video-layer {{
                position: absolute;
                top: -10%; left: 0;
                width: 100%; height: 120%;
                z-index: 1;
                opacity: 0.7;
            }}
            .db-video-layer iframe, .db-video-layer img {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                pointer-events: none;
            }}
            .db-content-layer {{
                position: absolute;
                bottom: 0; left: 0;
                width: 100%;
                padding: 25px 35px;
                z-index: 2;
                background: linear-gradient(to top, #000 30%, rgba(0,0,0,0.85) 60%, transparent 100%);
                animation: fadeInUp 0.6s ease-out 0.2s both;
            }}
            .db-title-row {{
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 8px;
            }}
            .db-title {{
                font-family: 'Bebas Neue', sans-serif;
                font-size: 2.5rem;
                line-height: 1;
                color: #fff;
                text-shadow: 2px 2px 8px rgba(0,0,0,1);
            }}
            .rating-circle {{
                position: relative;
                width: 48px;
                height: 48px;
                flex-shrink: 0;
            }}
            .rating-circle svg {{
                transform: rotate(-90deg);
            }}
            .rating-circle .bg {{
                fill: none;
                stroke: #204529;
                stroke-width: 4;
            }}
            .rating-circle .progress {{
                fill: none;
                stroke: {rating_color};
                stroke-width: 4;
                stroke-linecap: round;
                stroke-dasharray: 126;
                stroke-dashoffset: {126 - (126 * rating_pct / 100)};
                transition: stroke-dashoffset 1s ease-out;
            }}
            .rating-circle .value {{
                position: absolute;
                top: 50%; left: 50%;
                transform: translate(-50%, -50%);
                font-size: 0.85rem;
                font-weight: 700;
                color: #fff;
            }}
            .db-meta {{
                font-size: 0.85rem;
                color: #e50914;
                font-weight: 700;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
                animation: fadeInUp 0.6s ease-out 0.3s both;
            }}
            .db-overview {{
                font-size: 0.9rem;
                color: #ddd;
                line-height: 1.5;
                margin-bottom: 10px;
                display: -webkit-box;
                -webkit-line-clamp: 3;
                -webkit-box-orient: vertical;
                overflow: hidden;
                animation: fadeInUp 0.6s ease-out 0.4s both;
            }}
            .db-credits {{
                font-size: 0.8rem;
                color: #aaa;
                margin-bottom: 10px;
                animation: fadeInUp 0.6s ease-out 0.5s both;
            }}
            .db-credits strong {{ 
                color: #fff;
                transition: color 0.2s;
            }}
            .db-credits strong:hover {{
                color: #e50914;
            }}
            .db-providers {{
                margin-top: 10px;
                padding-top: 12px;
                border-top: 1px solid rgba(255,255,255,0.15);
                animation: fadeInUp 0.6s ease-out 0.6s both;
            }}
            .db-providers .label {{
                font-size: 0.7rem;
                color: #aaa;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-weight: bold;
            }}
            .db-providers img {{
                width: 38px;
                border-radius: 8px;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-right: 8px;
            }}
            .db-providers img:hover {{
                transform: scale(1.15);
                box-shadow: 0 4px 15px rgba(229,9,20,0.4);
            }}
            /* Credit links */
            .credit-link {{
                color: #fff;
                text-decoration: none;
                font-weight: bold;
                transition: color 0.2s;
            }}
            .credit-link:hover {{
                color: #e50914;
            }}
            /* Mute button */
            #muteBtn {{
                position: absolute;
                top: 15px;
                right: 15px;
                width: 44px;
                height: 44px;
                border-radius: 50%;
                background: rgba(0,0,0,0.7);
                border: 2px solid rgba(255,255,255,0.3);
                color: #fff;
                font-size: 20px;
                cursor: pointer;
                z-index: 100;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s;
            }}
            #muteBtn:hover {{
                background: rgba(229,9,20,0.8);
                border-color: #e50914;
                transform: scale(1.1);
            }}

        </style>
    </head>
    <body>
        <div class="dialog-billboard">
            <button id="muteBtn">🔇</button>
            {video_html}
            <div class="db-content-layer">
                <div class="db-title-row">
                    <div class="db-title">{rec.get('title')}</div>
                    <div class="rating-circle">
                        <svg width="48" height="48">
                            <circle class="bg" cx="24" cy="24" r="20"></circle>
                            <circle class="progress" cx="24" cy="24" r="20"></circle>
                        </svg>
                        <div class="value">{rating:.1f}</div>
                    </div>
                </div>
                <div class="db-meta">{year} • {runtime} • {str(genres).split(',')[0]}</div>
                <div class="db-overview">{overview_text}</div>
                {explanation_html}
                <div class="db-credits">Directed by {director_link} • Cast: {cast_links}</div>
                {provider_html}
            </div>
        </div>
        {youtube_js}
    </body>
    </html>
    '''
    
    # Render using components.html for JS execution
    components.html(billboard_html, height=600, scrolling=False)


def format_option(m):
    """Format movie for dropdown - just title and year."""
    title = m.get("title", "Unknown")
    year = m.get("release_date", "")[:4] if m.get("release_date") else ""
    return f"{title} ({year})" if year else title


# ===== APP MODES =====

# ===== CUSTOM CSS FOR LANDING PAGE =====
st.markdown("""
<style>
/* Glass Card for Landing Page */
.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    cursor: pointer;
    height: 300px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
}

.glass-card:hover {
    background: rgba(255, 255, 255, 0.1);
    transform: translateY(-10px) scale(1.02);
    border-color: rgba(229, 9, 20, 0.5);
    box-shadow: 0 20px 40px rgba(229, 9, 20, 0.2);
}

.card-icon {
    font-size: 4rem;
    margin-bottom: 20px;
}

.card-title {
    font-size: 2rem;
    font-weight: 700;
    color: #fff;
    margin-bottom: 10px;
}

.card-desc {
    color: #aaa;
    font-size: 1rem;
}

/* Hide default button styles for the clickable area hack */
.stButton button {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ===== NAVIGATION STATE =====
if "page" not in st.session_state:
    st.session_state.page = "home"

# Initialize monitoring state
if "monitoring_data" not in st.session_state:
    st.session_state.monitoring_data = {
        "kafka_events": [],
        "delta_state": {},
        "system_metrics": {}
    }


def go_home():
    st.session_state.page = "home"
    # Clear search results when going home
    if "recs" in st.session_state:
        del st.session_state.recs
    if "source_movie" in st.session_state:
        del st.session_state.source_movie
    if "selected_rec" in st.session_state:
        del st.session_state.selected_rec

def go_search():
    st.session_state.page = "search"

def go_chat():
    st.session_state.page = "chat"

def go_monitoring():
    st.session_state.page = "monitoring"

def go_console():
    st.session_state.page = "console"


@st.cache_data(ttl=3600)
def fetch_trending_movies():
    """Fetch trending movies from TMDB for the welcome page."""
    try:
        r = requests.get(
            "https://api.themoviedb.org/3/trending/movie/week",
            params={"api_key": TMDB_KEY},
            timeout=3
        )
        return r.json().get("results", [])
    except Exception:
        return []

# ===== PAGE 1: LANDING SCREEN (MAIN SCENE) =====
if st.session_state.page == "home":
    # 1. Google Fonts Import & Cinematic CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Montserrat:wght@300;400;600&display=swap');
    
    /* Animated Background Layer */
    .stApp {
        background: radial-gradient(circle at 60% 50%, #1a1a2e 0%, #000000 100%);
        background-attachment: fixed;
    }
    
    h1 {
        font-family: 'Bebas Neue', sans-serif !important;
        font-size: 3.5rem !important;
        line-height: 1 !important;
        margin-bottom: 5px !important;
        background: linear-gradient(to right, #ffffff, #a5a5a5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    p, button, div { font-family: 'Montserrat', sans-serif !important; }
    
    /* Compact Holographic Row Card */
    .holo-card-row {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 15px 20px;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    .holo-card-row:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(229, 9, 20, 0.5);
        transform: translateX(5px);
    }
    
    .holo-icon { font-size: 2rem; }
    .holo-text h3 { margin: 0; color: #fff; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 1px; }
    .holo-text p { margin: 0; color: #888; font-size: 0.8rem; }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Split Layout: Left (Controls) | Right (Visuals)
    st.markdown("<div style='margin-top: 2vh;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 7], gap="large")
    
    # LEFT COLUMN: Title & Tools
    with c1:
        st.markdown("<h1>MOVIE RECOMMENDATION<br>SYSTEM</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #888; font-size: 1rem; margin-bottom: 30px;'>AI-Powered Curator • Deep Search • Semantic Analysis</p>", unsafe_allow_html=True)
        
        # === NAVIGATION CARDS ===
        st.markdown("""
        <style>
        /* Style buttons to look like cards */
        div[data-testid="stButton"] button[kind="secondary"] {
            width: 100% !important;
            min-height: 80px !important;
            padding: 18px 20px !important;
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
            margin-bottom: 12px !important;
        }
        
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            border-color: #e50914 !important;
            background: rgba(229, 9, 20, 0.1) !important;
            transform: translateY(-4px) !important;
            box-shadow: 0 15px 40px rgba(229, 9, 20, 0.25), 0 0 20px rgba(229, 9, 20, 0.15) !important;
        }
        
        div[data-testid="stButton"] button[kind="secondary"] p {
            color: white !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            margin: 0 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Product Console Card
        if st.button("NOVA CONSOLE", key="nav_console", use_container_width=True):
            go_console()
            st.rerun()

        # Search Card
        if st.button("🔍  DEEP SEARCH", key="nav_search", use_container_width=True):
            go_search()
            st.rerun()
        
        # Chat Card
        if st.button("🧬  CINEBOT AI", key="nav_chat", use_container_width=True):
            go_chat()
            st.rerun()

        # Monitoring Card
        if st.button("📊  REAL-TIME MONITORING", key="nav_monitoring", use_container_width=True):
            go_monitoring()
            st.rerun()

    # RIGHT COLUMN: Visual Showcase (Trending)
    with c2:
        trending = fetch_trending_movies()
        
        if trending:
            # Initialize Slideshow State
            if "hero_index" not in st.session_state:
                st.session_state.hero_index = 0
                
            # Circular Buffer Logic
            # CLICKABLE IMAGE LOGIC (Invisible Overlay Button Hack)
            # We use st.button(type="primary") as a dedicated "Invisible Click Layer"
            
            # Current Hero Movie
            if "hero_index" not in st.session_state:
                 st.session_state.hero_index = 0
            
            hero = trending[st.session_state.hero_index]
            
            # FETCH FULL DETAILS
            trailer_key = fetch_trailer(hero["id"])
            credits = fetch_credits(hero["id"])
            details = fetch_tmdb_details(hero["id"])
            
            # Extract Metadata
            genres = ", ".join([g["name"] for g in details.get("genres", [])[:2]])
            runtime = f"{details.get('runtime', 0)} min" if details.get('runtime') else ""
            
            # HERO BILLBOARD LAYOUT
            st.markdown(f"""
            <style>
            .billboard-container {{
                background: linear-gradient(135deg, rgba(26, 26, 46, 0.9), rgba(0, 0, 0, 0.95));
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 20px;
                padding: 20px;
                display: flex;
                gap: 20px;
                height: 40vh;
                box-shadow: 0 10px 40px rgba(0,0,0,0.5);
                backdrop-filter: blur(10px);
            }}
            .billboard-video {{
                flex: 1.4; 
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.5);
                background: #000;
            }}
            .billboard-info {{
                flex: 1; 
                display: flex;
                flex-direction: column;
                justify-content: center;
                overflow-y: hidden;
            }}
            .bb-title {{
                font-family: 'Bebas Neue', sans-serif;
                font-size: 2.2rem;
                line-height: 1;
                margin-bottom: 8px;
                color: #fff;
                text-transform: uppercase;
            }}
            .bb-meta {{
                font-family: 'Montserrat', sans-serif;
                font-size: 0.75rem;
                color: #ea696f; 
                font-weight: 700;
                margin-bottom: 10px;
            }}
            .bb-desc {{
                font-family: 'Montserrat', sans-serif;
                font-size: 0.8rem;
                color: #ccc;
                line-height: 1.4;
                margin-bottom: 15px;
                display: -webkit-box;
                -webkit-line-clamp: 4;
                -webkit-box-orient: vertical;
                overflow: hidden;
            }}
            .bb-credits {{
                font-family: 'Montserrat', sans-serif;
                font-size: 0.7rem;
                color: #888;
                border-top: 1px solid rgba(255,255,255,0.1);
                padding-top: 8px;
            }}
            /* MOVIE CARD BUTTONS - Styled as labels */
            .movie-card-btn {{
                background: transparent !important;
                border: none !important;
                padding: 0 !important;
                margin: 0 !important;
                width: 100% !important;
            }}
            .movie-card-btn > button {{
                background: transparent !important;
                border: 2px solid transparent !important;
                border-radius: 12px !important;
                padding: 8px !important;
                width: 100% !important;
                transition: all 0.3s ease !important;
            }}
            .movie-card-btn > button:hover {{
                border-color: #e50914 !important;
                background: rgba(229, 9, 20, 0.1) !important;
                transform: scale(1.02);
            }}
            .movie-card-btn > button:focus {{
                border-color: #e50914 !important;
                box-shadow: 0 0 15px rgba(229, 9, 20, 0.4) !important;
            }}
            </style>
            """, unsafe_allow_html=True)
            
            # Render Billboard
            video_embed = ""
            if trailer_key:
                video_embed = f'<iframe src="https://www.youtube.com/embed/{trailer_key}?autoplay=1&mute=1&controls=0&disablekb=1&modestbranding=1&loop=1&playlist={trailer_key}" style="width:100%; height:100%; border:none; pointer-events: none;"></iframe>'
            else:
                poster_url = fetch_poster(hero.get("backdrop_path"))
                video_embed = f'<img src="{poster_url}" style="width:100%; height:100%; object-fit:cover;">'

            st.markdown(f"""
            <div class="billboard-container">
                <div class="billboard-video">
                    {video_embed}
                </div>
                <div class="billboard-info">
                    <div class="bb-title">{hero.get('title')}</div>
                    <div class="bb-meta">⭐ {hero.get('vote_average', 0):.1f} | {genres} | {runtime}</div>
                    <div class="bb-desc">{hero.get('overview')}</div>
                    <div class="bb-credits">
                        Directed by <strong>{credits.get('director')}</strong><br>
                        Starring: {credits.get('cast')}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # SUB-GRID - TRUE CLICKABLE IMAGES (No Buttons!)
            st.markdown("<div style='margin-bottom: 8px; color: #666; font-size: 0.75rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 15px;'>More Trending</div>", unsafe_allow_html=True)
            
            # Prepare image URLs for clickable_images
            poster_urls = [fetch_poster(m.get("poster_path")) for m in trending[:5]]
            
            # Clickable Images Component - returns index of clicked image
            clicked = clickable_images(
                paths=poster_urls,
                titles=[m.get('title', '') for m in trending[:5]],
                div_style={
                    "display": "flex", 
                    "justify-content": "center", 
                    "flex-wrap": "wrap",
                    "gap": "10px"
                },
                img_style={
                    "width": "18%",
                    "border-radius": "10px",
                    "cursor": "pointer",
                    "border": "2px solid transparent",
                    "transition": "all 0.3s ease",
                    "box-shadow": "0 4px 10px rgba(0,0,0,0.3)"
                },
                key="trending_selector"
            )
            
            # Handle click - update hero when image is clicked
            if clicked > -1 and clicked != st.session_state.hero_index:
                st.session_state.hero_index = clicked
                st.rerun()
        else:
            st.info("Loading trends...")


# ===== PAGE 1B: NOVA PRODUCT CONSOLE =====
elif st.session_state.page == "console":
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("Home", key="back_console"):
            go_home()
            st.rerun()

    st.title("Nova Console")
    st.caption("B2B recommendation intelligence control plane")

    context = api_get("/v1/platform/context") or {}
    health = api_get("/health") or {}
    usage = api_get("/v1/usage") or {}
    behavior = api_get("/v1/events/features", params={"limit": 10}) or {}
    experiments = api_get("/v1/experiments/metrics") or {}

    metric_cols = st.columns(5)
    metric_cols[0].metric("Mode", context.get("mode", "offline"))
    metric_cols[1].metric("Tenant", context.get("tenant_id", NOVA_TENANT_ID))
    metric_cols[2].metric("Catalog", context.get("catalog_id", NOVA_CATALOG_ID))
    metric_cols[3].metric("Items", f"{health.get('movie_count', 0):,}")
    metric_cols[4].metric(
        "Event Store",
        behavior.get("event_store", "jsonl"),
        "durable" if behavior.get("durable") else "local",
    )

    overview_tab, catalog_tab, quality_tab, events_tab, integration_tab = st.tabs(
        ["Overview", "Catalog Onboarding", "AI Quality", "Events", "Integration"]
    )

    with overview_tab:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("API Usage")
            usage_rows = [
                {"operation": key, "requests": value}
                for key, value in (usage.get("operation_counts") or {}).items()
            ]
            if usage_rows:
                usage_df = pd.DataFrame(usage_rows)
                fig = px.bar(
                    usage_df,
                    x="operation",
                    y="requests",
                    color="operation",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                fig.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Usage metrics will appear after API requests.")

        with right:
            st.subheader("Behavior Signals")
            event_counts = behavior.get("event_type_counts") or {}
            if event_counts:
                event_df = pd.DataFrame(
                    [{"event_type": key, "count": value} for key, value in event_counts.items()]
                )
                fig = px.pie(
                    event_df,
                    values="count",
                    names="event_type",
                    hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Safe,
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#ffffff"),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Behavior signals appear after event collection.")

        st.subheader("Top Trending Content")
        trending = behavior.get("trending_movies") or {}
        if trending:
            trend_df = pd.DataFrame(list(trending.values()))
            shown_cols = [col for col in ["movie_id", "content_id", "event_count", "views", "clicks", "ratings", "avg_rating"] if col in trend_df.columns]
            st.dataframe(trend_df[shown_cols], use_container_width=True, hide_index=True)
        else:
            st.caption("No event-backed ranking signals yet.")

    with catalog_tab:
        st.subheader("Catalog Onboarding")
        st.caption("Upload a CSV, map customer columns, preview quality issues, then store a raw upload manifest.")

        uploaded_catalog = st.file_uploader("Customer catalog CSV", type=["csv"], key="catalog_uploader")
        c_map_1, c_map_2, c_map_3 = st.columns(3)
        with c_map_1:
            id_col = st.text_input("Source ID column", value="id")
            title_col = st.text_input("Title column", value="title")
            description_col = st.text_input("Description column", value="overview")
        with c_map_2:
            genres_col = st.text_input("Genres column", value="genres")
            people_col = st.text_input("People/cast column", value="cast")
            language_col = st.text_input("Language column", value="original_language")
        with c_map_3:
            release_col = st.text_input("Release date column", value="release_date")
            rating_col = st.text_input("Rating column", value="vote_average")
            popularity_col = st.text_input("Popularity column", value="popularity")

        mapping = {
            "source_content_id": id_col,
            "title": title_col,
            "description": description_col,
            "genres": genres_col,
            "people": people_col,
            "language": language_col,
            "release_date": release_col,
            "rating": rating_col,
            "popularity": popularity_col,
        }

        if uploaded_catalog:
            csv_text = uploaded_catalog.getvalue().decode("utf-8", errors="replace")
            st.caption(f"{uploaded_catalog.name} - {len(csv_text):,} characters")

            action_cols = st.columns([1, 1])
            with action_cols[0]:
                if st.button("Preview Catalog Quality", key="preview_catalog", use_container_width=True):
                    st.session_state.catalog_profile = api_post(
                        "/v1/catalog/preview",
                        {
                            "filename": uploaded_catalog.name,
                            "csv_text": csv_text,
                            "column_mapping": mapping,
                            "sample_size": 10,
                        },
                        timeout=30,
                    )
            with action_cols[1]:
                if st.button("Store Upload Manifest", key="upload_catalog", use_container_width=True):
                    st.session_state.catalog_upload_result = api_post(
                        "/v1/catalog/upload",
                        {
                            "filename": uploaded_catalog.name,
                            "csv_text": csv_text,
                            "column_mapping": mapping,
                            "sample_size": 10,
                        },
                        timeout=30,
                    )

            profile = st.session_state.get("catalog_profile")
            upload_result = st.session_state.get("catalog_upload_result")
            if upload_result and not upload_result.get("error"):
                st.success(f"Stored upload {upload_result.get('upload_id')}")
                st.caption(upload_result.get("manifest_path"))
                profile = upload_result.get("profile") or profile
            elif upload_result and upload_result.get("error"):
                st.error(upload_result.get("error"))

            if profile:
                if profile.get("error"):
                    st.error(profile.get("error"))
                else:
                    p1, p2, p3, p4 = st.columns(4)
                    p1.metric("Rows Profiled", f"{profile.get('total_rows_profiled', 0):,}")
                    p2.metric("Valid Rows", f"{profile.get('valid_rows', 0):,}")
                    p3.metric("Quality", profile.get("quality_score", 0))
                    p4.metric("Ready", "Yes" if profile.get("ready_for_ingestion") else "No")

                    warnings = profile.get("warnings") or []
                    if warnings:
                        st.warning(", ".join(warnings))
                    else:
                        st.success("No blocking catalog quality warnings found.")

                    missing = profile.get("missing_mapped_columns") or []
                    if missing:
                        st.error(f"Missing mapped columns: {', '.join(missing)}")

                    samples = profile.get("samples") or []
                    if samples:
                        st.dataframe(pd.DataFrame(samples), use_container_width=True, hide_index=True)

                    with st.expander("Raw catalog profile"):
                        st.json(profile)
        else:
            st.info("Upload a CSV to start the customer catalog onboarding flow.")

    with quality_tab:
        st.subheader("Recommendation Quality")
        sample_size = st.slider("Sample size", min_value=5, max_value=100, value=25, step=5)
        k = st.slider("Neighbors per item", min_value=5, max_value=25, value=10, step=5)
        if st.button("Run Quality Check", key="run_quality_check", use_container_width=True):
            st.session_state.quality_report = api_get(
                "/v1/evaluation/recommendations",
                params={"sample_size": sample_size, "k": k},
                timeout=30,
            )

        quality = st.session_state.get("quality_report")
        if quality:
            q1, q2, q3, q4 = st.columns(4)
            vectors = quality.get("vectors", {})
            recs = quality.get("recommendations", {})
            q1.metric("Vector Rows", f"{vectors.get('vector_count', 0):,}")
            q2.metric("Dimensions", vectors.get("dimension", "-"))
            q3.metric("Coverage@K", recs.get("catalog_coverage_at_k", "-"))
            q4.metric("Genre Match", recs.get("genre_overlap_rate", "-"))

            quality_rows = [
                {"metric": "Self match", "value": recs.get("self_match_rate", 0)},
                {"metric": "Coverage@K", "value": recs.get("catalog_coverage_at_k", 0)},
                {"metric": "Genre overlap", "value": recs.get("genre_overlap_rate", 0)},
                {"metric": "Genre diversity", "value": recs.get("avg_genre_diversity_per_list", 0)},
            ]
            quality_df = pd.DataFrame(quality_rows)
            fig = px.bar(
                quality_df,
                x="metric",
                y="value",
                color="metric",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig.update_yaxes(range=[0, 1])
            fig.update_layout(
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#ffffff"),
                margin=dict(t=10, b=10, l=10, r=10),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            with st.expander("Raw quality report"):
                st.json(quality)
        else:
            st.info("Run a quality check to inspect the current AI artifacts.")

    with events_tab:
        st.subheader("Event Lab")
        c_event_1, c_event_2, c_event_3 = st.columns(3)
        with c_event_1:
            event_type = st.selectbox(
                "Event type",
                ["view", "click", "recommendation_impression", "rating", "search"],
            )
        with c_event_2:
            content_id = st.text_input("Content ID", value="demo-content-001")
        with c_event_3:
            rating_value = st.number_input("Rating", min_value=1.0, max_value=5.0, value=4.0, step=0.5)

        query_text = st.text_input("Search text", value="space adventure with strong visuals")
        if st.button("Send Test Event", key="send_console_event", use_container_width=True):
            payload = {
                "event_type": event_type,
                "content_id": content_id,
                "user_id": "console-user",
                "session_id": "console-session",
            }
            if event_type == "rating":
                payload["rating"] = rating_value
            if event_type == "search":
                payload.pop("content_id", None)
                payload["query_text"] = query_text

            result = api_post("/v1/events", payload)
            if result and not result.get("error"):
                store_label = result.get("event_store", "jsonl")
                durability = "durable" if result.get("durable") else "local"
                st.success(f"Accepted event {result.get('event_id')} ({store_label}, {durability})")
            else:
                st.error(result.get("error", "Event write failed") if result else "Event write failed")

        st.subheader("Current Behavior Features")
        st.json(behavior)

        experiment_rows = experiments.get("experiments") or []
        if experiment_rows:
            st.subheader("Experiment Outcomes")
            st.dataframe(pd.DataFrame(experiment_rows), use_container_width=True, hide_index=True)

        st.subheader("Personalization Probe")
        probe_user_id = st.text_input("User ID for personalized recommendations", value="console-user")
        if st.button("Get Personalized Recommendations", key="personalized_probe", use_container_width=True):
            st.session_state.personalized_probe = api_get(
                f"/v1/recommendations/user/{probe_user_id}",
                params={"n": 10},
                timeout=30,
            )
        personalized = st.session_state.get("personalized_probe")
        if personalized:
            st.dataframe(
                pd.DataFrame(personalized)[
                    [col for col in ["id", "title", "similarity_score", "retrieval_stage", "explanation_text"] if col in pd.DataFrame(personalized).columns]
                ],
                use_container_width=True,
                hide_index=True,
            )

    with integration_tab:
        st.subheader("Developer Integration")
        api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
        st.code(
            f"""curl -X POST "{api_url}/v1/events" \\
  -H "Content-Type: application/json" \\
  -H "X-Nova-API-Key: $NOVA_API_KEY" \\
  -d '{{
    "event_type": "view",
    "content_id": "customer-content-123",
    "user_id": "user-42",
    "session_id": "session-abc"
  }}'""",
            language="bash",
        )
        st.code(
            f"""curl "{api_url}/v1/recommendations/id/100?n=10" \\
  -H "X-Nova-API-Key: $NOVA_API_KEY" """,
            language="bash",
        )
        st.markdown(
            """
            Core product contract: tenant + catalog + content + behavior events.
            The demo uses TMDB movies, but the API and lakehouse model support
            customer catalogs through `tenant_id`, `catalog_id`, and `content_id`.
            """
        )


# ===== PAGE 2: SEARCH ENGINE =====
elif st.session_state.page == "search":
    # Header navigation
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("🏠 Home", key="back_search"):
            go_home()
            st.rerun()
            
    st.title("🔍 Search & Discover")
    
    # Pre-fetch all titles for instant autocomplete
    with st.spinner("Loading movie catalog..."):
        all_titles = fetch_all_movie_titles()
    
    movie_to_fetch = None
    
    if all_titles:
        # 1. Instant Dropdown Search (The primary UX)
        title_options = {t["title"]: t for t in all_titles}
        
        selected_title = st.selectbox(
            "Search by title", 
            options=list(title_options.keys()),
            index=None,
            placeholder="e.g. Inception, Avatar, The Dark Knight...",
        )
        
        if selected_title:
            # We instantly have the exact ID here
            selected_id = title_options[selected_title]["id"]
            
            # Fetch the full movie object using the ID endpoint
            try:
                api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
                r = requests.get(f"{api_url}/movie/{selected_id}", timeout=10)
                if r.ok:
                    movie_to_fetch = r.json()
            except requests.RequestException:
                st.error("Failed to load movie details.")
    else:
        st.warning("⚠️ Could not load the movie catalog. The recommendation engine may still be starting up.")
    
    # 2. Semantic Search (The fallback/advanced UX)
    with st.expander("🔎 Search by plot, genre, or description"):
        st.caption("Can't find the title? Describe the movie — e.g. *'time travel heist'* or *'romantic comedy set in Paris'*")
        search_query = st.text_input("Describe a movie...")
        
        if search_query and len(search_query) >= 2:
            with st.spinner("Analyzing semantic meaning..."):
                movies = ai_search_movies(search_query)
            
            if movies:
                options = {format_option(m): m for m in movies}
                selected_option = st.selectbox(f"Found {len(movies)} matches:", list(options.keys()), key="deep_search_select")
                if options.get(selected_option):
                    movie_to_fetch = options.get(selected_option)
            else:
                st.info("No text matches found. Try describing the plot differently!")
    
    
    # 3. Universal Display Logic (Triggered by either search method)
    if movie_to_fetch:
        movie = movie_to_fetch
        
        # Preview
        poster_url = fetch_poster(movie.get("poster_path"))
        credits = fetch_credits(movie.get("id"))
        
        # Highlight Card
        st.markdown(f"""
        <div style="display: flex; gap: 20px; background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; margin-top: 20px;">
            <img src="{poster_url}" width="120" style="border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);">
            <div>
                <div style="font-size: 1.5rem; font-weight: 700; margin-bottom: 5px;">{movie.get('title')}</div>
                <div style="color: #bbb; margin-bottom: 10px;">{movie.get('release_date', '')[:4]} • ⭐ {movie.get('vote_average', 0):.1f}/10</div>
                <div style="font-size: 0.9rem; line-height: 1.5; color: #ddd;">{movie.get('overview', '')}</div>
                <div style="margin-top: 10px; font-size: 0.8rem; color: #888;">🎭 {credits.get('cast', 'N/A')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Action Button
        if st.button("✨ Get Similar Recommendations", type="primary", use_container_width=True):
            st.session_state.selected_rec = None
            with st.spinner("Analysing semantics..."):
                # Call API
                try:
                    api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
                    r = requests.get(
                        f"{api_url}/v1/recommendations/id/{movie['id']}",
                        params={"n": 10},
                        headers=api_headers(),
                        timeout=30,
                    )
                    if r.ok:
                        result = r.json()
                        st.session_state.recs = result["recommendations"]
                        st.session_state.source_movie = movie
                    else:
                        st.error("API Error")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # Display Recommendations Grid (Shared Logic)
    if "recs" in st.session_state and st.session_state.recs:
        # Check if recs match current search context (optional, but keep it simple)
        recs = st.session_state.recs
        source = st.session_state.get("source_movie", {})
        
        st.markdown("---")
        st.subheader(f"Because you liked '{source.get('title', '...')}'")
        
        # Prepare data for clickable grid
        rec_posters = [fetch_poster(r.get("poster_path")) for r in recs]
        rec_titles = [f"{r.get('title')} ({int(r.get('similarity_score', 0)*100)}% Match)" for r in recs]
        
        # Clickable Images Grid (Matches Homepage Style)
        clicked_rec = clickable_images(
            paths=rec_posters,
            titles=rec_titles,
            div_style={
                "display": "flex", 
                "justify-content": "center", 
                "flex-wrap": "wrap",
                "gap": "15px",
                "margin-top": "20px"
            },
            img_style={
                "width": "18%", # roughly 5 per row
                "border-radius": "12px",
                "cursor": "pointer",
                "aspect-ratio": "2/3",
                "object-fit": "cover",
                "box-shadow": "0 4px 10px rgba(0,0,0,0.5)",
                "transition": "transform 0.3s ease"
            },
            key="rec_grid"
        )
        
        # Handle selection
        if clicked_rec > -1:
            show_movie_dialog(recs[clicked_rec])


# ===== PAGE 3: AI CHATBOT =====
elif st.session_state.page == "chat":
    # Header navigation
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("🏠 Home", key="back_chat"):
            go_home()
            st.rerun()

    st.title("🤖 CineBot Assistant")
    st.caption("Ask complex questions like: *'I want a thriller with a plot twist like Shutter Island'*")
    
    # Initialize Chat
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{"role": "assistant", "content": "Hello! I'm your AI movie expert. Ask me anything!"}]
    
    # Display History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    # Input
    if prompt := st.chat_input("Ask CineBot..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    recent_msgs = st.session_state.chat_history[-6:]
                    clean_msgs = [{"role": m["role"], "content": m["content"]} for m in recent_msgs if m["role"] != "system"]

                    api_url = st.session_state.get("API_URL", BACKEND_URLS[0])
                    r = requests.post(f"{api_url}/chat", json={"messages": clean_msgs}, timeout=60)

                    if r.ok:
                        response_text = r.json()["content"]
                        st.markdown(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                         st.error("AI Brain Offline.")
                except Exception as e:
                    st.error(f"Error: {e}")

# ===== PAGE 4: REAL-TIME MONITORING =====
elif st.session_state.page == "monitoring":
    # Header navigation
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("🏠 Home", key="back_monitoring"):
            go_home()
            st.rerun()

    st.title("📊 Real-Time Data Pipeline Monitoring")
    st.caption("Kafka event stream and Delta table state visualization")

    # Custom CSS for monitoring page
    st.markdown("""
    <style>
    /* Monitoring page specific styles */
    .monitoring-card {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .monitoring-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 15px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
    }
    .event-stream {
        height: 300px;
        overflow-y: auto;
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
        font-family: monospace;
        font-size: 0.8rem;
    }
    .event-item {
        padding: 8px;
        margin-bottom: 5px;
        border-radius: 5px;
        background: rgba(255, 255, 255, 0.05);
        border-left: 3px solid #e50914;
    }
    .event-item.view {
        border-left-color: #4CAF50;
    }
    .event-item.rating {
        border-left-color: #FFC107;
    }
    .event-timestamp {
        color: #888;
        font-size: 0.7rem;
        margin-bottom: 3px;
    }
    .event-details {
        color: #fff;
        font-size: 0.85rem;
    }
    .metric-card {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .chart-container {
        background: rgba(0, 0, 0, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize monitoring data if not exists
    if "monitoring_data" not in st.session_state:
        st.session_state.monitoring_data = {
            "kafka_events": [],
            "delta_state": {},
            "system_metrics": {
                "total_events": 0,
                "view_events": 0,
                "rating_events": 0,
                "last_event_time": None,
                "events_per_minute": 0
            }
        }

    # Function to simulate Kafka consumer (will be replaced with real implementation)
    def consume_kafka_events():
        """Simulate consuming Kafka events for demo purposes"""
        import random
        from datetime import datetime

        # Sample movie data for simulation
        sample_movies = [
            {"id": 1, "title": "The Shawshank Redemption"},
            {"id": 2, "title": "The Godfather"},
            {"id": 3, "title": "Pulp Fiction"},
            {"id": 4, "title": "The Dark Knight"},
            {"id": 5, "title": "Inception"},
            {"id": 6, "title": "Fight Club"},
            {"id": 7, "title": "Forrest Gump"},
            {"id": 8, "title": "The Matrix"},
            {"id": 9, "title": "Goodfellas"},
            {"id": 10, "title": "The Silence of the Lambs"}
        ]

        # Generate random event
        movie = random.choice(sample_movies)
        event_types = ["view", "rating"]
        event_type = random.choice(event_types)

        event = {
            "id": movie["id"],
            "title": movie["title"],
            "event_type": event_type,
            "user_id": random.randint(100, 999),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        if event_type == "rating":
            event["rating"] = random.randint(1, 5)

        return event

    # Function to simulate Delta table state
    def get_delta_table_state():
        """Simulate getting Delta table state for demo purposes"""
        import random
        from datetime import datetime, timedelta

        # Simulate some processing state
        states = ["idle", "processing", "updating", "optimizing"]
        current_state = random.choice(states)

        # Simulate last update time
        last_update = datetime.now() - timedelta(minutes=random.randint(0, 60))
        last_update_str = last_update.strftime("%Y-%m-%d %H:%M:%S")

        # Simulate record counts
        record_count = random.randint(10000, 50000)
        new_records = random.randint(0, 100)

        return {
            "status": current_state,
            "last_update": last_update_str,
            "record_count": record_count,
            "new_records": new_records,
            "processing_time": f"{random.randint(1, 10)}.{random.randint(0, 99)}s",
            "table_size": f"{random.randint(100, 999)}.{random.randint(0, 99)} MB"
        }

    # Function to update monitoring data
    def update_monitoring_data():
        """Update monitoring data with simulated values"""
        # Simulate Kafka events
        if len(st.session_state.monitoring_data["kafka_events"]) < 20:  # Limit to 20 events for display
            event = consume_kafka_events()
            st.session_state.monitoring_data["kafka_events"].insert(0, event)
        else:
            # Remove oldest event if we have too many
            st.session_state.monitoring_data["kafka_events"].pop()
            event = consume_kafka_events()
            st.session_state.monitoring_data["kafka_events"].insert(0, event)

        # Update metrics
        metrics = st.session_state.monitoring_data["system_metrics"]
        metrics["total_events"] += 1

        if event["event_type"] == "view":
            metrics["view_events"] += 1
        elif event["event_type"] == "rating":
            metrics["rating_events"] += 1

        metrics["last_event_time"] = event["timestamp"]

        # Calculate events per minute (simple simulation)
        if metrics["total_events"] > 1:
            metrics["events_per_minute"] = min(metrics["total_events"], 60)  # Cap at 60 for demo

        # Simulate Delta table state
        st.session_state.monitoring_data["delta_state"] = get_delta_table_state()

    # Update monitoring data periodically
    if st.session_state.get("last_monitoring_update", 0) < time.time() - 2:  # Update every 2 seconds
        update_monitoring_data()
        st.session_state.last_monitoring_update = time.time()

    # Layout: Two columns for metrics and event stream
    col1, col2 = st.columns([1, 1])

    with col1:
        # System Metrics
        st.markdown('<div class="monitoring-title">📈 System Metrics</div>', unsafe_allow_html=True)

        # Metrics grid
        m1, m2, m3, m4 = st.columns(4)

        metrics = st.session_state.monitoring_data["system_metrics"]

        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["total_events"]}</div>
                <div class="metric-label">Total Events</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["view_events"]}</div>
                <div class="metric-label">Views</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["rating_events"]}</div>
                <div class="metric-label">Ratings</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{metrics["events_per_minute"]}</div>
                <div class="metric-label">Events/Min</div>
            </div>
            """, unsafe_allow_html=True)

        # Delta Table State
        st.markdown('<div class="monitoring-title">🔄 Delta Table State</div>', unsafe_allow_html=True)

        delta_state = st.session_state.monitoring_data["delta_state"]

        d1, d2, d3 = st.columns(3)

        with d1:
            st.metric("Status", delta_state.get("status", "unknown"))
            st.metric("Last Update", delta_state.get("last_update", "never"))

        with d2:
            st.metric("Record Count", f"{delta_state.get('record_count', 0):,}")
            st.metric("New Records", delta_state.get("new_records", 0))

        with d3:
            st.metric("Processing Time", delta_state.get("processing_time", "0s"))
            st.metric("Table Size", delta_state.get("table_size", "0 MB"))

    with col2:
        # Event Stream
        st.markdown('<div class="monitoring-title">📡 Kafka Event Stream</div>', unsafe_allow_html=True)

        # Display event stream
        event_stream_html = '<div class="event-stream">'
        for event in st.session_state.monitoring_data["kafka_events"]:
            event_type = event.get("event_type", "unknown")
            event_class = f"event-item {event_type}"

            event_details = f"""
            <div class="event-timestamp">{event.get("timestamp", "")}</div>
            <div class="event-details">
                <strong>{event.get("title", "Unknown Movie")}</strong><br>
                {event_type.upper()} by user {event.get("user_id", "?")}
            """

            if event_type == "rating":
                event_details += f" (Rating: {event.get('rating', '?')}/5)"

            event_details += "</div>"

            event_stream_html += f'<div class="{event_class}">{event_details}</div>'
        event_stream_html += '</div>'

        st.markdown(event_stream_html, unsafe_allow_html=True)

        # Auto-refresh toggle
        if st.button("🔄 Refresh Now"):
            update_monitoring_data()
            st.rerun()

    # Charts section
    st.markdown('<div class="monitoring-title">📊 Data Pipeline Visualization</div>', unsafe_allow_html=True)

    # Create two columns for charts
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        # Event type distribution chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Event Type Distribution")

        # Prepare data for pie chart
        event_counts = {
            "Views": metrics["view_events"],
            "Ratings": metrics["rating_events"]
        }

        # Create pie chart
        import plotly.express as px
        fig = px.pie(
            values=list(event_counts.values()),
            names=list(event_counts.keys()),
            color=list(event_counts.keys()),
            color_discrete_map={"Views": "#4CAF50", "Ratings": "#FFC107"},
            hole=0.4
        )
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            showlegend=True
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        # Event timeline chart
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Event Timeline")

        # Prepare data for line chart (simulated)
        import pandas as pd
        import numpy as np

        # Create simulated timeline data
        now = datetime.now()
        time_points = [now - timedelta(minutes=i) for i in range(10, 0, -1)]
        event_counts = [metrics["events_per_minute"] * (1 + 0.2 * np.random.randn()) for _ in range(10)]
        event_counts = [max(0, int(count)) for count in event_counts]

        df = pd.DataFrame({
            "Time": time_points,
            "Events": event_counts
        })

        # Create line chart
        fig = px.line(
            df,
            x="Time",
            y="Events",
            line_shape="spline",
            color_discrete_sequence=["#e50914"]
        )
        fig.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ffffff"),
            xaxis=dict(showgrid=False, title=None),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title=None),
            showlegend=False
        )
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        st.markdown('</div>', unsafe_allow_html=True)

    # System Architecture Diagram
    st.markdown('<div class="monitoring-title">🏗️ Data Pipeline Architecture</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="monitoring-card">
        <div style="text-align: center; padding: 20px;">
            <svg width="100%" height="200" viewBox="0 0 800 200" style="background: rgba(0,0,0,0.3); border-radius: 10px; padding: 20px;">
                <!-- Kafka -->
                <rect x="50" y="80" width="120" height="60" rx="10" fill="#231F20" stroke="#e50914" stroke-width="2"/>
                <text x="110" y="50" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold">Kafka</text>
                <text x="110" y="70" text-anchor="middle" fill="#e50914" font-size="12">movie_events</text>
                <text x="110" y="115" text-anchor="middle" fill="#ffffff" font-size="10">Topic</text>

                <!-- Spark -->
                <rect x="250" y="80" width="120" height="60" rx="10" fill="#231F20" stroke="#4CAF50" stroke-width="2"/>
                <text x="310" y="50" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold">Spark</text>
                <text x="310" y="100" text-anchor="middle" fill="#4CAF50" font-size="10">Streaming</text>
                <text x="310" y="115" text-anchor="middle" fill="#4CAF50" font-size="10">Processing</text>

                <!-- Delta Lake -->
                <rect x="450" y="80" width="120" height="60" rx="10" fill="#231F20" stroke="#2196F3" stroke-width="2"/>
                <text x="510" y="50" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold">Delta</text>
                <text x="510" y="70" text-anchor="middle" fill="#2196F3" font-size="12">movies</text>
                <text x="510" y="115" text-anchor="middle" fill="#2196F3" font-size="10">Table</text>

                <!-- Streamlit -->
                <rect x="650" y="80" width="120" height="60" rx="10" fill="#231F20" stroke="#FF9800" stroke-width="2"/>
                <text x="710" y="50" text-anchor="middle" fill="#ffffff" font-size="14" font-weight="bold">Streamlit</text>
                <text x="710" y="100" text-anchor="middle" fill="#FF9800" font-size="10">Real-Time</text>
                <text x="710" y="115" text-anchor="middle" fill="#FF9800" font-size="10">Dashboard</text>

                <!-- Arrows -->
                <line x1="170" y1="110" x2="250" y2="110" stroke="#e50914" stroke-width="2" marker-end="url(#arrowhead)"/>
                <line x1="370" y1="110" x2="450" y2="110" stroke="#4CAF50" stroke-width="2" marker-end="url(#arrowhead)"/>
                <line x1="570" y1="110" x2="650" y2="110" stroke="#2196F3" stroke-width="2" marker-end="url(#arrowhead)"/>

                <!-- Arrowhead marker -->
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                        <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                    </marker>
                </defs>

                <!-- Labels -->
                <text x="110" y="170" text-anchor="middle" fill="#ffffff" font-size="10">1. User Events</text>
                <text x="310" y="170" text-anchor="middle" fill="#ffffff" font-size="10">2. Stream Processing</text>
                <text x="510" y="170" text-anchor="middle" fill="#ffffff" font-size="10">3. Data Storage</text>
                <text x="710" y="170" text-anchor="middle" fill="#ffffff" font-size="10">4. Visualization</text>
            </svg>
        </div>
        <div style="text-align: center; margin-top: 10px; color: #aaa; font-size: 0.8rem;">
            Real-time data flow: Kafka → Spark Streaming → Delta Lake → Streamlit Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; color: #666; font-size: 0.8rem;">
        <p>Real-time monitoring dashboard for Movie Recommendation System data pipeline</p>
        <p>Showing simulated data - connect to actual Kafka and Delta Lake for production use</p>
    </div>
    """, unsafe_allow_html=True)

