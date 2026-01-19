# Movie Recommendation System - Premium UI
# Run: streamlit run app.py

import streamlit as st
import requests
import time
import os
from st_clickable_images import clickable_images

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

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
API_URL = os.getenv("API_URL", "http://localhost:8000")
TMDB_KEY = os.getenv("TMDB_API_KEY")

# Validate TMDB Key
if not TMDB_KEY:
    st.warning("⚠️ TMDB_API_KEY not set. Posters and trailers will not load. Set it in your environment or Streamlit secrets.")


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
    Wake up the Render backend if it's sleeping.
    Retries for up to 60 seconds with visual feedback.
    """
    try:
        # Fast check first
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.ok:
            return True
    except requests.RequestException:
        pass

    # If failed, assume sleeping and start wake-up protocol
    with st.spinner("🚀 Waking up the recommendation engine... (This can take ~1 minute on Render Free Tier)"):
        # Max retry 60s
        for _ in range(30):
            try:
                r = requests.get(f"{API_URL}/health", timeout=5)
                if r.ok:
                    st.toast("✅ System Online!", icon="⚡")
                    return True
            except requests.RequestException:
                time.sleep(2)
        
    return False

# Initialize connection on app load
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = wake_up_backend()

def search_movies(query):
    """Search movies via API."""
    if not st.session_state.backend_ready:
        st.error("⚠️ Backend is unreachable. Please refresh or check Render dashboard.")
        return []
        
    try:
        r = requests.get(f"{API_URL}/search", params={"q": query, "limit": 100}, timeout=10)
        if r.ok:
            return r.json()
    except requests.RequestException:
        st.error("⚠️ Connection lost. Backend might be restarting.")
    return []


def get_recommendations(movie_id, n=10):
    """Get recommendations via API."""
    try:
        r = requests.get(f"{API_URL}/recommend/id/{movie_id}", params={"n": n}, timeout=30)
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


def go_home():
    st.session_state.page = "home"

def go_search():
    st.session_state.page = "search"

def go_chat():
    st.session_state.page = "chat"


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
        
        # === NAVIGATION: BUTTON STYLED AS CARD ===
        # The button IS the card - we style it to look like one
        st.markdown("""
        <style>
        /* Target the specific navigation buttons by their container */
        div[data-testid="stButton"]:has(button[kind="secondary"]) {
            margin-bottom: 12px;
        }
        
        /* Style the button to look like a premium card */
        div[data-testid="stButton"] button[kind="secondary"] {
            width: 100% !important;
            height: auto !important;
            min-height: 80px !important;
            padding: 18px 20px !important;
            
            /* Card appearance */
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 12px !important;
            
            /* Layout */
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            
            /* Animation */
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            cursor: pointer !important;
        }
        
        /* Hover effect - the key animation */
        div[data-testid="stButton"] button[kind="secondary"]:hover {
            border-color: #e50914 !important;
            background: rgba(229, 9, 20, 0.1) !important;
            transform: translateY(-4px) !important;
            box-shadow: 0 15px 40px rgba(229, 9, 20, 0.25), 0 0 20px rgba(229, 9, 20, 0.15) !important;
        }
        
        /* Button text styling - Title is larger, description is smaller */
        div[data-testid="stButton"] button[kind="secondary"] p {
            color: white !important;
            margin: 0 !important;
            text-align: left !important;
            white-space: pre-line !important;
            line-height: 1.6 !important;
        }
        
        /* First line (title) styling - larger and bolder */
        div[data-testid="stButton"] button[kind="secondary"] p::first-line {
            font-size: 1.1rem !important;
            font-weight: 700 !important;
            text-transform: uppercase !important;
            letter-spacing: 1.5px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Search Card Button (with description)
        if st.button("🔍  DEEP SEARCH\n\nFind matches by plot, vibe, or detailed queries", key="nav_search_card", use_container_width=True):
            go_search()
            st.rerun()
        
        # Chat Card Button (with description)
        if st.button("🧬  CINEBOT AI\n\nInteractive chat for complex recommendations", key="nav_chat_card", use_container_width=True):
            go_chat()
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


# ===== PAGE 2: SEARCH ENGINE =====
elif st.session_state.page == "search":
    # Header navigation
    c1, c2 = st.columns([1, 8])
    with c1:
        if st.button("🏠 Home", key="back_search"):
            go_home()
            st.rerun()
            
    st.title("🔍 Deep Search Engine")
    
    # Simple Search Interface
    search = st.text_input("Find movies by title, plot, or genre...", placeholder="Type 'Inception' or 'Time Travel'...")
    
    if search and len(search) >= 2:
        with st.spinner("Searching database..."):
            movies = search_movies(search)
        
        if movies:
            options = {format_option(m): m for m in movies}
            selected_option = st.selectbox(f"Found {len(movies)} matches:", list(options.keys()))
            movie = options.get(selected_option)
            
            if movie:
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
                            r = requests.get(f"{API_URL}/recommend/id/{movie['id']}/enriched", params={"n": 10}, timeout=30)
                            if r.ok:
                                result = r.json()
                                st.session_state.recs = result["recommendations"]
                                st.session_state.source_movie = movie
                            else:
                                st.error("API Error")
                        except Exception as e:
                            st.error(f"Connection Error: {e}")
        else:
            st.info("No text matches found. Try describing the plot!")

    # Display Recommendations Grid (Shared Logic)
    if "recs" in st.session_state and st.session_state.recs:
        # Check if recs match current search context (optional, but keep it simple)
        recs = st.session_state.recs
        source = st.session_state.get("source_movie", {})
        
        st.markdown("---")
        st.subheader(f"Because you liked '{source.get('title', '...')}'")
        
        # Grid Layout
        cols = st.columns(5)
        for idx, rec in enumerate(recs):
            with cols[idx % 5]:
                poster = fetch_poster(rec.get("poster_path"))
                title = rec.get("title")
                match = int(rec.get("similarity_score", 0) * 100)
                
                st.markdown(f"""
                <div style="margin-bottom: 10px; position: relative;">
                    <img src="{poster}" style="width: 100%; border-radius: 12px; aspect-ratio: 2/3; object-fit: cover;">
                    <div style="position: absolute; bottom: 8px; right: 8px; background: rgba(0,0,0,0.8); color: #4ade80; padding: 2px 8px; border-radius: 8px; font-size: 0.7rem; font-weight: bold;">{match}%</div>
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}</div>
                """, unsafe_allow_html=True)
                
                if st.button(f"Details", key=f"rec_{idx}", use_container_width=True):
                    st.session_state.selected_rec = rec
                    st.session_state.show_dialog = True


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
                    
                    r = requests.post(f"{API_URL}/chat", json={"messages": clean_msgs}, timeout=60)
                    
                    if r.ok:
                        response_text = r.json()["content"]
                        st.markdown(response_text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                         st.error("AI Brain Offline.")
                except Exception as e:
                    st.error(f"Error: {e}")


# Dialog logic (Shared for both modes)
if st.session_state.get("show_dialog") and st.session_state.get("selected_rec"):
    show_movie_dialog(st.session_state.selected_rec)
    st.session_state.show_dialog = False
