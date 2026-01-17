# Movie Recommendation System - Premium UI
# Run: streamlit run app.py

import streamlit as st
import requests
import time
import os

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


def search_movies(query):
    """Search movies via API."""
    try:
        r = requests.get(f"{API_URL}/search", params={"q": query, "limit": 100}, timeout=10)
        if r.ok:
            return r.json()
    except requests.RequestException:
        st.error("⚠️ Backend not running. Start: `uvicorn backend.main:app`")
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


# ===== MAIN APP =====

# Session state for selected recommendation
if "selected_rec" not in st.session_state:
    st.session_state.selected_rec = None

st.title("🎬 Movie Recommendation System")

# Search
search = st.text_input("Search for a movie", placeholder="Type movie name...")

if search and len(search) >= 2:
    movies = search_movies(search)
    
    if movies:
        options = {format_option(m): m for m in movies}
        selected = st.selectbox(f"Found {len(movies)} movies", list(options.keys()))
        movie = options.get(selected)
        
        if movie:
            # Preview with essential info (fetches credits for director/cast)
            poster_url = fetch_poster(movie.get("poster_path"))
            credits = fetch_credits(movie.get("id"))  # Cached, so fast after first call
            
            st.markdown(f"""
            <div style="display: flex; gap: 15px; align-items: flex-start; padding: 12px 0;">
                <img src="{poster_url}" width="80" style="border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.5);">
                <div style="font-size: 0.85rem; line-height: 1.5;">
                    <div style="font-size: 1rem; margin-bottom: 5px;"><b>⭐ {movie.get('vote_average', 0):.1f}/10</b> | {movie.get('genres', 'N/A')}</div>
                    <div style="color: #ccc;">🎬 Director: <b>{credits.get('director', 'N/A')}</b></div>
                    <div style="color: #ccc;">🎭 Cast: {credits.get('cast', 'N/A')}</div>
                    <div style="color: #888; font-size: 0.8rem; margin-top: 6px;">{movie.get('overview', '')[:180]}...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Get Recommendations button
            if st.button("🎯 Get Recommendations", type="primary"):
                st.session_state.selected_rec = None
                
                with st.spinner("Loading recommendations..."):
                    # Use the new ENRICHED endpoint - parallel fetch on backend!
                    try:
                        r = requests.get(
                            f"{API_URL}/recommend/id/{movie['id']}/enriched",
                            params={"n": 10},
                            timeout=30
                        )
                        if r.ok:
                            result = r.json()
                            if result.get("recommendations"):
                                st.session_state.recs = result["recommendations"]
                                st.session_state.source_movie = movie
                            else:
                                st.warning("No recommendations found.")
                        else:
                            st.error(f"API Error: {r.status_code}")
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No movies found. Try another search.")

# ===== RECOMMENDATIONS DISPLAY =====
if "recs" in st.session_state and st.session_state.recs:
    recs = st.session_state.recs
    source = st.session_state.get("source_movie", {})
    
    st.markdown("---")
    st.subheader(f"🎬 Movies Similar to '{source.get('title', 'Your Selection')}'")
    
    # CSS for card grid
    st.markdown("""
    <style>
    .movie-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 15px;
        padding: 10px 0;
    }
    .movie-card {
        background: rgba(20,20,30,0.9);
        border-radius: 10px;
        overflow: hidden;
        transition: transform 0.3s, box-shadow 0.3s;
        cursor: pointer;
        border: 2px solid transparent;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(229,9,20,0.4);
        border-color: #e50914;
    }
    .movie-card img {
        width: 100%;
        aspect-ratio: 2/3;
        object-fit: cover;
    }
    .movie-card-info {
        padding: 10px;
    }
    .movie-card-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #fff;
        margin-bottom: 5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .movie-card-match {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 0.75rem;
        font-weight: 700;
        display: inline-block;
        
    }
    .movie-card-genres {
        font-size: 0.7rem;
        color: #aaa;
        margin-top: 5px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Create clickable cards using Streamlit columns
    cols = st.columns(5)
    for idx, rec in enumerate(recs):
        with cols[idx % 5]:
            poster = fetch_poster(rec.get("poster_path"))
            match = int(rec.get("similarity_score", 0) * 100)
            title = rec.get("title", "Unknown")
            genres = rec.get("genres", "")[:25] + "..." if len(rec.get("genres", "")) > 25 else rec.get("genres", "")
            
            # Clickable card using button
            st.markdown(f"""
            <div style="background: rgba(20,20,30,0.9); border-radius: 10px; overflow: hidden; margin-bottom: 15px;">
                <img src="{poster}" style="width: 100%; aspect-ratio: 2/3; object-fit: cover;">
                <div style="padding: 8px;">
                    <div style="font-size: 0.85rem; font-weight: 600; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{title}</div>
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                        <span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.7rem; font-weight: 700;">{match}% Match</span>
                        <span style="font-size: 0.7rem; color: #aaa;">⭐{rec.get('vote_average', 0):.1f}</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            # Make entire card clickable via image button
            if st.button(f"▶ {title[:20]}", key=f"btn_{idx}", use_container_width=True):
                st.session_state.selected_rec = rec
                st.session_state.show_dialog = True


# ===== MOVIE DETAIL DIALOG (POPUP!) =====
@st.dialog("🎬 Movie Details", width="large")
def show_movie_dialog(rec):
    """Show movie details in a popup dialog with video background."""
    
    trailer_key = rec.get("trailer_key")
    poster = fetch_poster(rec.get("poster_path"))
    year = rec.get("release_date", "")[:4] if rec.get("release_date") else ""
    runtime = rec.get("runtime", 0)
    similarity = int(rec.get("similarity_score", 0) * 100)
    
    # Inject CSS for video background (using user's approach!)
    if trailer_key:
        st.markdown(
            f"""
            <style>
                /* Force the dialog content area to transparent so video shows */
                div[data-testid="stDialog"] div[data-testid="stVerticalBlock"] {{
                    background-color: transparent !important;
                    z-index: 1;
                }}
                
                /* Make dialog background transparent */
                div[data-testid="stDialog"] > div > div {{
                    background: rgba(0,0,0,0.85) !important;
                }}
                
                /* Position the video behind the content */
                #movie-bg-video {{
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    min-width: 100%;
                    min-height: 100%;
                    width: auto;
                    height: auto;
                    z-index: -1;
                    opacity: 0.35;
                    pointer-events: none;
                }}
                
                /* Style text to be readable over video */
                div[data-testid="stDialog"] .stMarkdown, 
                div[data-testid="stDialog"] .stButton {{
                    color: white !important;
                    text-shadow: 2px 2px 4px #000000;
                }}
            </style>
            
            <iframe id="movie-bg-video" 
                    src="https://www.youtube-nocookie.com/embed/{trailer_key}?autoplay=1&mute=1&loop=1&playlist={trailer_key}&controls=0&showinfo=0&modestbranding=1&iv_load_policy=3&rel=0&fs=0&disablekb=1&start=1"
                    allow="autoplay" frameborder="0">
            </iframe>
            """,
            unsafe_allow_html=True
        )
    
    # Layout with columns - content will appear over video
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(poster, use_container_width=True)
    
    with col2:
        # Unified HTML for tighter layout and control

        overview_text = rec.get('overview', 'No overview available.')
        providers = fetch_watch_providers(rec.get("id"))
        
        # Build Provider HTML if available
        provider_section = ""
        if providers:
            logos_html = ""
            for p in providers:
                logo_url = f"https://image.tmdb.org/t/p/w92{p['logo_path']}"
                name = p.get('provider_name', 'Unknown')
                logos_html += f"""<div style="display: flex; flex-direction: column; align-items: center; margin-right: 12px; min-width: 50px;"><img src="{logo_url}" title="{name}" style="width: 40px; height: 40px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"><span style="font-size: 0.65rem; margin-top: 4px; color: #bbb; width: 100%; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{name}</span></div>"""
            
            provider_section = f"""<div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.1);">
<div style="font-weight: 700; font-size: 0.9rem; margin-bottom: 8px; color: #ddd;">📺 Available to Watch:</div>
<div style="display: flex; overflow-x: auto; scrollbar-width: thin; align-items: flex-start;">{logos_html}</div>
</div>"""

        st.markdown(f"""<div style="color: white;">
<div style="font-size: 2rem; font-weight: 800; line-height: 1.2; text-shadow: 2px 2px 4px black;">{rec.get('title', 'Unknown')}</div>
<div style="font-size: 0.95rem; color: #ddd; margin: 5px 0 10px 0; font-weight: 500;">{year} • {runtime} min • ⭐ {round(rec.get('vote_average', 0), 1)}/10</div>
<div style="margin-bottom: 12px;"><span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 5px 12px; border-radius: 15px; font-weight: 700; font-size: 0.85rem; box-shadow: 0 4px 10px rgba(0,0,0,0.3);">🎯 {similarity}% Match</span></div>
<div style="background: rgba(0,0,0,0.3); padding: 12px; border-radius: 12px; backdrop-filter: blur(5px);">
<div style="margin-bottom: 6px; font-size: 0.9rem;"><b>🎬 Director:</b> <span style="color: #ccc;">{rec.get('director', 'N/A')}</span></div>
<div style="margin-bottom: 6px; font-size: 0.9rem;"><b>🎭 Cast:</b> <span style="color: #ccc;">{rec.get('cast', 'N/A')}</span></div>
<div style="margin-bottom: 10px; font-size: 0.9rem;"><b>🎪 Genres:</b> <span style="color: #ccc;">{rec.get('genres', 'N/A')}</span></div>
<div style="border-top: 1px solid rgba(255,255,255,0.1); margin: 10px 0;"></div>
<div style="font-weight: 700; font-size: 1rem; margin-bottom: 5px;">📖 Overview</div>
<div style="font-size: 0.95rem; line-height: 1.5; color: #ddd; max-height: 300px; overflow-y: auto;">{overview_text}</div>
</div>
{provider_section}
</div>""", unsafe_allow_html=True)


# Trigger dialog when movie is selected
if st.session_state.get("show_dialog") and st.session_state.get("selected_rec"):
    show_movie_dialog(st.session_state.selected_rec)
    st.session_state.show_dialog = False
