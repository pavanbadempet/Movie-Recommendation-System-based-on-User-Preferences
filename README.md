---
title: Movie Recs API
emoji: 🍿
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 7860
---

# Movie Recommendation Engine 🎬

> *Stop doom-scrolling and start watching.*

I built this because I was tired of Netflix recommending me the same 5 movies just because I watched *The Office* once. I wanted a recommendation engine that actually understands **context**—directors, eras, writing styles—not just "you liked X, here is X part 2".

This isn't just a wrapper around an API. It's a full-stack engine using **SBERT** (Sentence BERT) to understand semantic similarity in plot summaries, tailored with a custom re-ranking layer that I tweaked to feel like a movie buff's intuition.

## Top Features (The Good Stuff)

* **It actually understands plot:** Searching for "documentaries about minimalists" works, even if the word "minimalist" isn't in the title.
* **Smart Re-ranking:** It knows that if you like *Avatar*, you probably want to see *Avatar: The Way of Water*. But if you search for a generic sci-fi, it won't just dump all the sequels on you (thanks to MMR diversity).
* **Fast:** Searches 30k+ movies in <100ms. I use FAISS for this because standard cosine similarity was getting too slow.

## How to Run It

I wrote a single script to handle the messy stuff. You don't need to manually start the backend and frontend separately unless you want to.

### 1. The "One-Click" Setup

```bash
# This sets up the venv and installs dependencies
python manage.py setup
```

### 2. Start the App

```bash
# Fires up both the FastAPI backend and Streamlit frontend
python manage.py run
```

Then head to `http://localhost:8501`.

### 3. Data Updates (The ETL)

The movie data comes from TMDB. I have a pipeline that pulls fresh data daily. If you want to run it manually (e.g., to get today's releases):

```bash
python manage.py etl
```

*Note: The first run takes a while because it has to generate embeddings for thousands of movies. Grab a coffee.*

## Tech Stack & Why I Chose It

* **Backend:** FastAPI. Because it's fast and type-safe.
* **Search:** FAISS + SBERT. I tried TF-IDF first, but it failed at understanding context (e.g., "scary movie in space" didn't return *Alien*). SBERT fixed that.
* **Frontend:** Streamlit. I'm a backend engineer; I wanted a UI that looks good without writing 500 lines of React.
* **Deployment:** Render + Streamlit Cloud. Free tier heroes.

## Current Quirks / TODOs

* The "Wake Up" time on Render can be slow (free tier limits). I added a loading spinner so you know it hasn't crashed.
* I want to add user accounts eventually so you can save a "Watchlist".

## Contributing

Found a bug? Have an idea? Feel free to open an issue. I'm pretty active here. Check out `CONTRIBUTING.md` if you want to jump into the code.

---
*Built with 🍿 and Python.*
