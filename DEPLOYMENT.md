# Deployment Guide

How to deploy the backend and frontend.

## Prerequisites

- GitHub account with this repo pushed
- Render account (render.com) 
- Streamlit Cloud account (share.streamlit.io)

## Backend (Render)

1. Go to Render dashboard
2. New -> Web Service -> Connect your repo
3. Settings:
   - Name: `movie-recs-api`
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
4. Add env var: `PYTHON_VERSION` = `3.11.0`
5. Deploy

Takes 5-10 min. Copy the URL when done (e.g. `https://movie-recs-api.onrender.com`).

## Frontend (Cloudflare Pages)

Use the React frontend as the primary free static UI.

1. Go to Cloudflare Pages
2. Connect this repository
3. Recommended settings:
   - Root directory: `frontend`
   - Build command: `npm ci && npm run build`
   - Build output directory: `dist`
4. If you keep root directory as `/`, use:
   - Build command: `cd frontend && npm ci && npm run build`
   - Build output directory: `frontend/dist`
5. Add optional build variables only when overriding defaults:
   - `VITE_API_URL` = primary API gateway
   - `VITE_BACKUP_API_URL` = backup API gateway

The React frontend performs request-level API failover, so the UI can continue through one sleeping or slow free host.

## Frontend Backup (Streamlit Cloud)

1. Go to share.streamlit.io
2. New App -> Select your repo
3. Main file: `streamlit_app.py`
4. In Advanced Settings, add secret:
   ```toml
   API_URL = "https://movie-recs-api.onrender.com"
   TMDB_API_KEY = "your_tmdb_key"
   ```
5. Deploy

## Verify

- Primary frontend: Cloudflare Pages URL
- Backup frontend: `https://<your-app>.streamlit.app`
- Backend docs: `https://<your-api>.onrender.com/docs`
