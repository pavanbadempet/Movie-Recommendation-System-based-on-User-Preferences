# 🚀 Deployment Guide

This guide will help you put your Movie Recommendation System online!

We will use a split architecture:
1.  **Backend (API):** Hosted on **Render.com** (Free, Reliable).
2.  **Frontend (UI):** Hosted on **Streamlit Community Cloud** (Free, Built for Streamlit).

---

## ✅ Prerequisites
1.  **GitHub Account:** You must push this code to a GitHub repository.
2.  **Render Account:** Sign up at [render.com](https://render.com).
3.  **Streamlit Account:** Sign up at [share.streamlit.io](https://share.streamlit.io).

---

## Step 1: Push to GitHub
1.  Create a new **Public** repository on GitHub (e.g., `movie-recs`).
2.  Push your code:
    ```bash
    git init
    git add .
    git commit -m "Ready for deploy"
    git branch -M main
    git remote add origin https://github.com/<YOUR_USER>/movie-recs.git
    git push -u origin main
    ```
    *(Note: We have configured `.gitignore` to skip the huge 600MB legacy model file, so the upload should be fast).*

---

## Step 2: Deploy Backend (Render)
1.  Go to your [Render Dashboard](https://dashboard.render.com).
2.  Click **New +** -> **Web Service**.
3.  Connect your GitHub repository.
4.  **Configuration:**
    *   **Name:** `movie-recs-api`
    *   **Runtime:** `Python 3`
    *   **Build Command:** `pip install -r requirements.txt`
    *   **Start Command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5.  **Environment Variables:**
    *   Add Key: `PYTHON_VERSION` | Value: `3.11.0`
6.  Click **Deploy Web Service**.

**Wait 5-10 minutes.**
When finished, Render will give you a URL (e.g., `https://movie-recs-api.onrender.com`).
**Copy this URL.** You will need it for the frontend.

---

## Step 3: Deploy Frontend (Streamlit Cloud)
1.  Go to [share.streamlit.io](https://share.streamlit.io).
2.  Click **New App**.
3.  Select your GitHub repository (`movie-recs`).
4.  **Configuration:**
    *   **Main file path:** `streamlit_app.py`
5.  **Advanced Settings (Secrets):**
    *   We need to tell the frontend where the backend lives.
    *   Click "Advanced Settings"...
    *   in the "Environment variables" or "Secrets" box, add:
        ```toml
        API_URL = "https://movie-recs-api.onrender.com"
        ```
        *(Replace with your ACTUAL Render URL from Step 2)*.
6.  Click **Deploy!**

---

## 🎉 Done!
Your app is now live on the internet.
*   **Frontend:** `https://<your-app>.streamlit.app`
*   **Backend:** `https://<your-api>.onrender.com/docs`
