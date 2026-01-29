# Hosting Large Model Files for Free

This guide explains how to host the `sbert_embeddings.npy` file (104MB) externally to fix Streamlit Cloud deployment issues.

## The Problem

Streamlit Cloud has issues with Git LFS files. When it tries to clone the repository, it times out downloading large LFS files like `sbert_embeddings.npy`.

## Solution: Hugging Face Hub (Recommended)

Hugging Face offers **unlimited free storage** for model files.

### Step 1: Create a Hugging Face Account

1. Go to [huggingface.co](https://huggingface.co) and sign up
2. Verify your email

### Step 2: Create a Model Repository

1. Click your profile icon → "New Model"
2. Name it: `movie-recs-models` (or similar)
3. Set visibility to **Public** (for free hosting)
4. Click "Create Model"

### Step 3: Upload the Embeddings File

Option A: Web Upload
1. Go to your model page → "Files and versions" tab
2. Click "Add file" → "Upload files"
3. Upload `models/sbert_embeddings.npy`

Option B: CLI Upload
```bash
pip install huggingface_hub
huggingface-cli login  # Enter your token from huggingface.co/settings/tokens
huggingface-cli upload pavanbadempet/movie-recs-models models/sbert_embeddings.npy
```

### Step 4: Get the Direct Download URL

Your file URL will be:
```
https://huggingface.co/{username}/{repo}/resolve/main/sbert_embeddings.npy
```

Example:
```
https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/sbert_embeddings.npy
```

### Step 5: Configure Environment Variables

Add these to your deployment platforms:

**Streamlit Cloud:**
1. Go to your app settings → "Secrets"
2. Add:
```toml
EMBEDDINGS_URL = "https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/sbert_embeddings.npy"
```

**Render:**
1. Go to your service → "Environment"
2. Add:
   - Key: `EMBEDDINGS_URL`
   - Value: `https://huggingface.co/pavanbadempet/movie-recs-models/resolve/main/sbert_embeddings.npy`

### Step 6: Remove LFS Tracking (Optional)

To prevent future issues, you can remove LFS tracking for embeddings:

```bash
# Remove from LFS
git lfs untrack "*.npy"

# Update .gitattributes
# Remove the line: *.npy filter=lfs diff=lfs merge=lfs -text

# Commit the change
git add .gitattributes
git commit -m "chore: stop tracking .npy files in LFS"
```

## Alternative: GitHub Releases

If you prefer GitHub:

1. Go to your repo → "Releases" → "Create a new release"
2. Tag: `v1.0-models`
3. Upload `sbert_embeddings.npy` as a release asset
4. Use URL: `https://github.com/{user}/{repo}/releases/download/v1.0-models/sbert_embeddings.npy`

## How It Works

The `backend/model_loader.py` module:
1. Checks if embeddings file exists locally
2. If missing or too small (LFS pointer), downloads from configured URL
3. Caches the file for future use

This runs automatically when the backend starts.
