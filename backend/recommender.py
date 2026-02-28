"""
Recommendation engine.
This isn't just a database wrapper; it loads the FAISS index and handles the "fuzzy" logic 
of making recommendations feel personalized.
"""
import logging
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import faiss
import os
import json
from huggingface_hub import InferenceClient
from sklearn.feature_extraction.text import TfidfVectorizer

# Import model loader to handle external model downloads
from backend.model_loader import ensure_model_files

logger = logging.getLogger(__name__)

# Resolve paths relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"

# Ensure models are downloaded before proceeding
ensure_model_files(MODELS_DIR)


class Recommender:
    """
    The brain of the operation.
    It manages the FAISS index (for speed) and the metadata (for context).
    """
    
    def __init__(self):
        self._index: faiss.Index | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._movies: pd.DataFrame | None = None
        self._vectors: np.ndarray | None = None
    
    def load(self) -> "Recommender":
        """
        Loads the heavy artifacts.
        We use memory-mapping for the vectors so we don't blow up the RAM on the free tier.
        """
        logger.info("Loading recommendation engine...")
        
        # Load FAISS index
        index_path = MODELS_DIR / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found at {index_path}. Run the ETL pipeline first.")
        self._index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self._index.ntotal:,} vectors")
        
        # Load SBERT embeddings with memory-mapping (reads from disk, not RAM)
        vectors_path = MODELS_DIR / "sbert_embeddings.npy"
        if vectors_path.exists():
            # Memory-mapped mode: doesn't load entire array into RAM
            self._vectors = np.load(vectors_path, mmap_mode='r')
            logger.info(f"Loaded SBERT embeddings with shape {self._vectors.shape} (memory-mapped)")
        else:
            # Fallback to TF-IDF if SBERT not found
            vectors_path = MODELS_DIR / "tfidf_vectors.npy"
            if vectors_path.exists():
                self._vectors = np.load(vectors_path, mmap_mode='r')
                logger.warning("SBERT embeddings not found, using TF-IDF vectors.")
            else:
                logger.warning("No vectors found.")
        
        # Load movie metadata - only essential columns to save memory
        movies_path = DATA_DIR / "movies_transformed.parquet"
        if not movies_path.exists():
            movies_path = DATA_DIR / "movies.parquet"
        
        if movies_path.exists():
            # Only load columns we actually need for recommendations
            essential_cols = ['id', 'title', 'overview', 'genres', 'vote_average', 
                            'vote_count', 'popularity', 'release_date', 'poster_path',
                            'director', 'cast', 'original_language']
            try:
                self._movies = pd.read_parquet(movies_path, columns=essential_cols)
            except (KeyError, ValueError):
                # Fallback if some columns don't exist
                self._movies = pd.read_parquet(movies_path)
            logger.info(f"Loaded {len(self._movies):,} movies")
        else:
            raise FileNotFoundError("Movie data not found. Run the ETL pipeline first.")
        
        return self
    
    @property
    def movies(self) -> pd.DataFrame:
        """Get movie metadata DataFrame."""
        if self._movies is None:
            raise RuntimeError("Recommender not loaded. Call load() first.")
        return self._movies
    
    def get_movie_by_id(self, movie_id: int) -> dict | None:
        """Get movie details by TMDB ID."""
        matches = self._movies[self._movies["id"] == movie_id]
        if len(matches) == 0:
            return None
        return matches.iloc[0].to_dict()
    
    def get_movie_by_index(self, idx: int) -> dict:
        """Get movie details by DataFrame index."""
        return self._movies.iloc[idx].to_dict()
        
    def get_all_titles(self, limit: int = 5000) -> list[dict]:
        """
        Return a lightweight list of movie IDs and Titles for autocomplete.
        """
        if self._movies is None:
            return []
        
        # Extract necessary columns
        cols = ["id", "title"]
        if "release_date" in self._movies.columns:
            cols.append("release_date")
        if "popularity" in self._movies.columns:
            cols.append("popularity")
        if "genres" in self._movies.columns:
            cols.append("genres")
            
        titles_df = self._movies[cols].copy()
        
        # Append release year to the title for disambiguation
        if "release_date" in titles_df.columns:
            years = pd.to_datetime(titles_df["release_date"], errors="coerce").dt.year
            mask = years.notna() & (years > 0)
            titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " (" + years[mask].astype(int).astype(str) + ")"
            
        # Append genres to the title for extra context
        if "genres" in titles_df.columns:
            # Handle NaN/None in genres
            mask = titles_df["genres"].notna() & (titles_df["genres"] != "")
            # Take only the first 2 genres to keep it clean, if it's a comma-separated string
            def get_top_genres(g_str):
                try:
                    parts = str(g_str).split(",")
                    return ", ".join(p.strip() for p in parts[:2])
                except Exception:
                    return str(g_str)
            
            top_genres = titles_df.loc[mask, "genres"].apply(get_top_genres)
            titles_df.loc[mask, "title"] = titles_df.loc[mask, "title"] + " - " + top_genres
        
        # Sort by popularity so famous movies appear at the top instead of garbage punctuation
        if "popularity" in titles_df.columns:
            titles_df = titles_df.sort_values("popularity", ascending=False)
        else:
            titles_df = titles_df.sort_values("title")
            
        # Limit to the top N most popular movies to save bandwidth and browser memory
        if limit and limit > 0:
            titles_df = titles_df.head(limit)
        
        # Return only id and title

        return titles_df[["id", "title"]].to_dict(orient="records")
    
    def search_movies(self, query: str, limit: int = 20) -> list[dict]:
        """
        Standard text search, but with a few tweaks to make it feel smarter.
        We prioritize Titles, but also peek at Genres and Overviews so you can search for "action aliens".
        """
        """
        Search movies by title, overview, and genres (Deep Search).
        
        Args:
            query: Search query string
            limit: Maximum results to return
            
        Returns:
            List of matching movie dictionaries sorted by relevance
        """
        if not query:
            return []
            
        q_lower = query.lower()
        
        # 1. Title Match (Weight: 10)
        mask_title = self._movies["title"].str.lower().str.contains(q_lower, na=False)
        
        # 2. Overview Match (Weight: 3) - Allows searching by plot concepts
        mask_overview = self._movies["overview"].str.lower().str.contains(q_lower, na=False)
        
        # 3. Genre Match (Weight: 5)
        mask_genre = self._movies["genres"].str.lower().str.contains(q_lower, na=False)
        
        # Combine matches
        matches = self._movies[mask_title | mask_overview | mask_genre].copy()
        
        if len(matches) == 0:
            return []
            
        # Heuristic Scoring (The "Secret Sauce")
        # I tweaked these weights based on trial and error.
        # - Exact Match (+50): If you type "Avatar", you want "Avatar".
        # - Starts With (+20): "Ava..." should still show "Avatar".
        # - Popularity (*2.0): Hits usually beat indie films in search intent.
        
        matches["relevance"] = 0.0
        
        # Title Factors
        m_title = matches["title"].str.lower()
        matches.loc[m_title == q_lower, "relevance"] += 50.0
        matches.loc[m_title.str.startswith(q_lower), "relevance"] += 20.0
        matches.loc[m_title.str.contains(q_lower, regex=False), "relevance"] += 10.0
        
        # Other Factors
        # Note: We use the masks subsetted by the matches index
        matches.loc[mask_genre[matches.index], "relevance"] += 5.0
        matches.loc[mask_overview[matches.index], "relevance"] += 3.0
        
        # Popularity Boost
        matches["relevance"] += np.log1p(matches["popularity"]) * 2.0
        
        # Sort by relevance
        matches = matches.sort_values("relevance", ascending=False).head(limit)
        
        return matches.to_dict(orient="records")
    
    def recommend_by_index(self, movie_idx: int, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its DataFrame index.
        
        Args:
            movie_idx: Index of the movie in the DataFrame
            n: Number of recommendations
            
        Returns:
            List of recommended movie dictionaries with similarity scores
        """
        if self._vectors is None:
            raise RuntimeError("Vectors not loaded")
        
        # Get query vector
        query_vector = self._vectors[movie_idx].reshape(1, -1).astype(np.float32)
        query_vector = np.ascontiguousarray(query_vector)
        
        # Search (Fetch 100 candidates for re-ranking)
        # We fetch more than N to allow the business logic to re-order them
        fetch_k = 100
        
        # Configure IVF search
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = min(50, getattr(self._index, "nlist", 10))
            
        # Configure HNSW search (efSearch > k helps recall)
        if hasattr(self._index, "hnsw"):
            self._index.hnsw.efSearch = 200
        
        distances, indices = self._index.search(query_vector, fetch_k)
        
        # Get Query Metadata for Re-Ranking
        query_movie = self.get_movie_by_index(movie_idx)
        q_director = query_movie.get("director")
        q_title_tokens = set(query_movie["title"].lower().split())
        stop_words = {"the", "a", "an", "of", "and", "in", "to", "part", "vol", "volume", "chapter"}
        q_title_tokens -= stop_words
        
        # Build results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == movie_idx or idx < 0:
                continue
            
            cand = self.get_movie_by_index(idx)
            raw_score = float(dist)
            final_score = raw_score
            
            # --- BUSINESS LOGIC RE-RANKING ---
            
            # Director Match (+0.10) - Strong signal for stylistic similarity
            if q_director and cand.get("director") == q_director:
                final_score += 0.10
                
            # We are removing franchise string-matching heuristics because they overly heavily bias
            # the FAISS pool. The Hugging Face Llama-3 Reranker is now smart enough to detect true franchises
            # based on plot semantics and metadata without brute forcing scores.
            pass
            
            # Popularity Nudge (Log Scale)
            votes = cand.get("vote_count", 0)
            if votes > 0:
                final_score += 0.02 * np.log10(votes)

            # Genre Consistency Check
            # If the candidate shares NO genres with the query, it's likely a semantic drift (e.g. word match).
            # "Avatar" (Sci-Fi) vs "The Aviator" (Drama) -> No overlap.
            q_genres_str = str(query_movie.get("genres", "")).lower()
            cand_genres_str = str(cand.get("genres", "")).lower()
            
            # Simple set parsing (assuming comma separated)
            q_genre_set = {g.strip() for g in q_genres_str.split(",") if g.strip()}
            cand_genre_set = {g.strip() for g in cand_genres_str.split(",") if g.strip()}
            
            # Penalize if Disjoint (and query actually has genres)
            if q_genre_set and cand_genre_set.isdisjoint(q_genre_set):
                final_score -= 0.15 

            # Documentary Penalty (Unless Query is also a Documentary)
            # Users usually don't want "Making Of" videos when searching for feature films.
            is_query_doc = "documentary" in q_genres_str
            is_cand_doc = "documentary" in cand_genres_str
            
            if is_cand_doc and not is_query_doc:
                final_score -= 0.15 # Strong penalty to push them down
            
            # Quality-based score adjustments
            
            # Quality Boost (Favor well-rated films)
            cand_rating = cand.get("vote_average", 0) or 0
            cand_votes = cand.get("vote_count", 0) or 0
            if cand_rating > 0 and cand_votes > 100:
                # Combines rating quality with vote confidence
                quality_score = (cand_rating / 10) * np.log10(max(cand_votes, 1))
                final_score += 0.02 * quality_score  # Subtle but effective
            
            # Era Matching (Penalize large time gaps)
            try:
                q_year = int(str(query_movie.get("release_date", ""))[:4])
                c_year = int(str(cand.get("release_date", ""))[:4])
                year_gap = abs(q_year - c_year)
                
                if year_gap <= 5:
                    final_score += 0.03  # Same era boost
                elif year_gap >= 30:
                    final_score -= 0.05  # Different generation penalty
            except (ValueError, TypeError, IndexError):
                pass  # Skip if dates are invalid
            
            # Recency Boost (Slight preference for newer films)
            try:
                c_year = int(str(cand.get("release_date", ""))[:4])
                current_year = datetime.now().year
                years_old = current_year - c_year
                if years_old <= 5:
                    final_score += 0.02  # Recent film boost
            except (ValueError, TypeError, IndexError):
                pass
            
            # Same Language Preference
            q_lang = str(query_movie.get("original_language", "en")).lower()
            c_lang = str(cand.get("original_language", "en")).lower()
            if q_lang == c_lang:
                final_score += 0.02  # Same language slight boost
            
            # === EXPLAINABILITY (Why was this recommended?) ===
            explanation_tags = []
            

            
            # Director match
            if q_director and cand.get("director") == q_director:
                explanation_tags.append(f"Same director ({q_director})")
            
            # Genre overlap
            shared_genres = q_genre_set & cand_genre_set
            if shared_genres:
                top_genres = list(shared_genres)[:2]
                explanation_tags.append(f"Shared genres: {', '.join(g.title() for g in top_genres)}")
            
            # Era match
            try:
                q_year = int(str(query_movie.get("release_date", ""))[:4])
                c_year = int(str(cand.get("release_date", ""))[:4])
                if abs(q_year - c_year) <= 5:
                    explanation_tags.append(f"Same era ({c_year})")
            except (ValueError, TypeError, IndexError):
                pass
            
            # High quality
            if cand_rating >= 7.5 and cand_votes >= 1000:
                explanation_tags.append(f"Critically acclaimed ({cand_rating}/10)")
            
            # Same language (if not English - more notable)
            if q_lang == c_lang and q_lang != "en":
                explanation_tags.append(f"Same language ({c_lang.upper()})")
            
            # Default if no specific reasons found
            if not explanation_tags:
                explanation_tags.append("Similar themes and plot")
                
            cand["similarity_score"] = final_score
            cand["explanation"] = explanation_tags  # NEW: Add explanation
            cand["explanation_text"] = " • ".join(explanation_tags)  # Human-readable
            results.append(cand)
        
        # Sort by boosted score
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        top_candidates = results[:20]
        
        # === NEW 2026 SOTA: LLM-as-a-Judge Reranking ===
        # Pass the top 20 mathematically similar movies to the LLM to understand true aesthetic vibe
        try:
            llm_results = self._rerank_with_llm(query_movie, top_candidates, n)
            if llm_results and len(llm_results) > 0:
                return llm_results
        except Exception as e:
            logger.error(f"LLM Reranking failed, falling back to FAISS/MMR. Error: {e}")
            if results:
                results[0]["explanation_text"] = f"LLM Error: {str(e)[:200]}"
        
        # === MMR DIVERSITY (Maximal Marginal Relevance) ===
        # Prevents returning 5 nearly identical movies
        if len(results) > n and self._vectors is not None:
            diverse_results = self._apply_mmr(results, movie_idx, n, lambda_param=0.7)
            return diverse_results
        
        return results[:n]
    
    def _rerank_with_llm(self, query_movie: dict, candidates: list[dict], n: int = 10) -> list[dict]:
        """
        Uses Hugging Face Serverless Inference API (Llama-3) to semantically rerank candidates.
        """
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            logger.warning("HF_TOKEN missing. Skipping LLM reranking and falling back to FAISS/MMR.")
            raise ValueError("HF_TOKEN environment variable is not set or accessible.")
            
        client = InferenceClient(token=hf_token)
        
        # Prepare candidates for prompt
        cand_text = ""
        for i, c in enumerate(candidates):
            cand_text += f"[{i}] Title: {c.get('title')}, Genres: {c.get('genres')}, Plot: {c.get('overview', 'N/A')}\n"
            
        prompt = f"""You are an expert film critic. I will give you a QUERY MOVIE and a list of CANDIDATE MOVIES.
Your job is to select the {n} absolute best recommendations based on deep aesthetic similarity, themes, tropes, target audience, and vibe.
Ignore generic keyword matches. Focus on the actual experience of watching the movie.

QUERY MOVIE:
Title: {query_movie.get('title')}
Genres: {query_movie.get('genres')}
Plot: {query_movie.get('overview')}

CANDIDATE MOVIES:
{cand_text}

Output strictly in valid JSON format like this:
{{
  "recommendations": [
    {{"index": <candidate_index_from_above>, "explanation": "<a one-sentence explanation of why this matches the query aesthetic>"}}
  ]
}}
Do not write any other text except the JSON object.
"""
        # We use Qwen2.5-72B-Instruct because it is exceptionally strong at reasoning 
        # and strictly outputting JSON format, but more importantly, it is completely ungated.
        # The official Meta-Llama models throw 403 errors unless the user manually signs a license.
        # Note: Qwen2.5-72B-Instruct on the free inference API uses the conversational task.
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            model="Qwen/Qwen2.5-72B-Instruct", 
            max_tokens=1000, 
            temperature=0.1
        )
        
        # Parse the text string into JSON
        cleaned_response = response.choices[0].message.content.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:]
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-3]
            
        data = json.loads(cleaned_response)
        
        reranked_results = []
        for item in data.get("recommendations", []):
            idx = item.get("index")
            if isinstance(idx, int) and 0 <= idx < len(candidates):
                movie = candidates[idx]
                movie["explanation_text"] = "LLM Reranked: " + str(item.get("explanation", "Highly similar aesthetic vibe."))
                # Keep original similarity score but sort by the LLM's chosen order
                reranked_results.append(movie)
                
        if not reranked_results:
            raise ValueError(f"LLM returned no valid recommendations. Raw text: {cleaned_response[:100]}")
            
        return reranked_results[:n]
    
    def _apply_mmr(self, candidates: list[dict], query_idx: int, n: int, lambda_param: float = 0.7) -> list[dict]:
        """
        Apply Maximal Marginal Relevance for diversity.
        
        MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected))
        
        λ = 0.7 means 70% relevance, 30% diversity
        """
        if len(candidates) <= n:
            return candidates
        
        selected = []
        remaining = candidates.copy()
        
        # First pick: highest score (most relevant)
        selected.append(remaining.pop(0))
        
        while len(selected) < n and remaining:
            best_mmr = -float('inf')
            best_idx = 0
            
            for i, cand in enumerate(remaining):
                # Get candidate index in original DataFrame
                cand_matches = self._movies[self._movies["id"] == cand["id"]].index
                if len(cand_matches) == 0:
                    continue
                cand_idx = cand_matches[0]
                
                relevance = cand["similarity_score"]
                
                # Calculate max similarity to already selected
                max_sim_to_selected = 0
                for sel in selected:
                    sel_matches = self._movies[self._movies["id"] == sel["id"]].index
                    if len(sel_matches) == 0:
                        continue
                    sel_idx = sel_matches[0]
                    
                    # Cosine similarity between candidate and selected
                    # Cast to float32 for precision/speed (essential if vectors are float16)
                    v_cand = self._vectors[cand_idx].astype(np.float32)
                    v_sel = self._vectors[sel_idx].astype(np.float32)
                    sim = float(np.dot(v_cand, v_sel))
                    max_sim_to_selected = max(max_sim_to_selected, sim)
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected
    
    def recommend_by_id(self, movie_id: int, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its TMDB ID.
        
        Args:
            movie_id: TMDB movie ID
            n: Number of recommendations
            
        Returns:
            List of recommended movie dictionaries
        """
        # Find index of the movie
        matches = self._movies[self._movies["id"] == movie_id].index
        if len(matches) == 0:
            return []
        
        movie_idx = matches[0]
        return self.recommend_by_index(movie_idx, n)
    
    def recommend_by_title(self, title: str, n: int = 10) -> list[dict]:
        """
        Get recommendations for a movie by its title.
        
        Args:
            title: Movie title (case-insensitive)
            n: Number of recommendations
            
        Returns:
            List of recommended movie dictionaries
        """
        title_lower = title.lower()
        matches = self._movies[self._movies["title"].str.lower() == title_lower].index
        
        if len(matches) == 0:
            # Try partial match
            matches = self._movies[
                self._movies["title"].str.lower().str.contains(title_lower, na=False)
            ].index
        
        if len(matches) == 0:
            return []
        
        movie_idx = matches[0]
        return self.recommend_by_index(movie_idx, n)

    def semantic_search(self, query: str, n: int = 10) -> list[dict]:
        """
        Search movies by semantic meaning using the SBERT model + FAISS index.
        
        Unlike search_movies() which does text matching on titles,
        this encodes the query with the same model used for embeddings
        and searches the FAISS index directly.
        
        Args:
            query: Natural language query (e.g. "movies about space exploration")
            n: Number of results to return
            
        Returns:
            List of movie dictionaries with similarity scores
        """
        if not query or self._index is None or self._vectors is None:
            return []
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Use the same model that generated the embeddings
            model = _get_sbert_model()
            query_embedding = model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
            query_embedding = query_embedding.astype(np.float32)
            
            distances, indices = self._index.search(query_embedding, n)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self._movies):
                    continue
                movie = self._movies.iloc[idx].to_dict()
                movie["similarity_score"] = float(distances[0][i])
                results.append(movie)
            
            return results
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            # Fallback to text search
            return self.search_movies(query, limit=n)


# Lazy-loaded SBERT model for semantic search queries
_sbert_model = None

def _get_sbert_model():
    """Get or load the SBERT model (lazy singleton)."""
    global _sbert_model
    if _sbert_model is None:
        from sentence_transformers import SentenceTransformer
        _sbert_model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("Loaded SBERT model for semantic search")
    return _sbert_model


# Global singleton instance (lazy loaded)
_recommender: Recommender | None = None


def get_recommender() -> Recommender:
    """Get or create the global Recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = Recommender().load()
    return _recommender

