"""
Recommendation engine for the Movie Recommendation System.
Loads FAISS index and movie metadata for fast similarity search.
"""
import logging
from datetime import datetime
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

# Resolve paths relative to this file
MODELS_DIR = Path(__file__).parent.parent / "models"
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"


class Recommender:
    """Movie recommendation engine using FAISS similarity search."""
    
    def __init__(self):
        self._index: faiss.Index | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._movies: pd.DataFrame | None = None
        self._vectors: np.ndarray | None = None
    
    def load(self) -> "Recommender":
        """Load all required artifacts with minimal memory footprint."""
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
            raise FileNotFoundError(f"Movie data not found. Run the ETL pipeline first.")
        
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
    
    def search_movies(self, query: str, limit: int = 20) -> list[dict]:
        """
        Search movies by title.
        
        Args:
            query: Search query string
            limit: Maximum results to return
            
        Returns:
            List of matching movie dictionaries
        """
        query_lower = query.lower()
        matches = self._movies[
            self._movies["title"].str.lower().str.contains(query_lower, na=False)
        ].head(limit)
        
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
        if hasattr(self._index, "nprobe"):
            self._index.nprobe = min(50, getattr(self._index, "nlist", 10))
        
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
                
            # Franchise Detection (ONLY for exact franchise matches)
            # Example: "Avatar" → "Avatar: The Way of Water" 
            # This handles sequels/prequels that SBERT might not catch
            query_first_word = query_movie["title"].lower().split()[0]
            cand_first_word = cand["title"].lower().split()[0]
            
            # Only boost if:
            # 1. Same first word (franchise indicator)
            # 2. First word is substantial (4+ chars, not "The", "A", etc.)
            if (query_first_word == cand_first_word and 
                len(query_first_word) >= 4 and
                query_first_word not in {"the", "a", "an", "part"}):
                final_score += 0.25  # Strong franchise boost
            
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
            
            # Franchise match
            if (query_first_word == cand_first_word and 
                len(query_first_word) >= 4 and
                query_first_word not in {"the", "a", "an", "part"}):
                explanation_tags.append(f"Same franchise ({query_first_word.title()})")
            
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
        
        # === MMR DIVERSITY (Maximal Marginal Relevance) ===
        # Prevents returning 5 nearly identical movies
        if len(results) > n and self._vectors is not None:
            diverse_results = self._apply_mmr(results, movie_idx, n, lambda_param=0.7)
            return diverse_results
        
        return results[:n]
    
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
                    sim = float(np.dot(self._vectors[cand_idx], self._vectors[sel_idx]))
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


# Global singleton instance (lazy loaded)
_recommender: Recommender | None = None


def get_recommender() -> Recommender:
    """Get or create the global Recommender instance."""
    global _recommender
    if _recommender is None:
        _recommender = Recommender().load()
    return _recommender
