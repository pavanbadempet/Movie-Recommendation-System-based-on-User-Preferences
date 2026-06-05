import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LightweightRecommender:
    """
    A zero-capital, zero-data recommendation engine.
    Uses TF-IDF and Cosine Similarity for Content-Based Filtering.
    Runs entirely on CPU in milliseconds. No deep learning, no GPUs.
    """

    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None

    def fit(self, movies_df: pd.DataFrame, text_column: str = "overview"):
        """
        Trains the TF-IDF model on a corpus of movie descriptions.
        """
        print(f"Training LightweightRecommender on {len(movies_df)} items...")
        self.movies_df = movies_df.reset_index(drop=True)

        # Handle NaN values
        self.movies_df[text_column] = self.movies_df[text_column].fillna("")

        # Initialize TF-IDF Vectorizer
        # Removes english stop words (the, a, and) and maps words to frequencies
        tfidf = TfidfVectorizer(stop_words="english")

        # Fit and transform the text column
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df[text_column])

        # Compute the cosine similarity matrix
        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

        # Create a reverse map of indices and movie titles
        self.indices = pd.Series(self.movies_df.index, index=self.movies_df["title"]).drop_duplicates()
        print("Training complete. Matrix computed.")

    def get_recommendations(self, title: str, top_k: int = 5):
        """
        Retrieves the top K similar movies based on plot similarity.
        """
        if title not in self.indices:
            return f"Error: Movie '{title}' not found in the database."

        # Get the index of the movie that matches the title
        idx = self.indices[title]

        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = list(enumerate(self.cosine_sim[idx]))

        # Sort the movies based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the most similar movies (ignoring the 1st one, which is itself)
        sim_scores = sim_scores[1 : top_k + 1]

        # Get the movie indices
        movie_indices = [i[0] for i in sim_scores]
        scores = [i[1] for i in sim_scores]

        # Return the top K most similar movies
        results = self.movies_df.iloc[movie_indices][["title", "overview"]].copy()
        results["similarity_score"] = np.round(scores, 3)
        return results

    def get_trending(self, top_k: int = 5):
        """
        Heuristic baseline: Return movies with the highest average rating or popularity.
        Useful for new users with no history (Cold Start).
        """
        if "rating" in self.movies_df.columns:
            trending = self.movies_df.sort_values("rating", ascending=False).head(top_k)
            return trending[["title", "rating"]]
        return self.movies_df.head(top_k)[["title"]]


# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    # 1. Create Mock Data (Simulating a database)
    data = {
        "title": ["The Matrix", "Inception", "Interstellar", "The Notebook", "Titanic", "Hackers"],
        "overview": [
            "A computer hacker learns from mysterious rebels about the true nature of his reality.",
            "A thief who steals corporate secrets through the use of dream-sharing technology.",
            "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival.",
            "A poor yet passionate young man falls in love with a rich young woman.",
            "A seventeen-year-old aristocrat falls in love with a kind but poor artist aboard the luxurious, ill-fated R.M.S. Titanic.",
            "Hackers are blamed for making a virus that will capsize five oil tankers.",
        ],
        "rating": [8.7, 8.8, 8.6, 7.8, 7.9, 6.2],
    }

    mock_df = pd.DataFrame(data)

    # 2. Initialize and Train
    recommender = LightweightRecommender()
    recommender.fit(mock_df, text_column="overview")

    # 3. Test Recommendations
    print("\n--- Testing Content-Based Recommender ---")
    target_movie = "The Matrix"
    print(f"\nIf a user likes '{target_movie}', we recommend:")
    recs = recommender.get_recommendations(target_movie, top_k=2)
    print(recs.to_string(index=False))

    target_movie = "The Notebook"
    print(f"\nIf a user likes '{target_movie}', we recommend:")
    recs = recommender.get_recommendations(target_movie, top_k=2)
    print(recs.to_string(index=False))

    # 4. Test Cold Start / Trending
    print("\n--- Testing Cold Start (Trending) ---")
    print("For a brand new user, we show the highest rated:")
    print(recommender.get_trending(top_k=3).to_string(index=False))
