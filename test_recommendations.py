import os
import sys
from pathlib import Path
import asyncio

# Setup path
sys.path.append(str(Path(__file__).resolve().parent))

from backend.recommender import Recommender

def main():
    print("Loading Core Recommender and FAISS Index...")
    # This automatically loads data/processed/movies_transformed.parquet and data/faiss/semantic_index.bin
    recommender = Recommender()
    recommender.load()
    print("Recommender loaded successfully.")
    
    query = "A group of toys come to life and go on an adventure"
    print(f"\n--- Running Real-World Semantic Query ---")
    print(f"Query: '{query}'")
    
    try:
        # semantic_search is synchronous
        recs = recommender.semantic_search(query=query, n=5)
        
        print("\n--- Top 5 Neural Recommendations ---")
        for i, rec in enumerate(recs):
            title = rec.get("title", "Unknown").encode('ascii', 'ignore').decode()
            genres = rec.get("genres", "Unknown").encode('ascii', 'ignore').decode()
            score = rec.get("score", 0.0)
            print(f"{i+1}. {title} | Genres: {genres} | Score: {score:.4f}")
            
    except Exception as e:
        print(f"Error during recommendation test: {e}")

if __name__ == "__main__":
    main()
