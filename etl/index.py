"""
FAISS index module for efficient similarity search.
Builds and manages the approximate nearest neighbor index.
"""
import logging
from pathlib import Path

import numpy as np
import faiss

from etl.config import paths, data_config

logger = logging.getLogger(__name__)


def build_faiss_index(vectors: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """
    Build a FAISS index for efficient similarity search.
    
    Uses IVF (Inverted File) index for large datasets, 
    or flat index for smaller datasets.
    
    Args:
        vectors: 2D numpy array of shape (n_movies, n_features)
        use_gpu: Whether to use GPU acceleration (requires faiss-gpu)
        
    Returns:
        Trained FAISS index
    """
    n_samples, n_features = vectors.shape
    logger.info(f"Building FAISS index for {n_samples:,} vectors with {n_features} dimensions...")
    
    # Ensure vectors are contiguous float32 (FAISS requirement)
    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    
    # Choose index type based on dataset size
    if n_samples < 10000:
        # Small dataset: use flat (exact) index
        logger.info("Using Flat index for exact search")
        index = faiss.IndexFlatIP(n_features)  # Inner product (cosine on normalized)
    else:
        # Large dataset: use IVF with flat quantizer
        nlist = min(data_config.faiss_nlist, n_samples // 39)  # Rule of thumb
        logger.info(f"Using IVF index with {nlist} clusters")
        
        quantizer = faiss.IndexFlatIP(n_features)
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist, faiss.METRIC_INNER_PRODUCT)
        
        # Train the index
        logger.info("Training IVF index...")
        index.train(vectors)
    
    # Add vectors to index
    index.add(vectors)
    logger.info(f"Added {index.ntotal:,} vectors to index")
    
    return index


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    L2 normalize vectors for cosine similarity via inner product.
    
    Args:
        vectors: Raw TF-IDF vectors
        
    Returns:
        Normalized vectors
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    return vectors / norms


def save_index(index: faiss.Index, filename: str = "faiss.index") -> Path:
    """Save FAISS index to disk."""
    output_path = paths.models / filename
    faiss.write_index(index, str(output_path))
    logger.info(f"Saved FAISS index to {output_path}")
    return output_path


def load_index(filename: str = "faiss.index") -> faiss.Index:
    """Load FAISS index from disk."""
    index_path = paths.models / filename
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS index with {index.ntotal:,} vectors")
    return index


def search(index: faiss.Index, query_vector: np.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """
    Search for k nearest neighbors.
    
    Args:
        index: FAISS index
        query_vector: Query vector (1D or 2D)
        k: Number of neighbors to return
        
    Returns:
        Tuple of (distances, indices)
    """
    # Ensure 2D
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    query_vector = np.ascontiguousarray(query_vector.astype(np.float32))
    
    # Set nprobe for IVF index (number of clusters to search)
    if hasattr(index, "nprobe"):
        index.nprobe = min(10, index.nlist)
    
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


def build_index(vectors: np.ndarray | None = None) -> faiss.Index:
    """
    Main indexing pipeline: load vectors, normalize, build and save index.
    
    Args:
        vectors: Optional pre-loaded vectors. If None, loads from disk.
        
    Returns:
        Built FAISS index
    """
    logger.info("Starting index building...")
    
    # Load vectors if not provided
    if vectors is None:
        vectors_path = paths.models / "sbert_embeddings.npy"
        if not vectors_path.exists():
            vectors_path = paths.models / "tfidf_vectors.npy"
            
        vectors = np.load(vectors_path)
        logger.info(f"Loaded vectors with shape {vectors.shape} from {vectors_path.name}")
    
    # Normalize for cosine similarity
    vectors = normalize_vectors(vectors)
    
    # Build index
    index = build_faiss_index(vectors)
    
    # Save index
    save_index(index)
    
    logger.info("Index building complete")
    return index


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_index()
