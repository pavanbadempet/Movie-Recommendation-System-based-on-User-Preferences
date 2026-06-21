import logging
from pathlib import Path
import pickle
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class KnowledgeGraphEngine:
    """
    Constructs and queries a multi-modal Semantic Knowledge Graph.
    Nodes: Movies, Genres, Themes, Moods, Actors (PER), Locations (LOC).
    Edges: "HAS_THEME", "SET_IN", "STARRING", "HAS_MOOD".
    Enables multi-hop logical reasoning (e.g., "Find movies with the same moral dilemma theme").
    """

    def __init__(self):
        self.graph = nx.Graph()
        self.graph_path = MODELS_DIR / "knowledge_graph.gpickle"

    def build_graph(self, movies_data: list[dict[str, Any]], parsed_metadata: dict[int, dict[str, Any]]):
        """
        Constructs the Knowledge Graph from the raw movie data and the
        ContentUnderstandingEngine's extracted semantic features.

        parsed_metadata: { movie_id: { "themes": [...], "moods": [...], "entities": {"LOC": [...]} } }
        """
        logger.info(f"Building Knowledge Graph for {len(movies_data)} movies...")

        for movie in movies_data:
            m_id = f"MOVIE_{movie['id']}"
            title = movie.get("title", "Unknown")

            # Add Movie Node
            self.graph.add_node(m_id, type="MOVIE", title=title)

            # Add Genres
            for genre in movie.get("genres", []):
                g_id = f"GENRE_{genre}"
                self.graph.add_node(g_id, type="GENRE", name=genre)
                self.graph.add_edge(m_id, g_id, relation="HAS_GENRE")

            # Add Deep Semantic Features
            meta = parsed_metadata.get(movie["id"], {})

            for theme in meta.get("themes", []):
                t_id = f"THEME_{theme}"
                self.graph.add_node(t_id, type="THEME", name=theme)
                self.graph.add_edge(m_id, t_id, relation="EXPLORES_THEME")

            for mood in meta.get("moods", []):
                md_id = f"MOOD_{mood}"
                self.graph.add_node(md_id, type="MOOD", name=mood)
                self.graph.add_edge(m_id, md_id, relation="EVOKES_MOOD")

            entities = meta.get("entities", {})
            for loc in entities.get("LOC", []):
                l_id = f"LOC_{loc}"
                self.graph.add_node(l_id, type="LOCATION", name=loc)
                self.graph.add_edge(m_id, l_id, relation="SET_IN")

        logger.info(
            f"Knowledge Graph built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges."
        )
        self.save()

    def save(self):
        """Persists the graph to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, "wb") as f:
            pickle.dump(self.graph, f)

    def load(self) -> bool:
        """Loads the graph from disk."""
        if not self.graph_path.exists():
            return False
        try:
            with open(self.graph_path, "rb") as f:
                self.graph = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load Knowledge Graph: {e}")
            return False

    def rebuild_from_catalog(self, movies_df: Any, twins_path: Any) -> None:
        """Rebuild the knowledge graph from movies DataFrame and semantic twins parquet."""
        import json

        import polars as pl

        logger.info("Rebuilding Knowledge Graph from movies catalog and semantic twins...")
        self.graph = nx.Graph()

        # 1. Load twins parquet
        parsed_metadata = {}
        if twins_path and hasattr(twins_path, "exists") and twins_path.exists():
            try:
                twins_df = pl.read_parquet(str(twins_path))
                for row in twins_df.select(["id", "concepts", "emotional_arcs"]).iter_rows():
                    m_id = row[0]
                    try:
                        concepts = json.loads(row[1]) if isinstance(row[1], str) else list(row[1])
                    except Exception:
                        concepts = []
                    try:
                        moods = json.loads(row[2]) if isinstance(row[2], str) else list(row[2])
                    except Exception:
                        moods = []
                    parsed_metadata[m_id] = {"themes": concepts, "moods": moods}
            except Exception as e:
                logger.error(f"Failed to load/parse semantic twins: {e}")
        else:
            logger.warning(f"semantic_twins.parquet not found at {twins_path}. Rebuilding empty/genre-only graph.")

        # 2. Add movie nodes, genres, themes, moods
        movie_records = movies_df.to_dict(orient="records") if hasattr(movies_df, "to_dict") else []
        for movie in movie_records:
            m_id = f"MOVIE_{movie['id']}"
            title = movie.get("title", "Unknown")
            self.graph.add_node(m_id, type="MOVIE", title=title)

            # Add Genres
            genres_raw = movie.get("genres")
            if isinstance(genres_raw, str):
                if genres_raw.startswith("["):
                    try:
                        genres = json.loads(genres_raw)
                    except Exception:
                        genres = [g.strip() for g in genres_raw.split(",")]
                else:
                    genres = [g.strip() for g in genres_raw.split(",")]
            elif isinstance(genres_raw, list):
                genres = genres_raw
            else:
                genres = []

            for genre in genres:
                genre = genre.strip()
                if genre:
                    g_id = f"GENRE_{genre}"
                    self.graph.add_node(g_id, type="GENRE", name=genre)
                    self.graph.add_edge(m_id, g_id, relation="HAS_GENRE")

            # Add Themes & Moods
            try:
                m_id_raw = int(movie['id'])
            except (ValueError, TypeError):
                continue
            meta = parsed_metadata.get(m_id_raw, {})
            for theme in meta.get("themes", []):
                theme = theme.strip()
                if theme:
                    t_id = f"THEME_{theme}"
                    self.graph.add_node(t_id, type="THEME", name=theme)
                    self.graph.add_edge(m_id, t_id, relation="EXPLORES_THEME")

            for mood in meta.get("moods", []):
                mood = mood.strip()
                if mood:
                    md_id = f"MOOD_{mood}"
                    self.graph.add_node(md_id, type="MOOD", name=mood)
                    self.graph.add_edge(m_id, md_id, relation="EVOKES_MOOD")

        logger.info(
            f"Knowledge Graph rebuilt: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges."
        )
        try:
            self.save()
        except Exception as e:
            logger.warning(f"Could not persist rebuilt Knowledge Graph to disk: {e}")


    def find_thematically_similar(self, movie_id: int, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Multi-hop reasoning with TF-IDF weighting for shared attributes:
        Finds movies that share themes and moods, weighting rarer connections higher.
        """
        if not self.graph:
            return []

        m_id = f"MOVIE_{movie_id}"
        if m_id not in self.graph:
            return []

        # 1-hop: Find all themes/moods connected to the source movie
        shared_attributes = []
        for neighbor in self.graph.neighbors(m_id):
            edge_data = self.graph.get_edge_data(m_id, neighbor)
            if edge_data and edge_data.get("relation") in ["EXPLORES_THEME", "EVOKES_MOOD"]:
                shared_attributes.append(neighbor)

        if not shared_attributes:
            return []

        # Count total movie nodes for IDF calculation
        movie_nodes_count = sum(1 for n, d in self.graph.nodes(data=True) if d.get("type") == "MOVIE")
        if movie_nodes_count <= 0:
            movie_nodes_count = 75000

        # 2-hop: Find all other movies connected to those themes/moods
        movie_scores = {}
        import math
        for attr in shared_attributes:
            df = self.graph.degree(attr)
            # Ignore extremely common attributes (acting like stopwords)
            if df > 1000:
                continue
            # TF-IDF IDF calculation: log((N + 1) / (df + 1))
            w = math.log((movie_nodes_count + 1.0) / (df + 1.0))
            for potential_movie in self.graph.neighbors(attr):
                if potential_movie != m_id and self.graph.nodes[potential_movie].get("type") == "MOVIE":
                    try:
                        raw_id = int(potential_movie.split("_")[1])
                        movie_scores[raw_id] = movie_scores.get(raw_id, 0.0) + w
                    except (ValueError, IndexError):
                        continue

        # Sort by score descending
        sorted_movies = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_movies[:top_k]

    def get_neighbors(self, movie_id: int, n: int = 10) -> list[int]:
        """
        Get thematically similar movie IDs for retrieval pipeline compatibility.
        """
        results = self.find_thematically_similar(movie_id, top_k=n)
        return [m_id for m_id, _score in results]
