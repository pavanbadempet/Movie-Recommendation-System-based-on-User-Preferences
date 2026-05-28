import networkx as nx
import logging
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple

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
        
    def build_graph(self, movies_data: List[Dict[str, Any]], parsed_metadata: Dict[int, Dict[str, Any]]):
        """
        Constructs the Knowledge Graph from the raw movie data and the 
        ContentUnderstandingEngine's extracted semantic features.
        
        parsed_metadata: { movie_id: { "themes": [...], "moods": [...], "entities": {"LOC": [...]} } }
        """
        logger.info(f"Building Knowledge Graph for {len(movies_data)} movies...")
        
        for movie in movies_data:
            m_id = f"MOVIE_{movie['id']}"
            title = movie.get('title', 'Unknown')
            
            # Add Movie Node
            self.graph.add_node(m_id, type="MOVIE", title=title)
            
            # Add Genres
            for genre in movie.get('genres', []):
                g_id = f"GENRE_{genre}"
                self.graph.add_node(g_id, type="GENRE", name=genre)
                self.graph.add_edge(m_id, g_id, relation="HAS_GENRE")
                
            # Add Deep Semantic Features
            meta = parsed_metadata.get(movie['id'], {})
            
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
                
        logger.info(f"Knowledge Graph built: {self.graph.number_of_nodes()} Nodes, {self.graph.number_of_edges()} Edges.")
        self.save()

    def save(self):
        """Persists the graph to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
            
    def load(self) -> bool:
        """Loads the graph from disk."""
        if not self.graph_path.exists():
            return False
        try:
            with open(self.graph_path, 'rb') as f:
                self.graph = pickle.load(f)
            return True
        except Exception as e:
            logger.error(f"Failed to load Knowledge Graph: {e}")
            return False

    def find_thematically_similar(self, movie_id: int, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Multi-hop reasoning:
        Finds movies that share the highest number of Themes and Moods 
        with the target movie.
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
            
        # 2-hop: Find all other movies connected to those themes/moods
        movie_scores = {}
        for attr in shared_attributes:
            for potential_movie in self.graph.neighbors(attr):
                if potential_movie != m_id and self.graph.nodes[potential_movie].get("type") == "MOVIE":
                    # Extract raw ID
                    raw_id = int(potential_movie.split("_")[1])
                    movie_scores[raw_id] = movie_scores.get(raw_id, 0.0) + 1.0 # +1 point for each shared semantic feature
                    
        # Sort by score descending
        sorted_movies = sorted(movie_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_movies[:top_k]
