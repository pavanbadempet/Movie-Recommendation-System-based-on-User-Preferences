import logging
from typing import List, Dict, Any
import torch

logger = logging.getLogger(__name__)

class ContentUnderstandingEngine:
    """
    Deep Content Understanding Engine.
    Uses zero-shot classification to map raw movie plot overviews into structured
    semantic Themes and Moods, bridging the gap between keyword search and
    true natural language comprehension.
    """
    def __init__(self, device: str = None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = None
        self.ner_pipeline = None
        
        # Pre-defined universal themes
        self.themes = [
            "moral dilemma", "coming-of-age", "revenge", "found family", 
            "man vs nature", "corruption", "artificial intelligence", "time travel",
            "heist", "forbidden love", "survival", "betrayal"
        ]
        
        # Pre-defined universal moods
        self.moods = [
            "dark", "uplifting", "tense", "melancholic", "whimsical", 
            "gritty", "surreal", "epic", "claustrophobic", "heartwarming"
        ]

    def _load_models(self):
        """Lazy load HuggingFace pipelines to save memory during startup."""
        if self.classifier is None:
            try:
                from transformers import pipeline
                logger.info(f"Loading Zero-Shot Classifier on {self.device}...")
                # We use a lightweight zero-shot model for speed
                self.classifier = pipeline("zero-shot-classification", 
                                         model="cross-encoder/nli-distilroberta-base", 
                                         device=0 if self.device == "cuda" else -1)
                
                logger.info("Loading NER Pipeline...")
                # Tiny NER model to extract locations, orgs, people
                self.ner_pipeline = pipeline("ner", 
                                           model="dslim/bert-tiny-NER",
                                           aggregation_strategy="simple",
                                           device=0 if self.device == "cuda" else -1)
            except ImportError:
                logger.error("transformers library required for deep content understanding.")
                raise

    def extract_themes_and_moods(self, overview: str, top_k: int = 2) -> Dict[str, List[str]]:
        """
        Classifies the movie overview into the most probable themes and moods.
        """
        if not overview or len(overview.strip()) < 10:
            return {"themes": [], "moods": []}
            
        self._load_models()
        
        # 1. Extract Themes
        theme_results = self.classifier(overview, self.themes, multi_label=True)
        # Filter themes with > 0.5 confidence
        top_themes = [label for label, score in zip(theme_results['labels'], theme_results['scores']) if score > 0.5][:top_k]
        
        # 2. Extract Moods
        mood_results = self.classifier(overview, self.moods, multi_label=True)
        top_moods = [label for label, score in zip(mood_results['labels'], mood_results['scores']) if score > 0.5][:top_k]
        
        return {
            "themes": top_themes,
            "moods": top_moods
        }

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extracts Named Entities (Locations, Persons, Organizations) to build the Knowledge Graph.
        """
        if not text:
            return {"LOC": [], "PER": [], "ORG": []}
            
        self._load_models()
        
        entities = self.ner_pipeline(text)
        
        extracted = {"LOC": set(), "PER": set(), "ORG": set()}
        for ent in entities:
            ent_group = ent.get("entity_group")
            word = ent.get("word", "").strip()
            if ent_group in extracted and len(word) > 2:
                extracted[ent_group].add(word)
                
        return {k: list(v) for k, v in extracted.items()}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    engine = ContentUnderstandingEngine()
    test_plot = "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O., but his tragic past may doom the project and his team to disaster."
    
    print("\nAnalyzing: Inception")
    res = engine.extract_themes_and_moods(test_plot)
    print(f"Themes/Moods: {res}")
    
    ents = engine.extract_entities(test_plot)
    print(f"Entities: {ents}")
