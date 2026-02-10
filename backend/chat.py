import os
import logging
import google.generativeai as genai
from .recommender import get_recommender

logger = logging.getLogger(__name__)

# Configure Gemini
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    logger.warning("GOOGLE_API_KEY not set. GenAI features will be disabled.")

def generate_chat_response(messages: list[dict]) -> dict:
    """
    RAG Chatbot:
    1. Extract keywords from user message.
    2. Search vector DB for relevant movies (semantic search).
    3. Feed movies + user query to LLM to generate response.
    """
    if not GEMINI_KEY:
        return {"role": "assistant", "content": "I need a Google API Key to think! Please set GOOGLE_API_KEY in .env."}

    user_msg = messages[-1]["content"]
    
    # 1. RETRIEVAL (The "R" in RAG)
    # Use semantic search (FAISS + SBERT) for meaning-based retrieval
    recommender = get_recommender()
    
    try:
        # Semantic search encodes query with SBERT model and searches FAISS index
        results = recommender.semantic_search(user_msg, n=5)
        if not results:
            # Fallback to text-based search
            results = recommender.search_movies(user_msg, limit=5)
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        results = []

    # Format Context for LLM
    context_text = "Here are some movies from the database that might be relevant:\n"
    for m in results:
        context_text += f"- Title: {m['title']} ({m.get('release_date', '')[:4]})\n"
        context_text += f"  Director: {m.get('director', 'Unknown')}\n"
        context_text += f"  Overview: {m.get('overview', '')[:200]}...\n"
        context_text += f"  Rating: {m.get('vote_average', 'N/A')}\n\n"

    # 2. GENERATION (The "G" in RAG)
    # System Prompt
    system_prompt = """You are 'CineBot', an expert movie recommender AI.
    Your goal is to help users find great movies based on the provided context matches.
    
    Rules:
    1. ALWAYS use the provided movie context to answer if relevant.
    2. If the context matches the user's vaguely described mood, recommend them.
    3. Be enthusiastic, concise, and professional.
    4. If the user asks general questions, answer generally but try to tie it back to movies.
    5. Do not hallucinate movies not in the context unless you are suggesting general classics.
    """
    
    full_prompt = f"{system_prompt}\n\nCONTEXT:\n{context_text}\n\nUSER QUESTION: {user_msg}"
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        return {"role": "assistant", "content": response.text}
    except Exception as e:
        logger.error(f"GenAI generation failed: {e}")
        return {"role": "assistant", "content": "I'm having trouble connecting to my brain (Gemini API). Please try again."}
