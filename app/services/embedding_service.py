import openai
from typing import List, Optional
from app.utils.config import OPENAI_API_KEY
from langcache import LangCache
from app.utils.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, OPENAI_CHAT_MODEL
from app.utils.constants import OBJECT_TYPE
from app.utils.logger import get_logger

openai.api_key = OPENAI_API_KEY
logger = get_logger(__name__)

class EmbeddingService: 
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        
        if OPENAI_API_KEY:
            self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
            self.embeddings = self.openai_client.embeddings
        else:
            self.openai_client = None
            self.embeddings = None
            raise ValueError("OpenAI API key not configured")
            
        # Set OpenAI models
        self.embedding_model = OPENAI_EMBEDDING_MODEL
        self.chat_model = OPENAI_CHAT_MODEL

    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text, using cache if available"""
        try:
            
            print(f"Generating embedding for text: {text[:30]}... (without caching)")
            return self._get_embedding(text)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts, using cache where available"""
        try:
            print(f"Generating batch embeddings for {len(texts)} texts (without caching)")
            embeddings = []
            for text in texts:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            raise

    def _get_embedding(self, text: str) -> List[float]:
        """Call OpenAI API to generate embedding without caching"""
        try:
            response = self.embeddings.create(
                 input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
