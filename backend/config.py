import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"
    OPENAI_CHAT_MODEL = "gpt-4o-mini"
    
    # Pinecone Configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pinecone-search")
    
    # Application Settings
    MAX_SEARCH_RESULTS = 200
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    REDIS_SERVER_URL = os.getenv("REDIS_SERVER_URL")
    REDIS_CACHE_ID = os.getenv("REDIS_CACHE_ID")
    SIMILARITY_THRESHOLD = os.getenv("SIMILARITY_THRESHOLD")

    # OpenAI embedding dimension
    EMBEDDING_DIMENSION = 3072  # OpenAI text-embedding-3-large dimension
settings = Settings()