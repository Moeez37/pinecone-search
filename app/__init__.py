# Import key components for easier access
from app.services import (
    get_embedding_service,
    get_pinecone_service,
    get_cache_service,
    # get_llm_service
)

from app.api import api_router