from .embedding_service import EmbeddingService
from .pinecone_service import PineconeService
from .cache_service import CacheService
from .generate_text_service import GenerateTextService

from fastapi import Request
import logging
from app.utils.config import PINECONE_INDEX_NAME
# from .llm_service import LLMService

logger = logging.getLogger(__name__)
# Singleton instances
_embedding_service = None
_pinecone_service = None
_cache_service = None
_llm_service = None
_generate_text_service = None


def get_embedding_service(request:Request, use_cache=True):
    """
    Factory function to get a singleton instance of EmbeddingService
    """
    global _embedding_service
    print("checcking embedding is already exist",_embedding_service)
    if _embedding_service is None:
        print("creating new embedding service",_embedding_service)
        _embedding_service = EmbeddingService(use_cache=use_cache)
    else:
        print("using existing embedding service",_embedding_service)
    return _embedding_service


def get_pinecone_service(index_name=PINECONE_INDEX_NAME):
    """
    Factory function to get a singleton instance of PineconeService
    """
    global _pinecone_service
    print("checcking pinecone is already exist",_pinecone_service)
    if _pinecone_service is None:
        print("creating new pinecone service",_pinecone_service)
        _pinecone_service = PineconeService(index_name=index_name)
    else:
        print("using existing pinecone service",_pinecone_service)
    return _pinecone_service


def get_cache_service():
    """
    Factory function to get a singleton instance of CacheService
    """
    global _cache_service
    print("checcking cache is already exist")
    if _cache_service is None:
        print("creating new cache service")
        _cache_service = CacheService()
    else:
        print("using existing cache service")
    return _cache_service

def get_generate_text_service():
    """
    Factory function to get a singleton instance of GenerateTextService
    """
    global _generate_text_service
    if _generate_text_service is None:
        _generate_text_service = GenerateTextService()
    return _generate_text_service

# def get_llm_service():
    # """
    # Factory function to get a singleton instance of LLMService
    # """
    # global _llm_service
    # if _llm_service is None:
    #     _llm_service = LLMService()
    # return _llm_service