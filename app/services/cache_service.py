from app.utils.logger import get_logger
from typing import Any, Dict, Optional
import json
from langcache import LangCache
from app.utils.config import REDIS_LANGCACHE_SERVER_URL, REDIS_LANGCACHE_CACHE_ID, REDIS_LANGCACHE_API_KEY,REDIS_LANGCACHE_SIMILARITY_THRESHOLD
logger = get_logger(__name__)


class CacheService:
    def __init__(self):
        # try:
        self.client = LangCache(
            server_url=REDIS_LANGCACHE_SERVER_URL,
            cache_id=REDIS_LANGCACHE_CACHE_ID,
            api_key=REDIS_LANGCACHE_API_KEY,
        )
        logger.info("Connected to Redis cache using LangCache")
        # except Exception as e:
        #     logger.error(f"Failed to connect to Redis with LangCache: {e}")
        #     self.client = None

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Get a value from cache (synchronous)
        """
        if not self.client:
            logger.warning("Cache connection not available, skipping cache get")
            return None

        try:
            value = self.client.search(prompt=query, similarity_threshold=REDIS_LANGCACHE_SIMILARITY_THRESHOLD)  # sync method
            if value is not None and len(value.data) != 0:
                logger.info(f"Cache hit for query: {query}")
                return value
            logger.info(f"Cache miss for query: {query}")
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None

    async def set(self, query: str, value: Any, ex: int = 3600) -> bool:
        """
        Set a value in cache (asynchronous)
        """
        if not self.client:
            logger.warning("Cache connection not available, skipping cache set")
            return False
        try:
            self.client.set(
                prompt=query,
                response=json.dumps(value))  # async method
            logger.info(f"Cached value for query: {query}, expires in {ex}s")
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a value from cache (synchronous)
        """
        if not self.client:
            logger.warning("Cache connection not available, skipping cache delete")
            return False

        try:
            self.client.delete(key)
            logger.info(f"Deleted cache for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
