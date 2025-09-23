from langcache import LangCache
from config import settings
import logging
import json
import asyncio

logger = logging.getLogger(__name__)

class LangCacheService:
    """Service class to handle LangCache operations"""
    
    def __init__(self):
        """Initialize the LangCacheService"""
        self.server_url = settings.REDIS_SERVER_URL
        self.cache_id = settings.REDIS_CACHE_ID
        self.api_key = settings.LANGCHAIN_API_KEY
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
    def _create_lang_cache(self):
        """Create and return a LangCache instance
        
        Returns:
            LangCache: The LangCache instance or None if creation fails
        """
        try:
            return LangCache(
                server_url=self.server_url,
                cache_id=self.cache_id,
                api_key=self.api_key,
            )
        except Exception as e:
            logger.error(f"Error creating LangCache: {e}")
            return None
    
    def search(self, prompt, deserialize=True):
        """Search for a prompt in the cache
        
        Args:
            prompt (str): The prompt to search for
            deserialize (bool): Whether to deserialize the JSON response
            
        Returns:
            The search response (deserialized if deserialize=True) or None if not found
        """
        try:
            # Create LangCache instance
            with self._create_lang_cache() as lang_cache:
                if not lang_cache:
                    return None
                    
                logger.info(f"Searching cache for prompt: {prompt}")
                search_response = lang_cache.search(prompt=prompt, similarity_threshold=self.similarity_threshold)
                print("324563",search_response)
                if not search_response or len(search_response.data) == 0:
                    return None
                    
                # Process response if deserialization is requested
                if deserialize and hasattr(search_response, 'data') and search_response.data:
                    # Return first response for simplicity
                    if len(search_response.data) > 0 and hasattr(search_response.data[0], 'response') and len(search_response.data[0].response) > 0:
                        return search_response.data[0].response
                
                return search_response
        except Exception as e:
            logger.error(f"Error searching cache: {e}")
            return None
            
    def _process_response(self, response):
        """Process a response from the cache
        
        Args:
            response: The response to process
            
        Returns:
            The processed response
        """
        # Only try to deserialize if it's a string
        if isinstance(response, str):
            print("32456","3245",response)
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                logger.warning("Could not deserialize response")
                return response
        # If it's already a list or other type, return as is
        return response
    
    def set(self, prompt, response):
        """Set a prompt and response in the cache
        
        Args:
            prompt (str): The prompt to cache
            response: The response to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Serialize response if needed
        serialized_response = self._serialize_response(response)
        if serialized_response is None:
            return False
            
        try:
            # Create LangCache instance
            with self._create_lang_cache() as lang_cache:
                if not lang_cache:
                    return False
                    
                logger.info(f"Setting cache for prompt: {prompt}")
                lang_cache.set(prompt=prompt, response=serialized_response)
                return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
            
    def _serialize_response(self, response):
        """Serialize a response for caching
        
        Args:
            response: The response to serialize
            
        Returns:
            str: The serialized response or None if serialization fails
        """
        # If already a string, return as is
        if isinstance(response, str):
            return response
            
        # Convert to JSON string
        try:
            serialized = json.dumps(response)
            logger.info("Successfully serialized response to JSON string")
            return serialized
        except Exception as json_error:
            logger.error(f"Error serializing response: {json_error}")
            return None
            
    async def async_search(self, prompt, deserialize=True):
        """Async version of search method
        
        Args:
            prompt (str): The prompt to search for
            deserialize (bool): Whether to deserialize the JSON response
            
        Returns:
            The search response (deserialized if deserialize=True) or None if not found
        """
        try:
            return await asyncio.to_thread(self.search, prompt, deserialize)
        except Exception as e:
            logger.error(f"Error in async search: {e}")
            return None
            
    async def async_set(self, prompt, response):
        """Async version of set method
        
        Args:
            prompt (str): The prompt to cache
            response: The response to cache
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate response can be serialized before running in thread pool
            print("going to searilize")
            if not isinstance(response, str):
                print("not a string")
                serialized = self._serialize_response(response)
                print("serialized",serialized)
                if serialized is None:
                    return False
                    
            return await asyncio.to_thread(self.set, prompt, response)
        except Exception as e:
            logger.error(f"Error in async set: {e}")
            return False

# Create a singleton instance
lang_cache_service = LangCacheService()