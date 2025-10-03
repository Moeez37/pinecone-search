from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from app.models import SearchRequest, SearchResponse, SearchMatch
from app.services import get_embedding_service, get_pinecone_service, get_cache_service
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.services.cache_service import CacheService
from app.utils.constants import OBJECT_TYPE
from typing import List, Dict, Any, Optional
from app.utils.logger import get_logger
import json
import asyncio

router = APIRouter()
logger = get_logger(__name__)

@router.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    cache_service: CacheService = Depends(get_cache_service)
):
    try:
        query = request.query
        top_k = request.top_k
        query_namespaces = [request.namespace] if request.namespace != OBJECT_TYPE.ALL else [OBJECT_TYPE.PRODUCTS, OBJECT_TYPE.BLOGS, OBJECT_TYPE.STORES]
        location_id = request.location_id
        include_metadata = True


        print(f"Search request received for query: '{query}' with top_k={top_k}")
        
        cached_results = cache_service.get(query)

        cache_results = (json.loads(cached_results.data[0].response) if cached_results and cached_results.data and cached_results.data[0].response else None)
        
        if cached_results and cached_results.data and cached_results.data[0].response:
            print(f"Cache hit for search query: '{query}' and cached result len is '{(cache_results)}'")
            # Format results from cache - convert to SearchMatch objects
            cached_matches = []
            print(cache_results)
            for match in cache_results:
                search_match = SearchMatch(
                    id=match.get('id'),
                    metadata=match.get('metadata'),
                    score=match.get('score')
                )
                cached_matches.append(search_match)
                
            return SearchResponse(
                results=cached_matches,
                query_rewritten=None,
                total_results=len(cached_matches)
            )
        
        print(f"Cache miss for search query: '{query}', querying Pinecone")
        # Generate embedding for the query (langcache will handle embedding caching)
        query_embedding = embedding_service.generate_embedding(query)
        
        # Query Pinecone across all specified namespaces in parallel
        loop = asyncio.get_event_loop()
        tasks = []
        for namespace in query_namespaces:
            namespace_str = str(namespace)
            namespace_str = str(f"{request.location_id}-{namespace_str}" if namespace_str == OBJECT_TYPE.PRODUCTS else namespace_str)
            print(f"Querying namespace: {namespace_str}")
            tasks.append(loop.run_in_executor(None, pinecone_service.query, query_embedding, top_k, namespace_str))

        parallel_results = await asyncio.gather(*tasks)
        
        all_results = []
        for result in parallel_results:
            all_results.extend(result)
        # Sort combined results by score and limit to top_k
        # Convert to SearchMatch objects for response
        search_matches = []
        cache_results = []
        for match in all_results:
            result = {
                'id': match.get('id'),
                'metadata': match.get('metadata'),
                'score': match.get('score')
            }
            cache_results.append(result)
            search_match = SearchMatch(**result)
            search_matches.append(search_match)
            
        # Cache the results in the background
        background_tasks.add_task(cache_service.set, query, cache_results)
        print(f"Queued caching for search results for query: '{query}'")

        return SearchResponse(
            results=search_matches,
            query_rewritten=None,
            total_results=len(search_matches)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))