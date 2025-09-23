from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone_client import pinecone_client
from embeddings_service import embeddings_service
from config import settings
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from lang_cache_service import lang_cache_service
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PineCone Search API", version="1.0.0")

# Background task function for caching search results
async def cache_search_results(query: str, response: List[Dict]):
    """Background task to cache search results"""
    try:
        logger.info("Background task: Caching search results")
        cache_result = await lang_cache_service.async_set(
            prompt=query,
            response=response
        )
        if cache_result:
            logger.info("Background task: Successfully cached search results")
        else:
            logger.warning("Background task: Failed to cache search results")
    except Exception as cache_error:
        logger.error(f"Background task: Cache error: {cache_error}")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class SearchRequest(BaseModel):
    query: str
    location: Optional[str] = None
    techType: Optional[str] = "none"  # "products", "blogs", "all"
    rewrite_query: Optional[bool] = True  # Enable/disable query rewriting
    top_k: Optional[int] = 10  # Number of results to return

class SearchResult(BaseModel):
    id: str
    type: str  # "product", "blog", or "store"
    score: float
    metadata: Dict[str, Any]

class SearchResponse(BaseModel):
    results: List[SearchResult]
    query_rewritten: Optional[str] = None
    total_results: int

class IndexStats(BaseModel):
    total_vector_count: int
    dimension: int
    index_fullness: float
    namespaces: Dict[str, Any]

@app.get("/")
async def root():
    return {"message": "PineCone Search API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/search")
async def search(request: SearchRequest, background_tasks: BackgroundTasks):
    lang_cache_sdk = None
    """Search across all namespaces (product, blog, store) with parallel execution"""
    try:
        logger.info(f"Search request: {request.query} | Location: {request.location} | techType: {getattr(request, 'techType', None)}")

        should_use_cache = getattr(request, "techType", None) == "redis"

        # Use the LangCacheService to search for cached results
        search_response = None
        if should_use_cache:
            search_response = await lang_cache_service.async_search(prompt=request.query)
        
        # Check if we have a valid response with data
        if search_response and len(search_response) > 0:
            return {"results": json.loads(search_response), "query_rewritten": None, "total_results": len(search_response)}

        # Check Pinecone index
        if not pinecone_client.index:
            return {"results": [], "query_rewritten": None, "total_results": 0}

        # --- Query rewriting (disabled for now) ---
        rewritten_query = None
        search_query = request.query
        if False and request.rewrite_query:
            try:
                rewritten_query = embeddings_service.rewrite_query(request.query)
                search_query = rewritten_query
                logger.info(f"Query rewritten: '{request.query}' → '{rewritten_query}'")
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}, using original query")

        # --- Generate embedding ---
        query_embedding = embeddings_service.generate_embedding(search_query)

        # --- Always search across all namespaces ---
        namespaces = ["product", "blog", "store"]

        def search_namespace(namespace: str):
            logger.info(f"Searching namespace: {namespace}")
            results = pinecone_client.search(
                query_vector=query_embedding,
                top_k=min(request.top_k * 2, settings.MAX_SEARCH_RESULTS),
                include_metadata=True,
                namespace=namespace,
            )
            for r in results:
                r["namespace"] = namespace
                if not r["metadata"].get("type"):
                    r["metadata"]["type"] = namespace
            return results[:3]

        # --- Run namespace searches in parallel ---
        search_results = []
        with ThreadPoolExecutor(max_workers=len(namespaces)) as executor:
            futures = {executor.submit(search_namespace, ns): ns for ns in namespaces}
            for future in as_completed(futures):
                ns = futures[future]
                try:
                    res = future.result()
                    search_results.extend(res)
                    logger.info(f"Namespace '{ns}' returned {len(res)} results")
                except Exception as e:
                    logger.error(f"Search failed for namespace '{ns}': {e}")

        logger.info(f"Total results found across namespaces: {len(search_results)}")
        sorted_response = sorted(search_results, key=lambda x: x["score"], reverse=True)
        # --- Add caching to background tasks ---
        if should_use_cache:
            background_tasks.add_task(
                cache_search_results,
                request.query,
                sorted_response
            )
            logger.info("Added search results caching to background tasks")
        return {
            "results": sorted_response,
            "query_rewritten": rewritten_query,
            "total_results": len(sorted_response),
        }

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def get_store_info_by_location(location_id: str) -> Optional[Dict[str, Any]]:
    """Get store information by location ID"""
    try:
        stores_file = os.path.join(os.path.dirname(__file__), "data", "stores.json")
        with open(stores_file, 'r') as f:
            stores = json.load(f)
        
        for store in stores:
            if store.get('locationId') == location_id:
                return {
                    'locationId': store.get('locationId', ''),
                    'city': store.get('city', ''),
                    'state': store.get('state', {}).get('abbr', '') if store.get('state') else ''
                }
        return None
    except Exception as e:
        logger.error(f"Error loading store info for location {location_id}: {e}")
        return None

@app.get("/stores")
async def get_stores():
    """Get all available stores"""
    try:
        stores_file = os.path.join(os.path.dirname(__file__), "data", "stores.json")
        with open(stores_file, 'r') as f:
            stores = json.load(f)
        return {"stores": stores}
    except Exception as e:
        logger.error(f"Error loading stores: {e}")
        raise HTTPException(status_code=500, detail="Failed to load stores")

@app.get("/index/info")
async def get_index_info():
    """Get current Pinecone index information"""
    try:
        if not pinecone_client.index:
            raise HTTPException(status_code=503, detail="Pinecone client not available")
        
        return {
            "index_name": pinecone_client.index_name,
            "embedding_service": "openai",
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "status": "connected"
        }
    except Exception as e:
        logger.error(f"Error getting index info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get index info: {str(e)}")

@app.get("/index/stats")
async def get_index_stats():
    """Get Pinecone index statistics"""
    try:
        if not pinecone_client.index:
            return {"status": "error", "message": "Pinecone not available. Please check your API keys."}
        
        stats = pinecone_client.get_index_stats()
        # Convert namespaces to a simple dict for JSON serialization
        if "namespaces" in stats and stats["namespaces"]:
            serializable_namespaces = {}
            for ns_name, ns_data in stats["namespaces"].items():
                if hasattr(ns_data, 'vector_count'):
                    serializable_namespaces[ns_name] = {"vector_count": ns_data.vector_count}
                else:
                    serializable_namespaces[ns_name] = dict(ns_data) if ns_data else {}
            stats["namespaces"] = serializable_namespaces
        
        return {"status": "success", "stats": stats}
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/reingest")
async def reingest_data():
    """Reingest all data (useful for development)"""
    try:
        from data_ingestion import DataIngestion
        ingestion = DataIngestion()
        ingestion.ingest_all_data(clear_existing=True)
        return {"message": "Data reingestion completed successfully"}
    except Exception as e:
        logger.error(f"Reingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Reingestion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)