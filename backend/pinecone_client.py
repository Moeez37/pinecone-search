import os
import json
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PineconeClient:
    def __init__(self):
        try:
            print("here is my key of pinecone",settings.PINECONE_API_KEY)
            self.pc = Pinecone(api_key="pcsk_7PFDdr_JY4CmKFXhsvCVw268oP7PvXGry82DQqUMCmsdjvT51Y127tS36nBByCok9Kh9Zw")
            self.index_name = settings.PINECONE_INDEX_NAME
            self.index = None
            self._initialize_index()
        except Exception as e:
            logger.error(f"Warning: Failed to initialize Pinecone client: {e}")
            logger.error("Server will start but Pinecone features will be unavailable.")
            self.pc = None
            self.index = None
            self.index_name = None
    
    def _initialize_index(self):
        """Initialize or create Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = self.pc.list_indexes().names()
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Index {self.index_name} created successfully")
            else:
                logger.info(f"Using existing index: {self.index_name}")
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"Connected to index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Warning: Failed to initialize Pinecone: {e}")
            logger.error("Server will start but Pinecone features will be unavailable.")
            logger.error("Please check your PINECONE_API_KEY in settings")
            self.index = None
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]], namespace: Optional[str] = None) -> bool:
        """Upsert vectors to Pinecone index with namespace support"""
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return False
                
            # Convert vectors to the format expected by Pinecone
            pinecone_vectors = []
            for vector in vectors:
                pinecone_vectors.append({
                    "id": vector["id"],
                    "values": vector["values"],
                    "metadata": vector.get("metadata", {})
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(pinecone_vectors), batch_size):
                batch = pinecone_vectors[i:i + batch_size]
                upsert_params = {"vectors": batch}
                if namespace:
                    upsert_params["namespace"] = namespace
                
                self.index.upsert(**upsert_params)
                namespace_info = f" in namespace '{namespace}'" if namespace else ""
                logger.info(f"Upserted batch {i//batch_size + 1} with {len(batch)} vectors{namespace_info}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            return False
    
    def search(
        self, 
        query_vector: List[float], 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        namespace: Optional[str] = None,
        namespaces: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone with namespace support"""
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return []
            
            # If multiple namespaces are provided, search across all of them
            if namespaces:
                all_results = []
                for ns in namespaces:
                    search_params = {
                        "vector": query_vector,
                        "top_k": top_k * 2,  # Get more results per namespace
                        "include_metadata": include_metadata,
                        "namespace": ns
                    }
                    
                    if filter_dict:
                        search_params["filter"] = filter_dict
                    
                    results = self.index.query(**search_params)
                    logger.info(f"Namespace '{ns}' returned {len(results.matches)} results")
                    
                    # Format results and add namespace info
                    for match in results.matches:
                        result = {
                            "id": match.id,
                            "score": match.score,
                            "metadata": match.metadata if include_metadata else {},
                            "namespace": ns
                        }
                        all_results.append(result)
                
                logger.info(f"Total results from all namespaces: {len(all_results)}")
                # Sort by score and return top_k results
                all_results.sort(key=lambda x: x["score"], reverse=True)
                return all_results[:top_k * 2]  # Return more results for further processing
            
            # Single namespace or default search
            search_params = {
                "vector": query_vector,
                "top_k": top_k,
                "include_metadata": include_metadata
            }
            
            if namespace:
                search_params["namespace"] = namespace
            
            if filter_dict:
                search_params["filter"] = filter_dict
            
            results = self.index.query(**search_params)
            
            # Format results
            formatted_results = []
            for match in results.matches:
                result = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": match.metadata if include_metadata else {}
                }
                if namespace:
                    result["namespace"] = namespace
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    def delete_all(self) -> bool:
        """Delete all vectors from the index (useful for testing)"""
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return False
            
            self.index.delete(delete_all=True)
            logger.info("All vectors deleted from index")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            if not self.index:
                logger.error("Pinecone index not available")
                return {}
            
            stats = self.index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_fullness": stats.index_fullness,
                "namespaces": dict(stats.namespaces) if stats.namespaces else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {}

# Global instance
pinecone_client = PineconeClient()