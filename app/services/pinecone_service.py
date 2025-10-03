from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from app.utils.config import PINECONE_API_KEY, EMBEDDING_DIMENSION
from app.utils.logger import get_logger

logger = get_logger(__name__)

class PineconeService:
    def __init__(self, index_name: str):
        try: 
            logger.info("Initializing Pinecone", index_name=index_name)
            self.pc = Pinecone(api_key=PINECONE_API_KEY)
            # 1. Get list of indexes
            existing_indexes = self.pc.list_indexes().names()
            logger.info("Found existing indexes", indexes=existing_indexes)
            if index_name not in existing_indexes:
                logger.info("Index not found, creating new index", index_name=index_name)
                self.pc.create_index(
                    name=index_name,
                    dimension=EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info("Index created successfully", index_name=index_name)
                # 2. Connect to the index
            self.index = self.pc.Index(index_name)
            logger.info("Successfully connected to Pinecone index", index=str(self.index))
        except Exception as e:
            logger.error("Failed to initialize Pinecone", error=str(e))
            raise

    def upsert(self, vectors: List[Dict[str, Any]], namespace: str = None) -> Dict[str, Any]:
        """Upsert vectors to Pinecone index and return response"""
        try:
            batch_size = 100
            total_vectors = len(vectors)
            logger.info("Upserting vectors to Pinecone", count=total_vectors, namespace=namespace)

            responses = []
            for i in range(0, total_vectors, batch_size):
                batch = vectors[i:i + batch_size]
                batch_num = i//batch_size + 1
                batch_size = len(batch)
                logger.debug("Upserting batch", batch_num=batch_num, batch_size=batch_size)
                response = self.index.upsert(vectors=batch, namespace=namespace) if namespace else self.index.upsert(vectors=batch)
                responses.append(response)

            logger.info("Successfully upserted vectors", count=total_vectors)
            return {"total_upserted": total_vectors, "batches": responses}
        except Exception as e:
            logger.error("Error upserting vectors to Pinecone", error=str(e))
            raise

    def query(self, query_embedding: any, top_k: int = 5, namespace: str = None) -> Dict[str, Any]:
        """Query Pinecone index with vector"""
        try:
            logger.info("Querying Pinecone", top_k=top_k, namespace=namespace)
            if namespace:
                results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True, namespace=namespace, include_values=False)
            else:
                results = self.index.query(vector=query_embedding, top_k=top_k, include_metadata=True, include_values=False)

            matches = results.matches if hasattr(results, "matches") else results.get("matches", [])
            logger.info("Query completed", matches_count=len(matches))
            return matches
        except Exception as e:
            logger.error("Error querying Pinecone", error=str(e))
            raise

    def delete(self, ids: List[str]) -> Dict[str, Any]:
        """Delete vectors from Pinecone index by IDs"""
        try:
            logger.info("Deleting vectors from Pinecone", count=len(ids))
            response = self.index.delete(ids=ids)
            logger.info("Successfully deleted vectors", count=len(ids))
            return response
        except Exception as e:
            logger.error("Error deleting vectors from Pinecone", error=str(e))
            raise
