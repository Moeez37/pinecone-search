#!/usr/bin/env python3
"""
Test script to verify embedding dimensions with OpenAI API
"""

from embeddings_service import embeddings_service
from config import settings
from pinecone_client import pinecone_client

def test_embedding_dimension():
    print(f"Embedding Service: OpenAI")
    print(f"Expected Dimension: {settings.EMBEDDING_DIMENSION}")
    
    # Test embedding generation
    test_text = "red running shoes for athletes"
    print(f"\nGenerating embedding for: '{test_text}'")
    
    try:
        embedding = embeddings_service.generate_embedding(test_text)
        print(f"Generated embedding dimension: {len(embedding)}")
        print(f"Expected dimension: {settings.EMBEDDING_DIMENSION}")
        print(f"Dimensions match: {len(embedding) == settings.EMBEDDING_DIMENSION}")
        
        # Test Pinecone index creation
        print(f"\nPinecone index name: {pinecone_client.index_name}")
        print(f"Index initialized: {pinecone_client.index is not None}")
        
        if pinecone_client.index:
            stats = pinecone_client.get_index_stats()
            print(f"Index stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = test_embedding_dimension()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")