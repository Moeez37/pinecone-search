#!/usr/bin/env python3
"""
Smoke test script for Pinecone Search API

This script performs basic validation of the API endpoints:
- Health check
- Document ingestion (single and batch)
- Document search with cache validation

Usage:
    python smoke_test.py
"""

import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api"

# Test documents
TEST_DOCS = [
    {"id": "doc1", "text": "This is a test document about artificial intelligence and machine learning."},
    {"id": "doc2", "text": "Python is a popular programming language for data science and AI applications."},
    {"id": "doc3", "text": "Vector databases like Pinecone are used for similarity search and recommendation systems."},
]

# Test queries
TEST_QUERIES = [
    "What is artificial intelligence?",
    "Tell me about Python programming",
    "How are vector databases used?"
]

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f" {message} ")
    print("=" * 80)

def test_health_check():
    """Test the health check endpoint"""
    print_header("Testing Health Check Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print(f"✅ Health check successful: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed with status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed with error: {str(e)}")
        return False

def test_single_ingest():
    """Test single document ingestion"""
    print_header("Testing Single Document Ingestion")
    try:
        doc = TEST_DOCS[0]
        response = requests.post(
            f"{API_URL}/ingest/ingest",
            json=doc
        )
        if response.status_code == 200:
            print(f"✅ Single document ingestion successful: {response.json()}")
            return True
        else:
            print(f"❌ Single document ingestion failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Single document ingestion failed with error: {str(e)}")
        return False

def test_batch_ingest():
    """Test batch document ingestion"""
    print_header("Testing Batch Document Ingestion")
    try:
        # Use the remaining documents for batch ingestion
        docs = TEST_DOCS[1:]
        response = requests.post(
            f"{API_URL}/ingest/batch-ingest",
            json={"documents": docs}
        )
        if response.status_code == 200:
            print(f"✅ Batch document ingestion successful: {response.json()}")
            return True
        else:
            print(f"❌ Batch document ingestion failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch document ingestion failed with error: {str(e)}")
        return False

def test_search_with_cache():
    """Test search functionality with cache validation"""
    print_header("Testing Search with Cache Validation")
    try:
        # First search (cache miss)
        query = TEST_QUERIES[0]
        print(f"Performing first search with query: '{query}'")
        start_time = time.time()
        response1 = requests.post(
            f"{API_URL}/search/search",
            json={"query": query, "top_k": 3}
        )
        first_search_time = time.time() - start_time
        
        if response1.status_code != 200:
            print(f"❌ First search failed with status code: {response1.status_code}")
            print(f"Response: {response1.text}")
            return False
            
        print(f"✅ First search successful (cache miss) in {first_search_time:.4f} seconds")
        print(f"Results: {json.dumps(response1.json(), indent=2)}")
        
        # Second search with same query (should be cache hit)
        print(f"\nPerforming second search with same query (should be cache hit)")
        start_time = time.time()
        response2 = requests.post(
            f"{API_URL}/search/search",
            json={"query": query, "top_k": 3}
        )
        second_search_time = time.time() - start_time
        
        if response2.status_code != 200:
            print(f"❌ Second search failed with status code: {response2.status_code}")
            print(f"Response: {response2.text}")
            return False
            
        print(f"✅ Second search successful in {second_search_time:.4f} seconds")
        
        # Verify cache is working by comparing response times
        if second_search_time < first_search_time:
            print(f"✅ Cache appears to be working! Second search was faster ({second_search_time:.4f}s vs {first_search_time:.4f}s)")
        else:
            print(f"⚠️ Cache might not be working as expected. Second search was not faster ({second_search_time:.4f}s vs {first_search_time:.4f}s)")
        
        # Verify results are consistent
        if response1.json() == response2.json():
            print("✅ Search results are consistent between calls")
        else:
            print("⚠️ Search results differ between calls")
            
        return True
    except Exception as e:
        print(f"❌ Search test failed with error: {str(e)}")
        return False

def run_all_tests():
    """Run all smoke tests"""
    print_header("PINECONE SEARCH API SMOKE TESTS")
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Document Ingestion", test_single_ingest),
        ("Batch Document Ingestion", test_batch_ingest),
        ("Search with Cache Validation", test_search_with_cache)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\nRunning test: {name}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test '{name}' failed with unexpected error: {str(e)}")
            results.append((name, False))
    
    # Print summary
    print_header("TEST SUMMARY")
    all_passed = True
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        if not result:
            all_passed = False
        print(f"{status} - {name}")
    
    if all_passed:
        print("\n🎉 All tests passed! The API is functioning correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the logs above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())