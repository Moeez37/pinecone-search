import json
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pinecone_client import pinecone_client
from embeddings_service import embeddings_service
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self):
        self.data_dir = "data"
        self.stores = {}
        self.products = []
        self.blogs = []
    
    def load_json_data(self):
        """Load all JSON data files"""
        try:
            # Load stores
            with open(os.path.join(self.data_dir, "stores.json"), "r") as f:
                stores_list = json.load(f)
                self.stores = {store["slug"]: store for store in stores_list}
            
            # Load products
            with open(os.path.join(self.data_dir, "products.json"), "r") as f:
                self.products = json.load(f)
            
            # Load blogs
            with open(os.path.join(self.data_dir, "blogs.json"), "r") as f:
                self.blogs = json.load(f)
            
            logger.info(f"Loaded {len(self.stores)} stores, {len(self.products)} products, {len(self.blogs)} blogs")
            
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}")
            raise
    
    def create_product_vectors(self) -> List[Dict[str, Any]]:
        """Create vectors for all products in parallel"""
        vectors = []

        # Create a mapping of locationId to store info for faster lookup
        store_lookup = {}
        for store in self.stores.values():
            if store.get('locationId') or store.get("sweedId") or store.get("medicalStoreId") or store.get("recreationalStoreId"):
                store_lookup[store['locationId']] = store if store.get('locationId') else {}
                store_lookup[store['sweedId']] = store if store.get('sweedId') else {}
                store_lookup[store['medicalStoreId']] = store if store.get('medicalStoreId') else {}
                store_lookup[store['recreationalStoreId']] = store if store.get('recreationalStoreId') else {}
        
        def create_single_product_vector(product):
            try:
                # Find store info for this product
                store_info = store_lookup.get(product.get('store_id'))
                vector = embeddings_service.create_product_vector(product, store_info)
                logger.info(f"Created vector for product: {product.get('productId', 'unknown')}")
                return vector
            except Exception as e:
                logger.error(f"Error creating vector for product {product.get('productId', 'unknown')}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_product = {executor.submit(create_single_product_vector, product): product for product in self.products}
            for future in as_completed(future_to_product):
                result = future.result()
                if result is not None:
                    vectors.append(result)
        
        return vectors
    
    def create_blog_vectors(self) -> List[Dict[str, Any]]:
        """Create vectors for all blogs in parallel"""
        vectors = []
        
        def create_single_blog_vector(blog):
            try:
                vector = embeddings_service.create_blog_vector(blog)
                logger.info(f"Created vector for blog: {blog.get('title', blog.get('slug', blog.get('databaseId', 'unknown')))}")
                return vector
            except Exception as e:
                logger.error(f"Error creating vector for blog {blog.get('title', blog.get('slug', blog.get('databaseId', 'unknown')))}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_blog = {executor.submit(create_single_blog_vector, blog): blog for blog in self.blogs}
            
            for future in as_completed(future_to_blog):
                result = future.result()
                if result is not None:
                    vectors.append(result)
        
        return vectors
    
    def create_store_vectors(self) -> List[Dict[str, Any]]:
        """Create vectors for all stores in parallel"""
        vectors = []
        
        def create_single_store_vector(store):
            try:
                vector = embeddings_service.create_store_vector(store)
                logger.info(f"Created vector for store: {store.get('title', store.get('locationId', 'unknown'))}")
                return vector
            except Exception as e:
                logger.error(f"Error creating vector for store {store.get('title', store.get('locationId', 'unknown'))}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_store = {executor.submit(create_single_store_vector, store): store for store in self.stores.values()}
            
            for future in as_completed(future_to_store):
                result = future.result()
                if result is not None:
                    vectors.append(result)
        
        return vectors
    
    def ingest_all_data(self, clear_existing: bool = False):
        """Ingest all data into Pinecone in parallel namespaces"""
        try:
            logger.info("Starting data ingestion...")

            if clear_existing:
                logger.info("Clearing existing data...")
                pinecone_client.delete_all()

            # Load JSON data
            self.load_json_data()

            # Create vectors per type in parallel
            logger.info("Creating vectors for all data types in parallel...")
            
            def create_vectors_with_logging(vector_type, create_func):
                logger.info(f"Creating {vector_type} vectors...")
                vectors = create_func()
                logger.info(f"Completed creating {len(vectors)} {vector_type} vectors")
                return vectors
            
            # Run vector creation in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_type = {
                    executor.submit(create_vectors_with_logging, "product", self.create_product_vectors): "product",
                    executor.submit(create_vectors_with_logging, "blog", self.create_blog_vectors): "blog",
                    executor.submit(create_vectors_with_logging, "store", self.create_store_vectors): "store"
                }
                
                vector_results = {}
                for future in as_completed(future_to_type):
                    vector_type = future_to_type[future]
                    try:
                        vectors = future.result()
                        vector_results[vector_type] = vectors
                    except Exception as e:
                        logger.error(f"Error creating {vector_type} vectors: {e}")
                        vector_results[vector_type] = []
            
            product_vectors = vector_results.get("product", [])
            blog_vectors = vector_results.get("blog", [])
            store_vectors = vector_results.get("store", [])

            # Map namespace → vectors
            vector_batches = {
                "product": product_vectors,
                "blog": blog_vectors,
                "store": store_vectors,
            }

            def upsert_to_namespace(namespace, vectors):
                """Helper to upsert into Pinecone namespace"""
                if not vectors:
                    logger.warning(f"No vectors to upsert in namespace: {namespace}")
                    return False

                logger.info(f"Upserting {len(vectors)} vectors into namespace '{namespace}'...")
                success = pinecone_client.upsert_vectors(vectors, namespace=namespace)
                if success:
                    logger.info(f"✅ Successfully ingested {len(vectors)} vectors into '{namespace}'")
                    return True
                else:
                    logger.error(f"❌ Failed to upsert into namespace '{namespace}'")
                    return None

            # Run ingestion in parallel threads
            results = {}
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_ns = {
                    executor.submit(upsert_to_namespace, ns, vecs): ns
                    for ns, vecs in vector_batches.items()
                }
                for future in as_completed(future_to_ns):
                    ns = future_to_ns[future]
                    try:
                        result = future.result()
                        results[ns] = result
                    except Exception as e:
                        logger.error(f"Error in namespace '{ns}': {e}")
                        results[ns] = None

            logger.info(f"Final ingestion results: {results}")

        except Exception as e:
            logger.error(f"Error during data ingestion: {e}")
            raise
    
    def test_search(self, query: str = "red shoes"):
        """Test search functionality"""
        try:
            logger.info(f"Testing search with query: '{query}'")
            
            # Generate query embedding
            query_embedding = embeddings_service.generate_embedding(query)
            
            # Search
            results = pinecone_client.search(
                query_vector=query_embedding,
                top_k=5,
                include_metadata=True
            )
            
            logger.info(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                metadata = result['metadata']
                logger.info(f"{i}. {metadata.get('type', 'unknown').upper()}: {metadata.get('name', metadata.get('title', 'unknown'))} (Score: {result['score']:.3f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Error during test search: {e}")
            return []

def main():
    """Main function for running data ingestion"""
    ingestion = DataIngestion()
    
    try:
        # Ingest all data
        ingestion.ingest_all_data(clear_existing=True)
        
        # Test search
        print("\n" + "="*50)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*50)
        
        test_queries = [
            "red shoes",
            "ergonomic chair",
            "eco-friendly bottle",
            "fitness supplements"
        ]
        
        for query in test_queries:
            print(f"\nTesting: '{query}'")
            results = ingestion.test_search(query)
            print("-" * 30)
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    main()