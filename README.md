# Pinecone Search API

This project is a FastAPI application that uses Pinecone for semantic search.

## Installation

### Option 1: Local Installation

1. Clone the repository.
2. Create a `.env` file and add the following environment variables:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
REDIS_LANGCACHE_SERVER_URL=your_redis_langcache_server_url
REDIS_LANGCACHE_API_KEY=your_redis_langcache_api_key
REDIS_LANGCACHE_CACHE_ID=your_redis_langcache_cache_id
REDIS_LANGCACHE_SIMILARITY_THRESHOLD=0.9
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Docker Installation

1. Clone the repository.
2. Create a `.env` file with the environment variables as shown above.
3. Build and run the Docker container:

```bash
docker-compose up --build
```

## Running the application

### Local Run

```bash
uvicorn main:app --reload
```

### Docker Run

```bash
docker-compose up
```

The API will be available at http://localhost:8000. You can access the Swagger documentation at http://localhost:8000/docs.

## API Endpoints

### Ingest a single document

```bash
curl -X 'POST' \
  'http://localhost:8000/api/ingest/ingest' \
  -H 'Content-Type: application/json' \
  -d '{
  "id": "doc1",
  "text": "This is a sample document for testing",
  "metadata": {"category": "sample"}
}'
```

### Batch ingest multiple documents

```bash
curl -X 'POST' \
  'http://localhost:8000/api/ingest/batch-ingest' \
  -H 'Content-Type: application/json' \
  -d '{
  "documents": [
    {
      "id": "doc1",
      "text": "First sample document",
      "metadata": {"category": "sample"}
    },
    {
      "id": "doc2",
      "text": "Second sample document",
      "metadata": {"category": "sample"}
    }
  ]
}'
```

### Search documents

```bash
curl -X 'POST' \
  'http://localhost:8000/api/search/search' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "sample document",
  "top_k": 5
}'
```

## Redis Cache Integration

The application uses Redis for caching embeddings and search results to improve performance and reduce API calls.

### Configuration

To use Redis caching, ensure you have Redis installed and running. Set the following environment variables in your `.env` file:

```
REDIS_HOST=localhost  # Redis server host
REDIS_PORT=6379        # Redis server port
REDIS_DB=0             # Redis database number
REDIS_PASSWORD=        # Redis password (if required)
```

### Cache Behavior

- **Embedding Cache**: Generated embeddings are cached for 24 hours (86400 seconds)
- **Search Results Cache**: Search results are cached for 1 hour (3600 seconds)

The cache automatically handles:
- Storing and retrieving embeddings to reduce OpenAI API calls
- Storing and retrieving search results to reduce Pinecone queries
- Proper invalidation when documents are updated or deleted

## Running tests

### Unit Tests

To run the unit tests for services and API endpoints:

```bash
pytest tests/test_services.py tests/test_api.py -v
```

### Smoke Tests

For quick validation of the entire system, run the smoke test script:

```bash
python scripts/smoke_test.py
```

This script tests:
- Health check endpoint
- Single document ingestion
- Batch document ingestion
- Search functionality with cache validation

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=app tests/
```