import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from main import app

client = TestClient(app)


@pytest.fixture
def mock_embedding_service():
    with patch("app.api.ingest_routes.get_embedding_service") as mock:
        mock_service = MagicMock()
        mock_service.generate_embedding.return_value = [0.1, 0.2, 0.3]
        mock_service.generate_embeddings.return_value = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
        mock.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_pinecone_service():
    with patch("app.api.ingest_routes.get_pinecone_service") as mock:
        mock_service = MagicMock()
        mock_service.upsert.return_value = None
        mock_service.query.return_value = {
            "matches": [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "metadata": {"text": "Sample document", "category": "test"}
                },
                {
                    "id": "doc2",
                    "score": 0.8,
                    "metadata": {"text": "Another document", "category": "test"}
                }
            ]
        }
        mock.return_value = mock_service
        yield mock_service


# Test health endpoint
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "server running"}


# Test ingest endpoint
def test_ingest_document(mock_embedding_service, mock_pinecone_service):
    response = client.post(
        "/api/ingest/ingest",
        json={
            "id": "test-doc",
            "text": "This is a test document",
            "metadata": {"category": "test"}
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_embedding_service.generate_embedding.assert_called_once_with("This is a test document")
    mock_pinecone_service.upsert.assert_called_once()


# Test batch ingest endpoint
def test_batch_ingest_documents(mock_embedding_service, mock_pinecone_service):
    response = client.post(
        "/api/ingest/batch-ingest",
        json={
            "documents": [
                {
                    "id": "test-doc-1",
                    "text": "This is test document 1",
                    "metadata": {"category": "test"}
                },
                {
                    "id": "test-doc-2",
                    "text": "This is test document 2",
                    "metadata": {"category": "test"}
                }
            ]
        }
    )
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_embedding_service.generate_embeddings.assert_called_once()
    mock_pinecone_service.upsert.assert_called_once()


@pytest.fixture
def mock_cache_service():
    with patch("app.api.search_routes.get_cache_service") as mock:
        mock_service = MagicMock()
        mock_service.get.return_value = None
        mock_service.set.return_value = None
        mock.return_value = mock_service
        yield mock_service

# Test search endpoint - cache miss
def test_search_documents_cache_miss(mock_embedding_service, mock_pinecone_service, mock_cache_service):
    # Need to patch the search routes module separately
    with patch("app.api.search_routes.get_embedding_service", return_value=mock_embedding_service), \
         patch("app.api.search_routes.get_pinecone_service", return_value=mock_pinecone_service), \
         patch("app.api.search_routes.get_cache_service", return_value=mock_cache_service):
        
        # Set up cache miss
        mock_cache_service.get.return_value = None
        
        response = client.post(
            "/api/search/search",
            json={
                "query": "test query",
                "top_k": 2
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "matches" in result
        assert len(result["matches"]) == 2
        assert result["matches"][0]["id"] == "doc1"
        assert result["matches"][1]["id"] == "doc2"
        assert result["query"] == "test query"
        
        # Verify cache was checked
        mock_cache_service.get.assert_called_once()
        # Verify embedding was generated
        mock_embedding_service.generate_embedding.assert_called_with("test query")
        # Verify Pinecone was queried
        mock_pinecone_service.query.assert_called_with(vector=[0.1, 0.2, 0.3], top_k=2)
        # Verify results were cached
        mock_cache_service.set.assert_called_once()

# Test search endpoint - cache hit
def test_search_documents_cache_hit(mock_embedding_service, mock_pinecone_service, mock_cache_service):
    # Need to patch the search routes module separately
    with patch("app.api.search_routes.get_embedding_service", return_value=mock_embedding_service), \
         patch("app.api.search_routes.get_pinecone_service", return_value=mock_pinecone_service), \
         patch("app.api.search_routes.get_cache_service", return_value=mock_cache_service):
        
        # Set up cache hit
        cached_results = {
            "matches": [
                {
                    "id": "cached_doc1",
                    "score": 0.95,
                    "metadata": {"text": "Cached document 1", "category": "test"}
                },
                {
                    "id": "cached_doc2",
                    "score": 0.85,
                    "metadata": {"text": "Cached document 2", "category": "test"}
                }
            ],
            "query": "test query"
        }
        mock_cache_service.get.return_value = cached_results
        
        response = client.post(
            "/api/search/search",
            json={
                "query": "test query",
                "top_k": 2
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert "matches" in result
        assert len(result["matches"]) == 2
        assert result["matches"][0]["id"] == "cached_doc1"
        assert result["matches"][1]["id"] == "cached_doc2"
        assert result["query"] == "test query"
        
        # Verify cache was checked
        mock_cache_service.get.assert_called_once()
        # Verify embedding was NOT generated
        mock_embedding_service.generate_embedding.assert_not_called()
        # Verify Pinecone was NOT queried
        mock_pinecone_service.query.assert_not_called()