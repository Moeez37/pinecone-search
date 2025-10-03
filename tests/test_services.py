import pytest
from unittest.mock import MagicMock, patch
from app.services.embedding_service import EmbeddingService
from app.services.cache_service import CacheService
from app.services.pinecone_service import PineconeService

# EmbeddingService Tests

@pytest.fixture
def embedding_service(mocker):
    # Mock the langcache OpenAIEmbeddings
    mock_embeddings = MagicMock()
    mocker.patch('app.services.embedding_service.OpenAIEmbeddings', return_value=mock_embeddings)
    mocker.patch('app.services.embedding_service.Cache')
    
    service = EmbeddingService()
    service.embeddings = mock_embeddings
    return service

def test_embedding_service_generate_embedding_with_cache(embedding_service, mocker):
    # Mock the embed_query method
    embedding_service.embeddings.embed_query.return_value = [0.1, 0.2, 0.3]
    
    # Call generate_embedding
    embedding = embedding_service.generate_embedding("test text")
    
    # Verify langcache was used
    embedding_service.embeddings.embed_query.assert_called_once_with("test text")
    
    # Verify result
    assert embedding == [0.1, 0.2, 0.3]

def test_embedding_service_generate_embedding_no_cache(embedding_service, mocker):
    # Set use_cache to False
    embedding_service.use_cache = False
    
    # Mock OpenAI
    mock_openai = mocker.patch('app.services.embedding_service.openai')
    mock_embedding_response = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_openai.Embedding.create.return_value = mock_embedding_response
    
    # Call generate_embedding
    embedding = embedding_service.generate_embedding("test text")
    
    # Verify OpenAI was called directly
    mock_openai.Embedding.create.assert_called_once()
    
    # Verify result
    assert embedding == [0.1, 0.2, 0.3]

def test_embedding_service_generate_batch_embeddings(embedding_service, mocker):
    # Mock the embed_documents method
    embedding_service.embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6]
    ]
    
    # Call generate_batch_embeddings
    embeddings = embedding_service.generate_embeddings(["text1", "text2"])
    
    # Verify langcache batch method was used
    embedding_service.embeddings.embed_documents.assert_called_once_with(["text1", "text2"])
    
    # Verify results
    assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

# CacheService Tests

@pytest.fixture
def cache_service(mocker):
    # Mock the langcache Cache
    mock_cache = MagicMock()
    mocker.patch('app.services.cache_service.Cache', return_value=mock_cache)
    service = CacheService()
    service.client = mock_cache
    return service

def test_cache_service_set_get_delete(cache_service):
    # Test set
    cache_service.set("key", {"data": "value"}, ex=3600)
    cache_service.client.set.assert_called_once_with("key", {"data": "value"}, ttl=3600)

    # Test get
    cache_service.client.get.return_value = {"data": "value"}
    value = cache_service.get("key")
    assert value == {"data": "value"}
    cache_service.client.get.assert_called_once_with("key")

    # Test delete
    cache_service.delete("key")
    cache_service.client.delete.assert_called_once_with("key")

def test_cache_service_connection_error(mocker):
    # Test connection error handling
    mocker.patch('app.services.cache_service.Cache', side_effect=Exception("Connection error"))
    service = CacheService()
    assert service.client is None
    
    # Test methods with no client
    assert service.get("key") is None
    assert service.set("key", "value") is False
    assert service.delete("key") is False

# PineconeService Tests

@pytest.fixture
def pinecone_service(mocker):
    # Mock the Pinecone client
    mock_index = MagicMock()
    mocker.patch('app.services.pinecone_service.pinecone.Index', return_value=mock_index)
    
    # Mock the initialization
    mocker.patch('app.services.pinecone_service.pinecone.init')
    
    service = PineconeService()
    service.index = mock_index
    return service

def test_pinecone_service_init(mocker):
    # Mock pinecone module
    mock_pinecone = mocker.patch('app.services.pinecone_service.pinecone')
    mock_config = mocker.patch('app.services.pinecone_service.config')
    
    # Set config values
    mock_config.PINECONE_API_KEY = "test_api_key"
    mock_config.PINECONE_ENVIRONMENT = "test_env"
    mock_config.PINECONE_INDEX = "test_index"
    
    # Initialize service
    service = PineconeService()
    
    # Verify init was called with correct params
    mock_pinecone.init.assert_called_once_with(
        api_key="test_api_key",
        environment="test_env"
    )
    
    # Verify Index was created
    mock_pinecone.Index.assert_called_once_with("test_index")

def test_pinecone_service_upsert(pinecone_service):
    # Test data
    doc_id = "test_id"
    embedding = [0.1, 0.2, 0.3]
    metadata = {"text": "test document"}
    
    # Call upsert
    pinecone_service.upsert(doc_id, embedding, metadata)
    
    # Verify index.upsert was called correctly
    pinecone_service.index.upsert.assert_called_once_with(
        vectors=[{
            "id": doc_id,
            "values": embedding,
            "metadata": metadata
        }],
        namespace=""
    )

def test_pinecone_service_batch_upsert(pinecone_service):
    # Test data
    batch_data = [
        {"id": "doc1", "values": [0.1, 0.2, 0.3], "metadata": {"text": "document 1"}},
        {"id": "doc2", "values": [0.4, 0.5, 0.6], "metadata": {"text": "document 2"}}
    ]
    
    # Call batch upsert
    pinecone_service.batch_upsert(batch_data)
    
    # Verify index.upsert was called correctly
    pinecone_service.index.upsert.assert_called_once_with(
        vectors=batch_data,
        namespace=""
    )

def test_pinecone_service_query(pinecone_service):
    # Test data
    embedding = [0.1, 0.2, 0.3]
    top_k = 5
    
    # Mock query response
    mock_response = {
        "matches": [
            {"id": "doc1", "score": 0.9, "metadata": {"text": "document 1"}},
            {"id": "doc2", "score": 0.8, "metadata": {"text": "document 2"}}
        ]
    }
    pinecone_service.index.query.return_value = mock_response
    
    # Call query
    result = pinecone_service.query(embedding, top_k)
    
    # Verify index.query was called correctly
    pinecone_service.index.query.assert_called_once_with(
        vector=embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=""
    )
    
    # Verify result
    assert result == mock_response

def test_pinecone_service_delete(pinecone_service):
    # Test data
    doc_id = "test_id"
    
    # Call delete
    pinecone_service.delete(doc_id)
    
    # Verify index.delete was called correctly
    pinecone_service.index.delete.assert_called_once_with(
        ids=[doc_id],
        namespace=""
    )

def test_pinecone_service_initialization(pinecone_service):
    assert pinecone_service is not None
