from fastapi import APIRouter, Depends, HTTPException
from app.models import IngestRequest, BatchIngestRequest, StatusResponse
from app.services import get_embedding_service, get_pinecone_service, get_generate_text_service
from app.services.embedding_service import EmbeddingService
from app.services.pinecone_service import PineconeService
from app.services.generate_text_service import GenerateTextService
from app.utils.constants import OBJECT_TYPE
from app.utils.common import sanitize_metadata
from app.utils.logger import get_logger
# Use threading for parallel processing
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
        
router = APIRouter()
logger = get_logger(__name__)


@router.post("/ingest", response_model=StatusResponse)
async def ingest_document(
    request: IngestRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    generate_text_service: GenerateTextService = Depends(get_generate_text_service)
):
    try:
        metadata = request.metadata
        metadata = metadata.dict()
         # Generate text for the document
        text = generate_text_service.generate_text(metadata, request.type)
        
        # Generate embedding for the document
        embedding = embedding_service.generate_embedding(text)
        
        # Prepare vector for Pinecone
        vector = {
            "id": str(metadata.get("id",None) or metadata.get("databaseId",None) or metadata.get("zip",None)),
            "values": embedding,
            "metadata": sanitize_metadata(metadata)      
        }
        namespace = request.type if request.type != OBJECT_TYPE.PRODUCTS else f"{request.location_id or 'orphan'}-{request.type}"
        # Upsert to Pinecone
        # pinecone_service.upsert(vectors=[vector],namespace=namespace)
        logger.info("Upserting vector", namespace=namespace)
        logger.info("Successfully ingested document")
        return StatusResponse(status="success", message=f"{request.type} ingested successfully")
    except Exception as e:
        logger.error("Error ingesting document", type=request.type, error=str(e))       
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch-ingest", response_model=StatusResponse)
async def batch_ingest_documents(
    request: BatchIngestRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    pinecone_service: PineconeService = Depends(get_pinecone_service),
    generate_text_service: GenerateTextService = Depends(get_generate_text_service)
):
    try:
        metadata_list = request.metadata
        logger.info("Starting background batch ingestion", count=len(metadata_list), type=request.type)
        
        # Define batch size for Pinecone upserts
        batch_size = 100
        namespace = request.type if request.type != OBJECT_TYPE.PRODUCTS else f"{request.location_id or 'orphan'}-{request.type}"
        
        # Function to process the entire batch in background
        def process_batch_in_background():
            try:
                # Create a queue for vectors
                vector_queue = queue.Queue()
                
                # Function to process a single metadata item
                def process_metadata(metadata):
                    try:
                        # Generate text for the document
                        metadata = metadata.dict()
                        text = generate_text_service.generate_text(metadata, request.type)
                        logger.debug("Generated text for document", id=metadata.get("id",None) or metadata.get("databaseId",None) or metadata.get("zip",None))
                        # Generate embedding for the document
                        embedding = embedding_service.generate_embedding(text)
                        
                        # Prepare vector for Pinecone
                        vector = {
                            "id": str(metadata.get("id",None) or metadata.get("databaseId",None) or metadata.get("zip",None)),
                            "values": embedding,
                            "metadata": sanitize_metadata(metadata)
                        }
                        # Add to queue
                        vector_queue.put(vector)
                        logger.info("Processed document", id=metadata.get("id",None) or metadata.get("databaseId",None) or metadata.get("zip",None))
                    except Exception as e:
                        logger.error("Error processing document", error=str(e))
                
                # Function to batch upsert vectors to Pinecone
                def batch_upsert():
                    vectors_to_upsert = []
                    total_processed = 0
                    
                    while total_processed < len(metadata_list):
                        try:
                            # Get vector from queue with timeout
                            vector = vector_queue.get(timeout=1)
                            vectors_to_upsert.append(vector)
                            total_processed += 1
                            
                            # If we have a full batch or this is the last item, upsert to Pinecone
                            if len(vectors_to_upsert) >= batch_size or total_processed == len(metadata_list):
                                if vectors_to_upsert:
                                    logger.info("Upserting batch to Pinecone", count=len(vectors_to_upsert), namespace=namespace)
                                    pinecone_service.upsert(vectors=vectors_to_upsert, namespace=namespace)
                                    vectors_to_upsert = []
                        except queue.Empty:
                            # If queue is empty but we haven't processed all items, continue waiting
                            if total_processed < len(metadata_list):
                                continue
                            break
                
                # Start the batch upsert thread
                upsert_thread = threading.Thread(target=batch_upsert)
                upsert_thread.daemon = True  # Allow the program to exit even if this thread is running
                upsert_thread.start()
                
                # Process metadata items in parallel using ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=10) as executor:
                    executor.map(process_metadata, metadata_list)
                
                # Wait for the upsert thread to complete
                upsert_thread.join()
                
                logger.info("Successfully ingested documents", count=len(metadata_list), type=request.type)
            except Exception as e:
                logger.error("Error in background batch ingestion", error=str(e))
        
        # Start the background processing thread
        background_thread = threading.Thread(target=process_batch_in_background)
        background_thread.daemon = True  # Allow the program to exit even if this thread is running
        background_thread.start()
        
        # Immediately return success response
        return StatusResponse(
            status="success", 
            message=f"Batch ingestion of {len(metadata_list)} {request.type} documents started in background"
        )
    except Exception as e:
        logger.error("Error setting up batch ingestion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))