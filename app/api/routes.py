from fastapi import APIRouter
from app.api.ingest_routes import router as ingest_router
from app.api.search_routes import router as search_router
import logging
api_router = APIRouter()

api_router.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
logging.info("Ingest router included")
api_router.include_router(search_router, prefix="")