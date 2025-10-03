from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api import api_router
from app.utils.logger import get_logger
import logging
import sys
from typing import Any

logger = get_logger(__name__)


# Lifespan context manager (replaces @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Application starting up")
    yield
    # Shutdown logic
    logger.info("Application shutting down")


# Create FastAPI app with lifespan
app = FastAPI(
    title="Pinecone Search API",
    lifespan=lifespan
)

# Include API router
# (Optional) CORS Middleware – keep if you need cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An unexpected error occurred"}
    )

# Root endpoint

app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    logger.info("Health check endpoint called")
    return {"status": "server running"}
