# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core.config import settings
from .core.service import service
from .core.errors import APIError, ErrorResponse
from .api import chat_router, health_router

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan manager for FastAPI app"""
    # Startup
    logger.info("Starting API server...")
    logger.info(f"Model: {settings.model_name}")
    logger.info(f"Device: {settings.device}")
    
    # Initialize the model (optional - can be lazy loaded)
    try:
        await service.initialize()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize model on startup: {e}")
        logger.info("Model will be initialized on first request")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")


# Create FastAPI app
app = FastAPI(
    title="Fast-API",
    description="OpenAI-compatible API for diffusion language model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health_router, tags=["Health"])
app.include_router(chat_router, tags=["Chat"])


@app.exception_handler(APIError)
async def api_exception_handler(request: Request, exc: APIError):
    """Handle API-specific exceptions"""
    logger.error(f"API error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse.from_exception(exc).model_dump()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse.from_generic_exception(exc).model_dump()
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "API Server",
        "version": "1.0.0",
        "model": settings.model_name,
        "status": "running"
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        log_level=settings.log_level.lower(),
        reload=False
    )
