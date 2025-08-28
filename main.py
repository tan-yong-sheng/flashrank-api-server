#!/usr/bin/env python3
"""
FastAPI Cohere-Compatible Reranker using FlashRank

A high-performance, CPU-based reranking API that provides Cohere-compatible
endpoints while leveraging FlashRank's ultra-fast reranking models.
"""

import os
import logging
import uuid
from typing import List, Optional, Dict, Any, Set
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import uvicorn

try:
    from flashrank import Ranker, RerankRequest as FlashRankRequest
except ImportError:
    raise ImportError(
        "FlashRank not installed. Please install with: pip install flashrank"
    )

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global ranker instance (initialized on startup)
ranker_instance: Optional[Ranker] = None

# Available FlashRank models for reference
ALL_FLASHRANK_MODELS = {
    "ms-marco-TinyBERT-L-2-v2",
    "ms-marco-MiniLM-L-12-v2", 
    "rank-T5-flan",
    "ms-marco-MultiBERT-L-12",
    "ce-esci-MiniLM-L12-v2",
    "rank_zephyr_7b_v1_full",
    "miniReranker_arabic_v1"
}

# Models configured to be downloaded/available locally
CONFIGURED_MODELS: Set[str] = set()
DOWNLOADED_MODELS: Set[str] = set()
RANKER_CACHE: Dict[str, Ranker] = {}

# Configuration
class Config:
    @staticmethod
    def load_models() -> Set[str]:
        """Load configured models from environment"""
        models_env = os.getenv("FLASHRANK_MODELS", "ms-marco-TinyBERT-L-2-v2")
        models = {model.strip() for model in models_env.split(",") if model.strip()}
        
        # Validate models exist in our supported list
        invalid_models = models - ALL_FLASHRANK_MODELS
        if invalid_models:
            logger.warning(f"Invalid models in FLASHRANK_MODELS: {invalid_models}")
            models = models - invalid_models
        
        if not models:
            logger.warning("No valid models configured, using default")
            models = {"ms-marco-TinyBERT-L-2-v2"}
        
        return models
    
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "ms-marco-TinyBERT-L-2-v2")
    MAX_DOCUMENTS = int(os.getenv("MAX_DOCUMENTS", "1000"))
    DEFAULT_TOP_N = int(os.getenv("DEFAULT_TOP_N", "100"))
    MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))
    CACHE_DIR = os.getenv("FLASHRANK_CACHE_DIR", None)


# Pydantic models for request/response
class Document(BaseModel):
    text: str = Field(..., description="The document text to rerank")

class RerankRequest(BaseModel):
    model: str = Field(
        default=Config.DEFAULT_MODEL,
        description="The FlashRank model to use"
    )
    query: str = Field(..., description="The search query")
    documents: List[str] = Field(
        ..., 
        min_items=1,
        max_items=Config.MAX_DOCUMENTS,
        description="List of document texts to rerank"
    )
    top_n: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of documents to return"
    )
    max_tokens_per_doc: Optional[int] = Field(
        default=Config.MAX_LENGTH,
        ge=1,
        le=8192,
        description="Maximum tokens per document"
    )

    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        if v not in DOWNLOADED_MODELS:
            available_list = list(DOWNLOADED_MODELS) if DOWNLOADED_MODELS else list(CONFIGURED_MODELS)
            raise ValueError(
                f"Model '{v}' not available locally. Downloaded models: {available_list}"
            )
        return v

    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        if not v:
            raise ValueError("At least one document is required")
        return v


# V1 API Models (simpler response format)
class RerankV1Result(BaseModel):
    index: int = Field(..., description="Original index of the document")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")


class RerankV1Response(BaseModel):
    id: str = Field(..., description="Request ID")
    results: List[RerankV1Result] = Field(..., description="List of ranked documents")
    meta: Dict[str, Any] = Field(..., description="Metadata about the request")


# V2 API Models (with document wrapper)
class RerankResult(BaseModel):
    index: int = Field(..., description="Original index of the document")
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    document: Document = Field(..., description="The document object")


class RerankResponse(BaseModel):
    results: List[RerankResult] = Field(..., description="List of ranked documents")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = False
    model_name: Optional[str] = None


def download_and_test_models():
    """Download and test configured models"""
    global CONFIGURED_MODELS, DOWNLOADED_MODELS, RANKER_CACHE
    
    CONFIGURED_MODELS = Config.load_models()
    logger.info(f"Configured models: {list(CONFIGURED_MODELS)}")
    
    for model_name in CONFIGURED_MODELS:
        try:
            logger.info(f"Downloading and testing model: {model_name}")
            ranker = Ranker(
                model_name=model_name,
                max_length=Config.MAX_LENGTH,
                cache_dir=Config.CACHE_DIR
            )
            
            # Test the model with a simple query
            test_request = FlashRankRequest(
                query="test",
                passages=[{"id": 0, "text": "test document"}]
            )
            ranker.rerank(test_request)
            
            # If we get here, model works
            RANKER_CACHE[model_name] = ranker
            DOWNLOADED_MODELS.add(model_name)
            logger.info(f"Model {model_name} downloaded and verified successfully")
            
        except Exception as e:
            logger.error(f"Failed to download/test model {model_name}: {e}")
            continue
    
    if not DOWNLOADED_MODELS:
        logger.error("No models successfully downloaded!")
    else:
        logger.info(f"Successfully downloaded models: {list(DOWNLOADED_MODELS)}")


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup resources"""
    global CONFIGURED_MODELS, DOWNLOADED_MODELS, RANKER_CACHE
    
    # Startup
    logger.info("Initializing FlashRank...")
    download_and_test_models()
    
    if not DOWNLOADED_MODELS:
        logger.warning("No models available - API will return errors for rerank requests")
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    RANKER_CACHE.clear()
    DOWNLOADED_MODELS.clear()


# FastAPI app
app = FastAPI(
    title="FlashRank Cohere-Compatible Reranker",
    description="Ultra-fast CPU-based document reranking API compatible with Cohere's rerank endpoint",
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


def get_or_create_ranker(model_name: str) -> Ranker:
    """Get existing ranker or create a new one for the specified model"""
    global RANKER_CACHE, DOWNLOADED_MODELS
    
    # Check if model is available locally
    if model_name not in DOWNLOADED_MODELS:
        available_list = list(DOWNLOADED_MODELS) if DOWNLOADED_MODELS else list(CONFIGURED_MODELS)
        logger.warning(f"Model '{model_name}' not available locally. Available: {available_list}")
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model_name}' not available locally. Available models: {available_list}"
        )
    
    # Return cached ranker if available
    if model_name in RANKER_CACHE:
        return RANKER_CACHE[model_name]
    
    # Create new ranker for the requested model
    try:
        logger.info(f"Loading model: {model_name}")
        ranker = Ranker(
            model_name=model_name,
            max_length=Config.MAX_LENGTH,
            cache_dir=Config.CACHE_DIR
        )
        # Cache the ranker
        RANKER_CACHE[model_name] = ranker
        logger.info(f"Model {model_name} loaded and cached successfully")
        return ranker
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        # Remove from downloaded models if loading failed
        DOWNLOADED_MODELS.discard(model_name)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load reranking model '{model_name}': {str(e)}"
        )


def truncate_text(text: str, max_tokens: int) -> str:
    """Simple text truncation based on word count (approximation)"""
    if max_tokens <= 0:
        return text
    
    # Rough approximation: 1 token ≈ 0.75 words
    max_words = int(max_tokens * 0.75)
    words = text.split()
    
    if len(words) <= max_words:
        return text
    
    return " ".join(words[:max_words])


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=len(DOWNLOADED_MODELS) > 0,
        model_name=Config.DEFAULT_MODEL if Config.DEFAULT_MODEL in DOWNLOADED_MODELS else None
    )


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "service": "FlashRank Cohere-Compatible Reranker",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "rerank_v1": "/v1/rerank",
            "rerank_v2": "/v2/rerank",
            "health": "/health",
            "docs": "/docs"
        },
        "configured_models": list(CONFIGURED_MODELS),
        "downloaded_models": list(DOWNLOADED_MODELS)
    }


@app.post("/v1/rerank", response_model=RerankV1Response)
async def rerank_documents_v1(request: RerankRequest):
    """
    Rerank documents based on relevance to a query - V1 API format.
    
    Compatible with Cohere's v1 rerank API specification.
    Returns simpler response format without document wrapper.
    """
    try:
        logger.info(f"V1 Rerank request: model={request.model}, query_len={len(request.query)}, docs={len(request.documents)}")
        
        # Get or create ranker for the requested model
        ranker = get_or_create_ranker(request.model)
        
        # Prepare documents for FlashRank
        max_tokens = request.max_tokens_per_doc or Config.MAX_LENGTH
        passages = []
        
        for i, doc_text in enumerate(request.documents):
            # Truncate document if needed
            truncated_text = truncate_text(doc_text, max_tokens)
            
            passages.append({
                "id": i,
                "text": truncated_text,
                "meta": {"original_index": i}
            })
        
        # Create FlashRank request
        flashrank_request = FlashRankRequest(query=request.query, passages=passages)
        
        # Perform reranking
        try:
            results = ranker.rerank(flashrank_request)
        except Exception as e:
            logger.error(f"FlashRank reranking failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Reranking failed: {str(e)}"
            )
        
        # Convert results to Cohere V1 format (simpler format)
        rerank_results = []
        for result in results:
            original_index = result.get("meta", {}).get("original_index", result["id"])
            
            rerank_results.append(RerankV1Result(
                index=original_index,
                relevance_score=float(result["score"])
            ))
        
        # Apply top_n filtering if specified
        if request.top_n:
            rerank_results = rerank_results[:request.top_n]
        
        # Generate request ID and metadata
        request_id = str(uuid.uuid4())
        meta = {
            "api_version": {"version": "1"},
            "billed_units": {"search_units": 1}
        }
        
        logger.info(f"V1 Reranking completed: returned {len(rerank_results)} results")
        
        return RerankV1Response(
            id=request_id,
            results=rerank_results,
            meta=meta
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in v1 rerank endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/v2/rerank", response_model=RerankResponse)
async def rerank_documents(request: RerankRequest):
    """
    Rerank documents based on relevance to a query.
    
    Compatible with Cohere's rerank API specification.
    """
    try:
        logger.info(f"Rerank request: model={request.model}, query_len={len(request.query)}, docs={len(request.documents)}")
        
        # Get or create ranker for the requested model
        ranker = get_or_create_ranker(request.model)
        
        # Prepare documents for FlashRank
        max_tokens = request.max_tokens_per_doc or Config.MAX_LENGTH
        passages = []
        
        for i, doc_text in enumerate(request.documents):
            # Truncate document if needed
            truncated_text = truncate_text(doc_text, max_tokens)
            
            passages.append({
                "id": i,
                "text": truncated_text,
                "meta": {"original_index": i}
            })
        
        # Create FlashRank request
        flashrank_request = FlashRankRequest(query=request.query, passages=passages)
        
        # Perform reranking
        try:
            results = ranker.rerank(flashrank_request)
        except Exception as e:
            logger.error(f"FlashRank reranking failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Reranking failed: {str(e)}"
            )
        
        # Convert results to Cohere format
        rerank_results = []
        for result in results:
            original_index = result.get("meta", {}).get("original_index", result["id"])
            original_text = request.documents[original_index]
            
            rerank_results.append(RerankResult(
                index=original_index,
                relevance_score=float(result["score"]),
                document=Document(text=original_text)
            ))
        
        # Apply top_n filtering if specified
        if request.top_n:
            rerank_results = rerank_results[:request.top_n]
        
        logger.info(f"Reranking completed: returned {len(rerank_results)} results")
        
        return RerankResponse(results=rerank_results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in rerank endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/v2/models")
@app.get("/v1/models")
async def list_models():
    """List locally downloaded reranking models"""
    if not DOWNLOADED_MODELS:
        return {
            "models": [],
            "message": "No models downloaded locally",
            "configured_models": list(CONFIGURED_MODELS),
            "default": Config.DEFAULT_MODEL
        }
    
    return {
        "models": [
            {
                "name": model_name,
                "description": f"FlashRank {model_name} model (downloaded locally)",
                "status": "ready"
            }
            for model_name in sorted(DOWNLOADED_MODELS)
        ],
        "default": Config.DEFAULT_MODEL if Config.DEFAULT_MODEL in DOWNLOADED_MODELS else None,
        "total_downloaded": len(DOWNLOADED_MODELS)
    }


if __name__ == "__main__":
    # Configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")
    
    logger.info(f"Starting FlashRank Cohere-Compatible Reranker on {host}:{port}")
    logger.info(f"Default model: {Config.DEFAULT_MODEL}")
    logger.info(f"Max documents: {Config.MAX_DOCUMENTS}")
    logger.info(f"Max length: {Config.MAX_LENGTH}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False
    )