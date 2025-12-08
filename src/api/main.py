"""
FastAPI Application for RAG Finance System.

This module provides the REST API interface for the self-correcting RAG system,
including query endpoints, health checks, and metrics aggregation.

Features:
    - POST /query: Main query endpoint for financial questions
    - GET /health: Health check endpoint
    - GET /metrics: Aggregated metrics endpoint
    - GET /docs: Auto-generated Swagger UI documentation

Usage:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os
import time
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    query: str = Field(
        min_length=5,
        max_length=500,
        description="The financial question to ask",
        examples=["What was the company's total revenue in Q4 2023?"]
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier for tracking"
    )
    include_sources: bool = Field(
        default=True,
        description="Whether to include source documents in response"
    )


class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    answer: str = Field(description="The generated answer with citations")
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the answer"
    )
    verification_status: str = Field(
        description="Verification status: VERIFIED, UNCERTAIN, or FALSE"
    )
    sources: Optional[List[dict]] = Field(
        default=None,
        description="Source documents used (if requested)"
    )
    latency_ms: float = Field(description="Total processing time in milliseconds")
    tokens_used: int = Field(description="Total tokens consumed")
    cost_usd: float = Field(description="Estimated cost in USD")
    corrections_made: int = Field(description="Number of correction attempts")


class HealthResponse(BaseModel):
    """Response model for the /health endpoint."""
    status: str
    timestamp: str
    service: str
    version: str
    vector_store_status: str
    document_count: int


class MetricsResponse(BaseModel):
    """Response model for the /metrics endpoint."""
    total_queries: int
    total_tokens: int
    total_cost_usd: float
    total_corrections: int
    average_latency_ms: float
    queries_by_status: Dict[str, int]
    uptime_seconds: float


# =============================================================================
# Cost Tracker (Singleton)
# =============================================================================

class CostTracker:
    """
    Singleton class for tracking API costs and usage metrics.
    
    Provides aggregated cost tracking across all API calls.
    """
    
    _instance: Optional["CostTracker"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "CostTracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self._lock = threading.Lock()
        self.total_queries = 0
        self.total_tokens = 0
        self.total_cost_usd = 0.0
        self.total_corrections = 0
        self.latencies_ms: List[float] = []
        self.queries_by_status: Dict[str, int] = {}
        self.start_time = time.time()
        
        self._initialized = True
        logger.info("CostTracker initialized")
    
    def record_query(
        self,
        tokens: int,
        cost_usd: float,
        corrections: int,
        latency_ms: float,
        verification_status: str
    ) -> None:
        """Record metrics for a completed query."""
        with self._lock:
            self.total_queries += 1
            self.total_tokens += tokens
            self.total_cost_usd += cost_usd
            self.total_corrections += corrections
            self.latencies_ms.append(latency_ms)
            
            # Track queries by verification status
            status = verification_status or "UNKNOWN"
            self.queries_by_status[status] = self.queries_by_status.get(status, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics."""
        with self._lock:
            avg_latency = (
                sum(self.latencies_ms) / len(self.latencies_ms)
                if self.latencies_ms else 0.0
            )
            
            return {
                "total_queries": self.total_queries,
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "total_corrections": self.total_corrections,
                "average_latency_ms": round(avg_latency, 2),
                "queries_by_status": dict(self.queries_by_status),
                "uptime_seconds": round(time.time() - self.start_time, 2)
            }
    
    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        with cls._lock:
            cls._instance = None


# =============================================================================
# Singleton Instances
# =============================================================================

# Lazy-loaded singletons
_vector_store = None
_rag_system = None
_metrics_collector = None
_cost_tracker = None


def get_vector_store_instance():
    """Get or create the vector store singleton."""
    global _vector_store
    if _vector_store is None:
        from src.vector_store import get_vector_store
        mode = os.getenv("VECTOR_STORE_MODE", "chroma")
        _vector_store = get_vector_store(mode=mode)
        logger.info(f"Vector store initialized (mode={mode})")
    return _vector_store


def get_rag_system_instance():
    """Get or create the RAG system singleton."""
    global _rag_system
    if _rag_system is None:
        from src.orchestrator import SelfCorrectingRAG
        vector_store = get_vector_store_instance()
        max_corrections = int(os.getenv("MAX_CORRECTIONS", "2"))
        _rag_system = SelfCorrectingRAG(
            vector_store=vector_store,
            max_corrections=max_corrections
        )
        logger.info(f"SelfCorrectingRAG initialized (max_corrections={max_corrections})")
    return _rag_system


def get_metrics_collector_instance():
    """Get or create the metrics collector singleton."""
    global _metrics_collector
    if _metrics_collector is None:
        from src.observability.metrics import get_metrics_collector
        _metrics_collector = get_metrics_collector(service_name="rag-finance-api")
        logger.info("MetricsCollector initialized")
    return _metrics_collector


def get_cost_tracker_instance() -> CostTracker:
    """Get or create the cost tracker singleton."""
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# =============================================================================
# Application Lifecycle
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup/shutdown events.
    
    Startup:
        - Initialize vector store
        - Initialize RAG system
        - Initialize metrics collector
        - Initialize cost tracker
    
    Shutdown:
        - Log shutdown message
        - Cleanup resources
    """
    # Startup
    logger.info("=" * 60)
    logger.info("RAG Finance API starting up...")
    logger.info("=" * 60)
    
    try:
        # Initialize singletons (lazy loading on first request is also fine)
        # Pre-initialize for faster first request
        get_vector_store_instance()
        get_rag_system_instance()
        get_metrics_collector_instance()
        get_cost_tracker_instance()
        
        logger.info("All services initialized successfully")
        logger.info("API is ready to accept requests")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("=" * 60)
    logger.info("RAG Finance API shutting down...")
    logger.info("=" * 60)
    
    # Log final metrics
    cost_tracker = get_cost_tracker_instance()
    metrics = cost_tracker.get_metrics()
    logger.info(f"Final metrics: {metrics}")
    
    logger.info("Cleanup complete. Goodbye!")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title="RAG Finance API",
    description=(
        "Self-correcting Retrieval-Augmented Generation API for financial "
        "document question answering. Features fact-checking, automatic "
        "correction, and comprehensive observability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# =============================================================================
# Middleware
# =============================================================================

# CORS Middleware - Allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """
    Middleware to add a unique request ID to each request.
    
    Adds X-Request-ID header to response for request tracking.
    """
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """
    Middleware for request/response logging.
    
    Logs:
        - Incoming request method, path, and request ID
        - Response status code and processing time
    """
    request_id = getattr(request.state, "request_id", "unknown")
    start_time = time.perf_counter()
    
    # Log incoming request
    logger.info(
        f"[{request_id}] {request.method} {request.url.path} - Started"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate processing time
        process_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Log response
        logger.info(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {process_time_ms:.2f}ms"
        )
        
        # Add processing time header
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
        
        return response
        
    except Exception as e:
        process_time_ms = (time.perf_counter() - start_time) * 1000
        logger.error(
            f"[{request_id}] {request.method} {request.url.path} - "
            f"Error: {str(e)} - Duration: {process_time_ms:.2f}ms"
        )
        raise


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of the API and its dependencies.
    """
    try:
        vector_store = get_vector_store_instance()
        stats = vector_store.get_collection_stats()
        vector_store_status = "healthy"
        document_count = stats.get("count", 0)
    except Exception as e:
        logger.warning(f"Vector store health check failed: {e}")
        vector_store_status = f"unhealthy: {str(e)}"
        document_count = 0
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        service="rag-finance-api",
        version="1.0.0",
        vector_store_status=vector_store_status,
        document_count=document_count
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["System"])
async def get_metrics():
    """
    Aggregated metrics endpoint.
    
    Returns cumulative metrics for all processed queries.
    """
    cost_tracker = get_cost_tracker_instance()
    metrics = cost_tracker.get_metrics()
    
    return MetricsResponse(**metrics)


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_rag(request: QueryRequest, http_request: Request):
    """
    Main query endpoint for the RAG system.
    
    Processes a financial question through the self-correcting RAG pipeline:
    1. Retrieves relevant documents from the vector store
    2. Filters documents by relevance
    3. Generates an answer with citations
    4. Fact-checks the answer and corrects if needed
    
    Args:
        request: QueryRequest with the question and options
        http_request: FastAPI request object for accessing request metadata
    
    Returns:
        QueryResponse with the answer, metrics, and optional sources
    
    Raises:
        HTTPException 400: Invalid request (query too short/long)
        HTTPException 500: Internal processing error
    """
    request_id = getattr(http_request.state, "request_id", "unknown")
    
    logger.info(
        f"[{request_id}] Query received: {request.query[:100]}... "
        f"(user_id={request.user_id}, include_sources={request.include_sources})"
    )
    
    try:
        # Get the RAG system
        rag_system = get_rag_system_instance()
        
        # Execute query
        start_time = time.perf_counter()
        result = rag_system.query(request.query)
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Format sources if requested
        sources = None
        if request.include_sources and result.get("sources"):
            sources = []
            for content, score, reasoning in result["sources"]:
                # Truncate content to 200 characters
                truncated_content = (
                    content[:200] + "..." if len(content) > 200 else content
                )
                sources.append({
                    "content": truncated_content,
                    "relevance_score": round(score, 4),
                    "reasoning": reasoning[:100] if reasoning else None
                })
        
        # Record metrics
        metrics_collector = get_metrics_collector_instance()
        cost_tracker = get_cost_tracker_instance()
        
        cost_tracker.record_query(
            tokens=result.get("total_tokens", 0),
            cost_usd=result.get("total_cost_usd", 0.0),
            corrections=result.get("corrections_made", 0),
            latency_ms=latency_ms,
            verification_status=result.get("verification_status", "UNKNOWN")
        )
        
        logger.info(
            f"[{request_id}] Query completed: "
            f"status={result.get('verification_status')}, "
            f"confidence={result.get('confidence', 0):.2f}, "
            f"corrections={result.get('corrections_made', 0)}, "
            f"latency={latency_ms:.2f}ms"
        )
        
        return QueryResponse(
            answer=result.get("answer", ""),
            confidence=result.get("confidence", 0.0),
            verification_status=result.get("verification_status", "UNKNOWN"),
            sources=sources,
            latency_ms=round(latency_ms, 2),
            tokens_used=result.get("total_tokens", 0),
            cost_usd=round(result.get("total_cost_usd", 0.0), 6),
            corrections_made=result.get("corrections_made", 0)
        )
        
    except ValueError as e:
        # Validation errors
        logger.warning(f"[{request_id}] Validation error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request: {str(e)}"
        )
    
    except Exception as e:
        # Internal errors
        logger.error(
            f"[{request_id}] Internal error processing query: {e}",
            exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port} (reload={reload})")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )

