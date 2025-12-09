# =============================================================================
# Multi-Stage Dockerfile for RAG Finance System
# =============================================================================
# Stage 1: Builder - Install dependencies
# Stage 2: Runtime - Copy only necessary files
# Features: Non-root user, health check, minimal image size
# =============================================================================

# =============================================================================
# Stage 1: Builder
# =============================================================================
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.12-slim

# Set metadata
LABEL maintainer="RAG Finance System"
LABEL description="Production-grade multi-agent RAG system for financial document Q&A"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Application settings
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    # Vector store mode (chroma for local, pinecone for production)
    VECTOR_STORE_MODE=chroma \
    # Max correction attempts
    MAX_CORRECTIONS=2

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    # For healthcheck
    curl \
    # For ChromaDB
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 -m -s /sbin/nologin appuser

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser scripts/ ./scripts/
COPY --chown=appuser:appuser config/ ./config/

# Create necessary directories with proper permissions
RUN mkdir -p data/chroma_db data/raw data/processed && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

