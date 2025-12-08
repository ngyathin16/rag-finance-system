"""
Observability utilities for RAG Finance System.

This package provides:
- OpenTelemetry tracing with Jaeger export
- OpenTelemetry metrics with Prometheus export
- JSONL query logging for analysis

Usage:
    from src.observability import setup_tracing, MetricsCollector
    
    # Initialize tracing at application startup
    tracer = setup_tracing(service_name="rag-finance-system")
    
    # Get metrics collector
    metrics = MetricsCollector()
    
    # Record metrics after each query
    metrics.record_query(
        query="...",
        latency_seconds=2.5,
        tokens_used=1500,
        cost_usd=0.015,
        corrections=1,
        relevance_score=0.85,
        verification_status="VERIFIED"
    )
"""

from src.observability.tracing import (
    setup_tracing,
    get_tracer,
    shutdown_tracing,
    TracingConfigurationError,
)
from src.observability.metrics import (
    MetricsCollector,
    get_metrics_collector,
    MetricsConfigurationError,
)

__all__ = [
    # Tracing
    "setup_tracing",
    "get_tracer",
    "shutdown_tracing",
    "TracingConfigurationError",
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    "MetricsConfigurationError",
]
