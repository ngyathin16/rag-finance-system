"""
OpenTelemetry tracing configuration for RAG Finance System.

This module provides centralized tracing setup with:
- TracerProvider initialization
- OTLP exporter to Jaeger
- BatchSpanProcessor for efficient span export
- LangChain auto-instrumentation for LLM observability

Usage:
    from src.observability.tracing import setup_tracing
    
    tracer = setup_tracing(service_name="rag-finance-system")
    with tracer.start_as_current_span("my_operation") as span:
        span.set_attribute("key", "value")
        # ... your code ...
"""

import logging
import os
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)

# Global flag to prevent double initialization
_tracing_initialized = False


class TracingConfigurationError(Exception):
    """Raised when tracing configuration fails."""
    pass


def setup_tracing(
    service_name: str = "rag-finance-system",
    endpoint: Optional[str] = None
) -> trace.Tracer:
    """
    Initialize and configure OpenTelemetry tracing.
    
    Sets up a complete tracing pipeline with:
    - TracerProvider with service resource attributes
    - OTLP gRPC exporter configured for Jaeger
    - BatchSpanProcessor for efficient batched exports
    - LangChain auto-instrumentation for LLM call tracing
    
    Args:
        service_name: Name identifying this service in traces (default: "rag-finance-system")
        endpoint: OTLP endpoint URL. If not provided, reads from OTEL_EXPORTER_OTLP_ENDPOINT
                 environment variable. Falls back to "http://localhost:4317" for local dev.
    
    Returns:
        Configured Tracer instance for creating spans
    
    Raises:
        TracingConfigurationError: If tracing setup fails critically
    
    Environment Variables:
        OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint (e.g., "http://localhost:4317")
        OTEL_TRACES_EXPORTER: Optional, set to "none" to disable trace export
    
    Example:
        >>> from src.observability.tracing import setup_tracing
        >>> tracer = setup_tracing(service_name="my-service")
        >>> with tracer.start_as_current_span("process_request") as span:
        ...     span.set_attribute("user_id", "123")
        ...     result = process_data()
    """
    global _tracing_initialized
    
    # Prevent double initialization
    if _tracing_initialized:
        logger.debug("Tracing already initialized, returning existing tracer")
        return trace.get_tracer(service_name)
    
    # Resolve OTLP endpoint
    otlp_endpoint = endpoint or os.getenv(
        "OTEL_EXPORTER_OTLP_ENDPOINT",
        "http://localhost:4317"
    )
    
    # Check if tracing is disabled
    traces_exporter = os.getenv("OTEL_TRACES_EXPORTER", "").lower()
    if traces_exporter == "none":
        logger.info("Tracing disabled via OTEL_TRACES_EXPORTER=none")
        _tracing_initialized = True
        return trace.get_tracer(service_name)
    
    logger.info(f"Initializing OpenTelemetry tracing for service '{service_name}'")
    logger.info(f"OTLP endpoint: {otlp_endpoint}")
    
    try:
        # Create resource with service information
        resource = Resource.create({
            SERVICE_NAME: service_name,
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
        })
        
        # Initialize TracerProvider
        provider = TracerProvider(resource=resource)
        
        # Configure OTLP exporter to Jaeger
        try:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True  # Use insecure for local development
            )
            
            # Add BatchSpanProcessor for efficient export
            span_processor = BatchSpanProcessor(
                otlp_exporter,
                max_queue_size=2048,
                max_export_batch_size=512,
                schedule_delay_millis=5000,
            )
            provider.add_span_processor(span_processor)
            
            logger.info("OTLP exporter configured successfully")
            
        except Exception as e:
            logger.warning(
                f"Failed to configure OTLP exporter: {e}. "
                f"Traces will not be exported, but tracing will still work locally."
            )
        
        # Set as global TracerProvider
        trace.set_tracer_provider(provider)
        
        # Auto-instrument LangChain if available
        _instrument_langchain()
        
        _tracing_initialized = True
        logger.info("OpenTelemetry tracing initialized successfully")
        
        return trace.get_tracer(service_name)
        
    except Exception as e:
        logger.error(f"Failed to initialize tracing: {e}")
        raise TracingConfigurationError(f"Tracing setup failed: {e}") from e


def _instrument_langchain() -> None:
    """
    Auto-instrument LangChain for LLM observability.
    
    This provides automatic tracing for:
    - LLM calls (OpenAI, Anthropic, etc.)
    - Chain executions
    - Agent actions
    - Retriever operations
    """
    try:
        from opentelemetry.instrumentation.langchain import LangchainInstrumentor
        
        # Check if already instrumented
        instrumentor = LangchainInstrumentor()
        if not instrumentor.is_instrumented_by_opentelemetry:
            instrumentor.instrument()
            logger.info("LangChain auto-instrumentation enabled")
        else:
            logger.debug("LangChain already instrumented")
            
    except ImportError:
        logger.warning(
            "opentelemetry-instrumentation-langchain not installed. "
            "LangChain auto-instrumentation disabled. "
            "Install with: pip install opentelemetry-instrumentation-langchain"
        )
    except Exception as e:
        logger.warning(f"Failed to instrument LangChain: {e}")


def get_tracer(name: str = __name__) -> trace.Tracer:
    """
    Get a tracer instance for creating spans.
    
    Use this to get a tracer after setup_tracing() has been called.
    If tracing hasn't been initialized, returns a no-op tracer.
    
    Args:
        name: Tracer name, typically __name__ of the calling module
    
    Returns:
        Tracer instance
    
    Example:
        >>> from src.observability.tracing import get_tracer
        >>> tracer = get_tracer(__name__)
        >>> with tracer.start_as_current_span("my_span"):
        ...     do_work()
    """
    return trace.get_tracer(name)


def shutdown_tracing() -> None:
    """
    Gracefully shutdown the tracing provider.
    
    Call this during application shutdown to ensure all pending
    spans are exported before the application exits.
    """
    global _tracing_initialized
    
    provider = trace.get_tracer_provider()
    if hasattr(provider, 'shutdown'):
        try:
            provider.shutdown()
            logger.info("Tracing provider shutdown complete")
        except Exception as e:
            logger.warning(f"Error during tracing shutdown: {e}")
    
    _tracing_initialized = False

