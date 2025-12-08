"""
OpenTelemetry metrics collection for RAG Finance System.

This module provides centralized metrics collection with:
- Query counters (total queries, tokens, costs, corrections)
- Latency and relevance histograms
- JSONL logging for offline analysis

Usage:
    from src.observability.metrics import MetricsCollector
    
    metrics = MetricsCollector()
    metrics.record_query(
        query="What was the revenue?",
        latency_seconds=2.5,
        tokens_used=1500,
        cost_usd=0.015,
        corrections=1,
        relevance_score=0.85,
        verification_status="VERIFIED"
    )
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_LOG_PATH = Path("data/query_logs.jsonl")


class MetricsConfigurationError(Exception):
    """Raised when metrics configuration fails."""
    pass


class MetricsCollector:
    """
    OpenTelemetry metrics collector for RAG system observability.
    
    Collects and exports metrics for:
    - Query volume and patterns
    - Token usage and costs
    - Response latency
    - Relevance score distribution
    - Correction frequency
    
    Also writes detailed query logs to JSONL for offline analysis.
    
    Attributes:
        meter: OpenTelemetry meter for creating instruments
        log_path: Path to JSONL log file
    
    Example:
        >>> collector = MetricsCollector(service_name="rag-finance")
        >>> collector.record_query(
        ...     query="What was Q4 revenue?",
        ...     latency_seconds=3.2,
        ...     tokens_used=2000,
        ...     cost_usd=0.02,
        ...     corrections=0,
        ...     relevance_score=0.92,
        ...     verification_status="VERIFIED"
        ... )
    """
    
    _instance: Optional["MetricsCollector"] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs) -> "MetricsCollector":
        """Singleton pattern to ensure single metrics collector instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        service_name: str = "rag-finance-system",
        log_path: Optional[Path] = None,
        endpoint: Optional[str] = None
    ):
        """
        Initialize the metrics collector.
        
        Args:
            service_name: Service name for metric labels
            log_path: Path to JSONL log file (default: data/query_logs.jsonl)
            endpoint: OTLP metrics endpoint. If not provided, reads from
                     OTEL_EXPORTER_OTLP_ENDPOINT environment variable.
        
        Environment Variables:
            OTEL_EXPORTER_OTLP_ENDPOINT: OTLP collector endpoint
            OTEL_METRICS_EXPORTER: Set to "none" to disable metric export
        """
        # Prevent re-initialization
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        self.service_name = service_name
        self.log_path = log_path or DEFAULT_LOG_PATH
        self._file_lock = threading.Lock()
        
        # Ensure log directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenTelemetry metrics
        self._setup_metrics(endpoint)
        
        # Create metric instruments
        self._create_instruments()
        
        self._initialized = True
        logger.info(f"MetricsCollector initialized for service '{service_name}'")
        logger.info(f"Query logs will be written to: {self.log_path}")
    
    def _setup_metrics(self, endpoint: Optional[str] = None) -> None:
        """Set up OpenTelemetry metrics provider and exporter."""
        otlp_endpoint = endpoint or os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "http://localhost:4317"
        )
        
        # Check if metrics export is disabled
        metrics_exporter = os.getenv("OTEL_METRICS_EXPORTER", "").lower()
        if metrics_exporter == "none":
            logger.info("Metrics export disabled via OTEL_METRICS_EXPORTER=none")
            self.meter = metrics.get_meter(self.service_name)
            return
        
        try:
            # Create resource with service information
            resource = Resource.create({
                SERVICE_NAME: self.service_name,
                "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
                "deployment.environment": os.getenv("DEPLOYMENT_ENV", "development"),
            })
            
            # Configure OTLP exporter
            try:
                otlp_exporter = OTLPMetricExporter(
                    endpoint=otlp_endpoint,
                    insecure=True
                )
                
                metric_reader = PeriodicExportingMetricReader(
                    otlp_exporter,
                    export_interval_millis=30000,  # Export every 30 seconds
                )
                
                provider = MeterProvider(
                    resource=resource,
                    metric_readers=[metric_reader]
                )
                
                logger.info(f"OTLP metrics exporter configured: {otlp_endpoint}")
                
            except Exception as e:
                logger.warning(
                    f"Failed to configure OTLP metrics exporter: {e}. "
                    f"Metrics will be collected but not exported."
                )
                provider = MeterProvider(resource=resource)
            
            # Set as global MeterProvider
            metrics.set_meter_provider(provider)
            self.meter = metrics.get_meter(self.service_name)
            
        except Exception as e:
            logger.warning(f"Failed to initialize metrics provider: {e}")
            self.meter = metrics.get_meter(self.service_name)
    
    def _create_instruments(self) -> None:
        """Create OpenTelemetry metric instruments."""
        # Counters
        self.queries_counter = self.meter.create_counter(
            name="rag_queries_total",
            description="Total number of RAG queries processed",
            unit="1"
        )
        
        self.tokens_counter = self.meter.create_counter(
            name="tokens_used_total",
            description="Total number of tokens used across all LLM calls",
            unit="1"
        )
        
        self.cost_counter = self.meter.create_counter(
            name="cost_usd_total",
            description="Total cost in USD for LLM API calls",
            unit="USD"
        )
        
        self.corrections_counter = self.meter.create_counter(
            name="corrections_total",
            description="Total number of answer corrections performed",
            unit="1"
        )
        
        # Histograms
        self.latency_histogram = self.meter.create_histogram(
            name="query_latency_seconds",
            description="Query processing latency in seconds",
            unit="s"
        )
        
        self.relevance_histogram = self.meter.create_histogram(
            name="relevance_score_distribution",
            description="Distribution of document relevance scores",
            unit="1"
        )
        
        logger.debug("Metric instruments created")
    
    def record_query(
        self,
        query: str,
        latency_seconds: float,
        tokens_used: int,
        cost_usd: float,
        corrections: int,
        relevance_score: float,
        verification_status: str,
        answer: Optional[str] = None,
        source_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Record metrics for a completed RAG query.
        
        This method:
        1. Updates OpenTelemetry counters and histograms
        2. Writes detailed log entry to JSONL file
        
        Args:
            query: The user's query text
            latency_seconds: Total query processing time in seconds
            tokens_used: Total tokens consumed by LLM calls
            cost_usd: Estimated cost in USD
            corrections: Number of correction attempts made
            relevance_score: Average or max relevance score of retrieved docs
            verification_status: Final verification status (VERIFIED/UNCERTAIN/FALSE)
            answer: Optional generated answer (truncated in logs)
            source_count: Number of source documents used
            metadata: Optional additional metadata to include in logs
        
        Example:
            >>> metrics.record_query(
            ...     query="What was Q4 revenue?",
            ...     latency_seconds=2.8,
            ...     tokens_used=1800,
            ...     cost_usd=0.018,
            ...     corrections=1,
            ...     relevance_score=0.88,
            ...     verification_status="VERIFIED",
            ...     answer="The Q4 revenue was $4.2B...",
            ...     source_count=5
            ... )
        """
        # Common attributes for all metrics
        attributes = {
            "verification_status": verification_status,
            "had_corrections": str(corrections > 0).lower(),
        }
        
        # Update counters
        self.queries_counter.add(1, attributes)
        self.tokens_counter.add(tokens_used, attributes)
        self.cost_counter.add(cost_usd, attributes)
        
        if corrections > 0:
            self.corrections_counter.add(corrections, attributes)
        
        # Record histograms
        self.latency_histogram.record(latency_seconds, attributes)
        self.relevance_histogram.record(relevance_score, attributes)
        
        # Write to JSONL log
        self._write_log_entry(
            query=query,
            latency_seconds=latency_seconds,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
            corrections=corrections,
            relevance_score=relevance_score,
            verification_status=verification_status,
            answer=answer,
            source_count=source_count,
            metadata=metadata
        )
        
        logger.debug(
            f"Recorded metrics: latency={latency_seconds:.2f}s, "
            f"tokens={tokens_used}, cost=${cost_usd:.4f}, "
            f"corrections={corrections}, status={verification_status}"
        )
    
    def _write_log_entry(
        self,
        query: str,
        latency_seconds: float,
        tokens_used: int,
        cost_usd: float,
        corrections: int,
        relevance_score: float,
        verification_status: str,
        answer: Optional[str] = None,
        source_count: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Write a detailed log entry to the JSONL file.
        
        Thread-safe writing with file locking.
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "service": self.service_name,
            "query": query[:500] if query else "",  # Truncate long queries
            "answer_preview": answer[:200] if answer else None,  # Truncate answer
            "metrics": {
                "latency_seconds": round(latency_seconds, 4),
                "tokens_used": tokens_used,
                "cost_usd": round(cost_usd, 6),
                "corrections": corrections,
                "relevance_score": round(relevance_score, 4),
                "source_count": source_count,
            },
            "verification_status": verification_status,
            "metadata": metadata or {}
        }
        
        try:
            with self._file_lock:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.warning(f"Failed to write log entry to {self.log_path}: {e}")
    
    def increment_counter(
        self,
        counter_name: str,
        value: int = 1,
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increment a specific counter by name.
        
        Args:
            counter_name: Name of counter (queries, tokens, cost, corrections)
            value: Amount to increment by
            attributes: Optional metric attributes
        """
        attrs = attributes or {}
        
        counter_map = {
            "queries": self.queries_counter,
            "tokens": self.tokens_counter,
            "cost": self.cost_counter,
            "corrections": self.corrections_counter,
        }
        
        counter = counter_map.get(counter_name)
        if counter:
            counter.add(value, attrs)
        else:
            logger.warning(f"Unknown counter: {counter_name}")
    
    def record_latency(
        self,
        latency_seconds: float,
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a latency observation.
        
        Args:
            latency_seconds: Latency value in seconds
            attributes: Optional metric attributes
        """
        self.latency_histogram.record(latency_seconds, attributes or {})
    
    def record_relevance(
        self,
        score: float,
        attributes: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a relevance score observation.
        
        Args:
            score: Relevance score (0.0 to 1.0)
            attributes: Optional metric attributes
        """
        self.relevance_histogram.record(score, attributes or {})
    
    def get_log_path(self) -> Path:
        """Return the path to the JSONL log file."""
        return self.log_path
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance.
        
        Useful for testing or reconfiguration.
        """
        with cls._lock:
            cls._instance = None


def get_metrics_collector(
    service_name: str = "rag-finance-system",
    log_path: Optional[Path] = None
) -> MetricsCollector:
    """
    Get or create the MetricsCollector singleton.
    
    Convenience function to access the metrics collector.
    
    Args:
        service_name: Service name for metrics (only used on first call)
        log_path: Log file path (only used on first call)
    
    Returns:
        MetricsCollector singleton instance
    
    Example:
        >>> from src.observability.metrics import get_metrics_collector
        >>> metrics = get_metrics_collector()
        >>> metrics.record_query(...)
    """
    return MetricsCollector(service_name=service_name, log_path=log_path)

