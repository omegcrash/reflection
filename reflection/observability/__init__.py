# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 7: Observability Module

Comprehensive observability for production deployments:
- Prometheus-compatible metrics
- Structured JSON logging
- Distributed tracing (OpenTelemetry)
- Request/response correlation
- Audit logging

Components:
- metrics: Prometheus metrics registry
- logging: Structured logging with context
- tracing: Distributed tracing support
- middleware: FastAPI middleware for automatic instrumentation
"""

from .logging import (
    AuditLogger,
    ContextLogger,
    HumanFormatter,
    JSONFormatter,
    audit_logger,
    clear_request_context,
    configure_logging,
    get_logger,
    get_request_context,
    log_llm_request,
    log_request_end,
    log_request_start,
    mask_sensitive_data,
    set_request_context,
)
from .metrics import (
    LATENCY_BUCKETS,
    PROMETHEUS_AVAILABLE,
    TOKEN_BUCKETS,
    MetricsRegistry,
    get_metrics,
    init_metrics,
    timed,
)
from .middleware import (
    MetricsMiddleware,
    TracingMiddleware,
)
from .tracing import (
    OTEL_AVAILABLE,
    Tracer,
    get_tracer,
    init_tracer,
    trace_db_query,
    trace_llm_call,
    trace_redis_command,
    trace_request,
    trace_tool_execution,
)


def init_observability(
    app_name: str = "reflection",
    app_version: str = "0.0.0",
    environment: str = "development",
    log_level: str = "INFO",
    log_format: str = "json",
    otlp_endpoint: str = None,
):
    """
    Initialize all observability components.

    Args:
        app_name: Application name
        app_version: Application version
        environment: Environment (development, staging, production)
        log_level: Logging level
        log_format: Log format ("json" or "human")
        otlp_endpoint: OTLP collector endpoint for tracing
    """
    # Configure logging
    configure_logging(
        level=log_level,
        format=log_format,
        mask_sensitive=True,
        use_colors=(environment == "development"),
    )

    # Initialize metrics
    init_metrics(
        namespace="familiar",
        app_version=app_version,
        environment=environment,
    )

    # Initialize tracing
    init_tracer(
        service_name=app_name,
        service_version=app_version,
        environment=environment,
        otlp_endpoint=otlp_endpoint,
    )


__all__ = [
    # Main initializer
    "init_observability",
    # Metrics
    "MetricsRegistry",
    "get_metrics",
    "init_metrics",
    "timed",
    "PROMETHEUS_AVAILABLE",
    "LATENCY_BUCKETS",
    "TOKEN_BUCKETS",
    # Logging
    "configure_logging",
    "get_logger",
    "ContextLogger",
    "JSONFormatter",
    "HumanFormatter",
    "AuditLogger",
    "audit_logger",
    "set_request_context",
    "clear_request_context",
    "get_request_context",
    "log_request_start",
    "log_request_end",
    "log_llm_request",
    "mask_sensitive_data",
    # Tracing
    "Tracer",
    "get_tracer",
    "init_tracer",
    "trace_request",
    "trace_llm_call",
    "trace_db_query",
    "trace_redis_command",
    "trace_tool_execution",
    "OTEL_AVAILABLE",
    # Middleware
    "MetricsMiddleware",
    "TracingMiddleware",
]
