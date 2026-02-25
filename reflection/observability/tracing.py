# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 7: Observability - Distributed Tracing

OpenTelemetry-compatible tracing for:
- Request flow tracking
- LLM call tracing
- Database query tracing
- Cross-service correlation

Features:
- Automatic context propagation
- Span attributes for debugging
- Integration with Jaeger/Zipkin/OTLP
"""

import logging
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from .logging import (
    clear_request_context,
    set_request_context,
)

logger = logging.getLogger(__name__)

# Try to import OpenTelemetry
try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace import SpanKind

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logger.info(
        "opentelemetry not installed. Using fallback tracing. "
        "Install with: pip install opentelemetry-api opentelemetry-sdk"
    )


# ============================================================
# SPAN DATA STRUCTURE (FALLBACK)
# ============================================================


@dataclass
class SpanData:
    """Span data for fallback tracing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    kind: str
    start_time: datetime
    end_time: datetime | None = None
    status: str = "OK"
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "duration_ms": (
                (self.end_time - self.start_time).total_seconds() * 1000 if self.end_time else None
            ),
        }


# ============================================================
# FALLBACK SPAN CONTEXT
# ============================================================


class FallbackSpan:
    """Fallback span implementation when OpenTelemetry not available."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str | None = None,
        parent_span_id: str | None = None,
        kind: str = "INTERNAL",
    ):
        self.data = SpanData(
            trace_id=trace_id,
            span_id=span_id or uuid4().hex[:16],
            parent_span_id=parent_span_id,
            name=name,
            kind=kind,
            start_time=datetime.now(UTC),
        )
        self._ended = False

    def set_attribute(self, key: str, value: Any):
        """Set span attribute."""
        self.data.attributes[key] = value

    def set_attributes(self, attributes: dict[str, Any]):
        """Set multiple span attributes."""
        self.data.attributes.update(attributes)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None):
        """Add an event to the span."""
        self.data.events.append(
            {
                "name": name,
                "timestamp": datetime.now(UTC).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_status(self, status: str, description: str | None = None):
        """Set span status."""
        self.data.status = status
        if description:
            self.data.attributes["status_description"] = description

    def record_exception(self, exception: Exception):
        """Record an exception."""
        self.data.status = "ERROR"
        self.add_event(
            "exception",
            {
                "exception.type": type(exception).__name__,
                "exception.message": str(exception),
            },
        )

    def end(self):
        """End the span."""
        if not self._ended:
            self.data.end_time = datetime.now(UTC)
            self._ended = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val:
            self.record_exception(exc_val)
        self.end()
        return False


# ============================================================
# TRACER
# ============================================================


class Tracer:
    """
    Tracer for creating spans.

    Uses OpenTelemetry when available, falls back to simple implementation.
    """

    def __init__(self, name: str = "reflection"):
        self.name = name
        self._otel_tracer = None
        self._current_trace_id: str | None = None
        self._current_span_id: str | None = None

        if OTEL_AVAILABLE:
            self._otel_tracer = trace.get_tracer(name)

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: str = "INTERNAL",
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Any, None, None]:
        """
        Start a new span.

        Args:
            name: Span name
            kind: Span kind (INTERNAL, SERVER, CLIENT, PRODUCER, CONSUMER)
            attributes: Initial span attributes

        Yields:
            Span object
        """
        if OTEL_AVAILABLE and self._otel_tracer:
            # Use OpenTelemetry
            span_kind = getattr(SpanKind, kind, SpanKind.INTERNAL)
            with self._otel_tracer.start_as_current_span(
                name,
                kind=span_kind,
                attributes=attributes or {},
            ) as span:
                yield span
        else:
            # Use fallback
            trace_id = self._current_trace_id or uuid4().hex
            parent_span_id = self._current_span_id

            span = FallbackSpan(
                name=name,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                kind=kind,
            )

            if attributes:
                span.set_attributes(attributes)

            # Set as current
            old_span_id = self._current_span_id
            self._current_span_id = span.data.span_id

            try:
                yield span
            finally:
                span.end()
                self._current_span_id = old_span_id

                # Log span data
                logger.debug(
                    f"Span completed: {name}",
                    extra={"span": span.data.to_dict()},
                )

    def start_trace(self, trace_id: str | None = None) -> str:
        """Start a new trace and return the trace ID."""
        self._current_trace_id = trace_id or uuid4().hex
        self._current_span_id = None
        return self._current_trace_id

    def get_current_trace_id(self) -> str | None:
        """Get current trace ID."""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                return format(span.get_span_context().trace_id, "032x")
        return self._current_trace_id

    def get_current_span_id(self) -> str | None:
        """Get current span ID."""
        if OTEL_AVAILABLE:
            span = trace.get_current_span()
            if span and span.get_span_context().is_valid:
                return format(span.get_span_context().span_id, "016x")
        return self._current_span_id


# ============================================================
# GLOBAL TRACER
# ============================================================

_tracer: Tracer | None = None


def get_tracer() -> Tracer:
    """Get the global tracer."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer()
    return _tracer


def init_tracer(
    service_name: str = "reflection",
    service_version: str = "0.0.0",
    environment: str = "development",
    otlp_endpoint: str | None = None,
):
    """
    Initialize the global tracer.

    Args:
        service_name: Service name for traces
        service_version: Service version
        environment: Environment name
        otlp_endpoint: OTLP collector endpoint (optional)
    """
    global _tracer

    if OTEL_AVAILABLE:
        # Configure OpenTelemetry
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: service_name,
                ResourceAttributes.SERVICE_VERSION: service_version,
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
            }
        )

        provider = TracerProvider(resource=resource)

        # Add OTLP exporter if configured
        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info(f"OTLP tracing enabled: {otlp_endpoint}")
            except ImportError:
                logger.warning("OTLP exporter not available. Install opentelemetry-exporter-otlp")

        trace.set_tracer_provider(provider)
        logger.info("OpenTelemetry tracing initialized")

    _tracer = Tracer(service_name)
    logger.info(f"Tracer initialized (otel_available={OTEL_AVAILABLE})")


# ============================================================
# TRACING HELPERS
# ============================================================


@contextmanager
def trace_request(
    request_id: str,
    method: str,
    path: str,
    tenant_id: str | None = None,
    user_id: str | None = None,
):
    """
    Context manager for tracing an HTTP request.

    Sets up request context and creates a root span.
    """
    tracer = get_tracer()
    trace_id = tracer.start_trace()

    # Set logging context
    set_request_context(
        request_id=request_id,
        trace_id=trace_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    with tracer.start_span(
        f"{method} {path}",
        kind="SERVER",
        attributes={
            "http.method": method,
            "http.route": path,
            "http.request_id": request_id,
            "tenant.id": tenant_id,
            "user.id": user_id,
        },
    ) as span:
        try:
            yield span
        finally:
            clear_request_context()


@contextmanager
def trace_llm_call(
    provider: str,
    model: str,
    operation: str = "chat",
):
    """
    Context manager for tracing an LLM call.
    """
    tracer = get_tracer()

    with tracer.start_span(
        f"llm.{operation}",
        kind="CLIENT",
        attributes={
            "llm.provider": provider,
            "llm.model": model,
            "llm.operation": operation,
        },
    ) as span:
        yield span


@contextmanager
def trace_db_query(
    operation: str,
    table: str,
):
    """
    Context manager for tracing a database query.
    """
    tracer = get_tracer()

    with tracer.start_span(
        f"db.{operation}",
        kind="CLIENT",
        attributes={
            "db.operation": operation,
            "db.table": table,
            "db.system": "postgresql",
        },
    ) as span:
        yield span


@contextmanager
def trace_redis_command(command: str):
    """
    Context manager for tracing a Redis command.
    """
    tracer = get_tracer()

    with tracer.start_span(
        f"redis.{command}",
        kind="CLIENT",
        attributes={
            "db.system": "redis",
            "db.operation": command,
        },
    ) as span:
        yield span


@contextmanager
def trace_tool_execution(
    tool_name: str,
    tenant_tier: str,
):
    """
    Context manager for tracing tool execution.
    """
    tracer = get_tracer()

    with tracer.start_span(
        f"tool.{tool_name}",
        kind="INTERNAL",
        attributes={
            "tool.name": tool_name,
            "tenant.tier": tenant_tier,
        },
    ) as span:
        yield span


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Tracer
    "Tracer",
    "get_tracer",
    "init_tracer",
    # Fallback types
    "SpanData",
    "FallbackSpan",
    # Helpers
    "trace_request",
    "trace_llm_call",
    "trace_db_query",
    "trace_redis_command",
    "trace_tool_execution",
    # Availability flag
    "OTEL_AVAILABLE",
]
