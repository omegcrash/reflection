# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 7: Observability - Metrics Middleware

FastAPI middleware for automatic request metrics and tracing.
"""

import logging
import time
from collections.abc import Callable
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .logging import (
    clear_request_context,
    log_request_end,
    log_request_start,
    set_request_context,
)
from .metrics import get_metrics
from .tracing import get_tracer

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic request metrics collection.

    Collects:
    - Request count by method, endpoint, status
    - Request duration histogram
    - In-progress request gauge

    Usage:
        app.add_middleware(MetricsMiddleware)
    """

    def __init__(
        self,
        app,
        excluded_paths: list | None = None,
        normalize_paths: bool = True,
    ):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/metrics",
        ]
        self.normalize_paths = normalize_paths

    def _should_track(self, path: str) -> bool:
        """Check if path should be tracked."""
        return all(not path.startswith(excluded) for excluded in self.excluded_paths)

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path for metrics aggregation.

        Replaces UUIDs and numeric IDs with placeholders.
        """
        if not self.normalize_paths:
            return path

        import re

        # Replace UUIDs
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{uuid}",
            path,
            flags=re.IGNORECASE,
        )

        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)

        return path

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with metrics collection."""
        path = request.url.path
        method = request.method

        # Skip excluded paths
        if not self._should_track(path):
            return await call_next(request)

        # Normalize path for metrics
        normalized_path = self._normalize_path(path)

        # Get or generate request ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))
        request.state.request_id = request_id

        # Get tenant info if available
        tenant_id = getattr(request.state, "tenant_id", None)
        if tenant_id is None:
            tenant_id = request.headers.get("X-Tenant-ID")

        # Set request context for logging
        set_request_context(
            request_id=request_id,
            tenant_id=str(tenant_id) if tenant_id else None,
        )

        # Get metrics registry
        metrics = get_metrics()

        # Track in-progress
        metrics.inc_requests_in_progress(method, normalized_path)

        # Log request start
        log_request_start(method, path, request_id, str(tenant_id) if tenant_id else None)

        # Process request
        start_time = time.time()
        status_code = 500  # Default for exceptions

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
            logger.exception(f"Request failed: {method} {path}")
            raise

        finally:
            # Calculate duration
            duration = time.time() - start_time
            duration_ms = duration * 1000

            # Record metrics
            metrics.dec_requests_in_progress(method, normalized_path)
            metrics.record_http_request(
                method=method,
                endpoint=normalized_path,
                status_code=status_code,
                duration_seconds=duration,
            )

            # Log request end
            log_request_end(
                method,
                path,
                request_id,
                status_code,
                duration_ms,
                str(tenant_id) if tenant_id else None,
            )

            # Clear request context
            clear_request_context()


class TracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for distributed tracing.

    Creates a root span for each request and propagates context.

    Usage:
        app.add_middleware(TracingMiddleware)
    """

    def __init__(
        self,
        app,
        excluded_paths: list | None = None,
    ):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/metrics",
        ]

    def _should_trace(self, path: str) -> bool:
        """Check if path should be traced."""
        return all(not path.startswith(excluded) for excluded in self.excluded_paths)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with tracing."""
        path = request.url.path
        method = request.method

        # Skip excluded paths
        if not self._should_trace(path):
            return await call_next(request)

        # Get or generate trace context
        trace_id = request.headers.get("X-Trace-ID")
        request_id = getattr(request.state, "request_id", None) or str(uuid4())

        # Get tracer
        tracer = get_tracer()

        # Start trace
        if trace_id:
            tracer.start_trace(trace_id)
        else:
            trace_id = tracer.start_trace()

        # Set trace ID in logging context
        set_request_context(trace_id=trace_id)

        # Create root span
        with tracer.start_span(
            f"{method} {path}",
            kind="SERVER",
            attributes={
                "http.method": method,
                "http.route": path,
                "http.request_id": request_id,
            },
        ) as span:
            try:
                response = await call_next(request)

                # Add response attributes
                span.set_attribute("http.status_code", response.status_code)

                # Add trace headers to response
                response.headers["X-Trace-ID"] = trace_id

                return response

            except Exception as e:
                span.record_exception(e)
                span.set_status("ERROR", str(e))
                raise


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MetricsMiddleware",
    "TracingMiddleware",
]
