# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Request Context Middleware (v1.4.0)

Provides request ID propagation and context management:
- Generates or accepts X-Request-ID header
- Propagates request ID through logging
- Provides async context for request-scoped data
- Integrates with structured logging

Usage:
    # In FastAPI app
    from reflection.gateway.request_context import (
        RequestContextMiddleware,
        get_request_id,
        get_request_context,
    )

    app.add_middleware(RequestContextMiddleware)

    # In route handlers
    @app.get("/api/endpoint")
    async def endpoint():
        request_id = get_request_id()
        logger.info("Processing request", extra={"request_id": request_id})
"""

import logging
import secrets
import time
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Optional
from uuid import UUID

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


# ============================================================
# CONTEXT VARIABLES
# ============================================================

# Request ID for the current request
_request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Full request context
_request_context_var: ContextVar[Optional["RequestContext"]] = ContextVar(
    "request_context", default=None
)


@dataclass
class RequestContext:
    """
    Request-scoped context data.

    Available throughout the request lifecycle via get_request_context().
    """

    # Core identifiers
    request_id: str

    # Timing
    start_time: float = field(default_factory=time.time)
    start_datetime: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Request info
    method: str = ""
    path: str = ""
    client_ip: str | None = None
    user_agent: str | None = None

    # Tenant/user context (set by auth middleware)
    tenant_id: UUID | None = None
    user_id: UUID | None = None

    # Custom data
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since request started."""
        return (time.time() - self.start_time) * 1000

    def to_log_dict(self) -> dict[str, Any]:
        """Get dict suitable for structured logging."""
        return {
            "request_id": self.request_id,
            "method": self.method,
            "path": self.path,
            "client_ip": self.client_ip,
            "tenant_id": str(self.tenant_id) if self.tenant_id else None,
            "user_id": str(self.user_id) if self.user_id else None,
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ============================================================
# CONTEXT ACCESSORS
# ============================================================


def get_request_id() -> str:
    """
    Get the current request ID.

    Returns empty string if not in a request context.
    """
    return _request_id_var.get()


def get_request_context() -> RequestContext | None:
    """
    Get the current request context.

    Returns None if not in a request context.
    """
    return _request_context_var.get()


def set_tenant_context(tenant_id: UUID, user_id: UUID | None = None) -> None:
    """
    Set tenant/user context for the current request.

    Called by auth middleware after authentication.
    """
    ctx = get_request_context()
    if ctx:
        ctx.tenant_id = tenant_id
        ctx.user_id = user_id


def set_context_extra(key: str, value: Any) -> None:
    """
    Set extra data in the request context.

    Useful for adding custom metadata for logging.
    """
    ctx = get_request_context()
    if ctx:
        ctx.extra[key] = value


# ============================================================
# MIDDLEWARE
# ============================================================


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request ID propagation and context management.

    Features:
    - Generates unique request ID or uses X-Request-ID header
    - Sets context variables for the request lifecycle
    - Adds X-Request-ID to response headers
    - Logs request start/end with timing

    Configuration:
        middleware = RequestContextMiddleware(
            app,
            header_name="X-Request-ID",
            generate_id=lambda: secrets.token_hex(16),
            log_requests=True,
        )
    """

    def __init__(
        self,
        app,
        header_name: str = "X-Request-ID",
        generate_id: Callable[[], str] | None = None,
        log_requests: bool = True,
        trusted_proxies: list | None = None,
    ):
        """
        Initialize middleware.

        Args:
            app: ASGI application
            header_name: Header name for request ID
            generate_id: Function to generate request IDs
            log_requests: Whether to log request start/end
            trusted_proxies: List of trusted proxy IPs for X-Forwarded-For
        """
        super().__init__(app)
        self.header_name = header_name
        self.generate_id = generate_id or (lambda: secrets.token_hex(16))
        self.log_requests = log_requests
        self.trusted_proxies = set(trusted_proxies or [])

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Process request with context management."""

        # Get or generate request ID
        request_id = request.headers.get(self.header_name)
        if not request_id:
            request_id = self.generate_id()

        # Validate/sanitize request ID (prevent injection)
        request_id = self._sanitize_request_id(request_id)

        # Get client IP (handle proxies)
        client_ip = self._get_client_ip(request)

        # Create request context
        ctx = RequestContext(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
            user_agent=request.headers.get("User-Agent"),
        )

        # Set context variables
        request_id_token = _request_id_var.set(request_id)
        context_token = _request_context_var.set(ctx)

        # Store in request state for access in dependencies
        request.state.request_id = request_id
        request.state.request_context = ctx

        try:
            # Log request start
            if self.log_requests:
                logger.info(
                    f"Request started: {ctx.method} {ctx.path}",
                    extra={
                        "request_id": request_id,
                        "method": ctx.method,
                        "path": ctx.path,
                        "client_ip": client_ip,
                    },
                )

            # Process request
            response = await call_next(request)

            # Add request ID to response
            response.headers[self.header_name] = request_id

            # Log request end
            if self.log_requests:
                logger.info(
                    f"Request completed: {ctx.method} {ctx.path} "
                    f"status={response.status_code} duration={ctx.elapsed_ms:.2f}ms",
                    extra={
                        "request_id": request_id,
                        "method": ctx.method,
                        "path": ctx.path,
                        "status_code": response.status_code,
                        "duration_ms": round(ctx.elapsed_ms, 2),
                        "tenant_id": str(ctx.tenant_id) if ctx.tenant_id else None,
                    },
                )

            return response

        except Exception as e:
            # Log error with context
            logger.error(
                f"Request failed: {ctx.method} {ctx.path} error={type(e).__name__}",
                extra={
                    "request_id": request_id,
                    "method": ctx.method,
                    "path": ctx.path,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "duration_ms": round(ctx.elapsed_ms, 2),
                },
                exc_info=True,
            )
            raise

        finally:
            # Reset context variables
            _request_id_var.reset(request_id_token)
            _request_context_var.reset(context_token)

    def _sanitize_request_id(self, request_id: str) -> str:
        """
        Sanitize request ID to prevent log injection.

        - Limit length
        - Remove newlines and control characters
        - Allow only alphanumeric and dashes
        """
        # Limit length
        request_id = request_id[:64]

        # Remove dangerous characters
        sanitized = "".join(c for c in request_id if c.isalnum() or c in "-_")

        return sanitized or self.generate_id()

    def _get_client_ip(self, request: Request) -> str | None:
        """
        Get client IP, handling proxy headers.

        Trusts X-Forwarded-For only from trusted proxies.
        """
        client_host = request.client.host if request.client else None

        # Check if request is from trusted proxy
        if client_host in self.trusted_proxies:
            # Trust X-Forwarded-For
            forwarded_for = request.headers.get("X-Forwarded-For")
            if forwarded_for:
                # Take first IP (original client)
                return forwarded_for.split(",")[0].strip()

            # Try X-Real-IP
            real_ip = request.headers.get("X-Real-IP")
            if real_ip:
                return real_ip.strip()

        return client_host


# ============================================================
# LOGGING FILTER
# ============================================================


class RequestContextFilter(logging.Filter):
    """
    Logging filter that adds request context to log records.

    Usage:
        handler = logging.StreamHandler()
        handler.addFilter(RequestContextFilter())

        formatter = logging.Formatter(
            '%(asctime)s [%(request_id)s] %(name)s - %(message)s'
        )
        handler.setFormatter(formatter)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request context to log record."""
        ctx = get_request_context()

        record.request_id = ctx.request_id if ctx else "-"
        record.tenant_id = str(ctx.tenant_id) if ctx and ctx.tenant_id else "-"
        record.user_id = str(ctx.user_id) if ctx and ctx.user_id else "-"
        record.client_ip = ctx.client_ip if ctx else "-"

        return True


# ============================================================
# FASTAPI DEPENDENCY
# ============================================================


def get_request_id_dependency(request: Request) -> str:
    """
    FastAPI dependency to get request ID.

    Usage:
        @app.get("/api/endpoint")
        async def endpoint(request_id: str = Depends(get_request_id_dependency)):
            return {"request_id": request_id}
    """
    return getattr(request.state, "request_id", "") or get_request_id()


def get_request_context_dependency(request: Request) -> RequestContext | None:
    """
    FastAPI dependency to get request context.

    Usage:
        @app.get("/api/endpoint")
        async def endpoint(ctx: RequestContext = Depends(get_request_context_dependency)):
            return {"elapsed_ms": ctx.elapsed_ms}
    """
    return getattr(request.state, "request_context", None) or get_request_context()


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Context
    "RequestContext",
    "get_request_id",
    "get_request_context",
    "set_tenant_context",
    "set_context_extra",
    # Middleware
    "RequestContextMiddleware",
    # Logging
    "RequestContextFilter",
    # Dependencies
    "get_request_id_dependency",
    "get_request_context_dependency",
]
