# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
FastAPI Gateway Application

Main entry point for the Reflection API.

v1.4.0: Enhanced with RequestContextMiddleware for request ID propagation.
v1.5.0: Added job system and tenant lifecycle management.
v2.0.0: API versioning, Enterprise SSO, multi-region support.
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..core.async_base import Lifecycle
from ..core.executor import (
    ExecutorTimeoutError,
    QueueFullError,
    get_executor_pool,
    shutdown_executor_pool,
)
from ..core.settings import get_settings
from ..data.postgres import close_database, init_database

# Phase 7: Observability
from ..observability import (
    MetricsMiddleware,
    TracingMiddleware,
    init_observability,
)
from .auth_routes import router as auth_router
from .chat_routes import router as chat_router
from .chat_routes_v2 import router as chat_router_v2
from .health import router as health_router

# v1.5.0: Job system and lifecycle management
from .job_routes import router as job_router
from .lifecycle_routes import router as lifecycle_router
from .quota_middleware import QuotaMiddleware
from .quota_routes import router as quota_router
from .request_context import (
    RequestContextMiddleware,
)
from .routes import router as api_router
from .sso_routes import router as sso_router

# v2.0.0: API versioning, SSO, regions
from .versioning import version_router

logger = logging.getLogger(__name__)

settings = get_settings()


# ============================================================
# LIFECYCLE
# ============================================================

lifecycle = Lifecycle()


@lifecycle.on_startup
async def startup_observability():
    """Initialize observability (Phase 7)."""
    log_format = "human" if settings.is_development else "json"
    init_observability(
        app_name="reflection",
        app_version=settings.app_version,
        environment=settings.environment,
        log_level="DEBUG" if settings.is_development else "INFO",
        log_format=log_format,
    )
    logger.info("Observability initialized")


@lifecycle.on_startup
async def startup_database():
    """Initialize database connection pool."""
    await init_database()
    logger.info("Database initialized")


@lifecycle.on_startup
async def startup_redis():
    """Initialize Redis connection."""
    from ..data.redis import init_redis

    await init_redis()
    logger.info("Redis initialized")


@lifecycle.on_startup
async def startup_executor_pool():
    """Initialize tenant-tiered executor pool."""
    pool = get_executor_pool()
    logger.info(f"Executor pool initialized: {pool.get_metrics_summary()['by_tier'].keys()}")


@lifecycle.on_startup
async def startup_quota_service():
    """Initialize quota service with Redis backend."""
    from ..tenants.quota_service import init_quota_service

    redis_url = settings.redis_url if hasattr(settings, "redis_url") else None
    init_quota_service(redis_url=redis_url)
    logger.info("Quota service initialized")


@lifecycle.on_shutdown
async def shutdown_executor():
    """Shutdown executor pool gracefully."""
    await shutdown_executor_pool()
    logger.info("Executor pool shut down")


@lifecycle.on_shutdown
async def shutdown_database():
    """Close database connections."""
    await close_database()
    logger.info("Database closed")


@lifecycle.on_shutdown
async def shutdown_redis():
    """Close Redis connections."""
    from ..data.redis import close_redis

    await close_redis()
    logger.info("Redis closed")


@lifecycle.on_shutdown
async def shutdown_async_providers():
    """Shutdown async LLM provider pool."""
    from ..core.providers_async import shutdown_async_providers

    await shutdown_async_providers()
    logger.info("Async providers shut down")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """FastAPI lifespan context manager."""
    await lifecycle.startup()
    yield
    await lifecycle.shutdown()


# ============================================================
# APPLICATION
# ============================================================


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Reflection",
        description="Enterprise Multi-Tenant AI Companion Platform",
        version=settings.app_version,
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # --------------------------------------------------------
    # MIDDLEWARE
    # --------------------------------------------------------

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.security.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Context Middleware (v1.4.0)
    # Provides request ID propagation, timing, and context management
    app.add_middleware(
        RequestContextMiddleware,
        header_name="X-Request-ID",
        log_requests=True,
        trusted_proxies=["127.0.0.1", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"],
    )

    # --------------------------------------------------------
    # EXCEPTION HANDLERS
    # --------------------------------------------------------

    @app.exception_handler(QueueFullError)
    async def queue_full_handler(request: Request, exc: QueueFullError):
        """Handle queue full errors with 429 Too Many Requests."""
        logger.warning(
            f"Queue full for tier {exc.tier.value}: "
            f"queue_size={exc.queue_size}, "
            f"request_id={getattr(request.state, 'request_id', None)}"
        )
        return JSONResponse(
            status_code=429,
            content={
                "error": "Service temporarily at capacity. Please retry shortly.",
                "status_code": 429,
                "tier": exc.tier.value,
                "request_id": getattr(request.state, "request_id", None),
            },
            headers={"Retry-After": str(exc.retry_after_seconds)},
        )

    @app.exception_handler(ExecutorTimeoutError)
    async def executor_timeout_handler(request: Request, exc: ExecutorTimeoutError):
        """Handle executor timeout errors with 504 Gateway Timeout."""
        logger.error(
            f"Executor timeout for tier {exc.tier.value}: "
            f"timeout={exc.timeout_seconds}s, elapsed={exc.elapsed_seconds:.1f}s, "
            f"request_id={getattr(request.state, 'request_id', None)}"
        )
        return JSONResponse(
            status_code=504,
            content={
                "error": "Request processing timed out. Please try a simpler request.",
                "status_code": 504,
                "timeout_seconds": exc.timeout_seconds,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # Phase 6: Quota exceeded handler
    from ..tenants.quota_service import QuotaExceededError

    @app.exception_handler(QuotaExceededError)
    async def quota_exceeded_handler(request: Request, exc: QuotaExceededError):
        """Handle quota exceeded errors with 429 Too Many Requests."""
        logger.warning(
            f"Quota exceeded: {exc.quota_type.value} "
            f"({exc.current}/{exc.limit}), "
            f"request_id={getattr(request.state, 'request_id', None)}"
        )
        return JSONResponse(
            status_code=429,
            content=exc.to_dict(),
            headers={
                "Retry-After": str(exc.retry_after_seconds or 60),
                "X-RateLimit-Limit": str(exc.limit),
                "X-RateLimit-Remaining": "0",
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "status_code": exc.status_code,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")

        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "status_code": 500,
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    # --------------------------------------------------------
    # ROUTES
    # --------------------------------------------------------

    # Health check endpoints (no auth required)
    app.include_router(health_router, tags=["Health"])

    # v2.0.0: API version info endpoint
    app.include_router(version_router)

    # Authentication routes
    app.include_router(auth_router, prefix="/api/v1")

    # v2.0.0: SSO routes (SAML 2.0, OIDC)
    app.include_router(sso_router, prefix="/api/v2")

    # Chat routes v1 (full agent loop, sync path)
    app.include_router(chat_router, prefix="/api/v1")

    # Chat routes v2 (Phase 5: async path with dual routing)
    app.include_router(chat_router_v2, prefix="/api")

    # Quota routes (Phase 6: usage monitoring and admin)
    app.include_router(quota_router, prefix="/api/v1")

    # Job routes (v1.5.0: background jobs and data export)
    app.include_router(job_router, prefix="/api/v1")
    app.include_router(job_router, prefix="/api/v2")  # Also available in v2

    # Lifecycle routes (v1.5.0: tenant lifecycle management)
    app.include_router(lifecycle_router, prefix="/api/v1")
    app.include_router(lifecycle_router, prefix="/api/v2")  # Also available in v2

    # API routes
    app.include_router(api_router, prefix="/api/v1")

    # --------------------------------------------------------
    # QUOTA MIDDLEWARE (Phase 6)
    # --------------------------------------------------------
    # Note: Added after routes so it runs before route handlers
    # but we keep it optional for now - can be enabled in production
    if (
        settings.quota_middleware_enabled
        if hasattr(settings, "quota_middleware_enabled")
        else False
    ):
        app.add_middleware(
            QuotaMiddleware,
            excluded_paths=[
                "/health",
                "/ready",
                "/metrics",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/api/v1/auth",  # Don't rate limit auth
                "/api/v2/sso",  # Don't rate limit SSO
            ],
        )
        logger.info("Quota middleware enabled")

    # --------------------------------------------------------
    # OBSERVABILITY MIDDLEWARE (Phase 7)
    # --------------------------------------------------------
    # Metrics middleware - always enabled for monitoring
    app.add_middleware(
        MetricsMiddleware,
        excluded_paths=[
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
        normalize_paths=True,
    )

    # Tracing middleware - enabled in production for distributed tracing
    if not settings.is_development:
        app.add_middleware(
            TracingMiddleware,
            excluded_paths=[
                "/health",
                "/ready",
                "/metrics",
            ],
        )
        logger.info("Tracing middleware enabled")

    return app


# Create app instance
app = create_app()


# ============================================================
# EXPORTS
# ============================================================

__all__ = ["app", "create_app"]
