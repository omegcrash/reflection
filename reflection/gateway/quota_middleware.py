# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 6: Quota Middleware

FastAPI middleware for automatic quota enforcement.

Features:
- Pre-flight quota checks on all requests
- Automatic usage recording
- Rate limit headers in responses
- Quota exceeded error handling
"""

import logging
import time
from collections.abc import Callable
from uuid import UUID

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.executor import TenantTier
from ..tenants.quota_service import (
    QuotaExceededError,
    QuotaService,
    get_quota_service,
    get_tier_limits,
)

logger = logging.getLogger(__name__)


class QuotaMiddleware(BaseHTTPMiddleware):
    """
    Middleware for automatic quota enforcement.

    Checks quotas before processing requests and adds rate limit
    headers to responses.

    Usage:
        app.add_middleware(
            QuotaMiddleware,
            excluded_paths=["/health", "/ready", "/metrics"],
        )
    """

    def __init__(
        self,
        app,
        excluded_paths: list | None = None,
        quota_service: QuotaService | None = None,
    ):
        super().__init__(app)
        self.excluded_paths = excluded_paths or [
            "/health",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]
        self._quota_service = quota_service

    @property
    def quota_service(self) -> QuotaService:
        """Get quota service (lazy init)."""
        if self._quota_service is None:
            self._quota_service = get_quota_service()
        return self._quota_service

    def _should_check_quota(self, path: str) -> bool:
        """Determine if request path should have quota checked."""
        return all(not path.startswith(excluded) for excluded in self.excluded_paths)

    def _get_tenant_info(self, request: Request) -> tuple | None:
        """
        Extract tenant info from request.

        Returns: (tenant_id, tier) or None if not available
        """
        # Try to get from request state (set by auth)
        if hasattr(request.state, "tenant_id"):
            tenant_id = request.state.tenant_id
            tier_str = getattr(request.state, "tenant_tier", "free")
            tier = self._map_tier(tier_str)
            return (tenant_id, tier)

        # Try to get from headers (development/testing)
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            try:
                # Could be UUID or slug
                try:
                    tenant_id = UUID(tenant_header)
                except ValueError:
                    # Use hash of slug as pseudo-UUID for rate limiting
                    import hashlib

                    hash_bytes = hashlib.md5(tenant_header.encode()).digest()
                    tenant_id = UUID(bytes=hash_bytes)

                tier_header = request.headers.get("X-Tenant-Tier", "free")
                tier = self._map_tier(tier_header)
                return (tenant_id, tier)
            except Exception as e:
                logger.debug(f"Could not extract tenant from headers: {e}")

        return None

    def _map_tier(self, tier_str: str) -> TenantTier:
        """Map tier string to TenantTier enum."""
        mapping = {
            "free": TenantTier.FREE,
            "starter": TenantTier.FREE,
            "professional": TenantTier.PRO,
            "pro": TenantTier.PRO,
            "enterprise": TenantTier.ENTERPRISE,
            "business": TenantTier.ENTERPRISE,
        }
        return mapping.get(tier_str.lower(), TenantTier.FREE)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with quota checks."""
        # Skip quota check for excluded paths
        if not self._should_check_quota(request.url.path):
            return await call_next(request)

        # Get tenant info
        tenant_info = self._get_tenant_info(request)
        if tenant_info is None:
            # No tenant context - let auth middleware handle
            return await call_next(request)

        tenant_id, tier = tenant_info
        limits = get_tier_limits(tier)

        # Check if request is allowed
        try:
            result = await self.quota_service.check_request_allowed(
                tenant_id=tenant_id,
                tier=tier,
            )

            if not result.allowed:
                return self._quota_exceeded_response(result)

        except QuotaExceededError as e:
            return self._quota_exceeded_response_from_error(e)
        except Exception as e:
            # Log but don't block on quota check errors
            logger.error(f"Quota check failed: {e}")

        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration_ms = int((time.time() - start_time) * 1000)

        # Add rate limit headers
        try:
            usage = await self.quota_service.get_usage(tenant_id, tier)
            requests_usage = usage.get("usage", {}).get("requests", {}).get("per_minute", {})

            response.headers["X-RateLimit-Limit"] = str(limits.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(
                requests_usage.get("remaining", limits.requests_per_minute)
            )
            response.headers["X-RateLimit-Reset"] = str(60)  # Resets every minute
            response.headers["X-Request-Duration-Ms"] = str(duration_ms)
        except Exception as e:
            logger.debug(f"Could not add rate limit headers: {e}")

        return response

    def _quota_exceeded_response(self, result) -> JSONResponse:
        """Create response for quota exceeded."""
        retry_after = result.retry_after_seconds or 60

        return JSONResponse(
            status_code=429,
            content={
                "error": "quota_exceeded",
                "quota_type": result.quota_type.value if result.quota_type else "unknown",
                "current": result.current_value,
                "limit": result.limit_value,
                "message": result.message or "Rate limit exceeded",
                "retry_after_seconds": retry_after,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(result.limit_value),
                "X-RateLimit-Remaining": "0",
            },
        )

    def _quota_exceeded_response_from_error(self, error: QuotaExceededError) -> JSONResponse:
        """Create response from QuotaExceededError."""
        retry_after = error.retry_after_seconds or 60

        return JSONResponse(
            status_code=429,
            content=error.to_dict(),
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(error.limit),
                "X-RateLimit-Remaining": "0",
            },
        )


# ============================================================
# QUOTA CHECK DEPENDENCY
# ============================================================


class QuotaChecker:
    """
    FastAPI dependency for quota checking in routes.

    Usage:
        @router.post("/chat")
        async def chat(
            request: ChatRequest,
            quota: QuotaChecker = Depends(get_quota_checker),
        ):
            async with quota.request_context() as ctx:
                # Process request
                await quota.record_tokens(ctx, response.usage.total_tokens)
    """

    def __init__(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        service: QuotaService | None = None,
    ):
        self.tenant_id = tenant_id
        self.tier = tier
        self._service = service or get_quota_service()

    @property
    def service(self) -> QuotaService:
        return self._service

    async def check_allowed(self, estimated_tokens: int = 0) -> bool:
        """Check if request is allowed."""
        result = await self.service.check_request_allowed(self.tenant_id, self.tier)
        return result.allowed

    def request_context(self, estimated_tokens: int = 0):
        """Get context manager for quota tracking."""
        return self.service.request_context(
            self.tenant_id,
            self.tier,
            estimated_tokens=estimated_tokens,
        )

    async def record_tokens(self, ctx, tokens: int):
        """Record token usage."""
        await self.service.record_tokens(ctx, tokens)

    async def record_tool(self, ctx, tool_name: str):
        """Record tool execution."""
        await self.service.record_tool_execution(ctx, tool_name)

    async def check_tool_allowed(self, ctx, tool_name: str) -> bool:
        """Check if tool execution is allowed."""
        return await self.service.check_tool_allowed(ctx, tool_name)


async def get_quota_checker(request: Request) -> QuotaChecker:
    """FastAPI dependency for getting QuotaChecker."""
    # Get tenant info from request state (set by auth)
    tenant_id = getattr(request.state, "tenant_id", None)
    tier_str = getattr(request.state, "tenant_tier", "free")

    if tenant_id is None:
        # Try header fallback
        tenant_header = request.headers.get("X-Tenant-ID")
        if tenant_header:
            try:
                tenant_id = UUID(tenant_header)
            except ValueError:
                import hashlib

                hash_bytes = hashlib.md5(tenant_header.encode()).digest()
                tenant_id = UUID(bytes=hash_bytes)

    if tenant_id is None:
        raise HTTPException(status_code=401, detail="Tenant context required")

    tier_mapping = {
        "free": TenantTier.FREE,
        "pro": TenantTier.PRO,
        "professional": TenantTier.PRO,
        "enterprise": TenantTier.ENTERPRISE,
    }
    tier = tier_mapping.get(tier_str.lower(), TenantTier.FREE)

    return QuotaChecker(tenant_id, tier)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "QuotaMiddleware",
    "QuotaChecker",
    "get_quota_checker",
]
