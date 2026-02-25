# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 6: Quota Service

Enhanced quota management with:
- Tier-based default limits
- Pre-flight quota checks
- Atomic usage recording
- Usage alerts and notifications
- Admin quota overrides

This service wraps QuotaManager and provides the business logic
for quota enforcement across the platform.
"""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from ..core.executor import TenantTier
from .quotas import (
    InMemoryQuotaBackend,
    QuotaBackend,
    QuotaCheckResult,
    QuotaManager,
    QuotaType,
    RedisQuotaBackend,
    get_quota_manager,
    init_quota_manager,
)

logger = logging.getLogger(__name__)


# ============================================================
# TIER-BASED QUOTA LIMITS
# ============================================================


@dataclass
class QuotaLimits:
    """
    Quota limits configuration.

    These can be customized per-tenant or use tier defaults.
    """

    # Request limits
    requests_per_minute: int = 20
    requests_per_hour: int = 500
    requests_per_day: int = 5000

    # Token limits
    tokens_per_day: int = 100_000
    tokens_per_month: int = 1_000_000

    # Concurrency
    max_concurrent_requests: int = 5

    # Tool limits
    tool_executions_per_hour: int = 100
    shell_executions_per_hour: int = 20

    # Storage limits (MB)
    max_conversation_history_mb: int = 100
    max_memory_entries: int = 1000

    # Feature flags
    allow_tool_execution: bool = True
    allow_shell_execution: bool = False
    allow_file_upload: bool = True
    max_file_upload_mb: int = 10

    def to_dict(self) -> dict[str, Any]:
        return {
            "requests_per_minute": self.requests_per_minute,
            "requests_per_hour": self.requests_per_hour,
            "requests_per_day": self.requests_per_day,
            "tokens_per_day": self.tokens_per_day,
            "tokens_per_month": self.tokens_per_month,
            "max_concurrent_requests": self.max_concurrent_requests,
            "tool_executions_per_hour": self.tool_executions_per_hour,
            "shell_executions_per_hour": self.shell_executions_per_hour,
            "max_conversation_history_mb": self.max_conversation_history_mb,
            "max_memory_entries": self.max_memory_entries,
            "allow_tool_execution": self.allow_tool_execution,
            "allow_shell_execution": self.allow_shell_execution,
            "allow_file_upload": self.allow_file_upload,
            "max_file_upload_mb": self.max_file_upload_mb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuotaLimits":
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


# Tier-based default limits
TIER_LIMITS: dict[TenantTier, QuotaLimits] = {
    TenantTier.FREE: QuotaLimits(
        requests_per_minute=10,
        requests_per_hour=100,
        requests_per_day=500,
        tokens_per_day=50_000,
        tokens_per_month=500_000,
        max_concurrent_requests=2,
        tool_executions_per_hour=20,
        shell_executions_per_hour=0,  # No shell for free tier
        max_conversation_history_mb=50,
        max_memory_entries=100,
        allow_tool_execution=True,
        allow_shell_execution=False,
        allow_file_upload=True,
        max_file_upload_mb=5,
    ),
    TenantTier.PRO: QuotaLimits(
        requests_per_minute=30,
        requests_per_hour=1000,
        requests_per_day=10000,
        tokens_per_day=500_000,
        tokens_per_month=5_000_000,
        max_concurrent_requests=10,
        tool_executions_per_hour=500,
        shell_executions_per_hour=50,
        max_conversation_history_mb=500,
        max_memory_entries=5000,
        allow_tool_execution=True,
        allow_shell_execution=True,
        allow_file_upload=True,
        max_file_upload_mb=50,
    ),
    TenantTier.ENTERPRISE: QuotaLimits(
        requests_per_minute=100,
        requests_per_hour=5000,
        requests_per_day=50000,
        tokens_per_day=5_000_000,
        tokens_per_month=50_000_000,
        max_concurrent_requests=50,
        tool_executions_per_hour=5000,
        shell_executions_per_hour=500,
        max_conversation_history_mb=5000,
        max_memory_entries=50000,
        allow_tool_execution=True,
        allow_shell_execution=True,
        allow_file_upload=True,
        max_file_upload_mb=500,
    ),
}


def get_tier_limits(tier: TenantTier) -> QuotaLimits:
    """Get default quota limits for a tier."""
    return TIER_LIMITS.get(tier, TIER_LIMITS[TenantTier.FREE])


# ============================================================
# QUOTA EXCEEDED ERROR
# ============================================================


class QuotaExceededError(Exception):
    """Raised when a quota limit is exceeded."""

    def __init__(
        self,
        quota_type: QuotaType,
        current: int,
        limit: int,
        retry_after_seconds: int | None = None,
    ):
        self.quota_type = quota_type
        self.current = current
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds

        super().__init__(f"Quota exceeded: {quota_type.value} ({current}/{limit})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": "quota_exceeded",
            "quota_type": self.quota_type.value,
            "current": self.current,
            "limit": self.limit,
            "retry_after_seconds": self.retry_after_seconds,
            "message": str(self),
        }


# ============================================================
# QUOTA SERVICE
# ============================================================


@dataclass
class TenantQuotaContext:
    """
    Quota context for a tenant request.

    Used to track and enforce quotas throughout a request lifecycle.
    """

    tenant_id: UUID
    tier: TenantTier
    limits: QuotaLimits
    concurrent_token: str | None = None
    tokens_used: int = 0
    tools_executed: int = 0
    request_recorded: bool = False


class QuotaService:
    """
    High-level quota service for request lifecycle management.

    Usage:
        service = QuotaService(quota_manager)

        # Start request (checks and acquires concurrent slot)
        ctx = await service.start_request(tenant_id, tier)

        try:
            # Do work...
            await service.record_tokens(ctx, 1500)

            # Complete request
            await service.end_request(ctx)
        except Exception:
            await service.end_request(ctx, success=False)
    """

    def __init__(self, quota_manager: QuotaManager | None = None):
        self._manager = quota_manager

    @property
    def manager(self) -> QuotaManager:
        """Get the quota manager (lazy init)."""
        if self._manager is None:
            self._manager = get_quota_manager()
        return self._manager

    def get_limits(
        self, tier: TenantTier, custom_limits: dict[str, Any] | None = None
    ) -> QuotaLimits:
        """Get quota limits for a tier with optional overrides."""
        base = get_tier_limits(tier)

        if custom_limits:
            # Apply custom overrides
            merged = base.to_dict()
            merged.update(custom_limits)
            return QuotaLimits.from_dict(merged)

        return base

    async def check_request_allowed(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        limits: QuotaLimits | None = None,
    ) -> QuotaCheckResult:
        """
        Check if a request is allowed (pre-flight check).

        Checks:
        - Request rate limits (RPM, RPH, RPD)
        - Concurrent request limit

        Does NOT acquire the concurrent slot - use start_request for that.
        """
        if limits is None:
            limits = self.get_limits(tier)

        # Create a mock tenant object for QuotaManager
        tenant = _MockTenant(tenant_id, limits)

        # Check rate limits
        result = await self.manager.check_request(tenant)
        if not result.allowed:
            return result

        return QuotaCheckResult.allow()

    async def start_request(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        custom_limits: dict[str, Any] | None = None,
        estimated_tokens: int = 0,
    ) -> TenantQuotaContext:
        """
        Start a request with quota checks.

        Performs pre-flight checks and acquires concurrent slot.

        Raises:
            QuotaExceededError: If any quota check fails

        Returns:
            TenantQuotaContext for tracking the request
        """
        limits = self.get_limits(tier, custom_limits)
        tenant = _MockTenant(tenant_id, limits)

        # Check rate limits
        rate_result = await self.manager.check_request(tenant)
        if not rate_result.allowed:
            raise QuotaExceededError(
                rate_result.quota_type,
                rate_result.current_value,
                rate_result.limit_value,
                rate_result.retry_after_seconds,
            )

        # Check token budget if estimated
        if estimated_tokens > 0:
            token_result = await self.manager.check_tokens(tenant, estimated_tokens)
            if not token_result.allowed:
                raise QuotaExceededError(
                    token_result.quota_type,
                    token_result.current_value,
                    token_result.limit_value,
                    token_result.retry_after_seconds,
                )

        # Acquire concurrent slot
        concurrent_token = await self.manager.backend.acquire_semaphore(
            f"quota:{tenant_id}:concurrent",
            limits.max_concurrent_requests,
            timeout_seconds=300,  # 5 minute request timeout
        )

        if concurrent_token is None:
            raise QuotaExceededError(
                QuotaType.CONCURRENT_REQUESTS,
                limits.max_concurrent_requests,
                limits.max_concurrent_requests,
                retry_after_seconds=5,
            )

        # Record the request
        await self.manager.record_request(tenant)

        return TenantQuotaContext(
            tenant_id=tenant_id,
            tier=tier,
            limits=limits,
            concurrent_token=concurrent_token,
            request_recorded=True,
        )

    async def record_tokens(
        self,
        ctx: TenantQuotaContext,
        tokens: int,
    ):
        """Record token usage during a request."""
        tenant = _MockTenant(ctx.tenant_id, ctx.limits)
        await self.manager.record_tokens(tenant, tokens)
        ctx.tokens_used += tokens

    async def record_tool_execution(
        self,
        ctx: TenantQuotaContext,
        tool_name: str,
    ):
        """Record tool execution during a request."""
        tenant = _MockTenant(ctx.tenant_id, ctx.limits)
        await self.manager.record_tool_execution(tenant, tool_name)
        ctx.tools_executed += 1

    async def check_tool_allowed(
        self,
        ctx: TenantQuotaContext,
        tool_name: str,
    ) -> bool:
        """Check if a tool execution is allowed."""
        # Check feature flag
        if not ctx.limits.allow_tool_execution:
            return False

        if (
            tool_name in ("run_shell", "run_shell_async", "shell")
            and not ctx.limits.allow_shell_execution
        ):
            return False

        # Check rate limit
        tenant = _MockTenant(ctx.tenant_id, ctx.limits)
        result = await self.manager.check_tool_execution(tenant, tool_name)
        return result.allowed

    async def end_request(
        self,
        ctx: TenantQuotaContext,
        success: bool = True,
    ):
        """
        End a request and release resources.

        Always call this, even on failure, to release concurrent slot.
        """
        if ctx.concurrent_token:
            await self.manager.backend.release_semaphore(
                f"quota:{ctx.tenant_id}:concurrent",
                ctx.concurrent_token,
            )
            ctx.concurrent_token = None

        if not success:
            logger.debug(
                f"Request failed for tenant {ctx.tenant_id}, "
                f"tokens_used={ctx.tokens_used}, tools_executed={ctx.tools_executed}"
            )

    @asynccontextmanager
    async def request_context(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        custom_limits: dict[str, Any] | None = None,
        estimated_tokens: int = 0,
    ):
        """
        Context manager for quota-tracked requests.

        Usage:
            async with quota_service.request_context(tenant_id, tier) as ctx:
                # Do work...
                await quota_service.record_tokens(ctx, 1500)
        """
        ctx = await self.start_request(tenant_id, tier, custom_limits, estimated_tokens)
        try:
            yield ctx
            await self.end_request(ctx, success=True)
        except Exception:
            await self.end_request(ctx, success=False)
            raise

    async def get_usage(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        custom_limits: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get current usage summary for a tenant."""
        limits = self.get_limits(tier, custom_limits)
        tenant = _MockTenant(tenant_id, limits)

        usage = await self.manager.get_usage_summary(tenant)

        # Add percentage calculations
        for category in usage.values():
            for period in category.values():
                if isinstance(period, dict) and "current" in period and "limit" in period:
                    limit = period["limit"]
                    current = period["current"]
                    period["percentage"] = round((current / limit * 100) if limit > 0 else 0, 2)
                    period["remaining"] = max(0, limit - current)

        return {
            "tenant_id": str(tenant_id),
            "tier": tier.value,
            "usage": usage,
            "limits": limits.to_dict(),
            "timestamp": datetime.now(UTC).isoformat(),
        }

    async def get_usage_alerts(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        threshold_percent: float = 80.0,
    ) -> list[dict[str, Any]]:
        """
        Get usage alerts for quotas approaching limits.

        Returns list of quotas at or above threshold_percent.
        """
        usage_data = await self.get_usage(tenant_id, tier)
        alerts = []

        def check_category(category_name: str, category_data: dict):
            for period_name, period_data in category_data.items():
                if (
                    isinstance(period_data, dict)
                    and "percentage" in period_data
                    and period_data["percentage"] >= threshold_percent
                ):
                    alerts.append(
                        {
                            "category": category_name,
                            "period": period_name,
                            "current": period_data["current"],
                            "limit": period_data["limit"],
                            "percentage": period_data["percentage"],
                            "severity": "critical"
                            if period_data["percentage"] >= 95
                            else "warning",
                        }
                    )

        for category_name, category_data in usage_data["usage"].items():
            check_category(category_name, category_data)

        return sorted(alerts, key=lambda x: x["percentage"], reverse=True)


# ============================================================
# MOCK TENANT FOR QUOTA MANAGER
# ============================================================


class _MockTenantQuotas:
    """Mock quotas object for QuotaManager compatibility."""

    def __init__(self, limits: QuotaLimits):
        self.max_requests_per_minute = limits.requests_per_minute
        self.max_requests_per_hour = limits.requests_per_hour
        self.max_requests_per_day = limits.requests_per_day
        self.max_tokens_per_day = limits.tokens_per_day
        self.max_tokens_per_month = limits.tokens_per_month
        self.max_concurrent_requests = limits.max_concurrent_requests
        self.max_tool_executions_per_hour = limits.tool_executions_per_hour
        self.max_shell_executions_per_hour = limits.shell_executions_per_hour


class _MockTenant:
    """Mock tenant object for QuotaManager compatibility."""

    def __init__(self, tenant_id: UUID, limits: QuotaLimits):
        self.id = tenant_id
        self.quotas = _MockTenantQuotas(limits)


# ============================================================
# GLOBAL QUOTA SERVICE
# ============================================================

_quota_service: QuotaService | None = None


def get_quota_service() -> QuotaService:
    """Get the global quota service."""
    global _quota_service
    if _quota_service is None:
        _quota_service = QuotaService()
    return _quota_service


def init_quota_service(
    redis_url: str | None = None,
    backend: QuotaBackend | None = None,
):
    """
    Initialize the quota service.

    Args:
        redis_url: Redis URL for distributed quota storage
        backend: Custom quota backend (overrides redis_url)
    """
    global _quota_service

    if backend is None:
        if redis_url:
            backend = RedisQuotaBackend(redis_url)
            logger.info("Quota service initialized with Redis backend")
        else:
            backend = InMemoryQuotaBackend()
            logger.warning(
                "Quota service using in-memory backend. Use Redis for production deployments."
            )

    init_quota_manager(backend)
    _quota_service = QuotaService()

    logger.info("Quota service initialized")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Configuration
    "QuotaLimits",
    "TIER_LIMITS",
    "get_tier_limits",
    # Errors
    "QuotaExceededError",
    # Service
    "QuotaService",
    "TenantQuotaContext",
    "get_quota_service",
    "init_quota_service",
]
