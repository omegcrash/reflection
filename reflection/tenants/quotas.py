# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Quota Enforcement System

Enforces resource limits per tenant:
- Request rate limiting
- Token budgets
- Concurrent request limits
- Tool execution limits

Uses Redis for distributed state (supports horizontal scaling).

Architecture:
    Request → QuotaManager.check() → Allow/Deny → Execute → QuotaManager.record()
"""

import asyncio
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Tenant

logger = logging.getLogger(__name__)


class QuotaType(StrEnum):
    """Types of quotas that can be enforced."""

    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    TOKENS_PER_DAY = "tokens_per_day"
    TOKENS_PER_MONTH = "tokens_per_month"
    CONCURRENT_REQUESTS = "concurrent_requests"
    TOOL_EXECUTIONS_PER_HOUR = "tool_executions_per_hour"
    SHELL_EXECUTIONS_PER_HOUR = "shell_executions_per_hour"


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    quota_type: QuotaType | None = None
    current_value: int = 0
    limit_value: int = 0
    retry_after_seconds: int | None = None
    message: str = ""

    @classmethod
    def allow(cls) -> "QuotaCheckResult":
        return cls(allowed=True)

    @classmethod
    def deny(
        cls, quota_type: QuotaType, current: int, limit: int, retry_after: int | None = None
    ) -> "QuotaCheckResult":
        return cls(
            allowed=False,
            quota_type=quota_type,
            current_value=current,
            limit_value=limit,
            retry_after_seconds=retry_after,
            message=f"Quota exceeded: {quota_type.value} ({current}/{limit})",
        )


class QuotaBackend(ABC):
    """Abstract backend for quota storage."""

    @abstractmethod
    async def increment(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        """Increment a counter and return new value."""
        pass

    @abstractmethod
    async def get(self, key: str) -> int:
        """Get current counter value."""
        pass

    @abstractmethod
    async def acquire_semaphore(
        self, key: str, max_concurrent: int, timeout_seconds: int = 30
    ) -> str | None:
        """
        Acquire a slot in a distributed semaphore.

        Returns a token to release, or None if limit reached.
        """
        pass

    @abstractmethod
    async def release_semaphore(self, key: str, token: str):
        """Release a semaphore slot."""
        pass


class RedisQuotaBackend(QuotaBackend):
    """Redis-backed quota storage for distributed deployments."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self._redis = None

    async def _get_redis(self):
        """Lazy Redis connection."""
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(self.redis_url)
        return self._redis

    async def increment(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        """Increment counter with optional TTL."""
        redis = await self._get_redis()

        pipe = redis.pipeline()
        pipe.incrby(key, amount)
        if ttl_seconds:
            pipe.expire(key, ttl_seconds)

        results = await pipe.execute()
        return results[0]

    async def get(self, key: str) -> int:
        """Get counter value."""
        redis = await self._get_redis()
        value = await redis.get(key)
        return int(value) if value else 0

    async def acquire_semaphore(
        self, key: str, max_concurrent: int, timeout_seconds: int = 30
    ) -> str | None:
        """
        Acquire semaphore using Redis sorted set.

        Each slot is a member with score = expiry timestamp.
        """
        import uuid

        redis = await self._get_redis()

        token = uuid.uuid4().hex
        now = time.time()
        expiry = now + timeout_seconds

        # Clean expired slots and count current
        await redis.zremrangebyscore(key, "-inf", now)
        current = await redis.zcard(key)

        if current >= max_concurrent:
            return None

        # Add our slot
        await redis.zadd(key, {token: expiry})
        return token

    async def release_semaphore(self, key: str, token: str):
        """Release semaphore slot."""
        redis = await self._get_redis()
        await redis.zrem(key, token)


class InMemoryQuotaBackend(QuotaBackend):
    """
    In-memory quota storage for single-node deployments.

    WARNING: Does not persist across restarts and doesn't work
    with horizontal scaling. Use Redis for production.
    """

    def __init__(self):
        self._counters: dict[str, tuple[int, float]] = {}  # key -> (value, expiry)
        self._semaphores: dict[str, dict[str, float]] = {}  # key -> {token: expiry}
        self._lock = asyncio.Lock()

    async def increment(self, key: str, amount: int = 1, ttl_seconds: int | None = None) -> int:
        async with self._lock:
            now = time.time()

            # Get current value if not expired
            current, expiry = self._counters.get(key, (0, 0))
            if expiry > 0 and expiry < now:
                current = 0

            # Increment
            new_value = current + amount
            new_expiry = (now + ttl_seconds) if ttl_seconds else 0
            self._counters[key] = (new_value, new_expiry)

            return new_value

    async def get(self, key: str) -> int:
        async with self._lock:
            now = time.time()
            value, expiry = self._counters.get(key, (0, 0))
            if expiry > 0 and expiry < now:
                return 0
            return value

    async def acquire_semaphore(
        self, key: str, max_concurrent: int, timeout_seconds: int = 30
    ) -> str | None:
        import uuid

        async with self._lock:
            now = time.time()

            # Initialize or get semaphore
            if key not in self._semaphores:
                self._semaphores[key] = {}

            # Clean expired
            self._semaphores[key] = {t: e for t, e in self._semaphores[key].items() if e > now}

            # Check limit
            if len(self._semaphores[key]) >= max_concurrent:
                return None

            # Acquire
            token = uuid.uuid4().hex
            self._semaphores[key][token] = now + timeout_seconds
            return token

    async def release_semaphore(self, key: str, token: str):
        async with self._lock:
            if key in self._semaphores:
                self._semaphores[key].pop(token, None)


class QuotaManager:
    """
    Enforces quotas for tenants.

    Usage:
        quota_manager = QuotaManager(redis_backend)

        # Before processing request
        result = await quota_manager.check_request(tenant)
        if not result.allowed:
            raise QuotaExceededError(result)

        # After processing
        await quota_manager.record_usage(tenant, tokens=1500)
    """

    def __init__(self, backend: QuotaBackend):
        self.backend = backend

    def _key(self, tenant_id: str, quota_type: QuotaType, window: str = "") -> str:
        """Generate storage key for a quota."""
        base = f"quota:{tenant_id}:{quota_type.value}"
        return f"{base}:{window}" if window else base

    def _get_window(self, granularity: str) -> str:
        """Get current time window identifier."""
        now = datetime.now(UTC)
        if granularity == "minute":
            return now.strftime("%Y%m%d%H%M")
        elif granularity == "hour":
            return now.strftime("%Y%m%d%H")
        elif granularity == "day":
            return now.strftime("%Y%m%d")
        elif granularity == "month":
            return now.strftime("%Y%m")
        return ""

    def _get_ttl(self, granularity: str) -> int:
        """Get TTL in seconds for a window granularity."""
        ttls = {
            "minute": 120,  # Keep 2 minutes
            "hour": 7200,  # Keep 2 hours
            "day": 172800,  # Keep 2 days
            "month": 5184000,  # Keep 60 days
        }
        return ttls.get(granularity, 3600)

    async def check_request(
        self, tenant: "Tenant", check_concurrent: bool = True
    ) -> QuotaCheckResult:
        """
        Check if a request is allowed under quotas.

        Checks:
        - Requests per minute
        - Requests per hour
        - Requests per day
        - Concurrent requests (if check_concurrent=True)
        """
        quotas = tenant.quotas
        tenant_id = tenant.id

        # Check requests per minute
        rpm_key = self._key(tenant_id, QuotaType.REQUESTS_PER_MINUTE, self._get_window("minute"))
        rpm_current = await self.backend.get(rpm_key)
        if rpm_current >= quotas.max_requests_per_minute:
            return QuotaCheckResult.deny(
                QuotaType.REQUESTS_PER_MINUTE,
                rpm_current,
                quotas.max_requests_per_minute,
                retry_after=60,
            )

        # Check requests per hour
        rph_key = self._key(tenant_id, QuotaType.REQUESTS_PER_HOUR, self._get_window("hour"))
        rph_current = await self.backend.get(rph_key)
        if rph_current >= quotas.max_requests_per_hour:
            return QuotaCheckResult.deny(
                QuotaType.REQUESTS_PER_HOUR,
                rph_current,
                quotas.max_requests_per_hour,
                retry_after=3600,
            )

        # Check requests per day
        rpd_key = self._key(tenant_id, QuotaType.REQUESTS_PER_DAY, self._get_window("day"))
        rpd_current = await self.backend.get(rpd_key)
        if rpd_current >= quotas.max_requests_per_day:
            return QuotaCheckResult.deny(
                QuotaType.REQUESTS_PER_DAY,
                rpd_current,
                quotas.max_requests_per_day,
                retry_after=86400,
            )

        # Check concurrent requests
        if check_concurrent:
            concurrent_key = self._key(tenant_id, QuotaType.CONCURRENT_REQUESTS)
            token = await self.backend.acquire_semaphore(
                concurrent_key,
                quotas.max_concurrent_requests,
                timeout_seconds=300,  # 5 minute max request duration
            )
            if token is None:
                return QuotaCheckResult.deny(
                    QuotaType.CONCURRENT_REQUESTS,
                    quotas.max_concurrent_requests,
                    quotas.max_concurrent_requests,
                    retry_after=5,
                )
            # Return token via result for later release
            result = QuotaCheckResult.allow()
            result.metadata = {"concurrent_token": token}  # type: ignore
            return result

        return QuotaCheckResult.allow()

    async def check_tokens(self, tenant: "Tenant", tokens_needed: int) -> QuotaCheckResult:
        """Check if tenant has token budget remaining."""
        quotas = tenant.quotas
        tenant_id = tenant.id

        # Check daily token limit
        daily_key = self._key(tenant_id, QuotaType.TOKENS_PER_DAY, self._get_window("day"))
        daily_current = await self.backend.get(daily_key)
        if daily_current + tokens_needed > quotas.max_tokens_per_day:
            return QuotaCheckResult.deny(
                QuotaType.TOKENS_PER_DAY,
                daily_current,
                quotas.max_tokens_per_day,
                retry_after=86400,
            )

        # Check monthly token limit
        monthly_key = self._key(tenant_id, QuotaType.TOKENS_PER_MONTH, self._get_window("month"))
        monthly_current = await self.backend.get(monthly_key)
        if monthly_current + tokens_needed > quotas.max_tokens_per_month:
            return QuotaCheckResult.deny(
                QuotaType.TOKENS_PER_MONTH,
                monthly_current,
                quotas.max_tokens_per_month,
            )

        return QuotaCheckResult.allow()

    async def check_tool_execution(self, tenant: "Tenant", tool_name: str = "") -> QuotaCheckResult:
        """Check if tool execution is allowed."""
        quotas = tenant.quotas
        tenant_id = tenant.id

        # Check general tool limit
        tool_key = self._key(
            tenant_id, QuotaType.TOOL_EXECUTIONS_PER_HOUR, self._get_window("hour")
        )
        tool_current = await self.backend.get(tool_key)
        if tool_current >= quotas.max_tool_executions_per_hour:
            return QuotaCheckResult.deny(
                QuotaType.TOOL_EXECUTIONS_PER_HOUR,
                tool_current,
                quotas.max_tool_executions_per_hour,
                retry_after=3600,
            )

        # Check shell execution limit if applicable
        if tool_name in ("run_shell", "run_shell_async", "shell"):
            shell_key = self._key(
                tenant_id, QuotaType.SHELL_EXECUTIONS_PER_HOUR, self._get_window("hour")
            )
            shell_current = await self.backend.get(shell_key)
            if shell_current >= quotas.max_shell_executions_per_hour:
                return QuotaCheckResult.deny(
                    QuotaType.SHELL_EXECUTIONS_PER_HOUR,
                    shell_current,
                    quotas.max_shell_executions_per_hour,
                    retry_after=3600,
                )

        return QuotaCheckResult.allow()

    async def record_request(self, tenant: "Tenant"):
        """Record a request against quotas."""
        tenant_id = tenant.id

        # Increment all request counters
        await self.backend.increment(
            self._key(tenant_id, QuotaType.REQUESTS_PER_MINUTE, self._get_window("minute")),
            ttl_seconds=self._get_ttl("minute"),
        )
        await self.backend.increment(
            self._key(tenant_id, QuotaType.REQUESTS_PER_HOUR, self._get_window("hour")),
            ttl_seconds=self._get_ttl("hour"),
        )
        await self.backend.increment(
            self._key(tenant_id, QuotaType.REQUESTS_PER_DAY, self._get_window("day")),
            ttl_seconds=self._get_ttl("day"),
        )

    async def record_tokens(self, tenant: "Tenant", tokens: int):
        """Record token usage."""
        tenant_id = tenant.id

        await self.backend.increment(
            self._key(tenant_id, QuotaType.TOKENS_PER_DAY, self._get_window("day")),
            amount=tokens,
            ttl_seconds=self._get_ttl("day"),
        )
        await self.backend.increment(
            self._key(tenant_id, QuotaType.TOKENS_PER_MONTH, self._get_window("month")),
            amount=tokens,
            ttl_seconds=self._get_ttl("month"),
        )

    async def record_tool_execution(self, tenant: "Tenant", tool_name: str = ""):
        """Record tool execution."""
        tenant_id = tenant.id

        await self.backend.increment(
            self._key(tenant_id, QuotaType.TOOL_EXECUTIONS_PER_HOUR, self._get_window("hour")),
            ttl_seconds=self._get_ttl("hour"),
        )

        if tool_name in ("run_shell", "run_shell_async", "shell"):
            await self.backend.increment(
                self._key(tenant_id, QuotaType.SHELL_EXECUTIONS_PER_HOUR, self._get_window("hour")),
                ttl_seconds=self._get_ttl("hour"),
            )

    async def release_concurrent(self, tenant: "Tenant", token: str):
        """Release a concurrent request slot."""
        key = self._key(tenant.id, QuotaType.CONCURRENT_REQUESTS)
        await self.backend.release_semaphore(key, token)

    async def get_usage_summary(self, tenant: "Tenant") -> dict[str, any]:
        """Get current usage summary for a tenant."""
        tenant_id = tenant.id
        quotas = tenant.quotas

        return {
            "requests": {
                "per_minute": {
                    "current": await self.backend.get(
                        self._key(
                            tenant_id, QuotaType.REQUESTS_PER_MINUTE, self._get_window("minute")
                        )
                    ),
                    "limit": quotas.max_requests_per_minute,
                },
                "per_hour": {
                    "current": await self.backend.get(
                        self._key(tenant_id, QuotaType.REQUESTS_PER_HOUR, self._get_window("hour"))
                    ),
                    "limit": quotas.max_requests_per_hour,
                },
                "per_day": {
                    "current": await self.backend.get(
                        self._key(tenant_id, QuotaType.REQUESTS_PER_DAY, self._get_window("day"))
                    ),
                    "limit": quotas.max_requests_per_day,
                },
            },
            "tokens": {
                "per_day": {
                    "current": await self.backend.get(
                        self._key(tenant_id, QuotaType.TOKENS_PER_DAY, self._get_window("day"))
                    ),
                    "limit": quotas.max_tokens_per_day,
                },
                "per_month": {
                    "current": await self.backend.get(
                        self._key(tenant_id, QuotaType.TOKENS_PER_MONTH, self._get_window("month"))
                    ),
                    "limit": quotas.max_tokens_per_month,
                },
            },
            "tools": {
                "per_hour": {
                    "current": await self.backend.get(
                        self._key(
                            tenant_id, QuotaType.TOOL_EXECUTIONS_PER_HOUR, self._get_window("hour")
                        )
                    ),
                    "limit": quotas.max_tool_executions_per_hour,
                },
            },
        }


# Global quota manager instance
_quota_manager: QuotaManager | None = None


def get_quota_manager() -> QuotaManager:
    """Get the global quota manager."""
    global _quota_manager
    if _quota_manager is None:
        raise RuntimeError("Quota manager not initialized. Call init_quota_manager() first.")
    return _quota_manager


def init_quota_manager(backend: QuotaBackend | None = None):
    """Initialize the global quota manager."""
    global _quota_manager

    if backend is None:
        # Default to in-memory for development
        backend = InMemoryQuotaBackend()
        logger.warning("Using in-memory quota backend. Use Redis for production.")

    _quota_manager = QuotaManager(backend)
    logger.info("Quota manager initialized")


class QuotaExceededError(Exception):
    """Raised when a tenant exceeds their quota."""

    def __init__(self, result: QuotaCheckResult):
        self.result = result
        super().__init__(result.message)


# ---------------------------------------------------------------------------
# Synchronous quota enforcement (thread-safe, in-memory)
# ---------------------------------------------------------------------------

# key -> (count, window_start_timestamp)
_sync_counters: dict[str, tuple[int, float]] = {}
_sync_lock = threading.Lock()

# Default limits per quota type: (max_count, window_seconds)
_SYNC_DEFAULTS: dict[str, tuple[int, int]] = {
    "requests_per_minute": (60, 60),
    "requests_per_hour": (1_000, 3_600),
    "requests_per_day": (10_000, 86_400),
    "tokens_per_day": (1_000_000, 86_400),
    "tokens_per_month": (10_000_000, 2_592_000),
}


def set_sync_quota_limit(quota_type: str, max_count: int, window_seconds: int) -> None:
    """Override default limits for a quota type (useful for per-tier config)."""
    _SYNC_DEFAULTS[quota_type] = (max_count, window_seconds)


def check_quota(tenant_id, quota_type: str) -> bool:
    """
    Synchronous quota check with in-memory enforcement.

    Increments the counter for *tenant_id* / *quota_type* and raises
    :class:`QuotaExceededError` when the limit is exceeded.

    For distributed or async quota checks, use :class:`QuotaManager` directly.

    Args:
        tenant_id: Tenant UUID
        quota_type: Type of quota to check (e.g., "requests_per_day")

    Returns:
        True if quota allows

    Raises:
        QuotaExceededError: If the quota has been exceeded
    """
    limits = _SYNC_DEFAULTS.get(quota_type)
    if limits is None:
        # Unknown quota type — allow by default
        logger.debug("Unknown quota type %s for tenant %s — allowing", quota_type, tenant_id)
        return True

    max_count, window_seconds = limits
    key = f"{tenant_id}:{quota_type}"
    now = time.time()

    with _sync_lock:
        count, window_start = _sync_counters.get(key, (0, now))

        # Reset counter if the window has expired
        if now - window_start >= window_seconds:
            count = 0
            window_start = now

        count += 1
        _sync_counters[key] = (count, window_start)

    if count > max_count:
        retry_after = int(window_seconds - (now - window_start))
        # Map string quota_type to the enum if possible
        try:
            qt = QuotaType(quota_type)
        except ValueError:
            qt = QuotaType.REQUESTS_PER_DAY
        raise QuotaExceededError(
            QuotaCheckResult.deny(qt, count, max_count, retry_after=max(retry_after, 1))
        )

    return True


def record_usage(tenant_id, quota_type: str, amount: int = 1) -> None:
    """Record usage against a quota counter.

    This is a synchronous convenience function used by the services
    orchestrator to increment usage counters (e.g., tokens consumed).
    Schedules the async increment as a fire-and-forget task.
    """
    import asyncio

    async def _record():
        try:
            manager = get_quota_manager()
            await manager._backend.increment(str(tenant_id), quota_type, amount)
        except Exception:
            pass

    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_record())
    except RuntimeError:
        # No running loop — skip silently
        pass


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "QuotaManager",
    "QuotaBackend",
    "RedisQuotaBackend",
    "InMemoryQuotaBackend",
    "QuotaCheckResult",
    "QuotaExceededError",
    "QuotaType",
    "get_quota_manager",
    "init_quota_manager",
    "check_quota",
    "set_sync_quota_limit",
    "record_usage",
]
