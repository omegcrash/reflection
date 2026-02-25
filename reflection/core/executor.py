# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant-Tiered Executor Pool

Provides isolated thread pools per tenant tier to:
- Prevent noisy neighbor issues
- Implement backpressure via bounded queues
- Handle timeouts gracefully
- Enable tier-based resource allocation

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Async FastAPI Handlers              │
    ├─────────────────────────────────────────────────┤
    │            TenantExecutorPool                    │
    │  ┌───────────┐ ┌───────────┐ ┌───────────┐     │
    │  │  FREE     │ │   PRO     │ │ENTERPRISE │     │
    │  │ 2 workers │ │10 workers │ │50 workers │     │
    │  │ queue: 10 │ │queue: 100 │ │queue: 500 │     │
    │  └───────────┘ └───────────┘ └───────────┘     │
    ├─────────────────────────────────────────────────┤
    │          Sync Familiar Agent.chat()              │
    └─────────────────────────────────────────────────┘

This bridges the async FastAPI layer with synchronous Familiar Agent
calls while maintaining tenant isolation and backpressure.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypeVar
from uuid import UUID

logger = logging.getLogger(__name__)

T = TypeVar("T")


class TenantTier(StrEnum):
    """
    Tenant subscription tiers.

    Must match the tiers defined in tenants/models.py
    """

    FREE = "free"
    PRO = "professional"
    PROFESSIONAL = "professional"  # Alias for PRO
    ENTERPRISE = "enterprise"

    @classmethod
    def from_string(cls, value: str) -> "TenantTier":
        """Convert string to TenantTier, with fallback to FREE."""
        try:
            return cls(value.lower())
        except ValueError:
            logger.warning(f"Unknown tier '{value}', defaulting to FREE")
            return cls.FREE


@dataclass
class TierConfig:
    """
    Configuration for a tenant tier's executor pool.

    Attributes:
        max_workers: Maximum concurrent threads for this tier
        max_queue_size: Maximum pending requests before rejection
        timeout_seconds: Default timeout for operations
        priority: Higher = more priority (for future weighted scheduling)
    """

    max_workers: int
    max_queue_size: int
    timeout_seconds: int
    priority: int = 1

    def __post_init__(self):
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be >= 1")


# Default tier configurations based on production requirements
DEFAULT_TIER_CONFIGS: dict[TenantTier, TierConfig] = {
    TenantTier.FREE: TierConfig(
        max_workers=2,
        max_queue_size=10,
        timeout_seconds=30,
        priority=1,
    ),
    TenantTier.PROFESSIONAL: TierConfig(
        max_workers=10,
        max_queue_size=100,
        timeout_seconds=60,
        priority=2,
    ),
    TenantTier.ENTERPRISE: TierConfig(
        max_workers=50,
        max_queue_size=500,
        timeout_seconds=120,
        priority=3,
    ),
}


@dataclass
class ExecutorMetrics:
    """Metrics for a single tier's executor."""

    tier: TenantTier
    max_workers: int
    max_queue: int
    active_tasks: int = 0
    queued_tasks: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_timeouts: int = 0
    total_rejected: int = 0
    avg_latency_ms: float = 0.0

    @property
    def queue_utilization(self) -> float:
        """Queue utilization as percentage (0.0 to 1.0)."""
        if self.max_queue == 0:
            return 0.0
        return self.queued_tasks / self.max_queue

    @property
    def worker_utilization(self) -> float:
        """Worker utilization as percentage (0.0 to 1.0)."""
        if self.max_workers == 0:
            return 0.0
        return min(1.0, self.active_tasks / self.max_workers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "max_workers": self.max_workers,
            "max_queue": self.max_queue,
            "active_tasks": self.active_tasks,
            "queued_tasks": self.queued_tasks,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_timeouts": self.total_timeouts,
            "total_rejected": self.total_rejected,
            "queue_utilization": round(self.queue_utilization, 3),
            "worker_utilization": round(self.worker_utilization, 3),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }


class QueueFullError(Exception):
    """
    Raised when the executor queue is at capacity.

    Client should retry after a brief delay.
    """

    def __init__(
        self,
        message: str,
        tier: TenantTier,
        queue_size: int,
        retry_after_seconds: int = 5,
    ):
        super().__init__(message)
        self.tier = tier
        self.queue_size = queue_size
        self.retry_after_seconds = retry_after_seconds


class ExecutorTimeoutError(Exception):
    """
    Raised when an operation exceeds its timeout.
    """

    def __init__(
        self,
        message: str,
        tier: TenantTier,
        timeout_seconds: int,
        elapsed_seconds: float,
    ):
        super().__init__(message)
        self.tier = tier
        self.timeout_seconds = timeout_seconds
        self.elapsed_seconds = elapsed_seconds


class TierExecutor:
    """
    Thread pool executor for a single tenant tier.

    Manages:
    - Bounded thread pool
    - Backpressure via semaphore
    - Per-tier metrics
    """

    def __init__(self, tier: TenantTier, config: TierConfig):
        self.tier = tier
        self.config = config

        # Create thread pool
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_workers,
            thread_name_prefix=f"familiar-{tier.value}-",
        )

        # Semaphore for backpressure (limits queue depth)
        self._semaphore = asyncio.Semaphore(config.max_queue_size)

        # Explicit queue depth tracking (avoids accessing semaphore internals)
        self._queue_depth = 0

        # Metrics
        self._active_tasks = 0
        self._total_completed = 0
        self._total_failed = 0
        self._total_timeouts = 0
        self._total_rejected = 0
        self._latency_sum_ms = 0.0

        # Lock for metrics updates
        self._metrics_lock = asyncio.Lock()

        logger.info(
            f"TierExecutor initialized: tier={tier.value}, "
            f"workers={config.max_workers}, queue={config.max_queue_size}"
        )

    @property
    def queued_count(self) -> int:
        """Number of tasks waiting in queue."""
        # Use explicit counter instead of semaphore internals (_semaphore._value)
        # This is more stable across Python versions and async implementations
        return self._queue_depth

    async def run(
        self,
        func: Callable[..., T],
        args: tuple = (),
        kwargs: dict | None = None,
        timeout: float | None = None,
        tenant_id: UUID | None = None,
    ) -> T:
        """
        Run a function in this tier's thread pool.

        Args:
            func: Synchronous function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Override default timeout (seconds)
            tenant_id: For logging/metrics

        Returns:
            Function result

        Raises:
            QueueFullError: If queue is at capacity
            ExecutorTimeoutError: If execution exceeds timeout
            Exception: Any exception from the function
        """
        kwargs = kwargs or {}
        timeout = timeout or self.config.timeout_seconds

        # Try to acquire semaphore (provides backpressure)
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=5.0,  # Max wait to enter queue
            )
            # Track queue depth explicitly (acquired semaphore = in queue)
            async with self._metrics_lock:
                self._queue_depth += 1
        except TimeoutError as e:
            self._total_rejected += 1
            logger.warning(
                f"Queue full for tier {self.tier.value}, tenant={tenant_id}, rejecting request"
            )
            raise QueueFullError(
                f"Service busy for {self.tier.value} tier. Please retry shortly.",
                tier=self.tier,
                queue_size=self.config.max_queue_size,
                retry_after_seconds=5,
            ) from e

        # Track active tasks
        async with self._metrics_lock:
            self._active_tasks += 1

        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            # Run in thread pool with timeout
            result = await asyncio.wait_for(
                loop.run_in_executor(self._executor, lambda: func(*args, **kwargs)),
                timeout=timeout,
            )

            # Record success
            elapsed_ms = (time.time() - start_time) * 1000
            async with self._metrics_lock:
                self._total_completed += 1
                self._latency_sum_ms += elapsed_ms

            return result

        except TimeoutError as e:
            elapsed = time.time() - start_time
            async with self._metrics_lock:
                self._total_timeouts += 1

            logger.error(
                f"Timeout after {elapsed:.1f}s for tier {self.tier.value}, tenant={tenant_id}"
            )
            raise ExecutorTimeoutError(
                f"Operation timed out after {timeout}s",
                tier=self.tier,
                timeout_seconds=timeout,
                elapsed_seconds=elapsed,
            ) from e

        except Exception as e:
            async with self._metrics_lock:
                self._total_failed += 1

            logger.error(f"Error in tier {self.tier.value}, tenant={tenant_id}: {e}")
            raise

        finally:
            async with self._metrics_lock:
                self._active_tasks -= 1
                self._queue_depth -= 1  # Track queue depth explicitly
            self._semaphore.release()

    def get_metrics(self) -> ExecutorMetrics:
        """Get current metrics for this executor."""
        total_completed = self._total_completed
        avg_latency = self._latency_sum_ms / total_completed if total_completed > 0 else 0.0

        return ExecutorMetrics(
            tier=self.tier,
            max_workers=self.config.max_workers,
            max_queue=self.config.max_queue_size,
            active_tasks=self._active_tasks,
            queued_tasks=self.queued_count,
            total_completed=total_completed,
            total_failed=self._total_failed,
            total_timeouts=self._total_timeouts,
            total_rejected=self._total_rejected,
            avg_latency_ms=avg_latency,
        )

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool."""
        logger.info(f"Shutting down {self.tier.value} executor...")
        self._executor.shutdown(wait=wait)


class TenantExecutorPool:
    """
    Pool of tenant-tiered executors.

    Manages separate thread pools for each tenant tier, providing:
    - Resource isolation between tiers
    - Backpressure when overloaded
    - Timeout handling
    - Comprehensive metrics

    Usage:
        pool = TenantExecutorPool()

        # Run a sync function in tenant's tier pool
        result = await pool.run(
            tenant_id=uuid,
            tier=TenantTier.PROFESSIONAL,
            func=agent.chat,
            kwargs={"message": "Hello!"}
        )

        # Get pool metrics
        metrics = pool.get_all_metrics()

        # Shutdown
        await pool.shutdown()
    """

    def __init__(
        self,
        tier_configs: dict[TenantTier, TierConfig] | None = None,
    ):
        """
        Initialize the executor pool.

        Args:
            tier_configs: Custom configurations per tier (uses defaults if None)
        """
        self.tier_configs = tier_configs or DEFAULT_TIER_CONFIGS

        # Create executor per tier
        self._executors: dict[TenantTier, TierExecutor] = {}
        for tier, config in self.tier_configs.items():
            self._executors[tier] = TierExecutor(tier, config)

        self._shutdown = False

        logger.info(
            f"TenantExecutorPool initialized with tiers: "
            f"{', '.join(f'{t.value}={c.max_workers}w' for t, c in self.tier_configs.items())}"
        )

    def _get_executor(self, tier: TenantTier) -> TierExecutor:
        """Get executor for tier, defaulting to FREE if not found."""
        if tier not in self._executors:
            logger.warning(f"No executor for tier {tier}, using FREE")
            tier = TenantTier.FREE
        return self._executors[tier]

    async def run(
        self,
        tenant_id: UUID,
        tier: TenantTier,
        func: Callable[..., T],
        args: tuple = (),
        kwargs: dict | None = None,
        timeout: float | None = None,
    ) -> T:
        """
        Run a function in the appropriate tier's thread pool.

        Args:
            tenant_id: Tenant UUID (for logging/metrics)
            tier: Tenant's subscription tier
            func: Synchronous function to execute
            args: Positional arguments
            kwargs: Keyword arguments
            timeout: Override default timeout (seconds)

        Returns:
            Function result

        Raises:
            QueueFullError: If tier's queue is at capacity
            ExecutorTimeoutError: If execution exceeds timeout
            RuntimeError: If pool is shut down
        """
        if self._shutdown:
            raise RuntimeError("Executor pool is shut down")

        executor = self._get_executor(tier)
        return await executor.run(
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            tenant_id=tenant_id,
        )

    def get_metrics(self, tier: TenantTier) -> ExecutorMetrics:
        """Get metrics for a specific tier."""
        executor = self._get_executor(tier)
        return executor.get_metrics()

    def get_all_metrics(self) -> dict[str, ExecutorMetrics]:
        """Get metrics for all tiers."""
        return {tier.value: executor.get_metrics() for tier, executor in self._executors.items()}

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get a summary of pool metrics."""
        all_metrics = self.get_all_metrics()

        total_active = sum(m.active_tasks for m in all_metrics.values())
        total_queued = sum(m.queued_tasks for m in all_metrics.values())
        total_completed = sum(m.total_completed for m in all_metrics.values())
        total_failed = sum(m.total_failed for m in all_metrics.values())

        return {
            "total_active_tasks": total_active,
            "total_queued_tasks": total_queued,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "success_rate": (
                total_completed / (total_completed + total_failed)
                if (total_completed + total_failed) > 0
                else 1.0
            ),
            "by_tier": {name: metrics.to_dict() for name, metrics in all_metrics.items()},
        }

    async def shutdown(self, wait: bool = True):
        """
        Shutdown all executors.

        Args:
            wait: If True, wait for pending tasks to complete
        """
        if self._shutdown:
            return

        self._shutdown = True
        logger.info("Shutting down TenantExecutorPool...")

        for _tier, executor in self._executors.items():
            executor.shutdown(wait=wait)

        logger.info("TenantExecutorPool shut down complete")


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_executor_pool: TenantExecutorPool | None = None


def get_executor_pool() -> TenantExecutorPool:
    """
    Get or create the global executor pool.

    Returns:
        The singleton TenantExecutorPool instance
    """
    global _executor_pool
    if _executor_pool is None:
        _executor_pool = TenantExecutorPool()
    return _executor_pool


async def shutdown_executor_pool():
    """Shutdown the global executor pool."""
    global _executor_pool
    if _executor_pool is not None:
        await _executor_pool.shutdown()
        _executor_pool = None


def reset_executor_pool():
    """Reset the global executor pool (for testing)."""
    global _executor_pool
    if _executor_pool is not None:
        # Don't await - just mark as None for sync reset
        _executor_pool._shutdown = True
    _executor_pool = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TenantTier",
    "TierConfig",
    "ExecutorMetrics",
    "QueueFullError",
    "ExecutorTimeoutError",
    "TierExecutor",
    "TenantExecutorPool",
    "get_executor_pool",
    "shutdown_executor_pool",
    "reset_executor_pool",
    "DEFAULT_TIER_CONFIGS",
]
