# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Async Infrastructure Primitives

Provides:
- Lifecycle management
- Timeout contexts
- Resource pools
- Graceful shutdown
"""

import asyncio
import logging
import signal
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Generic, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def utcnow() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


# ============================================================
# TIMEOUT UTILITIES
# ============================================================


@asynccontextmanager
async def timeout_context(seconds: float, operation: str = "operation") -> AsyncIterator[None]:
    """
    Async context manager with timeout.

    Usage:
        async with timeout_context(30, "LLM call"):
            response = await provider.chat(...)
    """
    try:
        async with asyncio.timeout(seconds):
            yield
    except TimeoutError:
        logger.error(f"{operation} timed out after {seconds}s")
        raise


async def with_timeout(
    coro: Coroutine[Any, Any, T], seconds: float, default: T | None = None
) -> T | None:
    """
    Execute coroutine with timeout, return default on timeout.

    Usage:
        result = await with_timeout(slow_operation(), 5.0, default="fallback")
    """
    try:
        return await asyncio.wait_for(coro, timeout=seconds)
    except TimeoutError:
        return default


# ============================================================
# ASYNC INIT MIXIN
# ============================================================


class AsyncInitMixin:
    """
    Mixin for classes requiring async initialization.

    Usage:
        class MyService(AsyncInitMixin):
            async def _async_init(self):
                self.conn = await create_connection()

        service = await MyService.create(config)
    """

    async def _async_init(self) -> None:
        """Override for async initialization logic."""
        pass

    async def _async_cleanup(self) -> None:
        """Override for async cleanup logic."""
        pass

    @classmethod
    async def create(cls, *args, **kwargs) -> "AsyncInitMixin":
        """Factory method for async initialization."""
        instance = cls(*args, **kwargs)
        await instance._async_init()
        return instance


# ============================================================
# LIFECYCLE MANAGEMENT
# ============================================================


@dataclass
class Lifecycle:
    """
    Application lifecycle manager.

    Manages startup/shutdown hooks and signal handling.

    Usage:
        lifecycle = Lifecycle()

        @lifecycle.on_startup
        async def init_db():
            await db.connect()

        @lifecycle.on_shutdown
        async def close_db():
            await db.disconnect()

        await lifecycle.run(main())
    """

    _startup_hooks: list[Callable[[], Coroutine]] = field(default_factory=list)
    _shutdown_hooks: list[Callable[[], Coroutine]] = field(default_factory=list)
    _running: bool = False
    _shutdown_event: asyncio.Event | None = None

    def on_startup(self, func: Callable[[], Coroutine]) -> Callable[[], Coroutine]:
        """Decorator to register startup hook."""
        self._startup_hooks.append(func)
        return func

    def on_shutdown(self, func: Callable[[], Coroutine]) -> Callable[[], Coroutine]:
        """Decorator to register shutdown hook."""
        self._shutdown_hooks.append(func)
        return func

    async def startup(self) -> None:
        """Execute all startup hooks."""
        logger.info("Starting application...")
        for hook in self._startup_hooks:
            try:
                await hook()
            except Exception as e:
                logger.error(f"Startup hook {hook.__name__} failed: {e}")
                raise
        self._running = True
        logger.info("Application started")

    async def shutdown(self) -> None:
        """Execute all shutdown hooks in reverse order."""
        if not self._running:
            return

        logger.info("Shutting down application...")
        self._running = False

        for hook in reversed(self._shutdown_hooks):
            try:
                await hook()
            except Exception as e:
                logger.error(f"Shutdown hook {hook.__name__} failed: {e}")

        logger.info("Application shut down")

    async def run(self, main: Coroutine) -> None:
        """
        Run main coroutine with lifecycle management.

        Sets up signal handlers for graceful shutdown.
        """
        self._shutdown_event = asyncio.Event()
        loop = asyncio.get_event_loop()

        # Setup signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self._signal_handler(s)))

        try:
            await self.startup()
            await main
        finally:
            await self.shutdown()

    async def _signal_handler(self, sig: signal.Signals) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig.name}")
        if self._shutdown_event:
            self._shutdown_event.set()

    @property
    def is_running(self) -> bool:
        return self._running

    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        if self._shutdown_event:
            await self._shutdown_event.wait()


# ============================================================
# RESOURCE POOL
# ============================================================

# Type variable for resource pool
PoolT = TypeVar("PoolT")


class AsyncPool(Generic[PoolT]):
    """
    Generic async resource pool.

    Usage:
        pool = AsyncPool(create_connection, max_size=10)

        async with pool.acquire() as conn:
            await conn.execute(...)
    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, PoolT]],
        max_size: int = 10,
        cleanup: Callable[[PoolT], Coroutine[Any, Any, None]] | None = None,
    ):
        self._factory = factory
        self._cleanup = cleanup
        self._max_size = max_size
        self._pool: asyncio.Queue[PoolT] = asyncio.Queue(maxsize=max_size)
        self._created: int = 0
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[PoolT]:
        """Acquire a resource from the pool."""
        resource = await self._get()
        try:
            yield resource
        finally:
            await self._return(resource)

    async def _get(self) -> PoolT:
        """Get a resource, creating if necessary."""
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            pass

        async with self._lock:
            if self._created < self._max_size:
                resource = await self._factory()
                self._created += 1
                return resource

        # Wait for available resource
        return await self._pool.get()

    async def _return(self, resource: PoolT) -> None:
        """Return a resource to the pool."""
        try:
            self._pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool full, cleanup resource
            if self._cleanup:
                await self._cleanup(resource)
            async with self._lock:
                self._created -= 1

    async def close(self) -> None:
        """Close all pooled resources."""
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                if self._cleanup:
                    await self._cleanup(resource)
            except asyncio.QueueEmpty:
                break
        self._created = 0


# ============================================================
# BACKGROUND TASKS
# ============================================================


class BackgroundTasks:
    """
    Manager for background async tasks.

    Tracks running tasks and ensures cleanup on shutdown.
    """

    def __init__(self):
        self._tasks: set[asyncio.Task] = set()

    def create_task(self, coro: Coroutine, name: str | None = None) -> asyncio.Task:
        """Create and track a background task."""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def cancel_all(self, timeout: float = 5.0) -> None:
        """Cancel all running tasks."""
        for task in self._tasks:
            task.cancel()

        if self._tasks:
            await asyncio.wait(self._tasks, timeout=timeout)

    @property
    def count(self) -> int:
        return len(self._tasks)


# ============================================================
# RATE LIMITER
# ============================================================


class AsyncRateLimiter:
    """
    Token bucket rate limiter.

    Usage:
        limiter = AsyncRateLimiter(rate=10, per=1.0)  # 10 per second

        async with limiter:
            await make_request()
    """

    def __init__(self, rate: float, per: float = 1.0):
        self.rate = rate
        self.per = per
        self._tokens = rate
        self._last_update = asyncio.get_event_loop().time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_update
            self._tokens = min(self.rate, self._tokens + elapsed * (self.rate / self.per))
            self._last_update = now

            if self._tokens < 1:
                wait_time = (1 - self._tokens) * (self.per / self.rate)
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *args):
        pass


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "utcnow",
    "timeout_context",
    "with_timeout",
    "AsyncInitMixin",
    "Lifecycle",
    "AsyncPool",
    "BackgroundTasks",
    "AsyncRateLimiter",
]
