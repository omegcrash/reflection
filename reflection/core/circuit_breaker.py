# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 3: Circuit Breaker for LLM Providers

Implements the circuit breaker pattern to prevent cascading failures
when LLM providers are experiencing issues.

States:
- CLOSED: Normal operation, requests pass through
- OPEN: Provider is failing, requests fail fast
- HALF_OPEN: Testing if provider has recovered

Benefits:
- Fail fast when provider is down (no waiting for timeouts)
- Automatic recovery detection
- Tenant-aware (isolates failures per-tenant if needed)
- Observable via metrics and events
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(StrEnum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    # Failure threshold to trip the circuit
    failure_threshold: int = 5

    # Consecutive successes needed to close from half-open
    success_threshold: int = 2

    # How long to wait before testing (half-open)
    reset_timeout_seconds: float = 30.0

    # Time window for failure counting (rolling window)
    failure_window_seconds: float = 60.0

    # Maximum concurrent calls in half-open state
    half_open_max_calls: int = 1

    # Exceptions that should trip the circuit
    trip_on_exceptions: tuple = (
        Exception,  # Default: trip on any exception
    )

    # Exceptions to ignore (don't count as failures)
    ignore_exceptions: tuple = ()


@dataclass
class CircuitBreakerStats:
    """Statistics for circuit breaker monitoring."""

    state: CircuitState = CircuitState.CLOSED
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected while open
    last_failure_time: datetime | None = None
    last_success_time: datetime | None = None
    last_state_change: datetime | None = None
    consecutive_failures: int = 0
    consecutive_successes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "rejected_calls": self.rejected_calls,
            "last_failure_time": self.last_failure_time.isoformat()
            if self.last_failure_time
            else None,
            "last_success_time": self.last_success_time.isoformat()
            if self.last_success_time
            else None,
            "last_state_change": self.last_state_change.isoformat()
            if self.last_state_change
            else None,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
        }


class CircuitOpenError(Exception):
    """Raised when circuit is open and call is rejected."""

    def __init__(
        self,
        circuit_name: str,
        retry_after_seconds: float,
        message: str | None = None,
    ):
        self.circuit_name = circuit_name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            message or f"Circuit '{circuit_name}' is open. Retry after {retry_after_seconds:.1f}s"
        )


class CircuitBreaker:
    """
    Circuit breaker for protecting against cascading failures.

    Usage:
        # Create breaker
        breaker = CircuitBreaker("anthropic", config=CircuitBreakerConfig(
            failure_threshold=5,
            reset_timeout_seconds=30.0,
        ))

        # Use as context manager
        async with breaker:
            response = await provider.chat(...)

        # Or use decorator
        @breaker.protect
        async def call_provider():
            return await provider.chat(...)

    States:
        CLOSED -> OPEN: When failure_threshold exceeded
        OPEN -> HALF_OPEN: After reset_timeout_seconds
        HALF_OPEN -> CLOSED: When success_threshold reached
        HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[["CircuitBreaker", CircuitState, CircuitState], None]
        | None = None,
    ):
        """
        Initialize circuit breaker.

        Args:
            name: Identifier for this circuit (e.g., "anthropic", "openai")
            config: Configuration options
            on_state_change: Callback when state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.on_state_change = on_state_change

        # State
        self._state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

        # Tracking
        self._failure_times: list[float] = []  # Timestamps of recent failures
        self._consecutive_failures = 0
        self._consecutive_successes = 0
        self._half_open_calls = 0

        # Stats
        self.stats = CircuitBreakerStats()
        self._opened_at: float | None = None

    @property
    def state(self) -> CircuitState:
        """Get current state, checking for automatic transitions."""
        # State is checked; actual transitions happen in _check_state
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        return self.state == CircuitState.HALF_OPEN

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        if old_state == new_state:
            return

        self._state = new_state
        self.stats.state = new_state
        self.stats.last_state_change = datetime.now(UTC)

        logger.info(
            f"Circuit '{self.name}' state change: {old_state.value} -> {new_state.value}",
            extra={
                "circuit_name": self.name,
                "old_state": old_state.value,
                "new_state": new_state.value,
            },
        )

        # Reset counters based on new state
        if new_state == CircuitState.CLOSED:
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._failure_times.clear()
            self._opened_at = None
        elif new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._consecutive_successes = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._consecutive_successes = 0
            self._half_open_calls = 0

        # Callback
        if self.on_state_change:
            try:
                self.on_state_change(self, old_state, new_state)
            except Exception as e:
                logger.warning(f"Circuit state change callback failed: {e}")

    async def _check_state(self) -> None:
        """Check and possibly transition state."""
        async with self._lock:
            if (
                self._state == CircuitState.OPEN
                and self._opened_at
                and (time.time() - self._opened_at) >= self.config.reset_timeout_seconds
            ):
                self._transition_to(CircuitState.HALF_OPEN)

    def _record_success(self) -> None:
        """Record a successful call."""
        self.stats.total_calls += 1
        self.stats.successful_calls += 1
        self.stats.last_success_time = datetime.now(UTC)
        self._consecutive_failures = 0
        self._consecutive_successes += 1
        self.stats.consecutive_failures = 0
        self.stats.consecutive_successes = self._consecutive_successes

        # In half-open, check if we should close
        if (
            self._state == CircuitState.HALF_OPEN
            and self._consecutive_successes >= self.config.success_threshold
        ):
            self._transition_to(CircuitState.CLOSED)

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        now = time.time()

        self.stats.total_calls += 1
        self.stats.failed_calls += 1
        self.stats.last_failure_time = datetime.now(UTC)
        self._consecutive_successes = 0
        self._consecutive_failures += 1
        self.stats.consecutive_successes = 0
        self.stats.consecutive_failures = self._consecutive_failures

        # Add to rolling window
        self._failure_times.append(now)

        # Clean old failures outside window
        cutoff = now - self.config.failure_window_seconds
        self._failure_times = [t for t in self._failure_times if t > cutoff]

        # Check if we should open the circuit
        if self._state == CircuitState.CLOSED:
            if len(self._failure_times) >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

    def _should_allow_call(self) -> bool:
        """Check if a call should be allowed."""
        if self._state == CircuitState.CLOSED:
            return True
        elif self._state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open
            return self._half_open_calls < self.config.half_open_max_calls
        else:  # OPEN
            return False

    def _should_trip(self, error: Exception) -> bool:
        """Check if this exception should trip the circuit."""
        # First check ignore list
        if isinstance(error, self.config.ignore_exceptions):
            return False
        # Then check trip list
        return isinstance(error, self.config.trip_on_exceptions)

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        await self.before_call()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        if exc_val is None:
            await self.on_success()
        elif self._should_trip(exc_val):
            await self.on_failure(exc_val)
        # Don't suppress the exception
        return False

    async def before_call(self) -> None:
        """Called before making a protected call."""
        await self._check_state()

        if not self._should_allow_call():
            self.stats.rejected_calls += 1
            retry_after = 0.0
            if self._opened_at:
                retry_after = max(
                    0.0, self.config.reset_timeout_seconds - (time.time() - self._opened_at)
                )
            raise CircuitOpenError(
                circuit_name=self.name,
                retry_after_seconds=retry_after,
            )

        if self._state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1

    async def on_success(self) -> None:
        """Called after a successful call."""
        async with self._lock:
            self._record_success()

    async def on_failure(self, error: Exception) -> None:
        """Called after a failed call."""
        async with self._lock:
            self._record_failure(error)

    async def call(
        self,
        func: Callable[..., Awaitable[T]],
        *args,
        **kwargs,
    ) -> T:
        """
        Execute a function with circuit breaker protection.

        Args:
            func: Async function to call
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func

        Returns:
            Result of func

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Whatever func raises (after recording failure)
        """
        await self.before_call()

        try:
            result = await func(*args, **kwargs)
            await self.on_success()
            return result
        except Exception as e:
            if self._should_trip(e):
                await self.on_failure(e)
            raise

    def protect(self, func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        """
        Decorator to protect an async function with this circuit breaker.

        Usage:
            @breaker.protect
            async def call_llm():
                return await provider.chat(...)
        """

        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await self.call(func, *args, **kwargs)

        return wrapper

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics."""
        return self.stats.to_dict()

    async def reset(self) -> None:
        """Manually reset the circuit to closed state."""
        async with self._lock:
            self._transition_to(CircuitState.CLOSED)
            self._failure_times.clear()
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            self._opened_at = None


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers.

    Provides:
    - Provider-level breakers (e.g., "anthropic", "openai")
    - Tenant-level breakers (optional isolation)
    - Centralized stats collection
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()
        self.default_config = default_config or CircuitBreakerConfig()

    async def get_breaker(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name."""
        async with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(
                    name=name,
                    config=config or self.default_config,
                    on_state_change=self._on_state_change,
                )
            return self._breakers[name]

    def _on_state_change(
        self,
        breaker: CircuitBreaker,
        old_state: CircuitState,
        new_state: CircuitState,
    ) -> None:
        """Callback for state changes (for metrics/alerts)."""
        # Could emit metrics here
        pass

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    async def reset_all(self) -> None:
        """Reset all circuit breakers."""
        async with self._lock:
            for breaker in self._breakers.values():
                await breaker.reset()


# Global registry instance
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry(
            default_config=CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=2,
                reset_timeout_seconds=30.0,
                failure_window_seconds=60.0,
            )
        )
    return _registry


async def get_provider_circuit_breaker(provider_name: str) -> CircuitBreaker:
    """Get the circuit breaker for a specific provider."""
    registry = get_circuit_breaker_registry()
    return await registry.get_breaker(f"provider:{provider_name}")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "get_provider_circuit_breaker",
]
