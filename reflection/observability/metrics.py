# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 7: Observability - Metrics Module

Prometheus-compatible metrics for:
- Request latency and throughput
- LLM provider performance
- Token usage
- Quota utilization
- Error rates
- System resources

Uses prometheus_client library for proper metric types and exposition.
"""

import asyncio
import logging
import time
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Try to import prometheus_client
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning(
        "prometheus_client not installed. Metrics will use fallback. "
        "Install with: pip install prometheus-client"
    )


# ============================================================
# METRIC DEFINITIONS
# ============================================================

# Default buckets for latency histograms (in seconds)
LATENCY_BUCKETS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
)

# Token count buckets
TOKEN_BUCKETS = (
    10,
    50,
    100,
    250,
    500,
    1000,
    2500,
    5000,
    10000,
    25000,
    50000,
    100000,
)


class MetricsRegistry:
    """
    Central metrics registry for Reflection.

    Provides Prometheus-compatible metrics with graceful fallback
    when prometheus_client is not installed.
    """

    def __init__(self, namespace: str = "familiar"):
        self.namespace = namespace
        self._initialized = False
        self._fallback_counters: dict[str, float] = {}
        self._fallback_gauges: dict[str, float] = {}

        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

        self._initialized = True

    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics."""
        ns = self.namespace

        # ============================================================
        # HTTP REQUEST METRICS
        # ============================================================

        self.http_requests_total = Counter(
            f"{ns}_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
        )

        self.http_request_duration_seconds = Histogram(
            f"{ns}_http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            buckets=LATENCY_BUCKETS,
        )

        self.http_requests_in_progress = Gauge(
            f"{ns}_http_requests_in_progress",
            "HTTP requests currently in progress",
            ["method", "endpoint"],
        )

        # ============================================================
        # CHAT/LLM METRICS
        # ============================================================

        self.chat_requests_total = Counter(
            f"{ns}_chat_requests_total",
            "Total chat requests",
            ["tenant_tier", "provider", "model", "path", "status"],
        )

        self.chat_duration_seconds = Histogram(
            f"{ns}_chat_duration_seconds",
            "Chat request duration in seconds",
            ["tenant_tier", "provider", "path"],
            buckets=LATENCY_BUCKETS,
        )

        self.chat_tokens_input = Histogram(
            f"{ns}_chat_tokens_input",
            "Input tokens per chat request",
            ["tenant_tier", "provider", "model"],
            buckets=TOKEN_BUCKETS,
        )

        self.chat_tokens_output = Histogram(
            f"{ns}_chat_tokens_output",
            "Output tokens per chat request",
            ["tenant_tier", "provider", "model"],
            buckets=TOKEN_BUCKETS,
        )

        self.chat_tokens_total = Counter(
            f"{ns}_chat_tokens_total",
            "Total tokens consumed",
            ["tenant_tier", "provider", "model", "direction"],
        )

        self.chat_cost_usd_total = Counter(
            f"{ns}_chat_cost_usd_total",
            "Total cost in USD",
            ["tenant_tier", "provider", "model"],
        )

        self.chat_streaming_chunks = Counter(
            f"{ns}_chat_streaming_chunks_total",
            "Total streaming chunks sent",
            ["tenant_tier", "provider"],
        )

        # ============================================================
        # PROVIDER METRICS
        # ============================================================

        self.provider_requests_total = Counter(
            f"{ns}_provider_requests_total",
            "Total requests to LLM providers",
            ["provider", "model", "status"],
        )

        self.provider_duration_seconds = Histogram(
            f"{ns}_provider_duration_seconds",
            "LLM provider request duration",
            ["provider", "model"],
            buckets=LATENCY_BUCKETS,
        )

        self.provider_errors_total = Counter(
            f"{ns}_provider_errors_total",
            "Total LLM provider errors",
            ["provider", "error_type"],
        )

        self.provider_rate_limits_total = Counter(
            f"{ns}_provider_rate_limits_total",
            "Total rate limit hits from providers",
            ["provider"],
        )

        # ============================================================
        # EXECUTOR POOL METRICS
        # ============================================================

        self.executor_tasks_total = Counter(
            f"{ns}_executor_tasks_total",
            "Total executor tasks",
            ["tier", "status"],
        )

        self.executor_active_tasks = Gauge(
            f"{ns}_executor_active_tasks",
            "Currently active executor tasks",
            ["tier"],
        )

        self.executor_queued_tasks = Gauge(
            f"{ns}_executor_queued_tasks",
            "Currently queued executor tasks",
            ["tier"],
        )

        self.executor_task_duration_seconds = Histogram(
            f"{ns}_executor_task_duration_seconds",
            "Executor task duration",
            ["tier"],
            buckets=LATENCY_BUCKETS,
        )

        self.executor_queue_wait_seconds = Histogram(
            f"{ns}_executor_queue_wait_seconds",
            "Time spent waiting in executor queue",
            ["tier"],
            buckets=LATENCY_BUCKETS,
        )

        # ============================================================
        # QUOTA METRICS
        # ============================================================

        self.quota_checks_total = Counter(
            f"{ns}_quota_checks_total",
            "Total quota checks",
            ["tier", "quota_type", "result"],
        )

        self.quota_exceeded_total = Counter(
            f"{ns}_quota_exceeded_total",
            "Total quota exceeded events",
            ["tier", "quota_type"],
        )

        self.quota_usage_ratio = Gauge(
            f"{ns}_quota_usage_ratio",
            "Current quota usage ratio (0-1)",
            ["tier", "tenant_id", "quota_type"],
        )

        # ============================================================
        # DATABASE METRICS
        # ============================================================

        self.db_queries_total = Counter(
            f"{ns}_db_queries_total",
            "Total database queries",
            ["operation", "table"],
        )

        self.db_query_duration_seconds = Histogram(
            f"{ns}_db_query_duration_seconds",
            "Database query duration",
            ["operation", "table"],
            buckets=LATENCY_BUCKETS,
        )

        self.db_connections_active = Gauge(
            f"{ns}_db_connections_active",
            "Active database connections",
        )

        self.db_connections_idle = Gauge(
            f"{ns}_db_connections_idle",
            "Idle database connections",
        )

        # ============================================================
        # REDIS METRICS
        # ============================================================

        self.redis_commands_total = Counter(
            f"{ns}_redis_commands_total",
            "Total Redis commands",
            ["command"],
        )

        self.redis_command_duration_seconds = Histogram(
            f"{ns}_redis_command_duration_seconds",
            "Redis command duration",
            ["command"],
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
        )

        # ============================================================
        # AUTHENTICATION METRICS
        # ============================================================

        self.auth_attempts_total = Counter(
            f"{ns}_auth_attempts_total",
            "Total authentication attempts",
            ["method", "result"],
        )

        self.auth_tokens_issued_total = Counter(
            f"{ns}_auth_tokens_issued_total",
            "Total tokens issued",
            ["token_type"],
        )

        # ============================================================
        # TOOL EXECUTION METRICS
        # ============================================================

        self.tool_executions_total = Counter(
            f"{ns}_tool_executions_total",
            "Total tool executions",
            ["tier", "tool_name", "status"],
        )

        self.tool_duration_seconds = Histogram(
            f"{ns}_tool_duration_seconds",
            "Tool execution duration",
            ["tool_name"],
            buckets=LATENCY_BUCKETS,
        )

        # ============================================================
        # SYSTEM INFO
        # ============================================================

        self.app_info = Info(
            f"{ns}_app",
            "Application information",
        )

    def set_app_info(self, version: str, environment: str, **kwargs):
        """Set application info."""
        if PROMETHEUS_AVAILABLE:
            self.app_info.info(
                {
                    "version": version,
                    "environment": environment,
                    **kwargs,
                }
            )

    # ============================================================
    # RECORDING METHODS
    # ============================================================

    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float,
    ):
        """Record an HTTP request."""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=str(status_code),
            ).inc()

            self.http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration_seconds)
        else:
            key = f"http_requests:{method}:{endpoint}:{status_code}"
            self._fallback_counters[key] = self._fallback_counters.get(key, 0) + 1

    def inc_requests_in_progress(self, method: str, endpoint: str):
        """Increment in-progress requests gauge."""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

    def dec_requests_in_progress(self, method: str, endpoint: str):
        """Decrement in-progress requests gauge."""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

    def record_chat_request(
        self,
        tenant_tier: str,
        provider: str,
        model: str,
        path: str,
        status: str,
        duration_seconds: float,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost_usd: float = 0.0,
    ):
        """Record a chat request with all metrics."""
        if PROMETHEUS_AVAILABLE:
            self.chat_requests_total.labels(
                tenant_tier=tenant_tier,
                provider=provider,
                model=model,
                path=path,
                status=status,
            ).inc()

            self.chat_duration_seconds.labels(
                tenant_tier=tenant_tier,
                provider=provider,
                path=path,
            ).observe(duration_seconds)

            if tokens_input > 0:
                self.chat_tokens_input.labels(
                    tenant_tier=tenant_tier,
                    provider=provider,
                    model=model,
                ).observe(tokens_input)

                self.chat_tokens_total.labels(
                    tenant_tier=tenant_tier,
                    provider=provider,
                    model=model,
                    direction="input",
                ).inc(tokens_input)

            if tokens_output > 0:
                self.chat_tokens_output.labels(
                    tenant_tier=tenant_tier,
                    provider=provider,
                    model=model,
                ).observe(tokens_output)

                self.chat_tokens_total.labels(
                    tenant_tier=tenant_tier,
                    provider=provider,
                    model=model,
                    direction="output",
                ).inc(tokens_output)

            if cost_usd > 0:
                self.chat_cost_usd_total.labels(
                    tenant_tier=tenant_tier,
                    provider=provider,
                    model=model,
                ).inc(cost_usd)

    def record_streaming_chunk(self, tenant_tier: str, provider: str):
        """Record a streaming chunk."""
        if PROMETHEUS_AVAILABLE:
            self.chat_streaming_chunks.labels(
                tenant_tier=tenant_tier,
                provider=provider,
            ).inc()

    def record_provider_request(
        self,
        provider: str,
        model: str,
        status: str,
        duration_seconds: float,
    ):
        """Record an LLM provider request."""
        if PROMETHEUS_AVAILABLE:
            self.provider_requests_total.labels(
                provider=provider,
                model=model,
                status=status,
            ).inc()

            self.provider_duration_seconds.labels(
                provider=provider,
                model=model,
            ).observe(duration_seconds)

    def record_provider_error(self, provider: str, error_type: str):
        """Record an LLM provider error."""
        if PROMETHEUS_AVAILABLE:
            self.provider_errors_total.labels(
                provider=provider,
                error_type=error_type,
            ).inc()

    def record_provider_rate_limit(self, provider: str):
        """Record a provider rate limit hit."""
        if PROMETHEUS_AVAILABLE:
            self.provider_rate_limits_total.labels(provider=provider).inc()

    def record_executor_task(
        self,
        tier: str,
        status: str,
        duration_seconds: float,
        queue_wait_seconds: float = 0.0,
    ):
        """Record an executor task."""
        if PROMETHEUS_AVAILABLE:
            self.executor_tasks_total.labels(
                tier=tier,
                status=status,
            ).inc()

            self.executor_task_duration_seconds.labels(
                tier=tier,
            ).observe(duration_seconds)

            if queue_wait_seconds > 0:
                self.executor_queue_wait_seconds.labels(
                    tier=tier,
                ).observe(queue_wait_seconds)

    def set_executor_gauges(self, tier: str, active: int, queued: int):
        """Set executor pool gauges."""
        if PROMETHEUS_AVAILABLE:
            self.executor_active_tasks.labels(tier=tier).set(active)
            self.executor_queued_tasks.labels(tier=tier).set(queued)

    def record_quota_check(self, tier: str, quota_type: str, result: str):
        """Record a quota check."""
        if PROMETHEUS_AVAILABLE:
            self.quota_checks_total.labels(
                tier=tier,
                quota_type=quota_type,
                result=result,
            ).inc()

    def record_quota_exceeded(self, tier: str, quota_type: str):
        """Record a quota exceeded event."""
        if PROMETHEUS_AVAILABLE:
            self.quota_exceeded_total.labels(
                tier=tier,
                quota_type=quota_type,
            ).inc()

    def set_quota_usage(self, tier: str, tenant_id: str, quota_type: str, ratio: float):
        """Set quota usage ratio gauge."""
        if PROMETHEUS_AVAILABLE:
            self.quota_usage_ratio.labels(
                tier=tier,
                tenant_id=tenant_id,
                quota_type=quota_type,
            ).set(ratio)

    def record_auth_attempt(self, method: str, result: str):
        """Record an authentication attempt."""
        if PROMETHEUS_AVAILABLE:
            self.auth_attempts_total.labels(method=method, result=result).inc()

    def record_token_issued(self, token_type: str):
        """Record a token issuance."""
        if PROMETHEUS_AVAILABLE:
            self.auth_tokens_issued_total.labels(token_type=token_type).inc()

    def record_db_query(self, operation: str, table: str, duration_seconds: float):
        """Record a database query."""
        if PROMETHEUS_AVAILABLE:
            self.db_queries_total.labels(operation=operation, table=table).inc()
            self.db_query_duration_seconds.labels(operation=operation, table=table).observe(
                duration_seconds
            )

    def set_db_connections(self, active: int, idle: int):
        """Set database connection gauges."""
        if PROMETHEUS_AVAILABLE:
            self.db_connections_active.set(active)
            self.db_connections_idle.set(idle)

    def record_redis_command(self, command: str, duration_seconds: float):
        """Record a Redis command."""
        if PROMETHEUS_AVAILABLE:
            self.redis_commands_total.labels(command=command).inc()
            self.redis_command_duration_seconds.labels(command=command).observe(duration_seconds)

    def record_tool_execution(
        self,
        tier: str,
        tool_name: str,
        status: str,
        duration_seconds: float,
    ):
        """Record a tool execution."""
        if PROMETHEUS_AVAILABLE:
            self.tool_executions_total.labels(
                tier=tier,
                tool_name=tool_name,
                status=status,
            ).inc()

            self.tool_duration_seconds.labels(tool_name=tool_name).observe(duration_seconds)

    def generate_latest(self) -> bytes:
        """Generate Prometheus exposition format."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(REGISTRY)
        return b"# Prometheus metrics not available\n"

    def content_type(self) -> str:
        """Get Prometheus content type."""
        if PROMETHEUS_AVAILABLE:
            return CONTENT_TYPE_LATEST
        return "text/plain; charset=utf-8"


# ============================================================
# GLOBAL METRICS INSTANCE
# ============================================================

_metrics: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


def init_metrics(
    namespace: str = "familiar",
    app_version: str = "0.0.0",
    environment: str = "development",
):
    """Initialize the global metrics registry."""
    global _metrics
    _metrics = MetricsRegistry(namespace=namespace)
    _metrics.set_app_info(version=app_version, environment=environment)
    logger.info(f"Metrics initialized (prometheus_available={PROMETHEUS_AVAILABLE})")


# ============================================================
# DECORATOR HELPERS
# ============================================================


def timed(metric_name: str = "duration"):
    """Decorator to time function execution and log it."""

    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start
                logger.debug(f"{metric_name}: {func.__name__} took {duration:.3f}s")

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start
                logger.debug(f"{metric_name}: {func.__name__} took {duration:.3f}s")

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "MetricsRegistry",
    "get_metrics",
    "init_metrics",
    "timed",
    "PROMETHEUS_AVAILABLE",
    "LATENCY_BUCKETS",
    "TOKEN_BUCKETS",
]
