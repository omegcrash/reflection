# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Health Check and Metrics Endpoints (v1.4.0)

Kubernetes-compatible health and readiness probes,
plus Prometheus-compatible metrics with tenant labels.

Endpoints:
- /health/live - Liveness probe (is the app running?)
- /health/ready - Readiness probe (can the app serve traffic?)
- /health - Legacy combined health check
- /metrics/prometheus - Prometheus-format metrics with tenant labels

v1.4.0: Enhanced observability with tenant-labeled metrics.
"""

import logging
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel

from ..core.executor import get_executor_pool
from ..core.settings import get_settings
from ..data.postgres import get_database
from ..data.redis import get_redis
from ..observability.metrics import get_metrics

logger = logging.getLogger(__name__)

router = APIRouter()
settings = get_settings()

# Track startup time for uptime calculation
_startup_time = time.time()


# ============================================================
# RESPONSE MODELS
# ============================================================


class ComponentHealth(BaseModel):
    """Health status of a single component."""

    status: str  # healthy, degraded, unhealthy
    latency_ms: float | None = None
    error: str | None = None
    details: dict[str, Any] = {}


class LivenessResponse(BaseModel):
    """Liveness probe response."""

    status: str
    timestamp: str


class ReadinessResponse(BaseModel):
    """Readiness probe response."""

    status: str
    version: str
    environment: str
    uptime_seconds: float
    checks: dict[str, ComponentHealth]


class HealthResponse(BaseModel):
    """Combined health check response (legacy)."""

    status: str
    version: str
    environment: str
    checks: dict[str, Any] = {}


class MetricsResponse(BaseModel):
    """Metrics response."""

    executor_pool: dict[str, Any]
    database: dict[str, Any] = {}
    redis: dict[str, Any] = {}


# ============================================================
# LIVENESS PROBE
# ============================================================


@router.get("/health/live", response_model=LivenessResponse)
async def liveness_check():
    """
    Kubernetes liveness probe.

    Returns 200 if the application process is running.
    Does NOT check dependencies - use /health/ready for that.

    Use for:
    - Detecting stuck/deadlocked processes
    - Triggering container restart on failure

    Kubernetes config:
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 15
          failureThreshold: 3
    """
    return LivenessResponse(
        status="alive",
        timestamp=datetime.now(UTC).isoformat(),
    )


# ============================================================
# READINESS PROBE
# ============================================================


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness_check(
    response: Response,
    include_details: bool = Query(default=False, description="Include detailed check info"),
):
    """
    Kubernetes readiness probe.

    Returns 200 only if the application can serve traffic:
    - Database is connected and responsive
    - Redis is connected and responsive
    - Executor pool is not overloaded

    Use for:
    - Removing pod from service during startup
    - Removing pod during dependency failures
    - Load balancer health checks

    Kubernetes config:
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
          failureThreshold: 3
    """
    checks: dict[str, ComponentHealth] = {}
    all_healthy = True

    # Check database
    db_health = await _check_database(include_details)
    checks["database"] = db_health
    if db_health.status == "unhealthy":
        all_healthy = False

    # Check Redis
    redis_health = await _check_redis(include_details)
    checks["redis"] = redis_health
    if redis_health.status == "unhealthy":
        all_healthy = False

    # Check executor pool
    executor_health = _check_executor_pool(include_details)
    checks["executor"] = executor_health
    if executor_health.status == "unhealthy":
        all_healthy = False

    # Calculate uptime
    uptime = time.time() - _startup_time

    # Determine overall status
    has_degraded = any(c.status == "degraded" for c in checks.values())

    if not all_healthy:
        status = "not_ready"
        response.status_code = 503
    elif has_degraded:
        status = "degraded"
        # Still return 200 for degraded - pod can serve traffic
    else:
        status = "ready"

    return ReadinessResponse(
        status=status,
        version=settings.app_version,
        environment=settings.environment,
        uptime_seconds=round(uptime, 2),
        checks=checks,
    )


async def _check_database(include_details: bool = False) -> ComponentHealth:
    """Check database health."""
    start = time.time()
    try:
        db = get_database()
        health = await db.health_check()
        latency = (time.time() - start) * 1000

        status = health.get("status", "unknown")
        details = health if include_details else {}

        return ComponentHealth(
            status=status,
            latency_ms=round(latency, 2),
            details=details,
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Database health check failed: {e}")
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


async def _check_redis(include_details: bool = False) -> ComponentHealth:
    """Check Redis health."""
    start = time.time()
    try:
        redis = get_redis()
        health = await redis.health_check()
        latency = (time.time() - start) * 1000

        status = health.get("status", "unknown")
        details = health if include_details else {}

        return ComponentHealth(
            status=status,
            latency_ms=round(latency, 2),
            details=details,
        )
    except Exception as e:
        latency = (time.time() - start) * 1000
        logger.error(f"Redis health check failed: {e}")
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round(latency, 2),
            error=str(e),
        )


def _check_executor_pool(include_details: bool = False) -> ComponentHealth:
    """Check executor pool health."""
    try:
        pool = get_executor_pool()
        metrics = pool.get_metrics_summary()

        # Check if any tier is severely overloaded (>90% queue)
        overloaded_tiers = []
        degraded_tiers = []

        for tier_name, tier_metrics in metrics.get("by_tier", {}).items():
            utilization = tier_metrics.get("queue_utilization", 0)
            if utilization > 0.9:
                overloaded_tiers.append(tier_name)
            elif utilization > 0.7:
                degraded_tiers.append(tier_name)

        # Determine status
        if overloaded_tiers:
            status = "unhealthy"
        elif degraded_tiers:
            status = "degraded"
        else:
            status = "healthy"

        details = {}
        if include_details:
            details = {
                "total_active": metrics.get("total_active_tasks", 0),
                "total_queued": metrics.get("total_queued_tasks", 0),
                "overloaded_tiers": overloaded_tiers,
                "degraded_tiers": degraded_tiers,
            }

        return ComponentHealth(
            status=status,
            details=details,
        )
    except Exception as e:
        logger.error(f"Executor pool health check failed: {e}")
        return ComponentHealth(
            status="unhealthy",
            error=str(e),
        )


# ============================================================
# LEGACY HEALTH ENDPOINT
# ============================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Legacy health check (backward compatibility).

    For new deployments, use:
    - /health/live for liveness
    - /health/ready for readiness
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
    )


# ============================================================
# METRICS ENDPOINTS
# ============================================================


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics_json():
    """
    Get detailed metrics in JSON format.

    Returns executor pool, database, and Redis metrics.
    """
    result = {}

    # Executor pool metrics
    try:
        pool = get_executor_pool()
        result["executor_pool"] = pool.get_metrics_summary()
    except Exception as e:
        result["executor_pool"] = {"error": str(e)}

    # Database metrics
    try:
        db = get_database()
        result["database"] = await db.health_check()
    except Exception as e:
        result["database"] = {"error": str(e)}

    # Redis metrics
    try:
        redis = get_redis()
        result["redis"] = await redis.health_check()
    except Exception as e:
        result["redis"] = {"error": str(e)}

    return MetricsResponse(**result)


@router.get("/metrics/executor")
async def get_executor_metrics():
    """
    Get detailed executor pool metrics.

    Returns per-tier breakdown of:
    - Active tasks
    - Queued tasks
    - Completed/failed/timeout counts
    - Queue and worker utilization
    """
    try:
        pool = get_executor_pool()
        return pool.get_metrics_summary()
    except Exception as e:
        logger.error(f"Failed to get executor metrics: {e}")
        return {"error": str(e)}


@router.get("/metrics/prometheus")
async def get_prometheus_metrics(
    tenant_id: str | None = Query(default=None, description="Filter by tenant"),
):
    """
    Get metrics in Prometheus format with tenant labels.

    Returns plain text metrics compatible with Prometheus scraping.

    v1.4.0: Enhanced with tenant labels for multi-tenant observability.

    Prometheus config:
        - job_name: 'reflection'
          static_configs:
            - targets: ['localhost:8000']
          metrics_path: '/metrics/prometheus'
    """
    metrics = get_metrics()
    lines: list[str] = []

    # Get native Prometheus metrics
    try:
        prometheus_output = metrics.generate_latest()
        lines.append(prometheus_output.decode("utf-8"))
    except Exception as e:
        lines.append(f"# ERROR generating native metrics: {e}")

    # Add executor pool metrics with tier labels
    try:
        pool = get_executor_pool()
        summary = pool.get_metrics_summary()

        lines.append("")
        lines.append("# Executor pool metrics with tier labels")

        # Active tasks by tier
        lines.append("# HELP familiar_executor_active_tasks Active tasks in executor pool")
        lines.append("# TYPE familiar_executor_active_tasks gauge")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            active = tier_metrics.get("active_tasks", 0)
            lines.append(f'familiar_executor_active_tasks{{tier="{tier_name}"}} {active}')

        # Queued tasks by tier
        lines.append("# HELP familiar_executor_queued_tasks Queued tasks in executor pool")
        lines.append("# TYPE familiar_executor_queued_tasks gauge")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            queued = tier_metrics.get("queued_tasks", 0)
            lines.append(f'familiar_executor_queued_tasks{{tier="{tier_name}"}} {queued}')

        # Completed tasks by tier
        lines.append("# HELP familiar_executor_completed_total Completed tasks in executor pool")
        lines.append("# TYPE familiar_executor_completed_total counter")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            completed = tier_metrics.get("completed_tasks", 0)
            lines.append(f'familiar_executor_completed_total{{tier="{tier_name}"}} {completed}')

        # Failed tasks by tier
        lines.append("# HELP familiar_executor_failed_total Failed tasks in executor pool")
        lines.append("# TYPE familiar_executor_failed_total counter")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            failed = tier_metrics.get("failed_tasks", 0)
            lines.append(f'familiar_executor_failed_total{{tier="{tier_name}"}} {failed}')

        # Timeout tasks by tier
        lines.append("# HELP familiar_executor_timeout_total Timed out tasks in executor pool")
        lines.append("# TYPE familiar_executor_timeout_total counter")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            timeout = tier_metrics.get("timeout_tasks", 0)
            lines.append(f'familiar_executor_timeout_total{{tier="{tier_name}"}} {timeout}')

        # Queue utilization by tier
        lines.append("# HELP familiar_executor_queue_utilization Queue utilization ratio (0-1)")
        lines.append("# TYPE familiar_executor_queue_utilization gauge")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            util = tier_metrics.get("queue_utilization", 0)
            lines.append(f'familiar_executor_queue_utilization{{tier="{tier_name}"}} {util:.4f}')

        # Worker utilization by tier
        lines.append("# HELP familiar_executor_worker_utilization Worker utilization ratio (0-1)")
        lines.append("# TYPE familiar_executor_worker_utilization gauge")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            util = tier_metrics.get("worker_utilization", 0)
            lines.append(f'familiar_executor_worker_utilization{{tier="{tier_name}"}} {util:.4f}')

        # Average latency by tier
        lines.append("# HELP familiar_executor_avg_latency_seconds Average task latency")
        lines.append("# TYPE familiar_executor_avg_latency_seconds gauge")
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            latency = tier_metrics.get("avg_latency_ms", 0) / 1000
            lines.append(
                f'familiar_executor_avg_latency_seconds{{tier="{tier_name}"}} {latency:.6f}'
            )

        # Totals
        lines.append("")
        lines.append("# Executor pool totals")
        lines.append("# HELP familiar_executor_active_total Total active tasks")
        lines.append("# TYPE familiar_executor_active_total gauge")
        lines.append(f"familiar_executor_active_total {summary.get('total_active_tasks', 0)}")

        lines.append("# HELP familiar_executor_queued_total Total queued tasks")
        lines.append("# TYPE familiar_executor_queued_total gauge")
        lines.append(f"familiar_executor_queued_total {summary.get('total_queued_tasks', 0)}")

        # Update metrics registry gauges
        for tier_name, tier_metrics in summary.get("by_tier", {}).items():
            metrics.set_executor_gauges(
                tier=tier_name,
                active=tier_metrics.get("active_tasks", 0),
                queued=tier_metrics.get("queued_tasks", 0),
            )

    except Exception as e:
        logger.error(f"Failed to get executor metrics: {e}")
        lines.append(f"# ERROR getting executor metrics: {e}")

    # Application info
    lines.append("")
    lines.append("# Application info")
    lines.append("# HELP familiar_app_info Application information")
    lines.append("# TYPE familiar_app_info gauge")
    lines.append(
        f'familiar_app_info{{version="{settings.app_version}",'
        f'environment="{settings.environment}"}} 1'
    )

    # Uptime
    uptime = time.time() - _startup_time
    lines.append("# HELP familiar_uptime_seconds Application uptime in seconds")
    lines.append("# TYPE familiar_uptime_seconds gauge")
    lines.append(f"familiar_uptime_seconds {uptime:.2f}")

    return Response(
        content="\n".join(lines) + "\n",
        media_type=metrics.content_type(),
    )


@router.get("/version")
async def get_version():
    """Get application version info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "uptime_seconds": round(time.time() - _startup_time, 2),
    }
