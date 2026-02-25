# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Background Job System (v1.5.0)

Provides async job processing for long-running operations:
- Data export (GDPR)
- Bulk operations
- Scheduled tasks
- Tenant maintenance

Features:
- Redis-backed job queue
- Job status tracking
- Retry with exponential backoff
- Job cancellation
- Progress reporting

Usage:
    from reflection.jobs import JobService, Job, JobStatus

    # Create a job
    job = await job_service.create_job(
        job_type="data_export",
        tenant_id=tenant_id,
        params={"format": "json"},
    )

    # Check status
    status = await job_service.get_job(job.id)
    print(f"Status: {status.status}, Progress: {status.progress}%")

    # Cancel job
    await job_service.cancel_job(job.id)
"""

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum, StrEnum
from typing import Any, TypeVar
from uuid import UUID, uuid4

from redis.asyncio import Redis

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# JOB TYPES
# ============================================================


class JobStatus(StrEnum):
    """Job execution status."""

    PENDING = "pending"  # Waiting in queue
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed after retries
    CANCELLED = "cancelled"  # Cancelled by user
    EXPIRED = "expired"  # Expired before execution


class JobPriority(int, Enum):
    """Job priority levels."""

    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 100


@dataclass
class Job:
    """
    Background job definition.
    """

    id: str
    job_type: str
    tenant_id: UUID
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL

    # Parameters and result
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str | None = None

    # Progress tracking
    progress: int = 0  # 0-100
    progress_message: str | None = None

    # Timing
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    expires_at: datetime | None = None

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: int = 60

    # Metadata
    created_by: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "job_type": self.job_type,
            "tenant_id": str(self.tenant_id),
            "status": self.status.value,
            "priority": self.priority.value,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "progress": self.progress,
            "progress_message": self.progress_message,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "created_by": str(self.created_by) if self.created_by else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Job":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            job_type=data["job_type"],
            tenant_id=UUID(data["tenant_id"]),
            status=JobStatus(data["status"]),
            priority=JobPriority(data["priority"]),
            params=data.get("params", {}),
            result=data.get("result"),
            error=data.get("error"),
            progress=data.get("progress", 0),
            progress_message=data.get("progress_message"),
            created_at=datetime.fromisoformat(data["created_at"]),
            started_at=datetime.fromisoformat(data["started_at"])
            if data.get("started_at")
            else None,
            completed_at=datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 60),
            created_by=UUID(data["created_by"]) if data.get("created_by") else None,
            metadata=data.get("metadata", {}),
        )

    @property
    def is_terminal(self) -> bool:
        """Check if job is in a terminal state."""
        return self.status in (
            JobStatus.COMPLETED,
            JobStatus.FAILED,
            JobStatus.CANCELLED,
            JobStatus.EXPIRED,
        )

    @property
    def duration_seconds(self) -> float | None:
        """Get job duration in seconds."""
        if not self.started_at:
            return None
        end = self.completed_at or datetime.now(UTC)
        return (end - self.started_at).total_seconds()


# ============================================================
# JOB HANDLER TYPE
# ============================================================

JobHandler = Callable[[Job, "JobContext"], Coroutine[Any, Any, dict[str, Any]]]


@dataclass
class JobContext:
    """
    Context passed to job handlers.

    Provides methods for progress reporting and cancellation checking.
    """

    job: Job
    _service: "JobService"
    _cancelled: bool = False

    async def update_progress(self, progress: int, message: str | None = None) -> None:
        """Update job progress (0-100)."""
        await self._service.update_progress(self.job.id, progress, message)
        self.job.progress = progress
        if message:
            self.job.progress_message = message

    async def is_cancelled(self) -> bool:
        """Check if job has been cancelled."""
        if self._cancelled:
            return True

        # Check Redis for cancellation
        job = await self._service.get_job(self.job.id)
        if job and job.status == JobStatus.CANCELLED:
            self._cancelled = True
            return True

        return False

    def check_cancelled(self) -> None:
        """Raise exception if job is cancelled."""
        if self._cancelled:
            raise JobCancelledError(f"Job {self.job.id} was cancelled")


class JobCancelledError(Exception):
    """Raised when a job is cancelled during execution."""

    pass


# ============================================================
# JOB SERVICE
# ============================================================


class JobService:
    """
    Background job service.

    Manages job creation, execution, and monitoring.

    Usage:
        service = JobService(redis)

        # Register handlers
        service.register_handler("data_export", handle_data_export)

        # Create job
        job = await service.create_job(
            job_type="data_export",
            tenant_id=tenant_id,
            params={"format": "json"},
        )

        # Start worker (in background)
        asyncio.create_task(service.run_worker())
    """

    def __init__(
        self,
        redis: Redis,
        key_prefix: str = "jobs",
        default_ttl_hours: int = 24,
        poll_interval_seconds: float = 1.0,
    ):
        """
        Initialize job service.

        Args:
            redis: Redis client
            key_prefix: Prefix for Redis keys
            default_ttl_hours: Default job data retention
            poll_interval_seconds: Worker poll interval
        """
        self.redis = redis
        self.prefix = key_prefix
        self.default_ttl = timedelta(hours=default_ttl_hours)
        self.poll_interval = poll_interval_seconds

        self._handlers: dict[str, JobHandler] = {}
        self._running = False

    # --------------------------------------------------------
    # KEY HELPERS
    # --------------------------------------------------------

    def _job_key(self, job_id: str) -> str:
        return f"{self.prefix}:job:{job_id}"

    def _queue_key(self) -> str:
        return f"{self.prefix}:queue"

    def _tenant_jobs_key(self, tenant_id: UUID) -> str:
        return f"{self.prefix}:tenant:{tenant_id}"

    def _processing_key(self) -> str:
        return f"{self.prefix}:processing"

    # --------------------------------------------------------
    # HANDLER REGISTRATION
    # --------------------------------------------------------

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        """
        Register a handler for a job type.

        Args:
            job_type: Type of job (e.g., "data_export")
            handler: Async function that processes the job
        """
        self._handlers[job_type] = handler
        logger.info(f"Registered handler for job type: {job_type}")

    def get_registered_types(self) -> list[str]:
        """Get list of registered job types."""
        return list(self._handlers.keys())

    # --------------------------------------------------------
    # JOB CREATION
    # --------------------------------------------------------

    async def create_job(
        self,
        job_type: str,
        tenant_id: UUID,
        params: dict[str, Any] | None = None,
        priority: JobPriority = JobPriority.NORMAL,
        created_by: UUID | None = None,
        expires_in: timedelta | None = None,
        max_retries: int = 3,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """
        Create a new background job.

        Args:
            job_type: Type of job to create
            tenant_id: Tenant that owns the job
            params: Job parameters
            priority: Job priority
            created_by: User who created the job
            expires_in: Time until job expires
            max_retries: Maximum retry attempts
            metadata: Additional metadata

        Returns:
            Created job
        """
        if job_type not in self._handlers:
            raise ValueError(
                f"Unknown job type: {job_type}. Registered types: {list(self._handlers.keys())}"
            )

        job_id = str(uuid4())
        now = datetime.now(UTC)

        job = Job(
            id=job_id,
            job_type=job_type,
            tenant_id=tenant_id,
            status=JobStatus.PENDING,
            priority=priority,
            params=params or {},
            created_at=now,
            expires_at=now + expires_in if expires_in else None,
            max_retries=max_retries,
            created_by=created_by,
            metadata=metadata or {},
        )

        # Store job data
        ttl_seconds = int(self.default_ttl.total_seconds())
        await self.redis.setex(
            self._job_key(job_id),
            ttl_seconds,
            json.dumps(job.to_dict()),
        )

        # Add to queue (sorted by priority, then time)
        score = -priority.value * 1_000_000_000 + now.timestamp()
        await self.redis.zadd(self._queue_key(), {job_id: score})

        # Add to tenant's job list
        await self.redis.sadd(self._tenant_jobs_key(tenant_id), job_id)
        await self.redis.expire(self._tenant_jobs_key(tenant_id), ttl_seconds)

        logger.info(
            f"Created job {job_id}: type={job_type}, tenant={tenant_id}, priority={priority.name}"
        )

        return job

    # --------------------------------------------------------
    # JOB RETRIEVAL
    # --------------------------------------------------------

    async def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        data = await self.redis.get(self._job_key(job_id))
        if not data:
            return None

        try:
            return Job.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Invalid job data for {job_id}: {e}")
            return None

    async def get_tenant_jobs(
        self,
        tenant_id: UUID,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """
        Get jobs for a tenant.

        Args:
            tenant_id: Tenant ID
            status: Filter by status
            limit: Maximum jobs to return

        Returns:
            List of jobs
        """
        job_ids = await self.redis.smembers(self._tenant_jobs_key(tenant_id))

        jobs = []
        for job_id in list(job_ids)[: limit * 2]:  # Fetch extra for filtering
            job_id_str = job_id if isinstance(job_id, str) else job_id.decode()
            job = await self.get_job(job_id_str)

            if job and (status is None or job.status == status):
                jobs.append(job)
                if len(jobs) >= limit:
                    break

        # Sort by created_at descending
        jobs.sort(key=lambda j: j.created_at, reverse=True)

        return jobs

    # --------------------------------------------------------
    # JOB UPDATES
    # --------------------------------------------------------

    async def _save_job(self, job: Job) -> None:
        """Save job to Redis."""
        ttl = await self.redis.ttl(self._job_key(job.id))
        if ttl < 0:
            ttl = int(self.default_ttl.total_seconds())

        await self.redis.setex(
            self._job_key(job.id),
            ttl,
            json.dumps(job.to_dict()),
        )

    async def update_progress(
        self,
        job_id: str,
        progress: int,
        message: str | None = None,
    ) -> None:
        """Update job progress."""
        job = await self.get_job(job_id)
        if not job:
            return

        job.progress = min(100, max(0, progress))
        if message:
            job.progress_message = message

        await self._save_job(job)

    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a job.

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled, False if not cancellable
        """
        job = await self.get_job(job_id)
        if not job:
            return False

        if job.is_terminal:
            return False

        job.status = JobStatus.CANCELLED
        job.completed_at = datetime.now(UTC)

        await self._save_job(job)

        # Remove from queue if pending
        await self.redis.zrem(self._queue_key(), job_id)

        logger.info(f"Cancelled job {job_id}")
        return True

    # --------------------------------------------------------
    # WORKER
    # --------------------------------------------------------

    async def run_worker(self, worker_id: str | None = None) -> None:
        """
        Run the job worker loop.

        This should be started in a background task.
        """
        worker_id = worker_id or str(uuid4())[:8]
        self._running = True

        logger.info(f"Job worker {worker_id} started")

        while self._running:
            try:
                await self._process_next_job(worker_id)
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)

            await asyncio.sleep(self.poll_interval)

        logger.info(f"Job worker {worker_id} stopped")

    def stop_worker(self) -> None:
        """Signal worker to stop."""
        self._running = False

    async def _process_next_job(self, worker_id: str) -> bool:
        """
        Process the next job from the queue.

        Returns:
            True if a job was processed
        """
        # Get next job from queue (atomic pop)
        result = await self.redis.zpopmin(self._queue_key(), count=1)
        if not result:
            return False

        job_id = result[0][0]
        if isinstance(job_id, bytes):
            job_id = job_id.decode()

        job = await self.get_job(job_id)
        if not job:
            return False

        # Check if expired
        if job.expires_at and job.expires_at < datetime.now(UTC):
            job.status = JobStatus.EXPIRED
            job.completed_at = datetime.now(UTC)
            await self._save_job(job)
            logger.info(f"Job {job_id} expired")
            return True

        # Check if cancelled
        if job.status == JobStatus.CANCELLED:
            return True

        # Get handler
        handler = self._handlers.get(job.job_type)
        if not handler:
            logger.error(f"No handler for job type: {job.job_type}")
            job.status = JobStatus.FAILED
            job.error = f"No handler for job type: {job.job_type}"
            await self._save_job(job)
            return True

        # Start processing
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now(UTC)
        await self._save_job(job)

        logger.info(f"Worker {worker_id} processing job {job_id}: {job.job_type}")

        # Execute handler
        ctx = JobContext(job=job, _service=self)

        try:
            result = await handler(job, ctx)

            # Check for cancellation after completion
            if await ctx.is_cancelled():
                return True

            job.status = JobStatus.COMPLETED
            job.result = result
            job.progress = 100
            job.completed_at = datetime.now(UTC)

            logger.info(f"Job {job_id} completed in {job.duration_seconds:.2f}s")

        except JobCancelledError:
            logger.info(f"Job {job_id} was cancelled during execution")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)

            job.retry_count += 1

            if job.retry_count < job.max_retries:
                # Retry with exponential backoff
                delay = job.retry_delay_seconds * (2 ** (job.retry_count - 1))
                job.status = JobStatus.PENDING
                job.error = f"Attempt {job.retry_count} failed: {str(e)}"

                # Re-queue with delay
                retry_time = datetime.now(UTC) + timedelta(seconds=delay)
                score = -job.priority.value * 1_000_000_000 + retry_time.timestamp()
                await self.redis.zadd(self._queue_key(), {job_id: score})

                logger.info(
                    f"Job {job_id} queued for retry {job.retry_count}/{job.max_retries} in {delay}s"
                )
            else:
                job.status = JobStatus.FAILED
                job.error = f"Failed after {job.max_retries} attempts: {str(e)}"
                job.completed_at = datetime.now(UTC)

                logger.error(f"Job {job_id} failed permanently after {job.max_retries} attempts")

        await self._save_job(job)
        return True

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    async def get_queue_stats(self) -> dict[str, Any]:
        """Get queue statistics."""
        queue_size = await self.redis.zcard(self._queue_key())

        # Count by status (sample recent jobs)
        # This is approximate for performance

        return {
            "queue_size": queue_size,
            "registered_types": list(self._handlers.keys()),
            "worker_running": self._running,
        }


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_job_service: JobService | None = None


async def get_job_service() -> JobService | None:
    """Get the global job service."""
    global _job_service

    if _job_service is not None:
        return _job_service

    try:
        from ..data.redis import get_redis

        redis_manager = get_redis()

        try:
            redis = redis_manager.client
        except RuntimeError:
            await redis_manager.connect()
            redis = redis_manager.client

        _job_service = JobService(redis)
        logger.info("Job service initialized")

        return _job_service

    except Exception as e:
        logger.error(f"Failed to initialize job service: {e}")
        return None


async def init_job_service(redis: Redis) -> JobService:
    """Initialize the global job service with a Redis client."""
    global _job_service
    _job_service = JobService(redis)
    return _job_service


def reset_job_service() -> None:
    """Reset the global job service (for testing)."""
    global _job_service
    if _job_service:
        _job_service.stop_worker()
    _job_service = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "Job",
    "JobStatus",
    "JobPriority",
    "JobContext",
    "JobHandler",
    "JobCancelledError",
    # Service
    "JobService",
    "get_job_service",
    "init_job_service",
    "reset_job_service",
]
