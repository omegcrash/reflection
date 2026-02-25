# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Background Job API Routes (v1.5.0)

REST API for background job management:
- Create jobs (data export, bulk operations)
- Check job status
- Cancel jobs
- List tenant jobs

All endpoints require authentication.
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from ..core.settings import get_settings
from ..jobs import (
    Job,
    JobPriority,
    JobService,
    JobStatus,
    get_job_service,
)
from ..jobs.export_handlers import register_export_handlers
from .auth import TokenPayload, get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/jobs", tags=["Jobs"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class CreateJobRequest(BaseModel):
    """Request to create a background job."""

    job_type: str = Field(..., description="Type of job to create")
    params: dict[str, Any] = Field(default_factory=dict, description="Job parameters")
    priority: str = Field(default="normal", description="Job priority: low, normal, high, critical")


class JobResponse(BaseModel):
    """Job status response."""

    id: str
    job_type: str
    status: str
    priority: str
    progress: int
    progress_message: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        return cls(
            id=job.id,
            job_type=job.job_type,
            status=job.status.value,
            priority=job.priority.name.lower(),
            progress=job.progress,
            progress_message=job.progress_message,
            result=job.result,
            error=job.error,
            created_at=job.created_at.isoformat(),
            started_at=job.started_at.isoformat() if job.started_at else None,
            completed_at=job.completed_at.isoformat() if job.completed_at else None,
            retry_count=job.retry_count,
            max_retries=job.max_retries,
        )


class JobListResponse(BaseModel):
    """List of jobs response."""

    jobs: list[JobResponse]
    total: int


class QueueStatsResponse(BaseModel):
    """Queue statistics response."""

    queue_size: int
    registered_types: list[str]
    worker_running: bool


# ============================================================
# DEPENDENCIES
# ============================================================


async def get_initialized_job_service() -> JobService:
    """Get job service with handlers registered."""
    service = await get_job_service()
    if service is None:
        raise HTTPException(status_code=503, detail="Job service not available")

    # Register handlers if not already done
    if "tenant_data_export" not in service.get_registered_types():
        register_export_handlers(service)

    return service


# ============================================================
# ROUTES
# ============================================================


@router.post("", response_model=JobResponse)
async def create_job(
    request: CreateJobRequest,
    background_tasks: BackgroundTasks,
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Create a new background job.

    Available job types:
    - `tenant_data_export`: Export all tenant data (GDPR)
    - `user_data_export`: Export specific user's data

    The job will be processed asynchronously. Use GET /jobs/{job_id}
    to check status and retrieve results.
    """
    # Validate priority
    priority_map = {
        "low": JobPriority.LOW,
        "normal": JobPriority.NORMAL,
        "high": JobPriority.HIGH,
        "critical": JobPriority.CRITICAL,
    }

    priority = priority_map.get(request.priority.lower())
    if priority is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid priority: {request.priority}. "
            f"Valid options: {list(priority_map.keys())}",
        )

    # Validate job type
    registered = job_service.get_registered_types()
    if request.job_type not in registered:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown job type: {request.job_type}. Available types: {registered}",
        )

    try:
        job = await job_service.create_job(
            job_type=request.job_type,
            tenant_id=current_user.tenant_id,
            params=request.params,
            priority=priority,
            created_by=current_user.user_id,
        )

        logger.info(
            f"Job created: {job.id} type={request.job_type} tenant={current_user.tenant_id}"
        )

        return JobResponse.from_job(job)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Get job status and details.

    Returns current progress, status, and results (if completed).
    """
    job = await job_service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant ownership
    if job.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse.from_job(job)


@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Cancel a pending or running job.

    Jobs that have already completed cannot be cancelled.
    """
    job = await job_service.get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Verify tenant ownership
    if job.tenant_id != current_user.tenant_id:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.is_terminal:
        raise HTTPException(
            status_code=400, detail=f"Job already in terminal state: {job.status.value}"
        )

    success = await job_service.cancel_job(job_id)

    if not success:
        raise HTTPException(status_code=400, detail="Failed to cancel job")

    return {"status": "cancelled", "job_id": job_id}


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: str | None = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, le=100, description="Maximum jobs to return"),
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    List jobs for the current tenant.

    Optionally filter by status: pending, running, completed, failed, cancelled.
    """
    # Validate status filter
    status_filter = None
    if status:
        try:
            status_filter = JobStatus(status.lower())
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status: {status}. Valid options: {[s.value for s in JobStatus]}",
            ) from e

    jobs = await job_service.get_tenant_jobs(
        tenant_id=current_user.tenant_id,
        status=status_filter,
        limit=limit,
    )

    return JobListResponse(
        jobs=[JobResponse.from_job(j) for j in jobs],
        total=len(jobs),
    )


@router.get("/stats/queue", response_model=QueueStatsResponse)
async def get_queue_stats(
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Get job queue statistics.

    Shows queue size, registered job types, and worker status.
    """
    stats = await job_service.get_queue_stats()
    return QueueStatsResponse(**stats)


# ============================================================
# DATA EXPORT CONVENIENCE ENDPOINTS
# ============================================================


class DataExportRequest(BaseModel):
    """Request for data export."""

    format: str = Field(default="json", description="Export format: json, zip")
    include_conversations: bool = Field(default=True)
    include_messages: bool = Field(default=True)
    include_usage: bool = Field(default=True)
    include_agents: bool = Field(default=True)


@router.post("/export/tenant", response_model=JobResponse)
async def export_tenant_data(
    request: DataExportRequest,
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Export all tenant data (GDPR Article 20 - Data Portability).

    Creates a background job that exports:
    - Conversations and messages
    - Agent configurations
    - Usage records

    Returns a job ID to track progress.
    """
    job = await job_service.create_job(
        job_type="tenant_data_export",
        tenant_id=current_user.tenant_id,
        params={
            "format": request.format,
            "include_conversations": request.include_conversations,
            "include_messages": request.include_messages,
            "include_usage": request.include_usage,
            "include_agents": request.include_agents,
        },
        priority=JobPriority.NORMAL,
        created_by=current_user.user_id,
    )

    logger.info(f"Tenant data export job created: {job.id}")

    return JobResponse.from_job(job)


class UserExportRequest(BaseModel):
    """Request for user data export."""

    user_id: UUID = Field(..., description="User ID to export data for")
    format: str = Field(default="json", description="Export format: json")


@router.post("/export/user", response_model=JobResponse)
async def export_user_data(
    request: UserExportRequest,
    current_user: TokenPayload = Depends(get_current_user),
    job_service: JobService = Depends(get_initialized_job_service),
):
    """
    Export specific user's data (GDPR Article 15 - Right of Access).

    Creates a background job that exports all data for a specific user.

    Returns a job ID to track progress.
    """
    job = await job_service.create_job(
        job_type="user_data_export",
        tenant_id=current_user.tenant_id,
        params={
            "user_id": str(request.user_id),
            "format": request.format,
        },
        priority=JobPriority.NORMAL,
        created_by=current_user.user_id,
    )

    logger.info(f"User data export job created: {job.id} for user {request.user_id}")

    return JobResponse.from_job(job)
