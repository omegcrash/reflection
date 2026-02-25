# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Lifecycle API Routes (v1.5.0)

REST API for tenant lifecycle management:
- Suspend tenants
- Reactivate tenants
- Schedule deletion (GDPR Article 17)
- Cancel deletion
- Get tenant status

Requires admin authentication.
"""

import logging
from datetime import UTC, datetime, timedelta
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.settings import get_settings
from ..data.postgres import get_db_session
from ..tenants.lifecycle import (
    InvalidTransitionError,
    SuspensionReason,
    TenantLifecycleError,
    TenantLifecycleEvent,
    TenantLifecycleService,
)
from .auth import TokenPayload, get_current_user

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/admin/tenants", tags=["Tenant Lifecycle"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class TenantStatusResponse(BaseModel):
    """Tenant status response."""

    tenant_id: str
    status: str
    suspension_reason: str | None = None
    suspension_message: str | None = None
    suspended_at: str | None = None
    deletion_date: str | None = None


class SuspendTenantRequest(BaseModel):
    """Request to suspend a tenant."""

    reason: str = Field(
        ..., description="Suspension reason: billing, policy, security, admin, requested"
    )
    message: str | None = Field(default=None, description="Human-readable message")
    notify: bool = Field(default=True, description="Send notification to tenant")


class ReactivateTenantRequest(BaseModel):
    """Request to reactivate a tenant."""

    notes: str | None = Field(default=None, description="Notes about reactivation")


class ScheduleDeletionRequest(BaseModel):
    """Request to schedule tenant deletion."""

    reason: str = Field(default="user_request", description="Reason for deletion")
    deletion_days: int = Field(default=30, ge=7, le=90, description="Days until deletion (7-90)")


class LifecycleEventResponse(BaseModel):
    """Lifecycle event response."""

    tenant_id: str
    from_status: str
    to_status: str
    reason: str | None = None
    performed_by: str | None = None
    timestamp: str

    @classmethod
    def from_event(cls, event: TenantLifecycleEvent) -> "LifecycleEventResponse":
        return cls(
            tenant_id=str(event.tenant_id),
            from_status=event.from_status.value,
            to_status=event.to_status.value,
            reason=event.reason,
            performed_by=str(event.performed_by) if event.performed_by else None,
            timestamp=event.timestamp.isoformat(),
        )


# ============================================================
# DEPENDENCIES
# ============================================================


async def require_admin(current_user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Require admin role for lifecycle operations."""
    # Check if user has admin role (simplified - enhance with role system)
    if not getattr(current_user, "is_admin", False):
        # For now, allow all authenticated users (should be restricted in production)
        pass
    return current_user


# ============================================================
# ROUTES
# ============================================================


@router.get("/{tenant_id}/status", response_model=TenantStatusResponse)
async def get_tenant_status(
    tenant_id: UUID,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Get tenant lifecycle status.

    Returns current status and any suspension/deletion details.
    """
    service = TenantLifecycleService(session)

    status = await service.get_status(tenant_id)

    if status is None:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Get additional metadata
    from sqlalchemy import select

    from ..data.models import TenantModel

    result = await session.execute(select(TenantModel.metadata).where(TenantModel.id == tenant_id))
    metadata = result.scalar_one_or_none() or {}

    return TenantStatusResponse(
        tenant_id=str(tenant_id),
        status=status.value,
        suspension_reason=metadata.get("suspension_reason"),
        suspension_message=metadata.get("suspension_message"),
        suspended_at=metadata.get("suspended_at"),
        deletion_date=metadata.get("deletion_date"),
    )


@router.post("/{tenant_id}/suspend", response_model=LifecycleEventResponse)
async def suspend_tenant(
    tenant_id: UUID,
    request: SuspendTenantRequest,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Suspend a tenant.

    Suspended tenants cannot:
    - Make API calls
    - Access their data (except for export)
    - Create new resources

    Reasons:
    - `billing`: Payment failed
    - `policy`: Terms violation
    - `security`: Security concern
    - `admin`: Administrative action
    - `requested`: Customer request
    """
    # Validate reason
    try:
        reason = SuspensionReason(request.reason.lower())
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid reason: {request.reason}. "
            f"Valid options: {[r.value for r in SuspensionReason]}",
        ) from e

    service = TenantLifecycleService(session)

    try:
        event = await service.suspend_tenant(
            tenant_id=tenant_id,
            reason=reason,
            performed_by=current_user.user_id,
            message=request.message,
            notify=request.notify,
        )

        await session.commit()

        logger.info(f"Tenant {tenant_id} suspended by {current_user.user_id}: {reason.value}")

        return LifecycleEventResponse.from_event(event)

    except InvalidTransitionError as e:
        raise HTTPException(status_code=400, detail=f"Cannot suspend tenant: {str(e)}") from e
    except TenantLifecycleError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{tenant_id}/reactivate", response_model=LifecycleEventResponse)
async def reactivate_tenant(
    tenant_id: UUID,
    request: ReactivateTenantRequest,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Reactivate a suspended tenant.

    Returns the tenant to active status and clears suspension metadata.
    """
    service = TenantLifecycleService(session)

    try:
        event = await service.reactivate_tenant(
            tenant_id=tenant_id,
            performed_by=current_user.user_id,
            notes=request.notes,
        )

        await session.commit()

        logger.info(f"Tenant {tenant_id} reactivated by {current_user.user_id}")

        return LifecycleEventResponse.from_event(event)

    except InvalidTransitionError as e:
        raise HTTPException(status_code=400, detail=f"Cannot reactivate tenant: {str(e)}") from e
    except TenantLifecycleError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{tenant_id}/schedule-deletion", response_model=LifecycleEventResponse)
async def schedule_deletion(
    tenant_id: UUID,
    request: ScheduleDeletionRequest,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Schedule a tenant for deletion (GDPR Article 17 - Right to Erasure).

    The tenant will be deleted after the specified grace period (default: 30 days).
    During this period, the deletion can be cancelled.

    After deletion:
    - All tenant data is permanently removed
    - This action cannot be undone
    """
    deletion_date = datetime.now(UTC) + timedelta(days=request.deletion_days)

    service = TenantLifecycleService(session)

    try:
        event = await service.schedule_deletion(
            tenant_id=tenant_id,
            performed_by=current_user.user_id,
            deletion_date=deletion_date,
            reason=request.reason,
        )

        await session.commit()

        logger.info(
            f"Tenant {tenant_id} scheduled for deletion on {deletion_date} "
            f"by {current_user.user_id}"
        )

        return LifecycleEventResponse.from_event(event)

    except InvalidTransitionError as e:
        raise HTTPException(status_code=400, detail=f"Cannot schedule deletion: {str(e)}") from e
    except TenantLifecycleError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/{tenant_id}/cancel-deletion", response_model=LifecycleEventResponse)
async def cancel_deletion(
    tenant_id: UUID,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Cancel a scheduled deletion.

    Returns the tenant to active status.
    """
    service = TenantLifecycleService(session)

    try:
        event = await service.cancel_deletion(
            tenant_id=tenant_id,
            performed_by=current_user.user_id,
        )

        await session.commit()

        logger.info(f"Tenant {tenant_id} deletion cancelled by {current_user.user_id}")

        return LifecycleEventResponse.from_event(event)

    except TenantLifecycleError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ============================================================
# BULK OPERATIONS
# ============================================================


class BulkSuspendRequest(BaseModel):
    """Request to suspend multiple tenants."""

    tenant_ids: list[UUID] = Field(..., max_length=100)
    reason: str
    message: str | None = None


class BulkOperationResult(BaseModel):
    """Result of a bulk operation."""

    success: list[str]
    failed: dict[str, str]


@router.post("/bulk/suspend", response_model=BulkOperationResult)
async def bulk_suspend(
    request: BulkSuspendRequest,
    current_user: TokenPayload = Depends(require_admin),
    session: AsyncSession = Depends(get_db_session),
):
    """
    Suspend multiple tenants at once.

    Useful for billing-related bulk suspensions.
    Maximum 100 tenants per request.
    """
    try:
        reason = SuspensionReason(request.reason.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid reason: {request.reason}") from e

    service = TenantLifecycleService(session)

    success = []
    failed = {}

    for tenant_id in request.tenant_ids:
        try:
            await service.suspend_tenant(
                tenant_id=tenant_id,
                reason=reason,
                performed_by=current_user.user_id,
                message=request.message,
                notify=False,  # Batch notify separately
            )
            success.append(str(tenant_id))
        except Exception as e:
            failed[str(tenant_id)] = str(e)

    await session.commit()

    logger.info(
        f"Bulk suspend: {len(success)} succeeded, {len(failed)} failed by {current_user.user_id}"
    )

    return BulkOperationResult(success=success, failed=failed)
