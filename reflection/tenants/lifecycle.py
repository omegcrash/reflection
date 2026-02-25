# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Lifecycle Management (v1.5.0)

Manages tenant lifecycle states:
- Active: Normal operation
- Suspended: Temporarily disabled (billing, policy)
- Pending Deletion: Scheduled for data deletion (GDPR)
- Deleted: Soft-deleted, data removed

Provides:
- State transitions with validation
- Suspension reasons tracking
- Reactivation with checks
- Deletion scheduling (GDPR Article 17)
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ============================================================
# LIFECYCLE STATES
# ============================================================


class TenantStatus(StrEnum):
    """Tenant lifecycle status."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    PENDING_DELETION = "pending_deletion"
    DELETED = "deleted"


class SuspensionReason(StrEnum):
    """Reasons for tenant suspension."""

    BILLING = "billing"  # Payment failed
    POLICY = "policy"  # Terms violation
    SECURITY = "security"  # Security concern
    ADMIN = "admin"  # Administrative action
    REQUESTED = "requested"  # Customer request


@dataclass
class TenantLifecycleEvent:
    """Record of a lifecycle state change."""

    tenant_id: UUID
    from_status: TenantStatus
    to_status: TenantStatus
    reason: str | None = None
    performed_by: UUID | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": str(self.tenant_id),
            "from_status": self.from_status.value,
            "to_status": self.to_status.value,
            "reason": self.reason,
            "performed_by": str(self.performed_by) if self.performed_by else None,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


# ============================================================
# LIFECYCLE EXCEPTIONS
# ============================================================


class TenantLifecycleError(Exception):
    """Base exception for lifecycle errors."""

    pass


class InvalidTransitionError(TenantLifecycleError):
    """Invalid state transition."""

    def __init__(self, from_status: TenantStatus, to_status: TenantStatus):
        self.from_status = from_status
        self.to_status = to_status
        super().__init__(f"Invalid transition from {from_status.value} to {to_status.value}")


class TenantSuspendedError(TenantLifecycleError):
    """Operation blocked due to suspension."""

    def __init__(self, tenant_id: UUID, reason: str | None = None):
        self.tenant_id = tenant_id
        self.reason = reason
        super().__init__(f"Tenant {tenant_id} is suspended" + (f": {reason}" if reason else ""))


class TenantDeletedError(TenantLifecycleError):
    """Operation blocked due to deletion."""

    def __init__(self, tenant_id: UUID):
        self.tenant_id = tenant_id
        super().__init__(f"Tenant {tenant_id} is deleted")


# ============================================================
# LIFECYCLE SERVICE
# ============================================================


class TenantLifecycleService:
    """
    Service for managing tenant lifecycle.

    Handles state transitions, validation, and event logging.

    Usage:
        service = TenantLifecycleService(session)

        # Suspend tenant
        await service.suspend_tenant(
            tenant_id=tenant_id,
            reason=SuspensionReason.BILLING,
            performed_by=admin_id,
        )

        # Reactivate tenant
        await service.reactivate_tenant(
            tenant_id=tenant_id,
            performed_by=admin_id,
        )

        # Check status before operations
        await service.ensure_active(tenant_id)
    """

    # Valid state transitions
    VALID_TRANSITIONS = {
        TenantStatus.ACTIVE: {
            TenantStatus.SUSPENDED,
            TenantStatus.PENDING_DELETION,
        },
        TenantStatus.SUSPENDED: {
            TenantStatus.ACTIVE,
            TenantStatus.PENDING_DELETION,
        },
        TenantStatus.PENDING_DELETION: {
            TenantStatus.ACTIVE,  # Cancel deletion
            TenantStatus.DELETED,
        },
        TenantStatus.DELETED: set(),  # Terminal state
    }

    def __init__(self, session: AsyncSession):
        """
        Initialize lifecycle service.

        Args:
            session: Database session
        """
        self.session = session
        self._events: list[TenantLifecycleEvent] = []

    async def get_status(self, tenant_id: UUID) -> TenantStatus | None:
        """
        Get current tenant status.

        Returns None if tenant doesn't exist.
        """
        from ..data.models import TenantModel

        result = await self.session.execute(
            select(TenantModel.status).where(TenantModel.id == tenant_id)
        )
        row = result.scalar_one_or_none()

        if row is None:
            return None

        try:
            return TenantStatus(row)
        except ValueError:
            # Legacy status, treat as active
            return TenantStatus.ACTIVE

    async def ensure_active(self, tenant_id: UUID) -> None:
        """
        Ensure tenant is active, raise exception if not.

        Use this before operations that require active tenant.

        Raises:
            TenantSuspendedError: If tenant is suspended
            TenantDeletedError: If tenant is deleted
            TenantLifecycleError: If tenant doesn't exist
        """
        status = await self.get_status(tenant_id)

        if status is None:
            raise TenantLifecycleError(f"Tenant {tenant_id} not found")

        if status == TenantStatus.SUSPENDED:
            # Get suspension reason
            from ..data.models import TenantModel

            result = await self.session.execute(
                select(TenantModel.metadata).where(TenantModel.id == tenant_id)
            )
            metadata = result.scalar_one_or_none() or {}
            reason = metadata.get("suspension_reason")

            raise TenantSuspendedError(tenant_id, reason)

        if status in (TenantStatus.PENDING_DELETION, TenantStatus.DELETED):
            raise TenantDeletedError(tenant_id)

    def _validate_transition(
        self,
        from_status: TenantStatus,
        to_status: TenantStatus,
    ) -> None:
        """Validate a state transition is allowed."""
        valid = self.VALID_TRANSITIONS.get(from_status, set())

        if to_status not in valid:
            raise InvalidTransitionError(from_status, to_status)

    async def _update_status(
        self,
        tenant_id: UUID,
        new_status: TenantStatus,
        metadata_updates: dict[str, Any] | None = None,
    ) -> None:
        """Update tenant status in database."""

        from ..data.models import TenantModel

        update_data = {"status": new_status.value}

        if metadata_updates:
            # Merge with existing metadata
            result = await self.session.execute(
                select(TenantModel.metadata).where(TenantModel.id == tenant_id)
            )
            existing = result.scalar_one_or_none() or {}
            existing.update(metadata_updates)
            update_data["metadata"] = existing

        await self.session.execute(
            update(TenantModel).where(TenantModel.id == tenant_id).values(**update_data)
        )

    async def suspend_tenant(
        self,
        tenant_id: UUID,
        reason: SuspensionReason,
        performed_by: UUID | None = None,
        message: str | None = None,
        notify: bool = True,
    ) -> TenantLifecycleEvent:
        """
        Suspend a tenant.

        Suspended tenants cannot:
        - Make API calls
        - Access their data (except for export)
        - Create new resources

        Args:
            tenant_id: Tenant to suspend
            reason: Suspension reason
            performed_by: Admin who performed action
            message: Optional human-readable message
            notify: Whether to send notification

        Returns:
            Lifecycle event record

        Raises:
            InvalidTransitionError: If transition not allowed
        """
        current_status = await self.get_status(tenant_id)

        if current_status is None:
            raise TenantLifecycleError(f"Tenant {tenant_id} not found")

        self._validate_transition(current_status, TenantStatus.SUSPENDED)

        # Update status
        metadata_updates = {
            "suspension_reason": reason.value,
            "suspension_message": message,
            "suspended_at": datetime.now(UTC).isoformat(),
            "suspended_by": str(performed_by) if performed_by else None,
        }

        await self._update_status(
            tenant_id,
            TenantStatus.SUSPENDED,
            metadata_updates,
        )

        # Create event
        event = TenantLifecycleEvent(
            tenant_id=tenant_id,
            from_status=current_status,
            to_status=TenantStatus.SUSPENDED,
            reason=reason.value,
            performed_by=performed_by,
            metadata={"message": message} if message else {},
        )

        self._events.append(event)

        logger.info(f"Tenant {tenant_id} suspended: reason={reason.value}, by={performed_by}")

        # TODO: Send notification if notify=True

        return event

    async def reactivate_tenant(
        self,
        tenant_id: UUID,
        performed_by: UUID | None = None,
        notes: str | None = None,
    ) -> TenantLifecycleEvent:
        """
        Reactivate a suspended tenant.

        Args:
            tenant_id: Tenant to reactivate
            performed_by: Admin who performed action
            notes: Optional notes about reactivation

        Returns:
            Lifecycle event record

        Raises:
            InvalidTransitionError: If tenant not suspended
        """
        current_status = await self.get_status(tenant_id)

        if current_status is None:
            raise TenantLifecycleError(f"Tenant {tenant_id} not found")

        self._validate_transition(current_status, TenantStatus.ACTIVE)

        # Clear suspension metadata
        metadata_updates = {
            "suspension_reason": None,
            "suspension_message": None,
            "suspended_at": None,
            "suspended_by": None,
            "reactivated_at": datetime.now(UTC).isoformat(),
            "reactivated_by": str(performed_by) if performed_by else None,
        }

        await self._update_status(
            tenant_id,
            TenantStatus.ACTIVE,
            metadata_updates,
        )

        # Create event
        event = TenantLifecycleEvent(
            tenant_id=tenant_id,
            from_status=current_status,
            to_status=TenantStatus.ACTIVE,
            reason="reactivated",
            performed_by=performed_by,
            metadata={"notes": notes} if notes else {},
        )

        self._events.append(event)

        logger.info(f"Tenant {tenant_id} reactivated by {performed_by}")

        return event

    async def schedule_deletion(
        self,
        tenant_id: UUID,
        performed_by: UUID | None = None,
        deletion_date: datetime | None = None,
        reason: str = "user_request",
    ) -> TenantLifecycleEvent:
        """
        Schedule a tenant for deletion (GDPR Article 17).

        Default: 30 days from now (grace period for recovery).

        Args:
            tenant_id: Tenant to schedule for deletion
            performed_by: Who requested deletion
            deletion_date: When to delete (default: 30 days)
            reason: Reason for deletion

        Returns:
            Lifecycle event record
        """
        current_status = await self.get_status(tenant_id)

        if current_status is None:
            raise TenantLifecycleError(f"Tenant {tenant_id} not found")

        self._validate_transition(current_status, TenantStatus.PENDING_DELETION)

        if deletion_date is None:
            deletion_date = datetime.now(UTC) + timedelta(days=30)

        metadata_updates = {
            "deletion_scheduled_at": datetime.now(UTC).isoformat(),
            "deletion_date": deletion_date.isoformat(),
            "deletion_reason": reason,
            "deletion_requested_by": str(performed_by) if performed_by else None,
        }

        await self._update_status(
            tenant_id,
            TenantStatus.PENDING_DELETION,
            metadata_updates,
        )

        event = TenantLifecycleEvent(
            tenant_id=tenant_id,
            from_status=current_status,
            to_status=TenantStatus.PENDING_DELETION,
            reason=reason,
            performed_by=performed_by,
            metadata={"deletion_date": deletion_date.isoformat()},
        )

        self._events.append(event)

        logger.info(f"Tenant {tenant_id} scheduled for deletion on {deletion_date}")

        return event

    async def cancel_deletion(
        self,
        tenant_id: UUID,
        performed_by: UUID | None = None,
    ) -> TenantLifecycleEvent:
        """
        Cancel a scheduled deletion.

        Args:
            tenant_id: Tenant to cancel deletion for
            performed_by: Who cancelled

        Returns:
            Lifecycle event record
        """
        current_status = await self.get_status(tenant_id)

        if current_status != TenantStatus.PENDING_DELETION:
            raise TenantLifecycleError(f"Tenant {tenant_id} is not pending deletion")

        self._validate_transition(current_status, TenantStatus.ACTIVE)

        metadata_updates = {
            "deletion_scheduled_at": None,
            "deletion_date": None,
            "deletion_reason": None,
            "deletion_requested_by": None,
            "deletion_cancelled_at": datetime.now(UTC).isoformat(),
            "deletion_cancelled_by": str(performed_by) if performed_by else None,
        }

        await self._update_status(
            tenant_id,
            TenantStatus.ACTIVE,
            metadata_updates,
        )

        event = TenantLifecycleEvent(
            tenant_id=tenant_id,
            from_status=current_status,
            to_status=TenantStatus.ACTIVE,
            reason="deletion_cancelled",
            performed_by=performed_by,
        )

        self._events.append(event)

        logger.info(f"Tenant {tenant_id} deletion cancelled by {performed_by}")

        return event

    def get_events(self) -> list[TenantLifecycleEvent]:
        """Get lifecycle events from this session."""
        return self._events.copy()


# ============================================================
# MIDDLEWARE INTEGRATION
# ============================================================


async def check_tenant_active(
    tenant_id: UUID,
    session: AsyncSession,
) -> None:
    """
    FastAPI dependency to check tenant is active.

    Usage:
        @app.get("/api/endpoint")
        async def endpoint(
            tenant_id: UUID = Depends(get_tenant_id),
            session: AsyncSession = Depends(get_db_session),
            _: None = Depends(lambda: check_tenant_active(tenant_id, session)),
        ):
            ...
    """
    service = TenantLifecycleService(session)
    await service.ensure_active(tenant_id)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "TenantStatus",
    "SuspensionReason",
    "TenantLifecycleEvent",
    # Exceptions
    "TenantLifecycleError",
    "InvalidTransitionError",
    "TenantSuspendedError",
    "TenantDeletedError",
    # Service
    "TenantLifecycleService",
    "check_tenant_active",
]
