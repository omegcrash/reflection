# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 6: Quota API Routes

Endpoints for:
- Usage monitoring (tenants can view their usage)
- Quota alerts (approaching limits)
- Admin quota management (override limits)
"""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.executor import TenantTier
from ..data.postgres import get_db_session
from ..data.repositories import TenantRepository
from ..tenants.quota_service import (
    TIER_LIMITS,
    get_quota_service,
    get_tier_limits,
)
from .auth import JWTService, TokenExpiredError, TokenInvalidError, TokenType, get_jwt_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/quotas", tags=["Quotas"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class UsageMetric(BaseModel):
    """Single usage metric."""

    current: int
    limit: int
    percentage: float
    remaining: int


class UsageCategory(BaseModel):
    """Usage metrics for a category."""

    per_minute: UsageMetric | None = None
    per_hour: UsageMetric | None = None
    per_day: UsageMetric | None = None
    per_month: UsageMetric | None = None


class UsageResponse(BaseModel):
    """Current usage response."""

    tenant_id: str
    tier: str
    requests: UsageCategory
    tokens: UsageCategory
    tools: UsageCategory
    limits: dict[str, Any]
    timestamp: datetime


class AlertResponse(BaseModel):
    """Quota alert."""

    category: str
    period: str
    current: int
    limit: int
    percentage: float
    severity: str  # "warning" or "critical"


class QuotaLimitsResponse(BaseModel):
    """Quota limits configuration."""

    tier: str
    limits: dict[str, Any]

    # Breakdown by category
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    tokens_per_day: int
    tokens_per_month: int
    max_concurrent_requests: int
    tool_executions_per_hour: int
    shell_executions_per_hour: int

    # Features
    allow_tool_execution: bool
    allow_shell_execution: bool
    allow_file_upload: bool
    max_file_upload_mb: int


class QuotaOverrideRequest(BaseModel):
    """Request to override tenant quotas (admin only)."""

    requests_per_minute: int | None = Field(None, ge=1, le=10000)
    requests_per_hour: int | None = Field(None, ge=1, le=100000)
    requests_per_day: int | None = Field(None, ge=1, le=1000000)
    tokens_per_day: int | None = Field(None, ge=1000, le=100000000)
    tokens_per_month: int | None = Field(None, ge=10000, le=1000000000)
    max_concurrent_requests: int | None = Field(None, ge=1, le=1000)
    tool_executions_per_hour: int | None = Field(None, ge=0, le=100000)
    shell_executions_per_hour: int | None = Field(None, ge=0, le=10000)
    allow_tool_execution: bool | None = None
    allow_shell_execution: bool | None = None


class TierLimitsResponse(BaseModel):
    """Tier limits comparison."""

    tiers: dict[str, dict[str, Any]]


# ============================================================
# AUTHENTICATION
# ============================================================


async def get_auth_context(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-ID"),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> tuple[UUID, str, TenantTier]:
    """Get authentication context. Returns (tenant_id, role, tier)."""
    tenant_repo = TenantRepository(session)

    # Try JWT authentication
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

        if token.count(".") == 2:
            try:
                payload = jwt_service.decode_token(token, TokenType.ACCESS)
                tenant_id = UUID(payload.tenant_id)
                role = payload.role

                tenant = await tenant_repo.get_by_id(tenant_id)
                tier = _map_tier(tenant.tier if tenant else "free")

                return tenant_id, role, tier

            except (TokenExpiredError, TokenInvalidError) as e:
                raise HTTPException(status_code=401, detail=str(e)) from e

        # API key auth
        from ..data.repositories import APIKeyRepository

        api_key_repo = APIKeyRepository(session)

        api_key = await api_key_repo.validate_key(token)
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        tenant = await tenant_repo.get_by_id(api_key.tenant_id)
        if not tenant or tenant.status != "active":
            raise HTTPException(status_code=403, detail="Tenant not active")

        tier = _map_tier(tenant.tier)
        return tenant.id, "member", tier

    # X-Tenant-ID fallback
    if x_tenant_id:
        tenant = await tenant_repo.get_by_slug(x_tenant_id)
        if tenant:
            tier = _map_tier(tenant.tier)
            return tenant.id, "member", tier

    raise HTTPException(status_code=401, detail="Authorization required")


def _map_tier(tier_str: str) -> TenantTier:
    """Map tier string to TenantTier enum."""
    mapping = {
        "free": TenantTier.FREE,
        "starter": TenantTier.FREE,
        "professional": TenantTier.PRO,
        "pro": TenantTier.PRO,
        "enterprise": TenantTier.ENTERPRISE,
        "business": TenantTier.ENTERPRISE,
    }
    return mapping.get(tier_str.lower(), TenantTier.FREE)


# ============================================================
# USAGE ENDPOINTS
# ============================================================


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    Get current quota usage for your tenant.

    Returns usage metrics with percentages and remaining capacity.
    """
    tenant_id, role, tier = auth

    quota_service = get_quota_service()
    usage_data = await quota_service.get_usage(tenant_id, tier)

    def build_category(data: dict) -> UsageCategory:
        return UsageCategory(
            per_minute=UsageMetric(**data["per_minute"]) if "per_minute" in data else None,
            per_hour=UsageMetric(**data["per_hour"]) if "per_hour" in data else None,
            per_day=UsageMetric(**data["per_day"]) if "per_day" in data else None,
            per_month=UsageMetric(**data["per_month"]) if "per_month" in data else None,
        )

    return UsageResponse(
        tenant_id=usage_data["tenant_id"],
        tier=usage_data["tier"],
        requests=build_category(usage_data["usage"].get("requests", {})),
        tokens=build_category(usage_data["usage"].get("tokens", {})),
        tools=build_category(usage_data["usage"].get("tools", {})),
        limits=usage_data["limits"],
        timestamp=datetime.fromisoformat(usage_data["timestamp"].replace("Z", "+00:00")),
    )


@router.get("/alerts", response_model=list[AlertResponse])
async def get_alerts(
    threshold: float = Query(default=80.0, ge=0, le=100),
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    Get quota alerts for quotas approaching limits.

    Returns quotas at or above the specified threshold percentage.
    Default threshold is 80%.
    """
    tenant_id, role, tier = auth

    quota_service = get_quota_service()
    alerts = await quota_service.get_usage_alerts(tenant_id, tier, threshold)

    return [AlertResponse(**alert) for alert in alerts]


@router.get("/limits", response_model=QuotaLimitsResponse)
async def get_limits(
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    Get quota limits for your tenant.

    Shows both numeric limits and feature flags.
    """
    tenant_id, role, tier = auth

    limits = get_tier_limits(tier)

    return QuotaLimitsResponse(
        tier=tier.value,
        limits=limits.to_dict(),
        requests_per_minute=limits.requests_per_minute,
        requests_per_hour=limits.requests_per_hour,
        requests_per_day=limits.requests_per_day,
        tokens_per_day=limits.tokens_per_day,
        tokens_per_month=limits.tokens_per_month,
        max_concurrent_requests=limits.max_concurrent_requests,
        tool_executions_per_hour=limits.tool_executions_per_hour,
        shell_executions_per_hour=limits.shell_executions_per_hour,
        allow_tool_execution=limits.allow_tool_execution,
        allow_shell_execution=limits.allow_shell_execution,
        allow_file_upload=limits.allow_file_upload,
        max_file_upload_mb=limits.max_file_upload_mb,
    )


@router.get("/tiers", response_model=TierLimitsResponse)
async def get_tier_comparison():
    """
    Get quota limits for all tiers.

    Useful for showing upgrade benefits.
    No authentication required.
    """
    tiers = {}
    for tier, limits in TIER_LIMITS.items():
        tiers[tier.value] = limits.to_dict()

    return TierLimitsResponse(tiers=tiers)


# ============================================================
# ADMIN ENDPOINTS
# ============================================================


@router.get("/admin/{target_tenant_id}/usage", response_model=UsageResponse)
async def admin_get_usage(
    target_tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    [Admin] Get quota usage for any tenant.

    Requires admin or owner role.
    """
    tenant_id, role, tier = auth

    if role not in ("admin", "owner"):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        target_id = UUID(target_tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid tenant ID") from e

    # Get target tenant tier
    tenant_repo = TenantRepository(session)
    target_tenant = await tenant_repo.get_by_id(target_id)
    if not target_tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    target_tier = _map_tier(target_tenant.tier)

    quota_service = get_quota_service()
    usage_data = await quota_service.get_usage(target_id, target_tier)

    def build_category(data: dict) -> UsageCategory:
        return UsageCategory(
            per_minute=UsageMetric(**data["per_minute"]) if "per_minute" in data else None,
            per_hour=UsageMetric(**data["per_hour"]) if "per_hour" in data else None,
            per_day=UsageMetric(**data["per_day"]) if "per_day" in data else None,
            per_month=UsageMetric(**data["per_month"]) if "per_month" in data else None,
        )

    return UsageResponse(
        tenant_id=usage_data["tenant_id"],
        tier=usage_data["tier"],
        requests=build_category(usage_data["usage"].get("requests", {})),
        tokens=build_category(usage_data["usage"].get("tokens", {})),
        tools=build_category(usage_data["usage"].get("tools", {})),
        limits=usage_data["limits"],
        timestamp=datetime.fromisoformat(usage_data["timestamp"].replace("Z", "+00:00")),
    )


@router.put("/admin/{target_tenant_id}/override")
async def admin_override_quotas(
    target_tenant_id: str,
    override: QuotaOverrideRequest,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    [Admin] Override quota limits for a specific tenant.

    Requires admin or owner role.

    Overrides are stored in the tenant's config and take precedence
    over tier defaults.
    """
    tenant_id, role, tier = auth

    if role not in ("admin", "owner"):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        target_id = UUID(target_tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid tenant ID") from e

    tenant_repo = TenantRepository(session)
    target_tenant = await tenant_repo.get_by_id(target_id)
    if not target_tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Update tenant config with overrides
    current_config = target_tenant.config or {}
    quota_overrides = current_config.get("quota_overrides", {})

    # Apply non-None overrides
    override_dict = override.model_dump(exclude_none=True)
    quota_overrides.update(override_dict)

    current_config["quota_overrides"] = quota_overrides

    await tenant_repo.update(
        target_id,
        config=current_config,
    )
    await session.commit()

    logger.info(f"Admin {tenant_id} updated quotas for tenant {target_id}: {override_dict}")

    return {
        "status": "updated",
        "tenant_id": str(target_id),
        "overrides": quota_overrides,
    }


@router.delete("/admin/{target_tenant_id}/override")
async def admin_clear_overrides(
    target_tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """
    [Admin] Clear all quota overrides for a tenant.

    Tenant will use default tier limits.
    """
    tenant_id, role, tier = auth

    if role not in ("admin", "owner"):
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        target_id = UUID(target_tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid tenant ID") from e

    tenant_repo = TenantRepository(session)
    target_tenant = await tenant_repo.get_by_id(target_id)
    if not target_tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    # Remove quota overrides from config
    current_config = target_tenant.config or {}
    if "quota_overrides" in current_config:
        del current_config["quota_overrides"]

    await tenant_repo.update(
        target_id,
        config=current_config,
    )
    await session.commit()

    logger.info(f"Admin {tenant_id} cleared quota overrides for tenant {target_id}")

    return {
        "status": "cleared",
        "tenant_id": str(target_id),
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = ["router"]
