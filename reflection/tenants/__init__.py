# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

from .context import TenantContext, get_current_context, require_context, tenant_context
from .lifecycle import (
    InvalidTransitionError,
    SuspensionReason,
    TenantDeletedError,
    TenantLifecycleError,
    TenantLifecycleEvent,
    TenantLifecycleService,
    TenantSuspendedError,
    check_tenant_active,
)
from .lifecycle import (
    TenantStatus as TenantLifecycleStatus,
)
from .models import Tenant, TenantConfig, TenantQuotas, TenantStatus, TenantTier
from .quota_service import (
    TIER_LIMITS,
    QuotaExceededError,
    QuotaLimits,
    QuotaService,
    TenantQuotaContext,
    get_quota_service,
    get_tier_limits,
    init_quota_service,
)
from .quotas import QuotaCheckResult, QuotaManager

__all__ = [
    # Models
    "Tenant",
    "TenantTier",
    "TenantStatus",
    "TenantConfig",
    "TenantQuotas",
    # Context
    "TenantContext",
    "tenant_context",
    "get_current_context",
    "require_context",
    # Quotas (low-level)
    "QuotaManager",
    "QuotaCheckResult",
    # Quota Service (Phase 6 - high-level)
    "QuotaService",
    "QuotaLimits",
    "QuotaExceededError",
    "TenantQuotaContext",
    "TIER_LIMITS",
    "get_tier_limits",
    "get_quota_service",
    "init_quota_service",
    # Lifecycle (v1.5.0)
    "TenantLifecycleStatus",
    "SuspensionReason",
    "TenantLifecycleEvent",
    "TenantLifecycleService",
    "TenantLifecycleError",
    "InvalidTransitionError",
    "TenantSuspendedError",
    "TenantDeletedError",
    "check_tenant_active",
]
