# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Reflection - Enterprise Multi-Tenant AI Companion Platform

"From one, many. From chaos, order — at scale."

Reflection extends Familiar with enterprise multi-tenancy.
All core AI capabilities come from Familiar; this package adds:

- Multi-tenant isolation
- Per-tenant configuration
- Usage tracking & billing
- Enterprise channels (shared bots)
- PostgreSQL persistence
- Horizontal scaling via Redis

Quick Start:
    from reflection import TenantAgent, get_tenant_agent

    # Get agent for a tenant
    agent = get_tenant_agent(tenant_uuid)
    response = agent.chat("Hello!", user_id="user123")

Architecture:

    ┌─────────────────────────────────────────────────────┐
    │               Reflection (Enterprise)           │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Tenant Wrappers (isolation + billing)       │   │
    │  │  ┌─────────────────────────────────────┐    │   │
    │  │  │        Familiar Core (AI brain)      │    │   │
    │  │  │  - Providers (Anthropic, OpenAI)    │    │   │
    │  │  │  - Tools, Memory, Skills            │    │   │
    │  │  │  - Channels (Discord, Telegram)     │    │   │
    │  │  └─────────────────────────────────────┘    │   │
    │  └─────────────────────────────────────────────┘   │
    │  ┌─────────────────────────────────────────────┐   │
    │  │  Enterprise Layer                            │   │
    │  │  - PostgreSQL (tenants, users, usage)       │   │
    │  │  - Redis (quotas, sessions, cache)          │   │
    │  │  - FastAPI Gateway (REST + SSE)             │   │
    │  │  - JWT Authentication                        │   │
    │  └─────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────┘

All imports are lazy — ``import reflection`` succeeds without
enterprise dependencies (sqlalchemy, pydantic-settings, redis, etc.).
Heavy modules are loaded on first attribute access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

__version__ = "2.0.0"
__author__ = "Familiar AI"

if TYPE_CHECKING:
    from .core.settings import Settings as Settings
    from .core.settings import get_settings as get_settings
    from .tenant_wrappers import (
        DISCORD_AVAILABLE as DISCORD_AVAILABLE,
    )
    from .tenant_wrappers import (
        TEAMS_AVAILABLE as TEAMS_AVAILABLE,
    )
    from .tenant_wrappers import (
        TELEGRAM_AVAILABLE as TELEGRAM_AVAILABLE,
    )
    from .tenant_wrappers import (
        TenantAgent as TenantAgent,
    )
    from .tenant_wrappers import (
        TenantAgentPool as TenantAgentPool,
    )
    from .tenant_wrappers import (
        TenantChannelManager as TenantChannelManager,
    )
    from .tenant_wrappers import (
        TenantChannelRouter as TenantChannelRouter,
    )
    from .tenant_wrappers import (
        TenantConversationHistory as TenantConversationHistory,
    )
    from .tenant_wrappers import (
        TenantDiscordChannel as TenantDiscordChannel,
    )
    from .tenant_wrappers import (
        TenantMemory as TenantMemory,
    )
    from .tenant_wrappers import (
        TenantTeamsChannel as TenantTeamsChannel,
    )
    from .tenant_wrappers import (
        TenantTelegramChannel as TenantTelegramChannel,
    )
    from .tenant_wrappers import (
        TenantToolRegistry as TenantToolRegistry,
    )
    from .tenant_wrappers import (
        get_agent_pool as get_agent_pool,
    )
    from .tenant_wrappers import (
        get_channel_manager as get_channel_manager,
    )
    from .tenant_wrappers import (
        get_tenant_agent as get_tenant_agent,
    )
    from .tenant_wrappers import (
        get_tenant_tool_registry as get_tenant_tool_registry,
    )
    from .tenants.context import (
        TenantContext as TenantContext,
    )
    from .tenants.context import (
        get_current_context as get_current_context,
    )
    from .tenants.context import (
        get_current_tenant as get_current_tenant,
    )
    from .tenants.context import (
        set_current_tenant as set_current_tenant,
    )
    from .tenants.context import (
        tenant_context as tenant_context,
    )
    from .tenants.quotas import (
        QuotaManager as QuotaManager,
    )
    from .tenants.quotas import (
        check_quota as check_quota,
    )
    from .tenants.quotas import (
        get_quota_manager as get_quota_manager,
    )
    from .tenants.quotas import (
        init_quota_manager as init_quota_manager,
    )

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Settings
    "Settings": (".core.settings", "Settings"),
    "get_settings": (".core.settings", "get_settings"),
    # Tenant Wrappers (primary interface)
    "TenantAgent": (".tenant_wrappers", "TenantAgent"),
    "TenantAgentPool": (".tenant_wrappers", "TenantAgentPool"),
    "get_agent_pool": (".tenant_wrappers", "get_agent_pool"),
    "get_tenant_agent": (".tenant_wrappers", "get_tenant_agent"),
    "TenantMemory": (".tenant_wrappers", "TenantMemory"),
    "TenantConversationHistory": (".tenant_wrappers", "TenantConversationHistory"),
    "TenantToolRegistry": (".tenant_wrappers", "TenantToolRegistry"),
    "get_tenant_tool_registry": (".tenant_wrappers", "get_tenant_tool_registry"),
    "TenantChannelRouter": (".tenant_wrappers", "TenantChannelRouter"),
    "TenantChannelManager": (".tenant_wrappers", "TenantChannelManager"),
    "get_channel_manager": (".tenant_wrappers", "get_channel_manager"),
    # Channel availability
    "DISCORD_AVAILABLE": (".tenant_wrappers", "DISCORD_AVAILABLE"),
    "TELEGRAM_AVAILABLE": (".tenant_wrappers", "TELEGRAM_AVAILABLE"),
    "TEAMS_AVAILABLE": (".tenant_wrappers", "TEAMS_AVAILABLE"),
    "TenantDiscordChannel": (".tenant_wrappers", "TenantDiscordChannel"),
    "TenantTelegramChannel": (".tenant_wrappers", "TenantTelegramChannel"),
    "TenantTeamsChannel": (".tenant_wrappers", "TenantTeamsChannel"),
    # Tenant Context
    "TenantContext": (".tenants.context", "TenantContext"),
    "tenant_context": (".tenants.context", "tenant_context"),
    "get_current_context": (".tenants.context", "get_current_context"),
    "get_current_tenant": (".tenants.context", "get_current_tenant"),
    "set_current_tenant": (".tenants.context", "set_current_tenant"),
    # Quotas
    "QuotaManager": (".tenants.quotas", "QuotaManager"),
    "get_quota_manager": (".tenants.quotas", "get_quota_manager"),
    "init_quota_manager": (".tenants.quotas", "init_quota_manager"),
    "check_quota": (".tenants.quotas", "check_quota"),
}

__all__ = [
    "__version__",
    *_LAZY_IMPORTS.keys(),
]


def __getattr__(name: str) -> object:
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
