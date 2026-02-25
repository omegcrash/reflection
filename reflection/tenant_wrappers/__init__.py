# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Wrappers

Multi-tenant wrappers around Familiar's core components.
These thin wrappers add tenant isolation, usage tracking,
and per-tenant configuration.
"""

from .agent import (
    TenantAgent,
    TenantAgentPool,
    get_agent_pool,
    get_tenant_agent,
)
from .channels import (
    DISCORD_AVAILABLE,
    TEAMS_AVAILABLE,
    TELEGRAM_AVAILABLE,
    TenantChannelManager,
    TenantChannelRouter,
    get_channel_manager,
)
from .memory import (
    TenantConversationHistory,
    TenantMemory,
)
from .tools import (
    TenantToolRegistry,
    clear_tenant_registry,
    get_tenant_tool_registry,
)

# Conditionally import channel classes
if DISCORD_AVAILABLE:
    from .channels import TenantDiscordChannel
else:
    TenantDiscordChannel = None

if TELEGRAM_AVAILABLE:
    from .channels import TenantTelegramChannel
else:
    TenantTelegramChannel = None

if TEAMS_AVAILABLE:
    from .channels import TenantTeamsChannel
else:
    TenantTeamsChannel = None


__all__ = [
    # Agent
    "TenantAgent",
    "TenantAgentPool",
    "get_agent_pool",
    "get_tenant_agent",
    # Memory
    "TenantMemory",
    "TenantConversationHistory",
    # Tools
    "TenantToolRegistry",
    "get_tenant_tool_registry",
    "clear_tenant_registry",
    # Channels (always available)
    "TenantChannelRouter",
    "TenantChannelManager",
    "get_channel_manager",
    # Channel availability flags
    "DISCORD_AVAILABLE",
    "TELEGRAM_AVAILABLE",
    "TEAMS_AVAILABLE",
    # Optional channel classes (may be None)
    "TenantDiscordChannel",
    "TenantTelegramChannel",
    "TenantTeamsChannel",
]
