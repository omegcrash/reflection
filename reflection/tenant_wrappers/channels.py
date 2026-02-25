# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Channels

Multi-tenant wrappers around Familiar's channel implementations.
Adds:
- Per-tenant bot tokens
- Tenant routing for shared bots
- Usage tracking per channel
- Channel-specific tenant configuration

Note: Channel imports are optional - only available if the
corresponding familiar channel dependencies are installed.
"""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

logger = logging.getLogger(__name__)

# Try to import Familiar channels (optional dependencies)
DISCORD_AVAILABLE = False
TELEGRAM_AVAILABLE = False
TEAMS_AVAILABLE = False

try:
    from familiar.channels.discord import DiscordChannel

    DISCORD_AVAILABLE = True
except ImportError:
    DiscordChannel = None
    logger.debug("Discord channel not available (discord.py not installed)")

try:
    from familiar.channels.telegram import TelegramChannel

    TELEGRAM_AVAILABLE = True
except ImportError:
    TelegramChannel = None
    logger.debug("Telegram channel not available (python-telegram-bot not installed)")

try:
    from familiar.channels.teams import TeamsChannel

    TEAMS_AVAILABLE = True
except ImportError:
    TeamsChannel = None
    logger.debug("Teams channel not available (botbuilder not installed)")

if TYPE_CHECKING:
    from .agent import TenantAgent


class TenantChannelRouter:
    """
    Routes messages to appropriate tenant agent for shared bots.

    Useful when one bot serves multiple tenants based on:
    - Server/guild ID (Discord)
    - Chat/group ID (Telegram)
    - Team ID (Teams)

    Usage:
        router = TenantChannelRouter()
        router.register_server(server_id=123, tenant_id=uuid1)
        router.register_server(server_id=456, tenant_id=uuid2)

        tenant_id = router.get_tenant_for_server(123)  # Returns uuid1
    """

    def __init__(self):
        self._server_to_tenant: dict[str, UUID] = {}
        self._chat_to_tenant: dict[str, UUID] = {}
        self._tenant_agents: dict[UUID, TenantAgent] = {}

    def register_server(self, server_id: Any, tenant_id: UUID):
        """Register a server/guild to a tenant."""
        self._server_to_tenant[str(server_id)] = tenant_id
        logger.info(f"Registered server {server_id} to tenant {tenant_id}")

    def register_chat(self, chat_id: Any, tenant_id: UUID):
        """Register a chat/group to a tenant."""
        self._chat_to_tenant[str(chat_id)] = tenant_id
        logger.info(f"Registered chat {chat_id} to tenant {tenant_id}")

    def get_tenant_for_server(self, server_id: Any) -> UUID | None:
        """Get tenant ID for a server."""
        return self._server_to_tenant.get(str(server_id))

    def get_tenant_for_chat(self, chat_id: Any) -> UUID | None:
        """Get tenant ID for a chat."""
        return self._chat_to_tenant.get(str(chat_id))

    def get_agent(self, tenant_id: UUID) -> "TenantAgent":
        """Get or create agent for tenant."""
        if tenant_id not in self._tenant_agents:
            from .agent import get_tenant_agent

            self._tenant_agents[tenant_id] = get_tenant_agent(tenant_id)
        return self._tenant_agents[tenant_id]

    def route_message(
        self,
        message: str,
        server_id: Any | None = None,
        chat_id: Any | None = None,
        user_id: Any = "unknown",
        channel: str = "unknown",
    ) -> str | None:
        """Route message to appropriate tenant agent."""
        tenant_id = None
        if server_id:
            tenant_id = self.get_tenant_for_server(server_id)
        if not tenant_id and chat_id:
            tenant_id = self.get_tenant_for_chat(chat_id)

        if not tenant_id:
            logger.warning(f"No tenant found for server={server_id} chat={chat_id}")
            return None

        agent = self.get_agent(tenant_id)
        return agent.chat(message, user_id=user_id, channel=channel)


class TenantChannelManager:
    """
    Manages all channels for all tenants.

    Provides:
    - Channel lifecycle management
    - Tenant onboarding/offboarding
    - Health monitoring
    """

    def __init__(self):
        self._channels: dict[UUID, dict[str, Any]] = {}
        self._running: dict[UUID, dict[str, asyncio.Task]] = {}

    async def start_channel(self, tenant_id: UUID, channel_type: str, config: dict[str, Any]):
        """Start a channel for a tenant."""
        if tenant_id not in self._channels:
            self._channels[tenant_id] = {}
            self._running[tenant_id] = {}

        channel = None

        if channel_type == "discord" and DISCORD_AVAILABLE:
            channel = TenantDiscordChannel(
                tenant_id=tenant_id,
                token=config.get("token"),
                tenant_config=config,
            )
        elif channel_type == "telegram" and TELEGRAM_AVAILABLE:
            channel = TenantTelegramChannel(
                tenant_id=tenant_id,
                token=config.get("token"),
                tenant_config=config,
            )
        elif channel_type == "teams" and TEAMS_AVAILABLE:
            channel = TenantTeamsChannel(
                tenant_id=tenant_id,
                tenant_config=config,
            )
        else:
            raise ValueError(f"Channel type '{channel_type}' not available")

        self._channels[tenant_id][channel_type] = channel

        task = asyncio.create_task(channel.start())
        self._running[tenant_id][channel_type] = task

        logger.info(f"Started {channel_type} channel for tenant {tenant_id}")

    async def stop_channel(self, tenant_id: UUID, channel_type: str):
        """Stop a channel for a tenant."""
        if tenant_id in self._running and channel_type in self._running[tenant_id]:
            task = self._running[tenant_id][channel_type]
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

            del self._running[tenant_id][channel_type]
            del self._channels[tenant_id][channel_type]

            logger.info(f"Stopped {channel_type} channel for tenant {tenant_id}")

    def get_channel_status(self, tenant_id: UUID) -> dict[str, str]:
        """Get status of all channels for a tenant."""
        status = {}
        if tenant_id in self._running:
            for channel_type, task in self._running[tenant_id].items():
                if task.done():
                    status[channel_type] = "error" if task.exception() else "stopped"
                else:
                    status[channel_type] = "running"
        return status


# Global channel manager
_channel_manager: TenantChannelManager | None = None


def get_channel_manager() -> TenantChannelManager:
    """Get global channel manager."""
    global _channel_manager
    if _channel_manager is None:
        _channel_manager = TenantChannelManager()
    return _channel_manager


# Conditional channel class definitions
if DISCORD_AVAILABLE:

    class TenantDiscordChannel(DiscordChannel):
        """Multi-tenant Discord channel."""

        def __init__(
            self,
            tenant_id: UUID,
            agent=None,
            token: str | None = None,
            tenant_config: dict[str, Any] | None = None,
            **kwargs,
        ):
            self.tenant_id = tenant_id
            self.tenant_config = tenant_config or {}

            if agent is None:
                from .agent import get_tenant_agent

                agent = get_tenant_agent(tenant_id, tenant_config=tenant_config)

            super().__init__(agent=agent, token=token, **kwargs)
            logger.info(f"TenantDiscordChannel initialized for tenant {tenant_id}")
else:
    TenantDiscordChannel = None


if TELEGRAM_AVAILABLE:

    class TenantTelegramChannel(TelegramChannel):
        """Multi-tenant Telegram channel."""

        def __init__(
            self,
            tenant_id: UUID,
            agent=None,
            token: str | None = None,
            tenant_config: dict[str, Any] | None = None,
            **kwargs,
        ):
            self.tenant_id = tenant_id
            self.tenant_config = tenant_config or {}

            if agent is None:
                from .agent import get_tenant_agent

                agent = get_tenant_agent(tenant_id, tenant_config=tenant_config)

            super().__init__(agent=agent, token=token, **kwargs)
            logger.info(f"TenantTelegramChannel initialized for tenant {tenant_id}")
else:
    TenantTelegramChannel = None


if TEAMS_AVAILABLE:

    class TenantTeamsChannel(TeamsChannel):
        """Multi-tenant Microsoft Teams channel."""

        def __init__(
            self, tenant_id: UUID, agent=None, tenant_config: dict[str, Any] | None = None, **kwargs
        ):
            self.tenant_id = tenant_id
            self.tenant_config = tenant_config or {}

            if agent is None:
                from .agent import get_tenant_agent

                agent = get_tenant_agent(tenant_id, tenant_config=tenant_config)

            super().__init__(agent=agent, **kwargs)
            logger.info(f"TenantTeamsChannel initialized for tenant {tenant_id}")
else:
    TenantTeamsChannel = None


__all__ = [
    "TenantChannelRouter",
    "TenantChannelManager",
    "get_channel_manager",
    "DISCORD_AVAILABLE",
    "TELEGRAM_AVAILABLE",
    "TEAMS_AVAILABLE",
]

# Only export channel classes if available
if TenantDiscordChannel:
    __all__.append("TenantDiscordChannel")
if TenantTelegramChannel:
    __all__.append("TenantTelegramChannel")
if TenantTeamsChannel:
    __all__.append("TenantTeamsChannel")
