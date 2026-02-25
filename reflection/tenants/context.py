# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Context

The TenantContext is the critical mechanism that ensures all operations
are properly scoped to a tenant. It:

1. Flows through all async operations via contextvars
2. Provides access to tenant-specific resources
3. Enforces tenant isolation at the code level
4. Tracks usage for quota enforcement

Usage:
    async with tenant_context(tenant) as ctx:
        # All operations in this block are scoped to tenant
        agent = await ctx.get_agent()
        response = await agent.chat("Hello")
        # Usage automatically tracked

Pattern:
    Request → Authenticate → Load Tenant → Create Context → Execute → Cleanup
"""

import logging
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from ..core.agent_async import AsyncAgent
    from ..core.providers_async import AsyncLLMProvider
    from ..core.tools_async import AsyncToolRegistry
    from .models import Tenant, TenantAPIKey

logger = logging.getLogger(__name__)

# Context variable for current tenant - automatically propagates through async calls
_current_tenant_context: ContextVar[Optional["TenantContext"]] = ContextVar(
    "current_tenant_context", default=None
)

# Simple UUID-based tenant tracking (for sync wrappers)
_current_tenant_id: ContextVar[Any | None] = ContextVar("current_tenant_id", default=None)


def set_current_tenant(tenant_id: Any | None):
    """Set the current tenant ID (for sync code)."""
    _current_tenant_id.set(tenant_id)


def get_current_tenant_id() -> Any | None:
    """Get the current tenant ID."""
    # First check context
    ctx = _current_tenant_context.get()
    if ctx:
        return ctx.tenant_id
    # Fall back to simple ID
    return _current_tenant_id.get()


def utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class TenantUsage:
    """
    Tracks resource usage within a request/session.

    Accumulated and persisted at request completion.
    """

    tokens_input: int = 0
    tokens_output: int = 0
    tool_executions: int = 0
    llm_calls: int = 0
    request_count: int = 0

    def add_llm_usage(self, input_tokens: int, output_tokens: int):
        """Record LLM API usage."""
        self.tokens_input += input_tokens
        self.tokens_output += output_tokens
        self.llm_calls += 1

    def add_tool_execution(self):
        """Record tool execution."""
        self.tool_executions += 1

    def add_request(self):
        """Record a request."""
        self.request_count += 1

    @property
    def total_tokens(self) -> int:
        return self.tokens_input + self.tokens_output

    def to_dict(self) -> dict[str, int]:
        return {
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "tokens_total": self.total_tokens,
            "tool_executions": self.tool_executions,
            "llm_calls": self.llm_calls,
            "request_count": self.request_count,
        }


@dataclass
class TenantContext:
    """
    Runtime context for a tenant.

    Created per-request and provides:
    - Access to tenant configuration
    - Tenant-scoped agent instance
    - Usage tracking
    - Audit context

    All tenant-scoped operations should access the tenant through this context.
    """

    tenant: "Tenant"
    user_id: str | None = None
    api_key: Optional["TenantAPIKey"] = None
    request_id: str | None = None
    channel: str = "api"

    # Runtime state (not persisted)
    usage: TenantUsage = field(default_factory=TenantUsage)
    started_at: datetime = field(default_factory=utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Cached resources (lazily initialized)
    _agent: Optional["AsyncAgent"] = field(default=None, repr=False)
    _provider: Optional["AsyncLLMProvider"] = field(default=None, repr=False)
    _tools: Optional["AsyncToolRegistry"] = field(default=None, repr=False)

    @property
    def tenant_id(self) -> str:
        """Convenience accessor for tenant ID."""
        return self.tenant.id

    @property
    def is_active(self) -> bool:
        """Check if tenant is active."""
        return self.tenant.is_active

    async def get_agent(self) -> "AsyncAgent":
        """
        Get or create the async agent for this tenant.

        Agents are tenant-scoped and configured per tenant settings.
        """
        if self._agent is None:
            from ..core.agent_async import AsyncAgent

            # Create agent with tenant-specific config
            self._agent = await AsyncAgent.create_for_tenant(self.tenant)
            logger.debug(f"Created agent for tenant {self.tenant_id}")

        return self._agent

    async def get_provider(self) -> "AsyncLLMProvider":
        """Get the LLM provider for this tenant."""
        if self._provider is None:
            from ..core.providers_async import get_async_provider

            provider_name = self.tenant.config.default_provider
            self._provider = await get_async_provider(provider_name, self.tenant.config)

        return self._provider

    async def get_tools(self) -> "AsyncToolRegistry":
        """Get the tool registry for this tenant."""
        if self._tools is None:
            from ..core.tools_async import AsyncToolRegistry

            # Create tenant-scoped tool registry
            self._tools = AsyncToolRegistry(
                tenant_id=self.tenant_id,
                config=self.tenant.config,
            )

        return self._tools

    def track_tokens(self, input_tokens: int, output_tokens: int):
        """Track token usage (called by provider)."""
        self.usage.add_llm_usage(input_tokens, output_tokens)

    def track_tool(self):
        """Track tool execution (called by tool registry)."""
        self.usage.add_tool_execution()

    def get_database_schema(self) -> str:
        """
        Get the database schema for this tenant.

        Used for schema-per-tenant isolation.
        """
        return f"tenant_{self.tenant.slug}"

    def get_storage_prefix(self) -> str:
        """
        Get the storage prefix for this tenant.

        Used for object storage isolation.
        """
        return f"tenants/{self.tenant_id}/"

    def get_vector_namespace(self) -> str:
        """
        Get the vector database namespace for this tenant.

        Used for memory/embedding isolation.
        """
        return f"tenant:{self.tenant_id}"

    async def cleanup(self):
        """
        Clean up resources when context ends.

        Called automatically by async context manager.
        """
        # Close provider connections
        if self._provider:
            try:
                await self._provider.close()
            except Exception as e:
                logger.warning(f"Error closing provider: {e}")

        # Persist usage
        await self._persist_usage()

    async def _persist_usage(self):
        """Persist usage metrics to database."""
        # This would write to the usage tracking table
        logger.debug(
            f"Tenant {self.tenant_id} usage: "
            f"{self.usage.total_tokens} tokens, "
            f"{self.usage.tool_executions} tools"
        )

    def to_audit_context(self) -> dict[str, Any]:
        """Get context for audit logging."""
        return {
            "tenant_id": self.tenant_id,
            "tenant_name": self.tenant.name,
            "user_id": self.user_id,
            "api_key_id": self.api_key.id if self.api_key else None,
            "request_id": self.request_id,
            "channel": self.channel,
        }


def get_current_context() -> TenantContext | None:
    """
    Get the current tenant context.

    Returns None if not in a tenant context.

    Usage:
        ctx = get_current_context()
        if ctx:
            tenant_id = ctx.tenant_id
    """
    return _current_tenant_context.get()


def require_context() -> TenantContext:
    """
    Get the current tenant context or raise.

    Use this when tenant context is required.

    Usage:
        ctx = require_context()  # Raises if no context
        agent = await ctx.get_agent()
    """
    ctx = _current_tenant_context.get()
    if ctx is None:
        raise RuntimeError("No tenant context. Operations must be within a tenant_context().")
    return ctx


def get_current_tenant() -> Optional["Tenant"]:
    """Get the current tenant (convenience method)."""
    ctx = get_current_context()
    return ctx.tenant if ctx else None


def require_tenant() -> "Tenant":
    """Get the current tenant or raise."""
    return require_context().tenant


@asynccontextmanager
async def tenant_context(
    tenant: "Tenant",
    user_id: str | None = None,
    api_key: Optional["TenantAPIKey"] = None,
    request_id: str | None = None,
    channel: str = "api",
):
    """
    Async context manager for tenant-scoped operations.

    All operations within this context are scoped to the tenant.
    Usage is tracked and resources are cleaned up on exit.

    Usage:
        async with tenant_context(tenant, user_id="user123") as ctx:
            agent = await ctx.get_agent()
            response = await agent.chat("Hello")

    Args:
        tenant: The tenant for this context
        user_id: Optional user ID within the tenant
        api_key: Optional API key used for authentication
        request_id: Optional request ID for tracing
        channel: Channel identifier (api, slack, teams, etc.)

    Yields:
        TenantContext instance
    """
    import uuid

    # Generate request ID if not provided
    if request_id is None:
        request_id = f"req_{uuid.uuid4().hex[:16]}"

    # Create context
    ctx = TenantContext(
        tenant=tenant,
        user_id=user_id,
        api_key=api_key,
        request_id=request_id,
        channel=channel,
    )

    # Set as current context
    token = _current_tenant_context.set(ctx)

    try:
        logger.debug(f"Entering tenant context: {tenant.id} (request: {request_id})")
        yield ctx
    finally:
        # Cleanup
        await ctx.cleanup()

        # Reset context
        _current_tenant_context.reset(token)

        logger.debug(f"Exiting tenant context: {tenant.id} (tokens: {ctx.usage.total_tokens})")


class TenantMiddleware:
    """
    ASGI middleware for tenant context injection.

    Extracts tenant from request (subdomain, header, or API key)
    and injects TenantContext for the request lifecycle.
    """

    def __init__(self, app, tenant_resolver):
        self.app = app
        self.tenant_resolver = tenant_resolver

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        # Resolve tenant from request
        tenant, api_key = await self.tenant_resolver.resolve(scope)

        if tenant is None:
            # No tenant - return 401
            await send(
                {
                    "type": "http.response.start",
                    "status": 401,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "tenant_not_found"}',
                }
            )
            return

        if not tenant.is_active:
            # Tenant suspended
            await send(
                {
                    "type": "http.response.start",
                    "status": 403,
                    "headers": [[b"content-type", b"application/json"]],
                }
            )
            await send(
                {
                    "type": "http.response.body",
                    "body": b'{"error": "tenant_suspended"}',
                }
            )
            return

        # Run request within tenant context
        async with tenant_context(tenant, api_key=api_key) as ctx:
            # Inject context into scope for handlers
            scope["tenant_context"] = ctx
            await self.app(scope, receive, send)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "TenantContext",
    "TenantUsage",
    "tenant_context",
    "get_current_context",
    "require_context",
    "get_current_tenant",
    "get_current_tenant_id",
    "set_current_tenant",
    "require_tenant",
    "TenantMiddleware",
]
