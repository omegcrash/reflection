# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 5: Async Chat Routes (v2)

Chat endpoints using AsyncOrchestrator for:
- True async LLM calls (no thread pool for simple chat)
- Real-time streaming
- Automatic routing (async vs sync path)

These routes are API-compatible with v1 but use the new
dual-path architecture for better performance.
"""

import contextlib
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from reflection_core.security.trust import TrustLevel

from ..core.async_orchestrator import (
    AsyncOrchestrator,
)
from ..core.executor import TenantTier
from ..data.postgres import get_db_session
from ..data.repositories import (
    AgentRepository,
    TenantRepository,
)
from .auth import JWTService, TokenExpiredError, TokenInvalidError, TokenType, get_jwt_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v2", tags=["Chat V2 (Async)"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class AsyncChatRequest(BaseModel):
    """Request for async chat completion."""

    message: str = Field(..., min_length=1, max_length=100000)
    conversation_id: str | None = None
    agent_id: str | None = None
    stream: bool = False
    include_history: bool = True
    enable_tools: bool = False  # Set True to use sync path with full tool support
    system_prompt: str | None = None
    max_tokens: int = Field(default=4096, ge=1, le=200000)
    temperature: float = Field(default=0.7, ge=0, le=2)


class AsyncChatResponse(BaseModel):
    """Response from async chat completion."""

    text: str
    conversation_id: str
    message_id: str
    tool_results: list[dict] = []
    usage: dict
    iterations: int = 1
    latency_ms: int
    path_used: str  # "async" or "sync"
    cost_usd: str


class PathInfoResponse(BaseModel):
    """Information about execution path."""

    async_available: bool = True
    sync_available: bool = True
    recommended_path: str = "async"
    reason: str = "Simple chat uses async path for best performance"


# ============================================================
# AUTHENTICATION (reused from v1)
# ============================================================


async def get_auth_context_v2(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-ID"),
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> tuple[UUID, UUID | None, TrustLevel, TenantTier]:
    """
    Get authentication context with tenant tier.

    Returns: (tenant_id, user_id, trust_level, tenant_tier)
    """
    tenant_repo = TenantRepository(session)

    # Try JWT authentication
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]

        # Check if it's a JWT (contains dots) vs API key
        if token.count(".") == 2:
            try:
                payload = jwt_service.decode_token(token, TokenType.ACCESS)
                tenant_id = UUID(payload.tenant_id)
                user_id = UUID(payload.sub)

                trust_map = {
                    "owner": TrustLevel.OWNER,
                    "admin": TrustLevel.TRUSTED,
                    "member": TrustLevel.KNOWN,
                    "viewer": TrustLevel.STRANGER,
                }
                trust_level = trust_map.get(payload.role, TrustLevel.KNOWN)

                # Get tenant tier from database
                tenant = await tenant_repo.get_by_id(tenant_id)
                tier = _map_tenant_tier(tenant.tier if tenant else "free")

                return tenant_id, user_id, trust_level, tier

            except (TokenExpiredError, TokenInvalidError) as e:
                raise HTTPException(status_code=401, detail=str(e)) from e

        # API key authentication
        from ..data.repositories import APIKeyRepository

        api_key_repo = APIKeyRepository(session)

        api_key = await api_key_repo.validate_key(token)
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        tenant = await tenant_repo.get_by_id(api_key.tenant_id)
        if not tenant or tenant.status != "active":
            raise HTTPException(status_code=403, detail="Tenant not active")

        user_id = None
        if x_user_id:
            with contextlib.suppress(ValueError):
                user_id = UUID(x_user_id)

        tier = _map_tenant_tier(tenant.tier)
        return tenant.id, user_id, TrustLevel.KNOWN, tier

    # X-Tenant-ID fallback for development
    if x_tenant_id:
        tenant = await tenant_repo.get_by_slug(x_tenant_id)
        if not tenant:
            tenant = await tenant_repo.create(
                name=x_tenant_id.replace("-", " ").title(),
                slug=x_tenant_id,
                tier="professional",
                status="active",
                config={},
                quotas={"max_users": 100, "max_agents": 10},
            )
            await session.commit()

        user_id = None
        if x_user_id:
            with contextlib.suppress(ValueError):
                user_id = UUID(x_user_id)

        tier = _map_tenant_tier(tenant.tier)
        return tenant.id, user_id, TrustLevel.KNOWN, tier

    raise HTTPException(status_code=401, detail="Authorization required")


def _map_tenant_tier(tier_str: str) -> TenantTier:
    """Map tenant tier string to TenantTier enum."""
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
# CHAT ENDPOINTS
# ============================================================


@router.post("/chat/completions", response_model=AsyncChatResponse)
async def async_chat_completion(
    request: AsyncChatRequest,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context_v2),
):
    """
    Send a message with automatic async/sync path routing.

    By default, uses the async path for simple conversations.
    Set enable_tools=True to use the sync path with full tool support.

    Path Selection:
    - enable_tools=False → Async path (direct provider call, no thread pool)
    - enable_tools=True → Sync path (thread pool + full Familiar Agent)
    """
    tenant_id, user_id, trust_level, tenant_tier = auth

    agent_repo = AgentRepository(session)

    # Parse agent ID
    agent_id = None
    if request.agent_id:
        try:
            agent_id = UUID(request.agent_id)
            agent = await agent_repo.get_by_id(agent_id)
            if not agent or agent.tenant_id != tenant_id:
                raise HTTPException(status_code=404, detail="Agent not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid agent ID") from e

    # Parse conversation ID
    conversation_id = None
    if request.conversation_id:
        try:
            conversation_id = UUID(request.conversation_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid conversation ID") from e

    # Handle streaming
    if request.stream:

        async def event_generator():
            async with AsyncOrchestrator(
                session=session,
                tenant_id=tenant_id,
                tenant_tier=tenant_tier,
                user_id=user_id,
                trust_level=trust_level,
            ) as orchestrator:
                async for event in orchestrator.stream(
                    message=request.message,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    include_history=request.include_history,
                    enable_tools=request.enable_tools,
                    system_prompt=request.system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                ):
                    yield event.to_sse()

                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Non-streaming
    async with AsyncOrchestrator(
        session=session,
        tenant_id=tenant_id,
        tenant_tier=tenant_tier,
        user_id=user_id,
        trust_level=trust_level,
    ) as orchestrator:
        result = await orchestrator.chat(
            message=request.message,
            agent_id=agent_id,
            conversation_id=conversation_id,
            include_history=request.include_history,
            enable_tools=request.enable_tools,
            system_prompt=request.system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    return AsyncChatResponse(
        text=result.text,
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        tool_results=result.tool_results,
        usage=result.usage,
        iterations=result.iterations,
        latency_ms=result.latency_ms,
        path_used=result.path_used,
        cost_usd=result.cost_usd,
    )


@router.get("/chat/path-info", response_model=PathInfoResponse)
async def get_path_info():
    """
    Get information about available execution paths.

    Useful for understanding the dual-path architecture.
    """
    return PathInfoResponse(
        async_available=True,
        sync_available=True,
        recommended_path="async",
        reason=(
            "Async path uses direct LLM provider calls without thread pool overhead. "
            "Use sync path (enable_tools=True) when you need the full Familiar tool ecosystem."
        ),
    )


@router.post("/chat/simple")
async def simple_chat(
    message: str = Query(..., min_length=1, max_length=10000),
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context_v2),
):
    """
    Simplified chat endpoint for quick requests.

    Always uses async path. No conversation persistence.
    Useful for one-off queries, testing, or lightweight integrations.
    """
    tenant_id, user_id, trust_level, tenant_tier = auth

    async with AsyncOrchestrator(
        session=session,
        tenant_id=tenant_id,
        tenant_tier=tenant_tier,
        user_id=user_id,
        trust_level=trust_level,
    ) as orchestrator:
        result = await orchestrator.chat(
            message=message,
            include_history=False,
            enable_tools=False,
        )

    return {
        "text": result.text,
        "latency_ms": result.latency_ms,
        "path_used": result.path_used,
    }


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "router",
    "AsyncChatRequest",
    "AsyncChatResponse",
]
