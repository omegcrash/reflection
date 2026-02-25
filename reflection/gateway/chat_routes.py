# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Chat Routes with Full Agent Loop

Provides endpoints for:
- Chat completion with tool execution
- Streaming responses
- Conversation management
- Memory operations
"""

import contextlib
import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from reflection_core.security.trust import TrustLevel

from ..core.memory import MemoryService
from ..data.postgres import get_db_session
from ..data.repositories import (
    AgentRepository,
    ConversationRepository,
    MessageRepository,
    TenantRepository,
)
from ..services.orchestrator import AgentOrchestrator
from .auth import JWTService, TokenExpiredError, TokenInvalidError, TokenType, get_jwt_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""

    message: str = Field(..., min_length=1, max_length=100000)
    conversation_id: str | None = None
    agent_id: str | None = None
    stream: bool = False
    include_memory: bool = True
    max_tokens: int | None = Field(default=None, ge=1, le=200000)
    temperature: float | None = Field(default=None, ge=0, le=2)


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""

    text: str
    conversation_id: str
    message_id: str
    tool_results: list[dict] = []
    usage: dict
    iterations: int
    latency_ms: int


class ConversationResponse(BaseModel):
    """Conversation details."""

    id: str
    agent_id: str | None
    channel: str
    message_count: int
    total_tokens: int
    started_at: datetime
    last_message_at: datetime | None


class MessageResponse(BaseModel):
    """Message in a conversation."""

    id: str
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    created_at: datetime


class MemoryRequest(BaseModel):
    """Request to store a memory."""

    content: str = Field(..., min_length=1, max_length=2000)
    tags: list[str] = Field(default=[])


class MemoryResponse(BaseModel):
    """Memory entry response."""

    id: str
    content: str
    tags: list[str]
    created_at: datetime


# ============================================================
# AUTHENTICATION
# ============================================================


async def get_auth_context(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-ID"),
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    session: AsyncSession = Depends(get_db_session),
    jwt_service: JWTService = Depends(get_jwt_service),
) -> tuple[UUID, UUID | None, TrustLevel]:
    """
    Get authentication context.

    Returns:
        (tenant_id, user_id, trust_level)

    Authentication methods (in order of preference):
    1. JWT Bearer token - Full authentication with user context
    2. API Key - Service-level authentication
    3. X-Tenant-ID header - Development mode only (NEVER in production)
    """
    from ..core.settings import get_settings

    settings = get_settings()

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

                return tenant_id, user_id, trust_level

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

        return tenant.id, user_id, TrustLevel.KNOWN

    # X-Tenant-ID fallback - DEVELOPMENT MODE ONLY
    if x_tenant_id:
        # CRITICAL: Block X-Tenant-ID authentication in production
        if settings.is_production:
            logger.warning(
                "X-Tenant-ID authentication rejected in production environment",
                extra={
                    "x_tenant_id": x_tenant_id,
                    "environment": settings.environment,
                },
            )
            raise HTTPException(
                status_code=401,
                detail="X-Tenant-ID authentication is not allowed in production. "
                "Please use JWT or API key authentication.",
            )

        # Check if X-Tenant-ID auth is explicitly enabled
        if not settings.allow_x_tenant_id_auth:
            logger.warning(
                "X-Tenant-ID authentication rejected (not enabled)",
                extra={
                    "x_tenant_id": x_tenant_id,
                    "environment": settings.environment,
                },
            )
            raise HTTPException(
                status_code=401,
                detail="X-Tenant-ID authentication is disabled. "
                "Set ALLOW_X_TENANT_ID_AUTH=true for development, or use proper authentication.",
            )

        logger.info(
            f"Development mode: X-Tenant-ID authentication for '{x_tenant_id}'",
            extra={"environment": settings.environment},
        )

        tenant = await tenant_repo.get_by_slug(x_tenant_id)

        if not tenant:
            # Only auto-create if explicitly enabled
            if not settings.auto_create_tenants:
                raise HTTPException(
                    status_code=404,
                    detail=f"Tenant '{x_tenant_id}' not found. "
                    "Set AUTO_CREATE_TENANTS=true to auto-create in development.",
                )

            logger.warning(
                f"Auto-creating tenant in development mode: '{x_tenant_id}'",
                extra={"environment": settings.environment},
            )
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

        return tenant.id, user_id, TrustLevel.KNOWN

    raise HTTPException(status_code=401, detail="Authorization required")


# ============================================================
# CHAT ENDPOINTS
# ============================================================


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(
    request: ChatCompletionRequest,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """Send a message and get a response with full agent loop."""
    tenant_id, user_id, trust_level = auth

    agent_repo = AgentRepository(session)

    # Get agent
    agent_id = None
    if request.agent_id:
        try:
            agent_id = UUID(request.agent_id)
            agent = await agent_repo.get_by_id(agent_id)
            if not agent or agent.tenant_id != tenant_id:
                raise HTTPException(status_code=404, detail="Agent not found")
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid agent ID") from e
    else:
        agent = await agent_repo.get_default(tenant_id)
        if not agent:
            from ..core.settings import get_settings as _get_settings

            _settings = _get_settings()
            agent = await agent_repo.create(
                tenant_id=tenant_id,
                name="Default Assistant",
                provider=_settings.llm.default_provider,
                model=_settings.llm.default_model,
            )
            await session.flush()
        agent_id = agent.id

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
            async with AgentOrchestrator(
                session=session,
                tenant_id=tenant_id,
                user_id=user_id,
                trust_level=trust_level,
            ) as orchestrator:
                async for event in orchestrator.stream(
                    message=request.message,
                    agent_id=agent_id,
                    conversation_id=conversation_id,
                    include_memory=request.include_memory,
                ):
                    yield event.to_sse()

                yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    # Non-streaming
    async with AgentOrchestrator(
        session=session,
        tenant_id=tenant_id,
        user_id=user_id,
        trust_level=trust_level,
    ) as orchestrator:
        result = await orchestrator.chat(
            message=request.message,
            agent_id=agent_id,
            conversation_id=conversation_id,
            include_memory=request.include_memory,
        )

    return ChatCompletionResponse(
        text=result.text,
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        tool_results=result.tool_results,
        usage=result.usage,
        iterations=result.iterations,
        latency_ms=result.latency_ms,
    )


# ============================================================
# CONVERSATION ENDPOINTS
# ============================================================


@router.get("/conversations", response_model=list[ConversationResponse])
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """List conversations for the current user/tenant."""
    tenant_id, user_id, _ = auth

    conv_repo = ConversationRepository(session)

    if user_id:
        conversations = await conv_repo.list_by_user(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )
    else:
        conversations = await conv_repo.list_by_tenant(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset,
        )

    return [
        ConversationResponse(
            id=str(c.id),
            agent_id=str(c.agent_id) if c.agent_id else None,
            channel=c.channel,
            message_count=c.message_count,
            total_tokens=c.total_tokens,
            started_at=c.started_at,
            last_message_at=c.last_message_at,
        )
        for c in conversations
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """Get conversation details."""
    tenant_id, _, _ = auth

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from e

    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)

    if not conversation or conversation.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return ConversationResponse(
        id=str(conversation.id),
        agent_id=str(conversation.agent_id) if conversation.agent_id else None,
        channel=conversation.channel,
        message_count=conversation.message_count,
        total_tokens=conversation.total_tokens,
        started_at=conversation.started_at,
        last_message_at=conversation.last_message_at,
    )


@router.get("/conversations/{conversation_id}/messages", response_model=list[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """Get messages in a conversation."""
    tenant_id, _, _ = auth

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from e

    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)

    if not conversation or conversation.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    msg_repo = MessageRepository(session)
    messages = await msg_repo.list_by_conversation(
        conversation_id=conv_uuid,
        limit=limit,
        offset=offset,
    )

    return [
        MessageResponse(
            id=str(m.id),
            role=m.role,
            content=m.content,
            tool_calls=m.tool_calls,
            tool_results=m.tool_results,
            tokens_input=m.tokens_input,
            tokens_output=m.tokens_output,
            created_at=m.created_at,
        )
        for m in messages
    ]


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """Delete a conversation."""
    tenant_id, _, _ = auth

    try:
        conv_uuid = UUID(conversation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid conversation ID") from e

    conv_repo = ConversationRepository(session)
    conversation = await conv_repo.get_by_id(conv_uuid)

    if not conversation or conversation.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    await conv_repo.delete(conv_uuid)
    await session.commit()

    return {"status": "deleted", "conversation_id": conversation_id}


# ============================================================
# MEMORY ENDPOINTS
# ============================================================


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(
    request: MemoryRequest,
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """Store a new memory."""
    tenant_id, user_id, _ = auth

    memory_service = MemoryService()

    entry = await memory_service.store_memory(
        tenant_id=str(tenant_id),
        user_id=str(user_id) if user_id else None,
        content=request.content,
        tags=request.tags,
    )

    return MemoryResponse(
        id=entry.id,
        content=entry.content,
        tags=entry.tags,
        created_at=entry.created_at,
    )


@router.get("/memories", response_model=list[MemoryResponse])
async def list_memories(
    query: str | None = Query(default=None),
    limit: int = Query(default=10, ge=1, le=50),
    session: AsyncSession = Depends(get_db_session),
    auth: tuple = Depends(get_auth_context),
):
    """List or search memories."""
    tenant_id, user_id, _ = auth

    memory_service = MemoryService()

    if query:
        entries = await memory_service.search_memories(
            tenant_id=str(tenant_id),
            query=query,
            user_id=str(user_id) if user_id else None,
            limit=limit,
        )
    else:
        entries = await memory_service.get_recent_memories(
            tenant_id=str(tenant_id),
            user_id=str(user_id) if user_id else None,
            limit=limit,
        )

    return [
        MemoryResponse(
            id=e.id,
            content=e.content,
            tags=e.tags,
            created_at=e.created_at,
        )
        for e in entries
    ]
