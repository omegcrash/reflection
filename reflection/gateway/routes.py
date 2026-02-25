# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
API Routes with Database Persistence

All routes use repositories for database access.
"""

import contextlib
import logging
from datetime import UTC, datetime
from uuid import UUID

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.settings import DEFAULT_ANTHROPIC_MODEL
from ..data.models import APIKeyModel, TenantModel
from ..data.postgres import get_db_session
from ..data.repositories import (
    AgentRepository,
    APIKeyRepository,
    ConversationRepository,
    MessageRepository,
    TenantRepository,
    TenantUserRepository,
    UsageRepository,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================


class CreateTenantRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    slug: str = Field(..., min_length=1, max_length=63, pattern=r"^[a-z0-9-]+$")
    tier: str = Field(default="free", pattern=r"^(free|professional|enterprise)$")
    config: dict | None = None


class TenantResponse(BaseModel):
    id: str
    name: str
    slug: str
    tier: str
    status: str
    created_at: datetime


class TenantDetailResponse(TenantResponse):
    config: dict
    quotas: dict
    user_count: int = 0
    agent_count: int = 0


class CreateUserRequest(BaseModel):
    email: EmailStr
    display_name: str | None = None
    role: str = Field(default="member", pattern=r"^(owner|admin|member|viewer)$")
    trust_level: str = Field(default="known", pattern=r"^(stranger|known|trusted|owner)$")


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str | None
    role: str
    trust_level: str
    is_active: bool
    created_at: datetime


class CreateAPIKeyRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    permissions: list[str] = Field(default=["chat", "read"])
    expires_in_days: int | None = Field(default=None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    id: str
    name: str
    key_prefix: str
    permissions: list[str]
    is_active: bool
    expires_at: datetime | None
    created_at: datetime
    last_used_at: datetime | None


class APIKeyCreatedResponse(APIKeyResponse):
    key: str  # Only returned on creation!


class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    provider: str = Field(default="anthropic")
    model: str = Field(default=DEFAULT_ANTHROPIC_MODEL)
    system_prompt: str | None = None
    skills: list[str] = Field(default=[])
    max_tokens_per_request: int = Field(default=4096, ge=256, le=200000)
    temperature: float = Field(default=0.7, ge=0, le=2)


class AgentResponse(BaseModel):
    id: str
    name: str
    description: str | None
    provider: str
    model: str
    status: str
    skills: list[str]
    created_at: datetime


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=100000)
    conversation_id: str | None = None
    stream: bool = False


class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    usage: dict


class UsageResponse(BaseModel):
    period: dict
    totals: dict
    daily: list[dict] | None = None


# ============================================================
# AUTHENTICATION
# ============================================================


async def get_tenant_from_api_key(
    authorization: str | None = Header(None),
    x_tenant_id: str | None = Header(None, alias="X-Tenant-ID"),
    session: AsyncSession = Depends(get_db_session),
) -> tuple[TenantModel, APIKeyModel | None]:
    """
    Authenticate request and return tenant.

    Supports:
    - API key in Authorization header (Bearer token)
    - X-Tenant-ID header for development
    """
    api_key_repo = APIKeyRepository(session)
    tenant_repo = TenantRepository(session)

    # Try API key authentication first
    if authorization and authorization.startswith("Bearer "):
        api_key_str = authorization[7:]

        api_key = await api_key_repo.validate_key(api_key_str)
        if not api_key:
            raise HTTPException(status_code=401, detail="Invalid or expired API key")

        tenant = await tenant_repo.get_by_id(api_key.tenant_id)
        if not tenant:
            raise HTTPException(status_code=401, detail="Tenant not found")

        if tenant.status != "active":
            raise HTTPException(status_code=403, detail=f"Tenant is {tenant.status}")

        return tenant, api_key

    # Fall back to X-Tenant-ID for development
    if x_tenant_id:
        tenant = await tenant_repo.get_by_slug(x_tenant_id)
        if not tenant:
            # Auto-create for development
            tenant = await tenant_repo.create(
                name=x_tenant_id.replace("-", " ").title(),
                slug=x_tenant_id,
                tier="professional",
                status="active",
                config={},
                quotas={"max_users": 100, "max_agents": 10},
            )
            await session.commit()

        return tenant, None

    raise HTTPException(status_code=401, detail="Authorization header or X-Tenant-ID required")


async def require_tenant(
    auth: tuple = Depends(get_tenant_from_api_key),
) -> TenantModel:
    """Dependency that returns just the tenant."""
    return auth[0]


# ============================================================
# TENANT ROUTES
# ============================================================


@router.post("/tenants", response_model=TenantResponse, tags=["Tenants"])
async def create_tenant(
    request: CreateTenantRequest,
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new tenant."""
    tenant_repo = TenantRepository(session)

    existing = await tenant_repo.get_by_slug(request.slug)
    if existing:
        raise HTTPException(status_code=409, detail="Tenant slug already exists")

    tenant = await tenant_repo.create(
        name=request.name,
        slug=request.slug,
        tier=request.tier,
        status="active",
        config=request.config or {},
        quotas={"max_users": 100, "max_agents": 10},
    )

    await session.commit()
    logger.info(f"Created tenant: {tenant.slug} ({tenant.id})")

    return TenantResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        tier=tenant.tier,
        status=tenant.status,
        created_at=tenant.created_at,
    )


@router.get("/tenants/{tenant_id}", response_model=TenantDetailResponse, tags=["Tenants"])
async def get_tenant(
    tenant_id: str,
    session: AsyncSession = Depends(get_db_session),
):
    """Get tenant details."""
    tenant_repo = TenantRepository(session)

    try:
        uuid_id = UUID(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid tenant ID") from e

    tenant = await tenant_repo.get_by_id(uuid_id)
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")

    stats = await tenant_repo.get_usage_stats(uuid_id)

    return TenantDetailResponse(
        id=str(tenant.id),
        name=tenant.name,
        slug=tenant.slug,
        tier=tenant.tier,
        status=tenant.status,
        config=tenant.config,
        quotas=tenant.quotas,
        created_at=tenant.created_at,
        user_count=stats["user_count"],
        agent_count=stats["agent_count"],
    )


@router.get("/tenants", response_model=list[TenantResponse], tags=["Tenants"])
async def list_tenants(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    session: AsyncSession = Depends(get_db_session),
):
    """List all tenants."""
    tenant_repo = TenantRepository(session)
    tenants = await tenant_repo.list_active(limit=limit, offset=offset)

    return [
        TenantResponse(
            id=str(t.id),
            name=t.name,
            slug=t.slug,
            tier=t.tier,
            status=t.status,
            created_at=t.created_at,
        )
        for t in tenants
    ]


# ============================================================
# USER ROUTES
# ============================================================


@router.post("/users", response_model=UserResponse, tags=["Users"])
async def create_user(
    request: CreateUserRequest,
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new user in the tenant."""
    user_repo = TenantUserRepository(session)

    existing = await user_repo.get_by_email(tenant.id, request.email)
    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    user_count = await user_repo.count_by_tenant(tenant.id)
    max_users = tenant.quotas.get("max_users", 100)
    if user_count >= max_users:
        raise HTTPException(status_code=403, detail=f"User quota exceeded ({max_users})")

    user = await user_repo.create(
        tenant_id=tenant.id,
        email=request.email,
        display_name=request.display_name,
        role=request.role,
        trust_level=request.trust_level,
    )

    await session.commit()

    return UserResponse(
        id=str(user.id),
        email=user.email,
        display_name=user.display_name,
        role=user.role,
        trust_level=user.trust_level,
        is_active=user.is_active,
        created_at=user.created_at,
    )


@router.get("/users", response_model=list[UserResponse], tags=["Users"])
async def list_users(
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
    limit: int = Query(default=50, ge=1, le=100),
):
    """List users in the tenant."""
    user_repo = TenantUserRepository(session)
    users = await user_repo.list_by_tenant(tenant.id, limit=limit)

    return [
        UserResponse(
            id=str(u.id),
            email=u.email,
            display_name=u.display_name,
            role=u.role,
            trust_level=u.trust_level,
            is_active=u.is_active,
            created_at=u.created_at,
        )
        for u in users
    ]


# ============================================================
# API KEY ROUTES
# ============================================================


@router.post("/api-keys", response_model=APIKeyCreatedResponse, tags=["API Keys"])
async def create_api_key(
    request: CreateAPIKeyRequest,
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new API key. The key value is only returned once!"""
    api_key_repo = APIKeyRepository(session)

    expires_at = None
    if request.expires_in_days:
        from datetime import timedelta

        expires_at = datetime.now(UTC) + timedelta(days=request.expires_in_days)

    api_key, plaintext = await api_key_repo.create_key(
        tenant_id=tenant.id,
        name=request.name,
        permissions=request.permissions,
        expires_at=expires_at,
    )

    await session.commit()

    return APIKeyCreatedResponse(
        id=str(api_key.id),
        name=api_key.name,
        key_prefix=api_key.key_prefix,
        permissions=api_key.permissions,
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at,
        last_used_at=api_key.last_used_at,
        key=plaintext,
    )


@router.get("/api-keys", response_model=list[APIKeyResponse], tags=["API Keys"])
async def list_api_keys(
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """List API keys for the tenant."""
    api_key_repo = APIKeyRepository(session)
    keys = await api_key_repo.list_by_tenant(tenant.id)

    return [
        APIKeyResponse(
            id=str(k.id),
            name=k.name,
            key_prefix=k.key_prefix,
            permissions=k.permissions,
            is_active=k.is_active,
            expires_at=k.expires_at,
            created_at=k.created_at,
            last_used_at=k.last_used_at,
        )
        for k in keys
    ]


@router.delete("/api-keys/{key_id}", tags=["API Keys"])
async def revoke_api_key(
    key_id: str,
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Revoke an API key."""
    api_key_repo = APIKeyRepository(session)

    try:
        uuid_id = UUID(key_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid key ID") from e

    key = await api_key_repo.get_by_id(uuid_id)
    if not key or key.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="API key not found")

    await api_key_repo.revoke(uuid_id)
    await session.commit()

    return {"status": "revoked", "key_id": key_id}


# ============================================================
# AGENT ROUTES
# ============================================================


@router.post("/agents", response_model=AgentResponse, tags=["Agents"])
async def create_agent(
    request: CreateAgentRequest,
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Create a new agent."""
    agent_repo = AgentRepository(session)

    agent_count = await agent_repo.count_by_tenant(tenant.id)
    max_agents = tenant.quotas.get("max_agents", 10)
    if agent_count >= max_agents:
        raise HTTPException(status_code=403, detail=f"Agent quota exceeded ({max_agents})")

    agent = await agent_repo.create(
        tenant_id=tenant.id,
        name=request.name,
        description=request.description,
        provider=request.provider,
        model=request.model,
        system_prompt=request.system_prompt,
        skills=request.skills,
        max_tokens_per_request=request.max_tokens_per_request,
        temperature=float(request.temperature),
    )

    await session.commit()

    return AgentResponse(
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        provider=agent.provider,
        model=agent.model,
        status=agent.status,
        skills=agent.skills,
        created_at=agent.created_at,
    )


@router.get("/agents", response_model=list[AgentResponse], tags=["Agents"])
async def list_agents(
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """List agents for the tenant."""
    agent_repo = AgentRepository(session)
    agents = await agent_repo.list_by_tenant(tenant.id)

    return [
        AgentResponse(
            id=str(a.id),
            name=a.name,
            description=a.description,
            provider=a.provider,
            model=a.model,
            status=a.status,
            skills=a.skills,
            created_at=a.created_at,
        )
        for a in agents
    ]


@router.get("/agents/{agent_id}", response_model=AgentResponse, tags=["Agents"])
async def get_agent(
    agent_id: str,
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
):
    """Get agent details."""
    agent_repo = AgentRepository(session)

    try:
        uuid_id = UUID(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid agent ID") from e

    agent = await agent_repo.get_by_id(uuid_id)
    if not agent or agent.tenant_id != tenant.id:
        raise HTTPException(status_code=404, detail="Agent not found")

    return AgentResponse(
        id=str(agent.id),
        name=agent.name,
        description=agent.description,
        provider=agent.provider,
        model=agent.model,
        status=agent.status,
        skills=agent.skills,
        created_at=agent.created_at,
    )


# ============================================================
# CHAT ROUTES
# ============================================================


@router.post("/chat", tags=["Chat"])
async def chat(
    request: ChatRequest,
    tenant: TenantModel = Depends(require_tenant),
    x_user_id: str | None = Header(None, alias="X-User-ID"),
    session: AsyncSession = Depends(get_db_session),
):
    """Send a message to the default agent."""
    import json
    import time

    from ..core.providers import create_provider
    from ..core.settings import get_settings

    settings = get_settings()
    agent_repo = AgentRepository(session)
    conv_repo = ConversationRepository(session)
    msg_repo = MessageRepository(session)
    usage_repo = UsageRepository(session)

    # Get or create default agent
    agent = await agent_repo.get_default(tenant.id)
    if not agent:
        agent = await agent_repo.create(
            tenant_id=tenant.id,
            name="Default Assistant",
            provider=tenant.config.get("default_provider", "anthropic"),
            model=tenant.config.get("default_model", DEFAULT_ANTHROPIC_MODEL),
        )
        await session.flush()

    # Get or create conversation
    conversation = None
    if request.conversation_id:
        try:
            conv_uuid = UUID(request.conversation_id)
            conversation = await conv_repo.get_by_id(conv_uuid)
            if conversation and conversation.tenant_id != tenant.id:
                conversation = None
        except ValueError:
            pass

    if not conversation:
        user_uuid = None
        if x_user_id:
            with contextlib.suppress(ValueError):
                user_uuid = UUID(x_user_id)

        conversation = await conv_repo.create_conversation(
            tenant_id=tenant.id,
            agent_id=agent.id,
            user_id=user_uuid,
            channel="api",
        )
        await session.flush()

    # Save user message
    await msg_repo.add_message(
        conversation_id=conversation.id,
        role="user",
        content=request.message,
    )

    # Get conversation history
    messages = await conv_repo.get_recent_messages(conversation.id, limit=20)
    message_history = [{"role": m.role, "content": m.content} for m in messages]

    # Create provider
    provider = await create_provider(agent.provider, settings.llm)

    try:
        start_time = time.time()

        if request.stream:

            async def event_generator():
                full_response = ""

                async for event in provider.stream(
                    messages=message_history,
                    system=agent.system_prompt,
                    max_tokens=agent.max_tokens_per_request,
                    temperature=float(agent.temperature),
                ):
                    if event.type.value == "text":
                        full_response += event.content
                        yield f"data: {json.dumps({'type': 'text', 'content': event.content})}\n\n"
                    elif event.type.value == "message_end":
                        latency = int((time.time() - start_time) * 1000)
                        await msg_repo.add_message(
                            conversation_id=conversation.id,
                            role="assistant",
                            content=full_response,
                            latency_ms=latency,
                        )
                        await conv_repo.update_stats(conversation.id, message_count_delta=2)
                        await session.commit()
                        yield f"data: {json.dumps({'type': 'done', 'conversation_id': str(conversation.id)})}\n\n"

                yield "data: [DONE]\n\n"

            return StreamingResponse(event_generator(), media_type="text/event-stream")

        else:
            response = await provider.chat(
                messages=message_history,
                system=agent.system_prompt,
                max_tokens=agent.max_tokens_per_request,
                temperature=float(agent.temperature),
            )

            latency = int((time.time() - start_time) * 1000)

            # Save assistant message
            await msg_repo.add_message(
                conversation_id=conversation.id,
                role="assistant",
                content=response.text,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                latency_ms=latency,
            )

            # Update stats
            await conv_repo.update_stats(
                conversation.id,
                message_count_delta=2,
                token_delta=response.usage.total_tokens,
            )

            # Record usage
            await usage_repo.record_usage(
                tenant_id=tenant.id,
                agent_id=agent.id,
                conversation_id=conversation.id,
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
                provider=agent.provider,
                model=agent.model,
            )

            await session.commit()

            return ChatResponse(
                response=response.text,
                conversation_id=str(conversation.id),
                usage=response.usage.to_dict(),
            )
    finally:
        await provider.close()


# ============================================================
# USAGE ROUTES
# ============================================================


@router.get("/usage", response_model=UsageResponse, tags=["Usage"])
async def get_usage(
    tenant: TenantModel = Depends(require_tenant),
    session: AsyncSession = Depends(get_db_session),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    include_daily: bool = Query(default=False),
):
    """Get usage statistics for the tenant."""
    usage_repo = UsageRepository(session)

    start_dt = None
    end_dt = None

    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid start_date") from e

    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
        except ValueError as e:
            raise HTTPException(status_code=400, detail="Invalid end_date") from e

    totals = await usage_repo.get_tenant_usage(tenant.id, start_dt, end_dt)

    daily = None
    if include_daily:
        daily = await usage_repo.get_daily_usage(tenant.id, days=30)

    return UsageResponse(
        period={"start": start_date, "end": end_date},
        totals=totals,
        daily=daily,
    )
