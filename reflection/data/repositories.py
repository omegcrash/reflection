# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Repository pattern for tenant-isolated CRUD operations.

Every repository is constructed with an AsyncSession and provides
typed query methods. All list/get operations enforce tenant_id
isolation where applicable.
"""

import hashlib
import secrets
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import BaseModel
from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.settings import DEFAULT_ANTHROPIC_MODEL
from .models import (
    AgentModel,
    APIKeyModel,
    ConversationModel,
    MessageModel,
    TenantModel,
    TenantUserModel,
    UsageRecordModel,
)


def _utcnow() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# UsageEvent  (Pydantic model for in-flight usage tracking)
# ---------------------------------------------------------------------------


class UsageEvent(BaseModel):
    """Pydantic model for recording a usage event before persistence."""

    tenant_id: str
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tool_executions: int = 0
    cost_cents: int = 0


# ---------------------------------------------------------------------------
# TenantRepository
# ---------------------------------------------------------------------------


class TenantRepository:
    """CRUD for the tenants table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        name: str,
        slug: str,
        tier: str = "free",
        config: dict | None = None,
        quotas: dict | None = None,
    ) -> TenantModel:
        tenant = TenantModel(
            id=str(uuid.uuid4()),
            name=name,
            slug=slug,
            tier=tier,
            status="active",
            config=config or {},
            quotas=quotas or {},
        )
        self.session.add(tenant)
        await self.session.flush()
        return tenant

    async def get_by_id(self, tenant_id: uuid.UUID) -> TenantModel | None:
        result = await self.session.execute(
            select(TenantModel).where(TenantModel.id == str(tenant_id))
        )
        return result.scalar_one_or_none()

    async def get_by_slug(self, slug: str) -> TenantModel | None:
        result = await self.session.execute(select(TenantModel).where(TenantModel.slug == slug))
        return result.scalar_one_or_none()

    async def list_active(self) -> Sequence[TenantModel]:
        result = await self.session.execute(
            select(TenantModel).where(TenantModel.status == "active")
        )
        return result.scalars().all()

    async def update(self, tenant: TenantModel, **kwargs) -> TenantModel:
        for key, value in kwargs.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        tenant.updated_at = _utcnow()
        await self.session.flush()
        return tenant

    async def get_usage_stats(self, tenant_id: uuid.UUID) -> dict:
        """Aggregate usage stats for a tenant."""
        result = await self.session.execute(
            select(
                func.sum(UsageRecordModel.tokens_input).label("total_input"),
                func.sum(UsageRecordModel.tokens_output).label("total_output"),
                func.sum(UsageRecordModel.cost_cents).label("total_cost_cents"),
                func.count(UsageRecordModel.id).label("total_requests"),
            ).where(UsageRecordModel.tenant_id == str(tenant_id))
        )
        row = result.one()
        return {
            "total_input_tokens": row.total_input or 0,
            "total_output_tokens": row.total_output or 0,
            "total_cost_cents": row.total_cost_cents or 0,
            "total_requests": row.total_requests or 0,
        }


# ---------------------------------------------------------------------------
# TenantUserRepository
# ---------------------------------------------------------------------------


class TenantUserRepository:
    """CRUD for the tenant_users table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: uuid.UUID,
        email: str,
        role: str = "member",
        password_hash: str | None = None,
        display_name: str | None = None,
    ) -> TenantUserModel:
        user = TenantUserModel(
            id=str(uuid.uuid4()),
            tenant_id=str(tenant_id),
            email=email,
            role=role,
            password_hash=password_hash,
            display_name=display_name,
        )
        self.session.add(user)
        await self.session.flush()
        return user

    async def get_by_id(self, user_id: uuid.UUID) -> TenantUserModel | None:
        result = await self.session.execute(
            select(TenantUserModel).where(TenantUserModel.id == str(user_id))
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, tenant_id: uuid.UUID, email: str) -> TenantUserModel | None:
        result = await self.session.execute(
            select(TenantUserModel).where(
                and_(
                    TenantUserModel.tenant_id == str(tenant_id),
                    TenantUserModel.email == email,
                )
            )
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[TenantUserModel]:
        result = await self.session.execute(
            select(TenantUserModel)
            .where(TenantUserModel.tenant_id == str(tenant_id))
            .order_by(TenantUserModel.created_at)
        )
        return result.scalars().all()

    async def count_by_tenant(self, tenant_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(TenantUserModel.id)).where(
                TenantUserModel.tenant_id == str(tenant_id)
            )
        )
        return result.scalar_one()

    async def update(self, user: TenantUserModel, **kwargs) -> TenantUserModel:
        for key, value in kwargs.items():
            if hasattr(user, key):
                setattr(user, key, value)
        user.updated_at = _utcnow()
        await self.session.flush()
        return user

    async def set_password(self, user: TenantUserModel, password_hash: str) -> TenantUserModel:
        user.password_hash = password_hash
        user.updated_at = _utcnow()
        await self.session.flush()
        return user

    async def verify_email(self, user: TenantUserModel) -> TenantUserModel:
        user.email_verified = True
        user.updated_at = _utcnow()
        await self.session.flush()
        return user

    async def update_last_login(self, user: TenantUserModel) -> TenantUserModel:
        user.last_login_at = _utcnow()
        await self.session.flush()
        return user


# ---------------------------------------------------------------------------
# APIKeyRepository
# ---------------------------------------------------------------------------


class APIKeyRepository:
    """CRUD for the api_keys table.

    Keys are stored as SHA-256 hashes. The plaintext is only returned
    once at creation time.
    """

    KEY_PREFIX_LEN = 8

    def __init__(self, session: AsyncSession):
        self.session = session

    @staticmethod
    def _hash_key(plaintext: str) -> str:
        return hashlib.sha256(plaintext.encode()).hexdigest()

    async def create_key(
        self,
        tenant_id: uuid.UUID,
        name: str,
        permissions: list | None = None,
        expires_at: datetime | None = None,
    ) -> tuple[APIKeyModel, str]:
        """Create a new API key. Returns (model, plaintext)."""
        plaintext = f"me_{secrets.token_urlsafe(32)}"
        key_hash = self._hash_key(plaintext)

        api_key = APIKeyModel(
            id=str(uuid.uuid4()),
            tenant_id=str(tenant_id),
            name=name,
            key_hash=key_hash,
            key_prefix=plaintext[: self.KEY_PREFIX_LEN],
            permissions=permissions or ["chat", "read"],
            expires_at=expires_at,
        )
        self.session.add(api_key)
        await self.session.flush()
        return api_key, plaintext

    async def validate_key(self, plaintext: str) -> APIKeyModel | None:
        """Validate a plaintext API key. Returns the model if valid."""
        key_hash = self._hash_key(plaintext)
        result = await self.session.execute(
            select(APIKeyModel).where(
                and_(
                    APIKeyModel.key_hash == key_hash,
                    APIKeyModel.is_active == True,  # noqa: E712
                )
            )
        )
        api_key = result.scalar_one_or_none()
        if api_key is None:
            return None

        # Check expiration
        if api_key.expires_at and api_key.expires_at < _utcnow():
            return None

        # Update last_used_at
        api_key.last_used_at = _utcnow()
        await self.session.flush()
        return api_key

    async def get_by_id(self, key_id: uuid.UUID) -> APIKeyModel | None:
        result = await self.session.execute(
            select(APIKeyModel).where(APIKeyModel.id == str(key_id))
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[APIKeyModel]:
        result = await self.session.execute(
            select(APIKeyModel)
            .where(APIKeyModel.tenant_id == str(tenant_id))
            .order_by(APIKeyModel.created_at.desc())
        )
        return result.scalars().all()

    async def revoke(self, api_key: APIKeyModel) -> APIKeyModel:
        api_key.is_active = False
        await self.session.flush()
        return api_key


# ---------------------------------------------------------------------------
# AgentRepository
# ---------------------------------------------------------------------------


class AgentRepository:
    """CRUD for the agents table."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: uuid.UUID,
        name: str,
        provider: str = "anthropic",
        model: str = DEFAULT_ANTHROPIC_MODEL,
        system_prompt: str | None = None,
        **kwargs,
    ) -> AgentModel:
        agent = AgentModel(
            id=str(uuid.uuid4()),
            tenant_id=str(tenant_id),
            name=name,
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            **kwargs,
        )
        self.session.add(agent)
        await self.session.flush()
        return agent

    async def get_by_id(self, agent_id: uuid.UUID) -> AgentModel | None:
        result = await self.session.execute(
            select(AgentModel).where(AgentModel.id == str(agent_id))
        )
        return result.scalar_one_or_none()

    async def get_default(self, tenant_id: uuid.UUID) -> AgentModel | None:
        """Return the first active agent for a tenant."""
        result = await self.session.execute(
            select(AgentModel)
            .where(
                and_(
                    AgentModel.tenant_id == str(tenant_id),
                    AgentModel.status == "active",
                )
            )
            .order_by(AgentModel.created_at)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(self, tenant_id: uuid.UUID) -> Sequence[AgentModel]:
        result = await self.session.execute(
            select(AgentModel)
            .where(AgentModel.tenant_id == str(tenant_id))
            .order_by(AgentModel.created_at)
        )
        return result.scalars().all()

    async def count_by_tenant(self, tenant_id: uuid.UUID) -> int:
        result = await self.session.execute(
            select(func.count(AgentModel.id)).where(AgentModel.tenant_id == str(tenant_id))
        )
        return result.scalar_one()

    async def update(self, agent: AgentModel, **kwargs) -> AgentModel:
        for key, value in kwargs.items():
            if hasattr(agent, key):
                setattr(agent, key, value)
        agent.updated_at = _utcnow()
        await self.session.flush()
        return agent


# ---------------------------------------------------------------------------
# UsageRepository
# ---------------------------------------------------------------------------


class UsageRepository:
    """Record and query usage data."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(self, event: UsageEvent) -> UsageRecordModel:
        """Persist a UsageEvent to the database."""
        record = UsageRecordModel(
            id=str(uuid.uuid4()),
            tenant_id=event.tenant_id,
            agent_id=event.agent_id,
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            tokens_input=event.input_tokens,
            tokens_output=event.output_tokens,
            tool_executions=event.tool_executions,
            cost_cents=event.cost_cents,
            provider=event.provider,
            model=event.model,
        )
        self.session.add(record)
        await self.session.flush()
        return record

    async def record_usage(self, **kwargs) -> UsageRecordModel:
        """Convenience: record usage from keyword args."""
        record = UsageRecordModel(id=str(uuid.uuid4()), **kwargs)
        self.session.add(record)
        await self.session.flush()
        return record

    async def get_by_tenant(
        self,
        tenant_id: uuid.UUID,
        limit: int = 100,
    ) -> Sequence[UsageRecordModel]:
        result = await self.session.execute(
            select(UsageRecordModel)
            .where(UsageRecordModel.tenant_id == str(tenant_id))
            .order_by(UsageRecordModel.recorded_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_by_user(
        self,
        user_id: uuid.UUID,
        limit: int = 100,
    ) -> Sequence[UsageRecordModel]:
        result = await self.session.execute(
            select(UsageRecordModel)
            .where(UsageRecordModel.user_id == str(user_id))
            .order_by(UsageRecordModel.recorded_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_tenant_usage(
        self,
        tenant_id: uuid.UUID,
        since: datetime | None = None,
    ) -> dict:
        """Aggregate usage stats for a tenant, optionally since a date."""
        stmt = select(
            func.sum(UsageRecordModel.tokens_input).label("total_input"),
            func.sum(UsageRecordModel.tokens_output).label("total_output"),
            func.sum(UsageRecordModel.cost_cents).label("total_cost_cents"),
            func.count(UsageRecordModel.id).label("total_requests"),
        ).where(UsageRecordModel.tenant_id == str(tenant_id))

        if since:
            stmt = stmt.where(UsageRecordModel.recorded_at >= since)

        result = await self.session.execute(stmt)
        row = result.one()
        return {
            "total_input_tokens": row.total_input or 0,
            "total_output_tokens": row.total_output or 0,
            "total_cost_cents": row.total_cost_cents or 0,
            "total_requests": row.total_requests or 0,
        }

    async def get_daily_usage(self, tenant_id: uuid.UUID, days: int = 30) -> list[dict]:
        """Daily breakdown of usage for charting."""
        since = _utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(
                func.date(UsageRecordModel.recorded_at).label("day"),
                func.sum(UsageRecordModel.tokens_input).label("input_tokens"),
                func.sum(UsageRecordModel.tokens_output).label("output_tokens"),
                func.sum(UsageRecordModel.cost_cents).label("cost_cents"),
                func.count(UsageRecordModel.id).label("requests"),
            )
            .where(
                and_(
                    UsageRecordModel.tenant_id == str(tenant_id),
                    UsageRecordModel.recorded_at >= since,
                )
            )
            .group_by(func.date(UsageRecordModel.recorded_at))
            .order_by(func.date(UsageRecordModel.recorded_at))
        )
        return [
            {
                "day": str(row.day),
                "input_tokens": row.input_tokens or 0,
                "output_tokens": row.output_tokens or 0,
                "cost_cents": row.cost_cents or 0,
                "requests": row.requests or 0,
            }
            for row in result.all()
        ]

    async def get_monthly_spend(self, tenant_id: uuid.UUID) -> int:
        """Total cost in cents for the current calendar month."""
        now = _utcnow()
        month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = await self.session.execute(
            select(func.sum(UsageRecordModel.cost_cents)).where(
                and_(
                    UsageRecordModel.tenant_id == str(tenant_id),
                    UsageRecordModel.recorded_at >= month_start,
                )
            )
        )
        return result.scalar_one() or 0


# ============================================================
# CONVERSATION REPOSITORY
# ============================================================


class ConversationRepository:
    """CRUD for conversations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID | None = None,
        agent_id: uuid.UUID | None = None,
        channel: str = "api",
    ) -> ConversationModel:
        conv = ConversationModel(
            id=str(uuid.uuid4()),
            tenant_id=str(tenant_id),
            user_id=str(user_id) if user_id else None,
            agent_id=str(agent_id) if agent_id else None,
            channel=channel,
            message_count=0,
            total_tokens=0,
        )
        self.session.add(conv)
        await self.session.flush()
        return conv

    # Alias used by some call sites
    async def create_conversation(self, **kwargs) -> ConversationModel:
        return await self.create(**kwargs)

    async def get_by_id(self, conversation_id: uuid.UUID) -> ConversationModel | None:
        result = await self.session.execute(
            select(ConversationModel).where(ConversationModel.id == str(conversation_id))
        )
        return result.scalar_one_or_none()

    async def list_by_tenant(
        self,
        tenant_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ConversationModel]:
        result = await self.session.execute(
            select(ConversationModel)
            .where(ConversationModel.tenant_id == str(tenant_id))
            .order_by(ConversationModel.last_message_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    async def list_by_user(
        self,
        tenant_id: uuid.UUID,
        user_id: uuid.UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> Sequence[ConversationModel]:
        result = await self.session.execute(
            select(ConversationModel)
            .where(
                and_(
                    ConversationModel.tenant_id == str(tenant_id),
                    ConversationModel.user_id == str(user_id),
                )
            )
            .order_by(ConversationModel.last_message_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return result.scalars().all()

    # Aliases used by export_handlers
    async def get_by_tenant(
        self, tenant_id: uuid.UUID, limit: int = 100
    ) -> Sequence[ConversationModel]:
        return await self.list_by_tenant(tenant_id, limit=limit)

    async def get_by_user(
        self,
        user_id: uuid.UUID,
        tenant_id: uuid.UUID | None = None,
        limit: int = 100,
    ) -> Sequence[ConversationModel]:
        conditions = [ConversationModel.user_id == str(user_id)]
        if tenant_id:
            conditions.append(ConversationModel.tenant_id == str(tenant_id))
        result = await self.session.execute(
            select(ConversationModel)
            .where(and_(*conditions))
            .order_by(ConversationModel.last_message_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

    async def get_recent_messages(
        self,
        conversation_id: uuid.UUID,
        limit: int = 20,
    ) -> Sequence["MessageModel"]:
        """Get recent messages for a conversation (convenience shortcut)."""
        result = await self.session.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == str(conversation_id))
            .order_by(MessageModel.created_at.desc())
            .limit(limit)
        )
        # Return in chronological order
        return list(reversed(result.scalars().all()))

    async def update_stats(
        self,
        conversation_id: uuid.UUID,
        message_count_delta: int = 0,
        token_delta: int = 0,
    ) -> None:
        result = await self.session.execute(
            select(ConversationModel).where(ConversationModel.id == str(conversation_id))
        )
        conv = result.scalar_one_or_none()
        if conv:
            conv.message_count = (conv.message_count or 0) + message_count_delta
            conv.total_tokens = (conv.total_tokens or 0) + token_delta
            conv.last_message_at = _utcnow()

    async def delete(self, conversation_id: uuid.UUID) -> None:
        # Delete messages first, then conversation
        await self.session.execute(
            delete(MessageModel).where(MessageModel.conversation_id == str(conversation_id))
        )
        await self.session.execute(
            delete(ConversationModel).where(ConversationModel.id == str(conversation_id))
        )


# ============================================================
# MESSAGE REPOSITORY
# ============================================================


class MessageRepository:
    """CRUD for messages within conversations."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def add_message(
        self,
        conversation_id: uuid.UUID,
        role: str,
        content: str,
        latency_ms: int | None = None,
    ) -> MessageModel:
        msg = MessageModel(
            id=str(uuid.uuid4()),
            conversation_id=str(conversation_id),
            role=role,
            content=content,
            latency_ms=latency_ms,
        )
        self.session.add(msg)
        await self.session.flush()
        return msg

    async def get_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> Sequence[MessageModel]:
        result = await self.session.execute(
            select(MessageModel)
            .where(MessageModel.conversation_id == str(conversation_id))
            .order_by(MessageModel.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()

    # Alias
    async def list_by_conversation(
        self,
        conversation_id: uuid.UUID,
        limit: int = 100,
    ) -> Sequence[MessageModel]:
        return await self.get_by_conversation(conversation_id, limit=limit)
