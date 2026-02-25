# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Agent Orchestrator

Connects TenantAgent with database persistence:
- Loads agent configuration from database
- Persists messages and tool results
- Tracks usage for billing
- Manages conversation lifecycle

This is the main integration point between:
- Familiar's core agent functionality
- Reflection's tenant wrappers
- PostgreSQL persistence
"""

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

# Check if Familiar is available
FAMILIAR_AVAILABLE = False
try:
    from familiar.core.agent import Agent
    from familiar.core.providers import StreamEvent, StreamEventType
    from familiar.core.security import Capability, TrustLevel

    FAMILIAR_AVAILABLE = True
except ImportError:
    Agent = None
    StreamEvent = None
    StreamEventType = None
    TrustLevel = None
    Capability = None

# Import from Reflection data layer
from ..data.models import ConversationModel, MessageModel  # noqa: E402
from ..data.repositories import (  # noqa: E402
    ConversationRepository,
    MessageRepository,
    UsageRepository,
)

# Import tenant context
from ..tenants.quotas import record_usage  # noqa: E402

# Import tenant wrappers (only if Familiar available)  # noqa: E402
if FAMILIAR_AVAILABLE:
    from ..tenant_wrappers import TenantAgent, get_tenant_agent  # noqa: E402
else:
    TenantAgent = None  # type: ignore[assignment]
    get_tenant_agent = None

logger = logging.getLogger(__name__)


# ============================================================
# EVENTS
# ============================================================


@dataclass
class OrchestratorEvent:
    """Event emitted during orchestrated agent execution."""

    type: str  # text, tool_start, tool_end, thinking, done, error
    content: str = ""
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    tool_success: bool | None = None
    usage: dict[str, Any] | None = None
    conversation_id: str | None = None
    message_id: str | None = None
    iteration: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_output": self.tool_output,
            "tool_success": self.tool_success,
            "usage": self.usage,
            "conversation_id": self.conversation_id,
            "message_id": self.message_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        data = json.dumps(self.to_dict())
        return f"event: {self.type}\ndata: {data}\n\n"


@dataclass
class OrchestratorResult:
    """Final result of orchestrated execution."""

    response: str
    conversation_id: str
    message_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: int = 0
    iterations: int = 1
    latency_ms: int = 0


# ============================================================
# ORCHESTRATOR
# ============================================================


class AgentOrchestrator:
    """
    Orchestrates agent execution with database persistence.

    Usage:
        orchestrator = AgentOrchestrator(db_session)

        async for event in orchestrator.chat(
            tenant_id=uuid,
            user_id="user123",
            message="Hello!"
        ):
            print(event.to_dict())
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.conversation_repo = ConversationRepository(db)
        self.message_repo = MessageRepository(db)
        self.usage_repo = UsageRepository(db)

        # Cached agents per tenant
        self._agents: dict[UUID, TenantAgent] = {}

    def _get_agent(self, tenant_id: UUID, tenant_config: dict | None = None) -> TenantAgent:
        """Get or create agent for tenant."""
        if tenant_id not in self._agents:
            self._agents[tenant_id] = get_tenant_agent(
                tenant_id=tenant_id,
                tenant_config=tenant_config,
            )
        return self._agents[tenant_id]

    async def _get_or_create_conversation(
        self,
        tenant_id: UUID,
        user_id: str,
        conversation_id: str | None = None,
    ) -> ConversationModel:
        """Get existing conversation or create new one."""
        if conversation_id:
            conv = await self.conversation_repo.get_by_id(UUID(conversation_id))
            if conv and str(conv.tenant_id) == str(tenant_id):
                return conv

        # Create new conversation
        return await self.conversation_repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            title="New Conversation",
        )

    async def _persist_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> MessageModel:
        """Persist a message to database."""
        return await self.message_repo.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )

    async def _persist_usage(
        self,
        tenant_id: UUID,
        user_id: str,
        input_tokens: int,
        output_tokens: int,
        model: str,
        latency_ms: int,
    ):
        """Persist usage record for billing."""
        await self.usage_repo.create(
            tenant_id=tenant_id,
            user_id=user_id,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            model=model,
            latency_ms=latency_ms,
        )

        # Also update quota usage
        total_tokens = input_tokens + output_tokens
        record_usage(tenant_id, "tokens_per_day", total_tokens)

    async def _load_conversation_history(
        self,
        conversation_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Load recent messages from conversation."""
        messages = await self.message_repo.get_by_conversation(
            conversation_id=conversation_id,
            limit=limit,
        )

        # Convert to format expected by agent
        history = []
        for msg in reversed(messages):  # Oldest first
            history.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        return history

    async def chat(
        self,
        tenant_id: UUID,
        user_id: str,
        message: str,
        conversation_id: str | None = None,
        tenant_config: dict | None = None,
        stream: bool = True,
    ) -> AsyncIterator[OrchestratorEvent]:
        """
        Process a chat message with full orchestration.

        Yields events as the agent processes the request.
        Final event has type="done" with complete result.
        """
        start_time = datetime.now(UTC)

        # Get agent
        agent = self._get_agent(tenant_id, tenant_config)

        # Get or create conversation
        conversation = await self._get_or_create_conversation(
            tenant_id=tenant_id,
            user_id=user_id,
            conversation_id=conversation_id,
        )

        # Persist user message
        user_msg = await self._persist_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        # Load conversation history
        await self._load_conversation_history(conversation.id, limit=20)

        # Yield start event
        yield OrchestratorEvent(
            type="start",
            conversation_id=str(conversation.id),
            message_id=str(user_msg.id),
        )

        # Collect response
        full_response = []
        total_input_tokens = 0
        total_output_tokens = 0
        tool_calls = 0

        try:
            # Use agent's streaming if available
            if stream and hasattr(agent, "stream"):
                for event in agent.stream(
                    message=message,
                    user_id=user_id,
                    channel="api",
                    include_history=False,  # We loaded history separately
                ):
                    # Convert Familiar stream events to orchestrator events
                    if event.type == StreamEventType.TEXT:
                        full_response.append(event.content)
                        yield OrchestratorEvent(
                            type="text",
                            content=event.content,
                            conversation_id=str(conversation.id),
                        )

                    elif event.type == StreamEventType.TOOL_START:
                        tool_calls += 1
                        yield OrchestratorEvent(
                            type="tool_start",
                            tool_name=event.tool_name,
                            tool_input=event.tool_input,
                            conversation_id=str(conversation.id),
                        )

                    elif event.type == StreamEventType.TOOL_END:
                        yield OrchestratorEvent(
                            type="tool_end",
                            tool_name=event.tool_name,
                            tool_output=str(event.tool_output),
                            tool_success=event.tool_success,
                            conversation_id=str(conversation.id),
                        )

                    elif event.type == StreamEventType.DONE:
                        if hasattr(event, "usage"):
                            total_input_tokens = event.usage.get("input_tokens", 0)
                            total_output_tokens = event.usage.get("output_tokens", 0)

            else:
                # Non-streaming fallback
                response = agent.chat(
                    message=message,
                    user_id=user_id,
                    channel="api",
                    include_history=False,
                )
                full_response.append(response)

                yield OrchestratorEvent(
                    type="text",
                    content=response,
                    conversation_id=str(conversation.id),
                )

        except Exception as e:
            logger.error(f"Agent error: {e}")
            yield OrchestratorEvent(
                type="error",
                content=str(e),
                conversation_id=str(conversation.id),
            )
            return

        # Calculate latency
        latency_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

        # Join response
        final_response = "".join(full_response)

        # Persist assistant message
        assistant_msg = await self._persist_message(
            conversation_id=conversation.id,
            role="assistant",
            content=final_response,
        )

        # Persist usage
        await self._persist_usage(
            tenant_id=tenant_id,
            user_id=user_id,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            model=agent.provider.name if hasattr(agent, "provider") else "unknown",
            latency_ms=latency_ms,
        )

        # Yield done event with full result
        yield OrchestratorEvent(
            type="done",
            content=final_response,
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage={
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "tool_calls": tool_calls,
                "latency_ms": latency_ms,
            },
        )

    async def get_conversation(
        self,
        tenant_id: UUID,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        """Get conversation with messages."""
        conv = await self.conversation_repo.get_by_id(UUID(conversation_id))

        if not conv or str(conv.tenant_id) != str(tenant_id):
            return None

        messages = await self.message_repo.get_by_conversation(conv.id)

        return {
            "id": str(conv.id),
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "messages": [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ],
        }

    async def list_conversations(
        self,
        tenant_id: UUID,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List conversations for tenant/user."""
        conversations = await self.conversation_repo.list_by_tenant(
            tenant_id=tenant_id,
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

        return [
            {
                "id": str(c.id),
                "title": c.title,
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in conversations
        ]


# ============================================================
# FACTORY
# ============================================================


def create_orchestrator(db: AsyncSession) -> AgentOrchestrator:
    """Create orchestrator instance."""
    return AgentOrchestrator(db)


__all__ = [
    "AgentOrchestrator",
    "OrchestratorEvent",
    "OrchestratorResult",
    "create_orchestrator",
    "FAMILIAR_AVAILABLE",
]
