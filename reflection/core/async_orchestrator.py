# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 5: Async Orchestrator

Dual-path orchestration:
1. ASYNC PATH: Simple conversations use AsyncLLMProviderSDK directly
   - No thread pool overhead
   - True async I/O
   - ~80% of requests

2. SYNC PATH: Complex tool execution uses TenantExecutorPool
   - Full Familiar Agent with tool registry
   - Thread isolation per tenant tier
   - ~20% of requests

Decision logic:
- If request needs tools AND tenant has tools enabled → Sync path
- Otherwise → Async path

This closes the sync/async gap identified in the production guide
while preserving the full Familiar tool ecosystem.
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

# Familiar core imports
from reflection_core.security.trust import TrustLevel

from ..data.models import AgentModel, ConversationModel, MessageModel
from ..data.repositories import (
    AgentRepository,
    ConversationRepository,
    MessageRepository,
    UsageRepository,
)
from ..tenant_wrappers.agent import TenantAgent
from .executor import (
    ExecutorTimeoutError,
    QueueFullError,
    TenantTier,
    get_executor_pool,
)
from .providers_async import (
    AsyncLLMProviderSDK,
    LLMResponse,
    StreamEventType,
    Usage,
    get_async_provider_pool,
)

# Reflection imports
from .settings import get_settings
from .tokens import get_model_pricing, get_token_counter

logger = logging.getLogger(__name__)


# ============================================================
# ORCHESTRATOR EVENTS
# ============================================================


@dataclass
class AsyncOrchestratorEvent:
    """Event emitted during async orchestration."""

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
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class AsyncOrchestratorResult:
    """Final result from async orchestration."""

    text: str
    conversation_id: str
    message_id: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    iterations: int = 1
    latency_ms: int = 0
    path_used: str = "async"  # "async" or "sync"
    cost_usd: str = "0.00"


# ============================================================
# ASYNC ORCHESTRATOR
# ============================================================


class AsyncOrchestrator:
    """
    Enterprise orchestration with dual-path execution.

    Routing Decision:
    ┌─────────────────────────────────────────────────────────┐
    │ Request arrives                                          │
    │     │                                                    │
    │     ▼                                                    │
    │ Tools required?                                         │
    │     │                                                    │
    │   No ──► ASYNC PATH                                     │
    │     │    - AsyncLLMProviderSDK.chat()                   │
    │     │    - No thread pool                               │
    │     │    - True async I/O                               │
    │     │                                                    │
    │   Yes ─► SYNC PATH                                      │
    │          - TenantExecutorPool.run()                     │
    │          - TenantAgent.chat()                           │
    │          - Full tool ecosystem                          │
    └─────────────────────────────────────────────────────────┘

    Usage:
        async with AsyncOrchestrator(session, tenant_id, tier) as orch:
            # Simple chat - uses async path
            result = await orch.chat("What's 2+2?")

            # With tools - uses sync path automatically
            result = await orch.chat("Search for X", enable_tools=True)

            # Streaming
            async for event in orch.stream("Hello"):
                print(event.content)
    """

    def __init__(
        self,
        session: AsyncSession,
        tenant_id: UUID,
        tenant_tier: TenantTier = TenantTier.FREE,
        user_id: UUID | None = None,
        trust_level: TrustLevel = TrustLevel.KNOWN,
    ):
        self.session = session
        self.tenant_id = tenant_id
        self.tenant_tier = tenant_tier
        self.user_id = user_id
        self.trust_level = trust_level

        # Repositories
        self.agent_repo = AgentRepository(session)
        self.conversation_repo = ConversationRepository(session)
        self.message_repo = MessageRepository(session)
        self.usage_repo = UsageRepository(session)

        # Async provider (lazy loaded)
        self._async_provider: AsyncLLMProviderSDK | None = None

        # Sync agent (lazy loaded, for tool execution path)
        self._sync_agent: TenantAgent | None = None

        # Agent config from database
        self._agent_model: AgentModel | None = None

        # Executor pool (for sync path)
        self._executor_pool = get_executor_pool()

        # Async provider pool (for connection reuse)
        self._provider_pool = get_async_provider_pool()

        # State
        self._initialized = False
        self._conversation: ConversationModel | None = None

        # Settings
        self.settings = get_settings()

    async def __aenter__(self):
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()
        return False

    async def _initialize(self):
        """Initialize the orchestrator."""
        if self._initialized:
            return

        logger.debug(f"Initializing AsyncOrchestrator for tenant {self.tenant_id}")
        self._initialized = True

    async def _cleanup(self):
        """Clean up resources."""
        # Provider pool manages its own lifecycle
        pass

    async def _get_agent_config(self, agent_id: UUID | None = None) -> AgentModel:
        """Load agent configuration from database."""
        if self._agent_model is not None:
            return self._agent_model

        if agent_id:
            self._agent_model = await self.agent_repo.get_by_id(agent_id)
        else:
            self._agent_model = await self.agent_repo.get_default(self.tenant_id)

        if not self._agent_model:
            # Create default agent
            settings = get_settings()
            self._agent_model = await self.agent_repo.create(
                tenant_id=self.tenant_id,
                name="Default Assistant",
                provider=settings.llm.default_provider,
                model=settings.llm.default_model,
            )
            await self.session.flush()

        return self._agent_model

    async def _get_async_provider(self, agent_id: UUID | None = None) -> AsyncLLMProviderSDK:
        """Get async provider for direct LLM calls."""
        if self._async_provider is not None:
            return self._async_provider

        agent_config = await self._get_agent_config(agent_id)

        self._async_provider = await self._provider_pool.get_provider(
            provider_name=agent_config.provider,
            model=agent_config.model,
        )

        return self._async_provider

    async def _get_sync_agent(self, agent_id: UUID | None = None) -> TenantAgent:
        """Get sync agent for tool execution path."""
        if self._sync_agent is not None:
            return self._sync_agent

        agent_config = await self._get_agent_config(agent_id)

        tenant_config = {
            "agent_name": agent_config.name,
            "default_provider": agent_config.provider,
            "default_model": agent_config.model,
        }
        if agent_config.config:
            tenant_config.update(agent_config.config)

        self._sync_agent = TenantAgent(
            tenant_id=self.tenant_id,
            tenant_config=tenant_config,
        )

        return self._sync_agent

    async def _get_or_create_conversation(
        self,
        conversation_id: UUID | None = None,
        agent_id: UUID | None = None,
        channel: str = "api",
    ) -> ConversationModel:
        """Get or create conversation."""
        if conversation_id:
            conversation = await self.conversation_repo.get_by_id(
                conversation_id,
                tenant_id=self.tenant_id,
            )
            if conversation:
                return conversation

        # Create new conversation
        agent_config = await self._get_agent_config(agent_id)
        conversation = await self.conversation_repo.create(
            tenant_id=self.tenant_id,
            agent_id=agent_config.id,
            user_id=self.user_id,
            channel=channel,
        )
        await self.session.flush()

        return conversation

    async def _save_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
    ) -> MessageModel:
        """Save a message to the database."""
        message = await self.message_repo.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )
        await self.session.flush()
        return message

    async def _load_history(
        self,
        conversation_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Load conversation history for context."""
        messages = await self.message_repo.get_by_conversation(
            conversation_id,
            limit=limit,
        )

        return [
            {"role": msg.role, "content": msg.content}
            for msg in reversed(messages)  # Oldest first
        ]

    async def _record_usage(
        self,
        tokens_input: int,
        tokens_output: int,
        model: str,
        latency_ms: int,
        path: str = "async",
    ):
        """Record usage for billing and quotas."""
        # Record to database
        await self.usage_repo.record(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model,
            latency_ms=latency_ms,
            metadata={"path": path},
        )

        # Record to quota service (Phase 6)
        try:
            from ..tenants.quota_service import _MockTenant, get_quota_service

            quota_service = get_quota_service()
            limits = quota_service.get_limits(self.tenant_tier)
            tenant = _MockTenant(self.tenant_id, limits)
            await quota_service.manager.record_tokens(tenant, tokens_input + tokens_output)
        except Exception as e:
            # Don't fail request if quota recording fails
            logger.debug(f"Could not record quota usage: {e}")

    def _should_use_async_path(
        self,
        enable_tools: bool,
        tools: list[dict[str, Any]] | None,
    ) -> bool:
        """
        Determine whether to use async or sync path.

        Async path is used when:
        - Tools are not requested (enable_tools=False)
        - No specific tools are provided

        Sync path is used when:
        - Tools are enabled AND available
        """
        if not enable_tools:
            return True

        return not (tools and len(tools) > 0)

    async def chat(
        self,
        message: str,
        agent_id: UUID | None = None,
        conversation_id: UUID | None = None,
        channel: str = "api",
        include_history: bool = True,
        enable_tools: bool = False,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncOrchestratorResult:
        """
        Send a message with automatic path routing.

        Args:
            message: User message
            agent_id: Optional specific agent
            conversation_id: Optional existing conversation
            channel: Channel identifier (api, web, slack, etc.)
            include_history: Whether to include conversation history
            enable_tools: Whether to enable tool execution
            tools: Specific tools to enable (implies sync path)
            system_prompt: Optional system prompt override
            max_tokens: Maximum response tokens
            temperature: Sampling temperature

        Returns:
            AsyncOrchestratorResult with response and metadata
        """
        start_time = time.time()

        # Get conversation
        conversation = await self._get_or_create_conversation(
            conversation_id=conversation_id,
            agent_id=agent_id,
            channel=channel,
        )

        # Save user message
        await self._save_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        # Route to appropriate path
        if self._should_use_async_path(enable_tools, tools):
            result = await self._chat_async_path(
                message=message,
                conversation=conversation,
                agent_id=agent_id,
                include_history=include_history,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                start_time=start_time,
            )
        else:
            result = await self._chat_sync_path(
                message=message,
                conversation=conversation,
                agent_id=agent_id,
                channel=channel,
                include_history=include_history,
                tools=tools,
                start_time=start_time,
            )

        await self.session.commit()
        return result

    async def _chat_async_path(
        self,
        message: str,
        conversation: ConversationModel,
        agent_id: UUID | None,
        include_history: bool,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> AsyncOrchestratorResult:
        """
        Async path: Direct provider call, no thread pool.

        This is the fast path for simple conversations.
        """
        logger.debug(f"Using ASYNC path for tenant {self.tenant_id}")

        # Get async provider
        provider = await self._get_async_provider(agent_id)
        agent_config = await self._get_agent_config(agent_id)

        # Build messages
        messages = []
        if include_history:
            history = await self._load_history(conversation.id)
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        # Get system prompt
        system = system_prompt
        if not system and agent_config.config:
            system = agent_config.config.get("system_prompt")
        if not system:
            system = f"You are {agent_config.name}, a helpful AI assistant."

        try:
            # Direct async call - no thread pool!
            response: LLMResponse = await provider.chat(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            response_text = response.text
            usage = response.usage

        except Exception as e:
            logger.error(f"Async provider error for tenant {self.tenant_id}: {e}")
            response_text = f"I apologize, but I encountered an error: {str(e)}"
            usage = Usage()

        # Save assistant message
        latency_ms = int((time.time() - start_time) * 1000)

        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            tokens_input=usage.input_tokens,
            tokens_output=usage.output_tokens,
        )

        # Record usage
        await self._record_usage(
            tokens_input=usage.input_tokens,
            tokens_output=usage.output_tokens,
            model=agent_config.model,
            latency_ms=latency_ms,
            path="async",
        )

        # Calculate cost
        pricing = get_model_pricing(agent_config.model)
        if pricing:
            cost = (usage.input_tokens / 1_000_000) * float(pricing.input_price_per_million) + (
                usage.output_tokens / 1_000_000
            ) * float(pricing.output_price_per_million)
        else:
            cost = 0.0

        return AsyncOrchestratorResult(
            text=response_text,
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage=usage.to_dict(),
            latency_ms=latency_ms,
            path_used="async",
            cost_usd=f"{cost:.6f}",
        )

    async def _chat_sync_path(
        self,
        message: str,
        conversation: ConversationModel,
        agent_id: UUID | None,
        channel: str,
        include_history: bool,
        tools: list[dict[str, Any]] | None,
        start_time: float,
    ) -> AsyncOrchestratorResult:
        """
        Sync path: Thread pool execution with full Familiar Agent.

        Used when tool execution is required.
        """
        logger.debug(f"Using SYNC path for tenant {self.tenant_id}")

        # Get sync agent
        agent = await self._get_sync_agent(agent_id)
        agent_config = await self._get_agent_config(agent_id)

        try:
            # Execute in tenant's tiered thread pool
            response = await self._executor_pool.run(
                tenant_id=self.tenant_id,
                tier=self.tenant_tier,
                func=agent.chat,
                kwargs={
                    "message": message,
                    "user_id": str(self.user_id) if self.user_id else "api",
                    "channel": channel,
                    "include_history": include_history,
                },
            )
            response_text = response

        except QueueFullError:
            logger.warning(f"Queue full for tenant {self.tenant_id}")
            return AsyncOrchestratorResult(
                text="I'm currently experiencing high demand. Please try again in a moment.",
                conversation_id=str(conversation.id),
                message_id="",
                usage={},
                latency_ms=int((time.time() - start_time) * 1000),
                path_used="sync",
            )

        except ExecutorTimeoutError as e:
            logger.error(f"Timeout for tenant {self.tenant_id}: {e.elapsed_seconds:.1f}s")
            return AsyncOrchestratorResult(
                text="Your request took too long to process. Please try again.",
                conversation_id=str(conversation.id),
                message_id="",
                usage={},
                latency_ms=int((time.time() - start_time) * 1000),
                path_used="sync",
            )

        except Exception as e:
            logger.error(f"Sync agent error for tenant {self.tenant_id}: {e}")
            response_text = f"I apologize, but I encountered an error: {str(e)}"

        # Calculate usage (estimate since sync path doesn't return usage directly)
        token_counter = get_token_counter()
        tokens_input = token_counter.count_tokens(message, agent_config.model).count
        tokens_output = token_counter.count_tokens(response_text, agent_config.model).count

        latency_ms = int((time.time() - start_time) * 1000)

        # Save assistant message
        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        # Record usage
        await self._record_usage(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=agent_config.model,
            latency_ms=latency_ms,
            path="sync",
        )

        return AsyncOrchestratorResult(
            text=response_text,
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage={
                "input_tokens": tokens_input,
                "output_tokens": tokens_output,
                "total_tokens": tokens_input + tokens_output,
            },
            latency_ms=latency_ms,
            path_used="sync",
        )

    async def stream(
        self,
        message: str,
        agent_id: UUID | None = None,
        conversation_id: UUID | None = None,
        channel: str = "api",
        include_history: bool = True,
        enable_tools: bool = False,
        tools: list[dict[str, Any]] | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[AsyncOrchestratorEvent]:
        """
        Stream response with true async streaming.

        Uses async provider's stream() method for real-time token delivery.
        Falls back to chunked sync response if tools are required.
        """
        start_time = time.time()

        # Get conversation
        conversation = await self._get_or_create_conversation(
            conversation_id=conversation_id,
            agent_id=agent_id,
            channel=channel,
        )

        # Save user message
        await self._save_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        if self._should_use_async_path(enable_tools, tools):
            async for event in self._stream_async_path(
                message=message,
                conversation=conversation,
                agent_id=agent_id,
                include_history=include_history,
                system_prompt=system_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                start_time=start_time,
            ):
                yield event
        else:
            async for event in self._stream_sync_path(
                message=message,
                conversation=conversation,
                agent_id=agent_id,
                channel=channel,
                include_history=include_history,
                start_time=start_time,
            ):
                yield event

        await self.session.commit()

    async def _stream_async_path(
        self,
        message: str,
        conversation: ConversationModel,
        agent_id: UUID | None,
        include_history: bool,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        start_time: float,
    ) -> AsyncIterator[AsyncOrchestratorEvent]:
        """True async streaming using provider SDK."""
        logger.debug(f"Using ASYNC STREAM for tenant {self.tenant_id}")

        provider = await self._get_async_provider(agent_id)
        agent_config = await self._get_agent_config(agent_id)

        # Build messages
        messages = []
        if include_history:
            history = await self._load_history(conversation.id)
            messages.extend(history)
        messages.append({"role": "user", "content": message})

        # Get system prompt
        system = system_prompt
        if not system and agent_config.config:
            system = agent_config.config.get("system_prompt")
        if not system:
            system = f"You are {agent_config.name}, a helpful AI assistant."

        collected_text = ""
        final_usage: Usage | None = None

        try:
            async for event in provider.stream(
                messages=messages,
                system=system,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                if event.type == StreamEventType.TEXT_DELTA:
                    collected_text += event.text
                    yield AsyncOrchestratorEvent(
                        type="text",
                        content=event.text,
                        conversation_id=str(conversation.id),
                    )

                elif event.type == StreamEventType.CONTENT_BLOCK_START and event.tool_call:
                    yield AsyncOrchestratorEvent(
                        type="tool_start",
                        tool_name=event.tool_call.name,
                        tool_input=event.tool_call.input,
                        conversation_id=str(conversation.id),
                    )

                elif event.type == StreamEventType.CONTENT_BLOCK_STOP and event.tool_call:
                    yield AsyncOrchestratorEvent(
                        type="tool_end",
                        tool_name=event.tool_call.name,
                        tool_input=event.tool_call.input,
                        conversation_id=str(conversation.id),
                    )

                elif event.type == StreamEventType.MESSAGE_STOP:
                    final_usage = event.usage

                elif event.type == StreamEventType.ERROR:
                    yield AsyncOrchestratorEvent(
                        type="error",
                        content=event.error or "Unknown error",
                        conversation_id=str(conversation.id),
                    )

        except Exception as e:
            logger.error(f"Stream error for tenant {self.tenant_id}: {e}")
            yield AsyncOrchestratorEvent(
                type="error",
                content=str(e),
                conversation_id=str(conversation.id),
            )
            collected_text = f"Error: {e}"
            final_usage = Usage()

        # Save and record
        latency_ms = int((time.time() - start_time) * 1000)

        tokens_input = final_usage.input_tokens if final_usage else 0
        tokens_output = final_usage.output_tokens if final_usage else 0

        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=collected_text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        await self._record_usage(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=agent_config.model,
            latency_ms=latency_ms,
            path="async_stream",
        )

        # Final done event
        yield AsyncOrchestratorEvent(
            type="done",
            content="",
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage={
                "input_tokens": tokens_input,
                "output_tokens": tokens_output,
                "total_tokens": tokens_input + tokens_output,
            },
        )

    async def _stream_sync_path(
        self,
        message: str,
        conversation: ConversationModel,
        agent_id: UUID | None,
        channel: str,
        include_history: bool,
        start_time: float,
    ) -> AsyncIterator[AsyncOrchestratorEvent]:
        """Chunked streaming for sync path (tool execution)."""
        logger.debug(f"Using SYNC STREAM (chunked) for tenant {self.tenant_id}")

        yield AsyncOrchestratorEvent(
            type="thinking",
            content="Processing with tools...",
            conversation_id=str(conversation.id),
        )

        agent = await self._get_sync_agent(agent_id)
        agent_config = await self._get_agent_config(agent_id)

        try:
            response = await self._executor_pool.run(
                tenant_id=self.tenant_id,
                tier=self.tenant_tier,
                func=agent.chat,
                kwargs={
                    "message": message,
                    "user_id": str(self.user_id) if self.user_id else "api",
                    "channel": channel,
                    "include_history": include_history,
                },
            )

            # Simulate streaming by chunking
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                yield AsyncOrchestratorEvent(
                    type="text",
                    content=chunk,
                    conversation_id=str(conversation.id),
                )
                # Small delay for streaming effect
                await asyncio.sleep(0.01)

            response_text = response

        except QueueFullError:
            yield AsyncOrchestratorEvent(
                type="error",
                content="Service is busy. Please try again.",
                conversation_id=str(conversation.id),
            )
            response_text = "Service temporarily unavailable."

        except ExecutorTimeoutError:
            yield AsyncOrchestratorEvent(
                type="error",
                content="Request timed out.",
                conversation_id=str(conversation.id),
            )
            response_text = "Request timed out."

        except Exception as e:
            yield AsyncOrchestratorEvent(
                type="error",
                content=str(e),
                conversation_id=str(conversation.id),
            )
            response_text = f"Error: {e}"

        # Save and record
        latency_ms = int((time.time() - start_time) * 1000)

        token_counter = get_token_counter()
        tokens_input = token_counter.count_tokens(message, agent_config.model).count
        tokens_output = token_counter.count_tokens(response_text, agent_config.model).count

        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response_text,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        await self._record_usage(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=agent_config.model,
            latency_ms=latency_ms,
            path="sync_stream",
        )

        yield AsyncOrchestratorEvent(
            type="done",
            content="",
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage={
                "input_tokens": tokens_input,
                "output_tokens": tokens_output,
                "total_tokens": tokens_input + tokens_output,
            },
        )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "AsyncOrchestrator",
    "AsyncOrchestratorEvent",
    "AsyncOrchestratorResult",
]
