# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Agent Orchestrator

Enterprise orchestration layer that:
- Uses Familiar's core Agent for LLM interactions
- Adds database persistence for messages/conversations
- Tracks usage for billing
- Manages conversation lifecycle
- Integrates tenant-scoped memory
- Uses tenant-tiered executor pools for isolation
"""

import json
import logging
import secrets
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Optional
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

# Import from reflection_core (shared primitives)
from reflection_core.security.trust import TrustLevel

# Import from familiar (core AI)
try:
    from familiar.core.agent import Agent

    FAMILIAR_AVAILABLE = True
except ImportError:
    FAMILIAR_AVAILABLE = False
    Agent = None

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
from .settings import get_settings
from .usage_alerts import (
    get_usage_alert_service,
)
from .usage_calculator import (
    get_usage_calculator,
)

logger = logging.getLogger(__name__)


# ============================================================
# ORCHESTRATOR EVENTS
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
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        return f"data: {json.dumps(self.to_dict())}\n\n"


@dataclass
class OrchestratorResult:
    """Final result from orchestrated agent execution."""

    text: str
    conversation_id: str
    message_id: str
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    iterations: int = 1
    latency_ms: int = 0
    memories_created: int = 0
    cost_usd: str = "0.00"  # Cost as string for JSON serialization


class AgentOrchestrator:
    """
    Enterprise orchestration layer for AI agents.

    Wraps Familiar's Agent with:
    - Database persistence (PostgreSQL)
    - Usage tracking for billing
    - Tenant isolation via tiered executor pools
    - Conversation management
    - Backpressure handling

    Usage:
        async with AgentOrchestrator(session, tenant_id, tenant_tier) as orch:
            result = await orch.chat("Hello!")

        # Or streaming
        async with AgentOrchestrator(session, tenant_id, tenant_tier) as orch:
            async for event in orch.stream("Hello!"):
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

        # Agent (lazy loaded)
        self._agent: TenantAgent | None = None
        self._agent_model: AgentModel | None = None

        # Executor pool (shared across orchestrators)
        self._executor_pool = get_executor_pool()

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

        logger.debug(f"Initializing orchestrator for tenant {self.tenant_id}")
        self._initialized = True

    async def _cleanup(self):
        """Clean up resources."""
        pass

    async def _get_agent(self, agent_id: UUID | None = None) -> TenantAgent:
        """Get or create the tenant agent."""
        if self._agent is not None:
            return self._agent

        # Load agent config from database
        if agent_id:
            self._agent_model = await self.agent_repo.get_by_id(agent_id)
        else:
            self._agent_model = await self.agent_repo.get_default(self.tenant_id)

        if not self._agent_model:
            # Create default agent with centralized model setting
            settings = get_settings()
            self._agent_model = await self.agent_repo.create(
                tenant_id=self.tenant_id,
                name="Default Assistant",
                provider=settings.llm.default_provider,
                model=settings.llm.default_model,
            )
            await self.session.flush()

        # Create TenantAgent with config from database
        tenant_config = {
            "agent_name": self._agent_model.name,
            "default_provider": self._agent_model.provider,
            "default_model": self._agent_model.model,
        }
        if self._agent_model.config:
            tenant_config.update(self._agent_model.config)

        self._agent = TenantAgent(
            tenant_id=self.tenant_id,
            tenant_config=tenant_config,
            db_session=self.session,
        )

        return self._agent

    async def _get_or_create_conversation(
        self,
        conversation_id: UUID | None = None,
        agent_id: UUID | None = None,
        channel: str = "api",
    ) -> ConversationModel:
        """Get or create conversation."""
        if conversation_id:
            conv = await self.conversation_repo.get_by_id(conversation_id)
            if conv and conv.tenant_id == self.tenant_id:
                self._conversation = conv
                return conv

        # Create new conversation
        await self._get_agent(agent_id)
        self._conversation = await self.conversation_repo.create(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            agent_id=self._agent_model.id if self._agent_model else None,
            channel=channel,
        )
        await self.session.flush()
        return self._conversation

    async def _load_history(
        self,
        conversation_id: UUID,
        limit: int = 20,
    ) -> list[dict[str, str]]:
        """Load conversation history from database."""
        messages = await self.message_repo.get_by_conversation(
            conversation_id,
            limit=limit,
        )

        history = []
        for msg in messages:
            history.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                }
            )

        return history

    async def _save_message(
        self,
        conversation_id: UUID,
        role: str,
        content: str,
        tokens_input: int = 0,
        tokens_output: int = 0,
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> MessageModel:
        """Save message to database."""
        message = await self.message_repo.create(
            conversation_id=conversation_id,
            role=role,
            content=content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            tool_calls=tool_calls,
            tool_results=tool_results,
        )
        await self.session.flush()

        # Update conversation stats
        await self.conversation_repo.update_stats(
            conversation_id,
            tokens_input + tokens_output,
        )

        return message

    async def _record_usage(
        self,
        tokens_input: int,
        tokens_output: int,
        model: str,
        latency_ms: int,
        cost_usd: Optional["Decimal"] = None,
    ):
        """
        Record usage for billing and check budget alerts (v1.3.0).

        This method:
        1. Records usage to database
        2. Checks if budget thresholds are exceeded
        3. Triggers alerts if needed
        """
        from decimal import Decimal

        # Calculate cost if not provided
        if cost_usd is None:
            from .tokens import get_model_pricing

            pricing = get_model_pricing(model)
            if pricing:
                cost_usd = pricing.calculate_cost(tokens_input, tokens_output)
            else:
                cost_usd = Decimal("0")

        await self.usage_repo.create(
            tenant_id=self.tenant_id,
            user_id=self.user_id,
            resource_type="llm",
            resource_id=model,
            quantity=tokens_input + tokens_output,
            unit="tokens",
            metadata={
                "input_tokens": tokens_input,
                "output_tokens": tokens_output,
                "model": model,
                "latency_ms": latency_ms,
                "cost_usd": str(cost_usd),
            },
        )

        # Check budget alerts asynchronously (don't block response)
        try:
            await self._check_budget_alerts(tokens_input + tokens_output, cost_usd)
        except Exception as e:
            logger.warning(f"Failed to check budget alerts: {e}")

    async def _check_budget_alerts(
        self,
        tokens_used: int,
        cost_usd: "Decimal",
    ):
        """
        Check usage against budget thresholds and trigger alerts (v1.3.0).

        This runs after each request to check if warning/critical
        thresholds have been crossed.
        """
        from ..tenants.quota_service import get_quota_service, get_tier_limits

        try:
            # Get current usage summary
            quota_service = get_quota_service()
            usage_data = await quota_service.get_usage(
                self.tenant_id,
                self.tenant_tier,
            )

            # Get limits
            limits = get_tier_limits(self.tenant_tier)

            # Build alert data
            alert_data = {}

            usage_info = usage_data.get("usage", {})

            # Token alerts
            if "tokens" in usage_info:
                tokens = usage_info["tokens"]
                if "day" in tokens:
                    alert_data["tokens_daily"] = {
                        "current": tokens["day"].get("current", 0),
                        "limit": limits.tokens_per_day,
                    }
                if "month" in tokens:
                    alert_data["tokens_monthly"] = {
                        "current": tokens["month"].get("current", 0),
                        "limit": limits.tokens_per_month,
                    }

            # Request alerts
            if "requests" in usage_info:
                requests = usage_info["requests"]
                if "day" in requests:
                    alert_data["requests_daily"] = {
                        "current": requests["day"].get("current", 0),
                        "limit": limits.requests_per_day,
                    }

            # Check and send alerts
            if alert_data:
                alert_service = get_usage_alert_service()
                await alert_service.check_and_alert(
                    tenant_id=self.tenant_id,
                    usage_data=alert_data,
                )

        except Exception as e:
            # Don't fail the request if alerting fails
            logger.debug(f"Budget alert check failed (non-critical): {e}")

    async def chat(
        self,
        message: str,
        agent_id: UUID | None = None,
        conversation_id: UUID | None = None,
        channel: str = "api",
        include_history: bool = True,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> OrchestratorResult:
        """
        Send a message and get a response.

        Uses Familiar's Agent for the actual LLM interaction,
        executed in a tenant-tiered thread pool for isolation.

        Args:
            message: User's message
            agent_id: Specific agent to use (optional)
            conversation_id: Continue existing conversation (optional)
            channel: Source channel (api, discord, etc.)
            include_history: Include conversation history
            max_tokens: Override max tokens
            temperature: Override temperature

        Returns:
            OrchestratorResult with response and metadata

        Raises:
            QueueFullError: If tenant's tier queue is full
            ExecutorTimeoutError: If request times out
        """
        start_time = time.time()

        # Get agent
        agent = await self._get_agent(agent_id)

        # Get or create conversation
        conversation = await self._get_or_create_conversation(
            conversation_id=conversation_id,
            agent_id=agent_id,
            channel=channel,
        )

        # Save user message
        user_msg = await self._save_message(
            conversation_id=conversation.id,
            role="user",
            content=message,
        )

        # Load history if enabled
        history = []
        if include_history and conversation_id:
            history = await self._load_history(conversation.id)

        # Execute agent.chat() in tenant's tiered thread pool
        # This prevents sync LLM calls from blocking the async event loop
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
        except QueueFullError:
            logger.warning(
                f"Queue full for tenant {self.tenant_id} (tier={self.tenant_tier.value})"
            )
            # Return graceful error response
            return OrchestratorResult(
                text="I'm currently experiencing high demand. Please try again in a moment.",
                conversation_id=str(conversation.id),
                message_id=str(user_msg.id),
                usage={},
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except ExecutorTimeoutError as e:
            logger.error(f"Timeout for tenant {self.tenant_id}: {e.elapsed_seconds:.1f}s")
            # Return graceful timeout response
            return OrchestratorResult(
                text="Your request took too long to process. Please try a simpler question or try again later.",
                conversation_id=str(conversation.id),
                message_id=str(user_msg.id),
                usage={},
                latency_ms=int((time.time() - start_time) * 1000),
            )
        except Exception as e:
            # Log full error details for debugging (server-side only)
            import traceback

            request_id = secrets.token_hex(8)
            logger.error(
                f"Agent error for tenant {self.tenant_id}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "tenant_id": str(self.tenant_id),
                    "user_id": str(self.user_id) if self.user_id else None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            # Return sanitized message to user - NEVER expose internal errors
            response = (
                "I apologize, but I encountered an unexpected error processing your request. "
                f"Please try again. If the problem persists, contact support with reference: {request_id}"
            )

        # Calculate accurate token usage using unified calculator (v1.3.0)
        model = self._agent_model.model if self._agent_model else get_settings().llm.default_model
        usage_calc = get_usage_calculator(model)

        usage_record = usage_calc.calculate(
            input_message=message,
            output_text=response,
            system_prompt=self._agent_model.system_prompt if self._agent_model else None,
            history=history if history else None,
        )

        tokens_input = usage_record.input_tokens
        tokens_output = usage_record.output_tokens

        # Save assistant message
        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        # Record usage with cost (v1.3.0)
        latency_ms = int((time.time() - start_time) * 1000)
        await self._record_usage(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model,
            latency_ms=latency_ms,
            cost_usd=usage_record.cost_usd,
        )

        await self.session.commit()

        return OrchestratorResult(
            text=response,
            conversation_id=str(conversation.id),
            message_id=str(assistant_msg.id),
            usage=usage_record.to_dict(),
            latency_ms=latency_ms,
            cost_usd=str(usage_record.cost_usd or "0.00"),
        )

    async def stream(
        self,
        message: str,
        agent_id: UUID | None = None,
        conversation_id: UUID | None = None,
        channel: str = "api",
        include_history: bool = True,
    ) -> AsyncIterator[OrchestratorEvent]:
        """
        Stream response with events.

        Executes agent in tiered thread pool and streams results.
        """
        start_time = time.time()

        # Get agent and conversation
        agent = await self._get_agent(agent_id)
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

        # Yield thinking event
        yield OrchestratorEvent(
            type="thinking",
            content="Processing your request...",
            conversation_id=str(conversation.id),
        )

        # Execute in tiered thread pool
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

            # Simulate streaming by chunking the response
            chunk_size = 20
            for i in range(0, len(response), chunk_size):
                chunk = response[i : i + chunk_size]
                yield OrchestratorEvent(
                    type="text",
                    content=chunk,
                    conversation_id=str(conversation.id),
                )

        except QueueFullError:
            yield OrchestratorEvent(
                type="error",
                content="Service is busy. Please try again in a moment.",
                conversation_id=str(conversation.id),
            )
            response = "Service temporarily unavailable."

        except ExecutorTimeoutError as e:
            yield OrchestratorEvent(
                type="error",
                content=f"Request timed out after {e.timeout_seconds}s.",
                conversation_id=str(conversation.id),
            )
            response = "Request timed out."

        except Exception as e:
            # Log full error details for debugging (server-side only)
            import traceback

            request_id = secrets.token_hex(8)
            logger.error(
                f"Stream error for tenant {self.tenant_id}",
                exc_info=True,
                extra={
                    "request_id": request_id,
                    "tenant_id": str(self.tenant_id),
                    "user_id": str(self.user_id) if self.user_id else None,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
            # Return sanitized message to user - NEVER expose internal errors
            yield OrchestratorEvent(
                type="error",
                content=f"An unexpected error occurred. Reference: {request_id}",
                conversation_id=str(conversation.id),
            )
            response = "An unexpected error occurred."

        # Save and record - use unified UsageCalculator for consistent billing (v1.3.0)
        model_name = (
            self._agent_model.model if self._agent_model else get_settings().llm.default_model
        )
        usage_calc = get_usage_calculator(model_name)

        # Calculate usage consistently with chat() method
        usage_record = usage_calc.calculate(
            input_message=message,
            output_text=response,
            system_prompt=self._agent_model.system_prompt if self._agent_model else None,
            history=None,  # History loaded separately if needed
        )

        tokens_input = usage_record.input_tokens
        tokens_output = usage_record.output_tokens
        latency_ms = int((time.time() - start_time) * 1000)

        assistant_msg = await self._save_message(
            conversation_id=conversation.id,
            role="assistant",
            content=response,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
        )

        await self._record_usage(
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            model=model_name,
            latency_ms=latency_ms,
            cost_usd=usage_record.cost_usd,
        )

        await self.session.commit()

        # Final done event
        yield OrchestratorEvent(
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
    "AgentOrchestrator",
    "OrchestratorEvent",
    "OrchestratorResult",
    "FAMILIAR_AVAILABLE",
]
