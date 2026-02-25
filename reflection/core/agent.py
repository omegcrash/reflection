# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Async Agent

The agent orchestrates:
- LLM provider interactions
- Tool execution loops
- Memory retrieval
- Conversation management

This is the core intelligence layer.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from reflection_core.security.trust import Capability, TrustLevel, get_capabilities_for_trust

from .async_base import AsyncInitMixin
from .providers import (
    AsyncLLMProvider,
    StreamEventType,
    ToolCall,
    Usage,
    create_provider,
)
from .settings import DEFAULT_ANTHROPIC_MODEL, get_settings
from .tools import AsyncToolRegistry, ToolResult, create_default_tools

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
# ============================================================


class MessageRole(StrEnum):
    """Message roles in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class Message:
    """A message in a conversation."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # For tool results
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for LLM API."""
        msg: dict[str, Any] = {
            "role": self.role.value,
            "content": self.content,
        }
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass
class Conversation:
    """A conversation with history."""

    id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Metadata
    tenant_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    channel: str = "api"

    # Stats
    total_tokens: int = 0

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def get_messages_for_llm(self) -> list[dict[str, Any]]:
        """Get messages formatted for LLM API."""
        return [msg.to_dict() for msg in self.messages if msg.role != MessageRole.SYSTEM]


@dataclass
class AgentResponse:
    """Response from an agent chat."""

    text: str
    conversation_id: str
    tool_results: list[ToolResult] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    iterations: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "conversation_id": self.conversation_id,
            "tool_results": [tr.to_dict() for tr in self.tool_results],
            "usage": self.usage.to_dict(),
            "iterations": self.iterations,
        }


class AgentEventType(StrEnum):
    """Types of streaming events from agent."""

    TEXT = "text"
    TOOL_START = "tool_start"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    DONE = "done"
    ERROR = "error"


@dataclass
class AgentEvent:
    """A streaming event from the agent."""

    type: AgentEventType
    content: str = ""
    tool_name: str | None = None
    tool_result: ToolResult | None = None
    usage: Usage | None = None


# ============================================================
# AGENT CONFIGURATION
# ============================================================


@dataclass
class AgentConfig:
    """Configuration for an agent instance."""

    # Identity
    name: str = "Assistant"
    system_prompt: str = "You are a helpful AI assistant."

    # Provider
    provider: str = "anthropic"
    model: str = DEFAULT_ANTHROPIC_MODEL

    # Behavior
    max_iterations: int = 15
    max_tokens: int = 4096
    temperature: float = 0.7

    # Tools
    enable_tools: bool = True
    tool_choice: str = "auto"  # auto, required, none

    # Timeouts
    request_timeout: float = 300.0
    tool_timeout: float = 30.0


# ============================================================
# ASYNC AGENT
# ============================================================


class AsyncAgent(AsyncInitMixin):
    """
    Async AI Agent.

    Orchestrates LLM interactions with tool execution in an
    agentic loop. Supports streaming and parallel tool execution.

    Usage:
        agent = await AsyncAgent.create(config)
        response = await agent.chat("Hello!")

        async for event in agent.stream("Tell me about the weather"):
            print(event)

        await agent.close()
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        provider: AsyncLLMProvider | None = None,
        tools: AsyncToolRegistry | None = None,
        tenant_id: str | None = None,
    ):
        self.config = config or AgentConfig()
        self._provider = provider
        self._tools = tools
        self.tenant_id = tenant_id

        # Conversation storage (in-memory for Step 1)
        self._conversations: dict[str, Conversation] = {}

    async def _async_init(self) -> None:
        """Async initialization."""
        # Initialize provider if not provided
        if self._provider is None:
            settings = get_settings()
            self._provider = await create_provider(self.config.provider, settings.llm)

        # Initialize tools if not provided
        if self._tools is None:
            self._tools = create_default_tools()

        logger.info(f"Agent initialized: {self.config.name}")

    async def _async_cleanup(self) -> None:
        """Cleanup resources."""
        if self._provider:
            await self._provider.close()

    async def close(self) -> None:
        """Close the agent and release resources."""
        await self._async_cleanup()

    @property
    def provider(self) -> AsyncLLMProvider:
        """Get the LLM provider."""
        if self._provider is None:
            raise RuntimeError("Agent not initialized")
        return self._provider

    @property
    def tools(self) -> AsyncToolRegistry:
        """Get the tool registry."""
        if self._tools is None:
            raise RuntimeError("Agent not initialized")
        return self._tools

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Get a conversation by ID."""
        return self._conversations.get(conversation_id)

    def create_conversation(
        self,
        user_id: str | None = None,
        channel: str = "api",
    ) -> Conversation:
        """Create a new conversation."""
        conv_id = f"conv_{uuid.uuid4().hex[:16]}"

        conversation = Conversation(
            id=conv_id,
            tenant_id=self.tenant_id,
            user_id=user_id,
            channel=channel,
        )

        self._conversations[conv_id] = conversation
        return conversation

    async def chat(
        self,
        message: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        trust_level: TrustLevel = TrustLevel.KNOWN,
        capabilities: set[Capability] | None = None,
    ) -> AgentResponse:
        """
        Send a message and get a response.

        Executes the full agent loop including tool use.

        Args:
            message: User message
            conversation_id: Optional existing conversation
            user_id: User identifier
            trust_level: User's trust level for tool permissions
            capabilities: Explicit user capabilities

        Returns:
            AgentResponse with text and metadata
        """
        # Get or create conversation
        if conversation_id:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                conversation = self.create_conversation(user_id)
        else:
            conversation = self.create_conversation(user_id)

        # Add user message
        conversation.add_message(
            Message(
                role=MessageRole.USER,
                content=message,
            )
        )

        # Get effective capabilities
        if capabilities is None:
            capabilities = set(get_capabilities_for_trust(trust_level))

        # Get tool schemas based on permissions
        tool_schemas = None
        if self.config.enable_tools:
            tool_schemas = self.tools.get_schemas(trust_level, capabilities)

        # Agent loop
        total_usage = Usage()
        all_tool_results: list[ToolResult] = []
        iterations = 0
        final_text = ""

        while iterations < self.config.max_iterations:
            iterations += 1

            # Call LLM
            response = await self.provider.chat(
                messages=conversation.get_messages_for_llm(),
                system=self.config.system_prompt,
                tools=tool_schemas,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )

            # Accumulate usage
            total_usage.input_tokens += response.usage.input_tokens
            total_usage.output_tokens += response.usage.output_tokens

            # Check for tool calls
            if response.has_tool_calls:
                # Add assistant message with tool calls
                conversation.add_message(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=response.text,
                        tool_calls=response.tool_calls,
                    )
                )

                # Execute tools in parallel
                tool_calls_input = [
                    {"name": tc.name, "arguments": tc.input} for tc in response.tool_calls
                ]

                results = await self.tools.execute_parallel(
                    tool_calls_input,
                    trust_level,
                    capabilities,
                )

                all_tool_results.extend(results)

                # Add tool results to conversation
                for tc, result in zip(response.tool_calls, results, strict=False):
                    output = result.output if result.success else f"Error: {result.error}"
                    conversation.add_message(
                        Message(
                            role=MessageRole.TOOL,
                            content=str(output),
                            tool_call_id=tc.id,
                        )
                    )

                # Continue loop to process tool results
                continue

            # No tool calls - we have the final response
            final_text = response.text

            # Add assistant response
            conversation.add_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=final_text,
                )
            )

            break

        # Update conversation stats
        conversation.total_tokens += total_usage.total_tokens

        return AgentResponse(
            text=final_text,
            conversation_id=conversation.id,
            tool_results=all_tool_results,
            usage=total_usage,
            iterations=iterations,
        )

    async def stream(
        self,
        message: str,
        conversation_id: str | None = None,
        user_id: str | None = None,
        trust_level: TrustLevel = TrustLevel.KNOWN,
        capabilities: set[Capability] | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """
        Stream a response with real-time events.

        Yields AgentEvents as the response is generated.

        Usage:
            async for event in agent.stream("Hello"):
                if event.type == AgentEventType.TEXT:
                    print(event.content, end="")
        """
        # Get or create conversation
        if conversation_id:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                conversation = self.create_conversation(user_id)
        else:
            conversation = self.create_conversation(user_id)

        # Add user message
        conversation.add_message(
            Message(
                role=MessageRole.USER,
                content=message,
            )
        )

        # Get capabilities
        if capabilities is None:
            capabilities = set(get_capabilities_for_trust(trust_level))

        # Get tool schemas
        tool_schemas = None
        if self.config.enable_tools:
            tool_schemas = self.tools.get_schemas(trust_level, capabilities)

        # Stream from provider
        collected_text = ""
        collected_tool_calls: list[ToolCall] = []

        async for event in self.provider.stream(
            messages=conversation.get_messages_for_llm(),
            system=self.config.system_prompt,
            tools=tool_schemas,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        ):
            if event.type == StreamEventType.TEXT:
                collected_text += event.content
                yield AgentEvent(
                    type=AgentEventType.TEXT,
                    content=event.content,
                )

            elif event.type == StreamEventType.TOOL_USE_START:
                if event.tool_call:
                    yield AgentEvent(
                        type=AgentEventType.TOOL_START,
                        tool_name=event.tool_call.name,
                    )

            elif event.type == StreamEventType.TOOL_USE_END:
                if event.tool_call:
                    collected_tool_calls.append(event.tool_call)

                    # Execute tool
                    result = await self.tools.execute(
                        event.tool_call.name,
                        event.tool_call.input,
                        trust_level,
                        capabilities,
                    )

                    yield AgentEvent(
                        type=AgentEventType.TOOL_RESULT,
                        tool_name=event.tool_call.name,
                        tool_result=result,
                    )

            elif event.type == StreamEventType.MESSAGE_END:
                break

        # Add final message to conversation
        if collected_text:
            conversation.add_message(
                Message(
                    role=MessageRole.ASSISTANT,
                    content=collected_text,
                    tool_calls=collected_tool_calls if collected_tool_calls else None,
                )
            )

        # Yield done event
        yield AgentEvent(type=AgentEventType.DONE)


# ============================================================
# FACTORY
# ============================================================


async def create_agent(
    config: AgentConfig | None = None,
    tenant_id: str | None = None,
) -> AsyncAgent:
    """
    Create and initialize an async agent.

    Args:
        config: Optional agent configuration
        tenant_id: Optional tenant ID for multi-tenancy

    Returns:
        Initialized AsyncAgent
    """
    agent = AsyncAgent(config=config, tenant_id=tenant_id)
    await agent._async_init()
    return agent


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "MessageRole",
    "Message",
    "Conversation",
    "AgentResponse",
    "AgentEventType",
    "AgentEvent",
    "AgentConfig",
    # Agent
    "AsyncAgent",
    "create_agent",
]
