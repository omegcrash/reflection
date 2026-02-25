# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 5: Async LLM Providers (Official SDK Implementation)

True async providers using official SDKs:
- anthropic.AsyncAnthropic for Claude
- openai.AsyncOpenAI for GPT

These provide:
- Native connection pooling
- Automatic retries with backoff
- Proper SSE stream parsing
- Accurate token counts from responses

This module is used by AsyncOrchestrator for the async chat path,
bypassing the thread pool for simple conversations.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from reflection_core import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    get_provider_circuit_breaker,
)
from .settings import DEFAULT_ANTHROPIC_MODEL, LLMSettings, get_settings

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS (shared with sync providers)
# ============================================================


class StopReason(StrEnum):
    """Reason the model stopped generating."""

    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"


@dataclass
class ToolCall:
    """A tool call from the model."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class Usage:
    """Token usage statistics from provider response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = field(default_factory=Usage)
    model: str = ""

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class StreamEventType(StrEnum):
    """Types of streaming events."""

    MESSAGE_START = "message_start"
    CONTENT_BLOCK_START = "content_block_start"
    TEXT_DELTA = "text_delta"
    INPUT_JSON_DELTA = "input_json_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_DELTA = "message_delta"
    MESSAGE_STOP = "message_stop"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A streaming event from the model."""

    type: StreamEventType
    text: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None
    stop_reason: StopReason | None = None
    error: str | None = None


# ============================================================
# ABSTRACT ASYNC PROVIDER
# ============================================================


class AsyncLLMProviderSDK(ABC):
    """
    Abstract base for async LLM providers using official SDKs.

    Unlike the httpx-based providers, these use:
    - anthropic.AsyncAnthropic
    - openai.AsyncOpenAI

    Which provide native async support, connection pooling,
    and proper error handling.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Current model name."""
        pass

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Async chat completion."""
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming chat completion."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close provider resources."""
        pass


# ============================================================
# ANTHROPIC ASYNC PROVIDER (Official SDK)
# ============================================================


class AsyncAnthropicProviderSDK(AsyncLLMProviderSDK):
    """
    Async Anthropic Claude provider using official SDK.

    Uses anthropic.AsyncAnthropic which provides:
    - Native async/await support
    - Connection pooling
    - Automatic retries with exponential backoff
    - Proper SSE stream parsing
    - Accurate token counts

    Usage:
        provider = AsyncAnthropicProviderSDK(api_key="sk-ant-...")
        response = await provider.chat([{"role": "user", "content": "Hello"}])

        # Streaming
        async for event in provider.stream([{"role": "user", "content": "Hello"}]):
            if event.type == StreamEventType.TEXT_DELTA:
                print(event.text, end="")
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout: float = 300.0,
        max_retries: int = 2,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self._client = None
        self._api_key = api_key
        self._max_retries = max_retries
        self._circuit_breaker = circuit_breaker

    @property
    def name(self) -> str:
        return f"anthropic-sdk:{self.model}"

    @property
    def model_name(self) -> str:
        return self.model

    async def _get_client(self):
        """Lazy client initialization."""
        if self._client is None:
            try:
                import anthropic
            except ImportError as e:
                raise ProviderError(
                    "anthropic package not installed. Run: pip install anthropic",
                    provider=self.name,
                ) from e

            self._client = anthropic.AsyncAnthropic(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=self._max_retries,
            )
        return self._client

    async def close(self) -> None:
        """Close the async client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """
        Async chat completion with Claude.

        Returns accurate token counts from the API response.
        Uses circuit breaker to fail fast when provider is unavailable.
        """
        import anthropic

        # Check circuit breaker before calling
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        client = await self._get_client()

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._convert_messages(messages),
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.messages.create(**kwargs)
            result = self._parse_response(response)

            # Record success
            if self._circuit_breaker:
                await self._circuit_breaker.on_success()

            return result

        except anthropic.AuthenticationError as e:
            error = ProviderAuthenticationError(
                f"Anthropic authentication failed: {e}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except anthropic.RateLimitError as e:
            error = ProviderRateLimitError(
                f"Anthropic rate limit exceeded: {e}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except anthropic.APIConnectionError as e:
            error = ProviderConnectionError(
                f"Failed to connect to Anthropic: {e}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except anthropic.APIStatusError as e:
            error = ProviderResponseError(
                f"Anthropic API error: {e.status_code} - {e.message}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """
        Async streaming with Claude.

        Yields events for:
        - Text deltas
        - Tool use start/delta/end
        - Message completion with final usage

        Uses circuit breaker to fail fast when provider is unavailable.
        """
        import anthropic

        # Check circuit breaker before calling
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        client = await self._get_client()

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": self._convert_messages(messages),
        }

        if system:
            kwargs["system"] = system

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        stream_success = False

        try:
            async with client.messages.stream(**kwargs) as stream:
                current_tool: dict[str, Any] | None = None

                async for event in stream:
                    # Message start
                    if event.type == "message_start":
                        yield StreamEvent(type=StreamEventType.MESSAGE_START)

                    # Content block start (text or tool_use)
                    elif event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {
                                "id": block.id,
                                "name": block.name,
                                "input": "",
                            }
                            yield StreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_START,
                                tool_call=ToolCall(id=block.id, name=block.name, input={}),
                            )

                    # Content deltas
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.text)
                        elif delta.type == "input_json_delta" and current_tool:
                            current_tool["input"] += delta.partial_json
                            yield StreamEvent(
                                type=StreamEventType.INPUT_JSON_DELTA, text=delta.partial_json
                            )

                    # Content block stop
                    elif event.type == "content_block_stop":
                        if current_tool:
                            import json

                            try:
                                input_dict = json.loads(current_tool["input"])
                            except json.JSONDecodeError:
                                input_dict = {}

                            yield StreamEvent(
                                type=StreamEventType.CONTENT_BLOCK_STOP,
                                tool_call=ToolCall(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    input=input_dict,
                                ),
                            )
                            current_tool = None

                    # Message delta (stop reason)
                    elif event.type == "message_delta":
                        stop_reason = StopReason.END_TURN
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            stop_reason = StopReason(event.delta.stop_reason)

                        usage = None
                        if hasattr(event, "usage") and event.usage:
                            usage = Usage(output_tokens=event.usage.output_tokens)

                        yield StreamEvent(
                            type=StreamEventType.MESSAGE_DELTA, stop_reason=stop_reason, usage=usage
                        )

                    # Message stop
                    elif event.type == "message_stop":
                        # Get final message for complete usage
                        final_message = await stream.get_final_message()
                        yield StreamEvent(
                            type=StreamEventType.MESSAGE_STOP,
                            usage=Usage(
                                input_tokens=final_message.usage.input_tokens,
                                output_tokens=final_message.usage.output_tokens,
                            ),
                        )
                        # Mark stream as successful
                        stream_success = True

            # Record success after stream completes
            if self._circuit_breaker and stream_success:
                await self._circuit_breaker.on_success()

        except anthropic.APIError as e:
            # Record failure
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(e)
            yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
            raise

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert internal message format to Anthropic format."""
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool results
            if role == "user" and isinstance(content, list):
                anthropic_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            anthropic_content.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": item["tool_use_id"],
                                    "content": item.get("content", ""),
                                }
                            )
                        elif item.get("type") == "text":
                            anthropic_content.append(item)
                        else:
                            anthropic_content.append({"type": "text", "text": str(item)})
                    else:
                        anthropic_content.append({"type": "text", "text": str(item)})
                converted.append({"role": "user", "content": anthropic_content})

            # Handle assistant messages with tool calls
            elif role == "assistant" and isinstance(content, list):
                converted.append({"role": "assistant", "content": content})

            # Simple text messages
            else:
                converted.append({"role": role, "content": str(content)})

        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert tools to Anthropic format."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {})),
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> LLMResponse:
        """Parse Anthropic response to internal format."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=block.input))

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=StopReason(response.stop_reason)
            if response.stop_reason
            else StopReason.END_TURN,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_creation_input_tokens=getattr(
                    response.usage, "cache_creation_input_tokens", 0
                )
                or 0,
                cache_read_input_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            ),
            model=response.model,
        )


# ============================================================
# OPENAI ASYNC PROVIDER (Official SDK)
# ============================================================


class AsyncOpenAIProviderSDK(AsyncLLMProviderSDK):
    """
    Async OpenAI GPT provider using official SDK.

    Uses openai.AsyncOpenAI which provides:
    - Native async/await support
    - Connection pooling
    - Automatic retries
    - Proper SSE stream parsing
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        timeout: float = 300.0,
        max_retries: int = 2,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.model = model
        self.timeout = timeout
        self._client = None
        self._api_key = api_key
        self._max_retries = max_retries
        self._circuit_breaker = circuit_breaker

    @property
    def name(self) -> str:
        return f"openai-sdk:{self.model}"

    @property
    def model_name(self) -> str:
        return self.model

    async def _get_client(self):
        """Lazy client initialization."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai", provider=self.name
                ) from e

            self._client = AsyncOpenAI(
                api_key=self._api_key,
                timeout=self.timeout,
                max_retries=self._max_retries,
            )
        return self._client

    async def close(self) -> None:
        """Close the async client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Async chat completion with GPT with circuit breaker protection."""
        import openai

        # Check circuit breaker before calling
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        client = await self._get_client()

        # Build messages with system
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(self._convert_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        try:
            response = await client.chat.completions.create(**kwargs)
            result = self._parse_response(response)

            # Record success
            if self._circuit_breaker:
                await self._circuit_breaker.on_success()

            return result

        except openai.AuthenticationError as e:
            error = ProviderAuthenticationError(
                f"OpenAI authentication failed: {e}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except openai.RateLimitError as e:
            error = ProviderRateLimitError(f"OpenAI rate limit exceeded: {e}", provider=self.name)
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except openai.APIConnectionError as e:
            error = ProviderConnectionError(f"Failed to connect to OpenAI: {e}", provider=self.name)
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e
        except openai.APIStatusError as e:
            error = ProviderResponseError(
                f"OpenAI API error: {e.status_code} - {e.message}", provider=self.name
            )
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(error)
            raise error from e

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming with GPT with circuit breaker protection."""
        import openai

        # Check circuit breaker before calling
        if self._circuit_breaker:
            await self._circuit_breaker.before_call()

        client = await self._get_client()

        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        openai_messages.extend(self._convert_messages(messages))

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        stream_success = False

        try:
            stream = await client.chat.completions.create(**kwargs)

            current_tool_calls: dict[int, dict[str, Any]] = {}

            async for chunk in stream:
                # Handle usage (comes at the end with stream_options)
                if chunk.usage:
                    yield StreamEvent(
                        type=StreamEventType.MESSAGE_STOP,
                        usage=Usage(
                            input_tokens=chunk.usage.prompt_tokens,
                            output_tokens=chunk.usage.completion_tokens,
                        ),
                    )
                    stream_success = True
                    continue

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta

                # Text content
                if delta.content:
                    yield StreamEvent(type=StreamEventType.TEXT_DELTA, text=delta.content)

                # Tool calls
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index

                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc.id or "",
                                "name": tc.function.name if tc.function else "",
                                "arguments": "",
                            }
                            if tc.function and tc.function.name:
                                yield StreamEvent(
                                    type=StreamEventType.CONTENT_BLOCK_START,
                                    tool_call=ToolCall(
                                        id=current_tool_calls[idx]["id"],
                                        name=current_tool_calls[idx]["name"],
                                        input={},
                                    ),
                                )

                        if tc.function and tc.function.arguments:
                            current_tool_calls[idx]["arguments"] += tc.function.arguments
                            yield StreamEvent(
                                type=StreamEventType.INPUT_JSON_DELTA, text=tc.function.arguments
                            )

                # Finish reason
                if choice.finish_reason:
                    # Emit tool call completions
                    for tc_data in current_tool_calls.values():
                        import json

                        try:
                            input_dict = json.loads(tc_data["arguments"])
                        except json.JSONDecodeError:
                            input_dict = {}

                        yield StreamEvent(
                            type=StreamEventType.CONTENT_BLOCK_STOP,
                            tool_call=ToolCall(
                                id=tc_data["id"], name=tc_data["name"], input=input_dict
                            ),
                        )

                    stop_reason = StopReason.END_TURN
                    if choice.finish_reason == "tool_calls":
                        stop_reason = StopReason.TOOL_USE
                    elif choice.finish_reason == "length":
                        stop_reason = StopReason.MAX_TOKENS
                    elif choice.finish_reason == "stop":
                        stop_reason = StopReason.END_TURN

                    yield StreamEvent(type=StreamEventType.MESSAGE_DELTA, stop_reason=stop_reason)
                    stream_success = True

            # Record success after stream completes
            if self._circuit_breaker and stream_success:
                await self._circuit_breaker.on_success()

        except openai.APIError as e:
            # Record failure
            if self._circuit_breaker:
                await self._circuit_breaker.on_failure(e)
            yield StreamEvent(type=StreamEventType.ERROR, error=str(e))
            raise

    def _convert_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to OpenAI format."""
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Handle tool results
            if role == "tool":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", ""),
                        "content": str(content),
                    }
                )
            # Handle assistant with tool calls
            elif role == "assistant" and msg.get("tool_calls"):
                converted.append(
                    {
                        "role": "assistant",
                        "content": content if content else None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc.get("arguments", "{}"),
                                },
                            }
                            for tc in msg["tool_calls"]
                        ],
                    }
                )
            else:
                converted.append({"role": role, "content": str(content)})

        return converted

    def _convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to OpenAI function format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", tool.get("parameters", {})),
                },
            }
            for tool in tools
        ]

    def _parse_response(self, response) -> LLMResponse:
        """Parse OpenAI response."""
        import json

        message = response.choices[0].message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    input_dict = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    input_dict = {}

                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, input=input_dict))

        finish_reason = response.choices[0].finish_reason
        stop_reason = StopReason.END_TURN
        if finish_reason == "tool_calls":
            stop_reason = StopReason.TOOL_USE
        elif finish_reason == "length":
            stop_reason = StopReason.MAX_TOKENS

        return LLMResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
            model=response.model,
        )


# ============================================================
# PROVIDER FACTORY
# ============================================================


async def create_async_provider(
    provider_name: str,
    settings: LLMSettings | None = None,
    use_circuit_breaker: bool = True,
) -> AsyncLLMProviderSDK:
    """
    Create an async LLM provider by name.

    Args:
        provider_name: Provider name (anthropic, openai)
        settings: Optional LLM settings
        use_circuit_breaker: Whether to enable circuit breaker (default: True)

    Returns:
        Configured AsyncLLMProviderSDK instance with circuit breaker
    """
    if settings is None:
        settings = get_settings().llm

    # Get circuit breaker for this provider
    circuit_breaker = None
    if use_circuit_breaker:
        circuit_breaker = await get_provider_circuit_breaker(provider_name)

    name_lower = provider_name.lower()

    if name_lower in ("anthropic", "claude"):
        api_key = settings.anthropic_api_key
        if not api_key:
            # Try environment variable
            import os

            api_key = os.environ.get("ANTHROPIC_API_KEY")

        if not api_key:
            raise ProviderError("ANTHROPIC_API_KEY not configured", provider=provider_name)

        return AsyncAnthropicProviderSDK(
            api_key=api_key,
            model=settings.anthropic_model,
            timeout=float(settings.request_timeout),
            circuit_breaker=circuit_breaker,
        )

    elif name_lower in ("openai", "gpt"):
        api_key = settings.openai_api_key
        if not api_key:
            import os

            api_key = os.environ.get("OPENAI_API_KEY")

        if not api_key:
            raise ProviderError("OPENAI_API_KEY not configured", provider=provider_name)

        return AsyncOpenAIProviderSDK(
            api_key=api_key,
            model=settings.openai_model,
            timeout=float(settings.request_timeout),
            circuit_breaker=circuit_breaker,
        )

    else:
        raise ProviderError(
            f"Unknown async provider: {provider_name}. Supported: anthropic, openai",
            provider=provider_name,
        )


# ============================================================
# PROVIDER POOL (Connection Reuse)
# ============================================================


class AsyncProviderPool:
    """
    Pool of async providers for connection reuse.

    Maintains a single provider instance per (provider_name, model) tuple
    to reuse HTTP connections across requests.
    """

    def __init__(self):
        self._providers: dict[str, AsyncLLMProviderSDK] = {}

    async def get_provider(
        self,
        provider_name: str,
        model: str | None = None,
        settings: LLMSettings | None = None,
    ) -> AsyncLLMProviderSDK:
        """Get or create a provider instance."""
        if settings is None:
            settings = get_settings().llm

        # Determine model
        if model is None:
            if provider_name.lower() in ("anthropic", "claude"):
                model = settings.anthropic_model
            elif provider_name.lower() in ("openai", "gpt"):
                model = settings.openai_model
            else:
                model = "default"

        key = f"{provider_name}:{model}"

        if key not in self._providers:
            provider = await create_async_provider(provider_name, settings)
            self._providers[key] = provider

        return self._providers[key]

    async def close_all(self):
        """Close all providers."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


# Global provider pool
_provider_pool: AsyncProviderPool | None = None


def get_async_provider_pool() -> AsyncProviderPool:
    """Get the global async provider pool."""
    global _provider_pool
    if _provider_pool is None:
        _provider_pool = AsyncProviderPool()
    return _provider_pool


async def shutdown_async_providers():
    """Shutdown all async providers."""
    global _provider_pool
    if _provider_pool is not None:
        await _provider_pool.close_all()
        _provider_pool = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Models
    "StopReason",
    "ToolCall",
    "Usage",
    "LLMResponse",
    "StreamEventType",
    "StreamEvent",
    # Providers
    "AsyncLLMProviderSDK",
    "AsyncAnthropicProviderSDK",
    "AsyncOpenAIProviderSDK",
    # Factory
    "create_async_provider",
    # Pool
    "AsyncProviderPool",
    "get_async_provider_pool",
    "shutdown_async_providers",
    # Circuit Breaker (re-exported for convenience)
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitOpenError",
    "get_provider_circuit_breaker",
]
