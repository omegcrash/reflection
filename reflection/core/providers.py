# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Async LLM Providers

Fully async implementations for:
- Anthropic (Claude)
- OpenAI (GPT-4)
- Ollama (local models)
"""

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import httpx

from reflection_core import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)

from .async_base import timeout_context
from .settings import DEFAULT_ANTHROPIC_MODEL, LLMSettings, get_settings

logger = logging.getLogger(__name__)


# ============================================================
# DATA MODELS
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
    """Token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: StopReason = StopReason.END_TURN
    usage: Usage = field(default_factory=Usage)
    raw_response: dict[str, Any] | None = None

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class StreamEventType(StrEnum):
    """Types of streaming events."""

    TEXT = "text"
    TOOL_USE_START = "tool_use_start"
    TOOL_USE_DELTA = "tool_use_delta"
    TOOL_USE_END = "tool_use_end"
    MESSAGE_START = "message_start"
    MESSAGE_END = "message_end"
    ERROR = "error"


@dataclass
class StreamEvent:
    """A streaming event from the model."""

    type: StreamEventType
    content: str = ""
    tool_call: ToolCall | None = None
    usage: Usage | None = None


# ============================================================
# ABSTRACT PROVIDER
# ============================================================


class AsyncLLMProvider(ABC):
    """
    Abstract base for async LLM providers.

    All providers implement both chat and streaming methods.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging."""
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
# ANTHROPIC PROVIDER
# ============================================================


class AsyncAnthropicProvider(AsyncLLMProvider):
    """Async Anthropic Claude provider."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return f"anthropic:{self.model}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy client initialization."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.anthropic.com",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Async chat completion with Claude."""
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = self._format_tools(tools)

        try:
            async with timeout_context(self.timeout, "Anthropic API"):
                response = await client.post("/v1/messages", json=payload)

                if response.status_code == 401:
                    raise ProviderAuthenticationError(
                        "Invalid Anthropic API key", provider=self.name
                    )
                elif response.status_code == 429:
                    raise ProviderRateLimitError(
                        "Anthropic rate limit exceeded", provider=self.name
                    )
                elif response.status_code >= 500:
                    raise ProviderConnectionError(
                        f"Anthropic server error: {response.status_code}", provider=self.name
                    )

                response.raise_for_status()
                data = response.json()

        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                f"Failed to connect to Anthropic: {e}", provider=self.name
            ) from e

        return self._parse_response(data)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming with Claude."""
        client = await self._get_client()

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
        }

        if system:
            payload["system"] = system

        if tools:
            payload["tools"] = self._format_tools(tools)

        async with client.stream("POST", "/v1/messages", json=payload) as response:
            response.raise_for_status()

            current_tool: dict[str, Any] | None = None

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                if event_type == "content_block_start":
                    block = data.get("content_block", {})
                    if block.get("type") == "tool_use":
                        current_tool = {
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "input": "",
                        }
                        yield StreamEvent(
                            type=StreamEventType.TOOL_USE_START,
                            tool_call=ToolCall(
                                id=current_tool["id"], name=current_tool["name"], input={}
                            ),
                        )

                elif event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        yield StreamEvent(type=StreamEventType.TEXT, content=delta.get("text", ""))
                    elif delta.get("type") == "input_json_delta" and current_tool:
                        current_tool["input"] += delta.get("partial_json", "")

                elif event_type == "content_block_stop":
                    if current_tool:
                        try:
                            input_dict = json.loads(current_tool["input"])
                        except json.JSONDecodeError:
                            input_dict = {}

                        yield StreamEvent(
                            type=StreamEventType.TOOL_USE_END,
                            tool_call=ToolCall(
                                id=current_tool["id"], name=current_tool["name"], input=input_dict
                            ),
                        )
                        current_tool = None

                elif event_type == "message_stop":
                    yield StreamEvent(type=StreamEventType.MESSAGE_END)

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for Anthropic API."""
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("input_schema", tool.get("parameters", {})),
            }
            for tool in tools
        ]

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse Anthropic response."""
        text_parts = []
        tool_calls = []

        for block in data.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        input=block.get("input", {}),
                    )
                )

        usage = data.get("usage", {})

        return LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            stop_reason=StopReason(data.get("stop_reason", "end_turn")),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
            ),
            raw_response=data,
        )


# ============================================================
# OPENAI PROVIDER
# ============================================================


class AsyncOpenAIProvider(AsyncLLMProvider):
    """Async OpenAI GPT provider."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        timeout: float = 300.0,
    ):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Async chat completion with GPT."""
        client = await self._get_client()

        formatted_messages = list(messages)
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        try:
            async with timeout_context(self.timeout, "OpenAI API"):
                response = await client.post("/v1/chat/completions", json=payload)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ProviderAuthenticationError(
                    "Invalid OpenAI API key", provider=self.name
                ) from e
            elif e.response.status_code == 429:
                raise ProviderRateLimitError(
                    "OpenAI rate limit exceeded", provider=self.name
                ) from e
            raise

        return self._parse_response(data)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming with GPT."""
        client = await self._get_client()

        formatted_messages = list(messages)
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": formatted_messages,
            "stream": True,
        }

        if tools:
            payload["tools"] = self._format_tools(tools)

        async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue

                if line == "data: [DONE]":
                    yield StreamEvent(type=StreamEventType.MESSAGE_END)
                    break

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                if "content" in delta and delta["content"]:
                    yield StreamEvent(type=StreamEventType.TEXT, content=delta["content"])

    def _format_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format tools for OpenAI API."""
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

    def _parse_response(self, data: dict[str, Any]) -> LLMResponse:
        """Parse OpenAI response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = []
        for tc in message.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", ""),
                    name=tc.get("function", {}).get("name", ""),
                    input=json.loads(tc.get("function", {}).get("arguments", "{}")),
                )
            )

        usage = data.get("usage", {})

        return LLMResponse(
            text=message.get("content", "") or "",
            tool_calls=tool_calls,
            stop_reason=StopReason.TOOL_USE if tool_calls else StopReason.END_TURN,
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
            ),
            raw_response=data,
        )


# ============================================================
# OLLAMA PROVIDER
# ============================================================


class AsyncOllamaProvider(AsyncLLMProvider):
    """Async Ollama local provider."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 600.0,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Async chat with local Ollama."""
        client = await self._get_client()

        formatted_messages = list(messages)
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except httpx.ConnectError as e:
            raise ProviderConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {e}", provider=self.name
            ) from e

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            tool_calls=[],
            stop_reason=StopReason.END_TURN,
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            ),
            raw_response=data,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> AsyncIterator[StreamEvent]:
        """Async streaming with Ollama."""
        client = await self._get_client()

        formatted_messages = list(messages)
        if system:
            formatted_messages.insert(0, {"role": "system", "content": system})

        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()

            async for line in response.aiter_lines():
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if data.get("done"):
                    yield StreamEvent(type=StreamEventType.MESSAGE_END)
                    break

                content = data.get("message", {}).get("content", "")
                if content:
                    yield StreamEvent(type=StreamEventType.TEXT, content=content)


# ============================================================
# FACTORY
# ============================================================


async def create_provider(name: str, settings: LLMSettings | None = None) -> AsyncLLMProvider:
    """
    Create an async LLM provider by name.

    Args:
        name: Provider name (anthropic, openai, ollama)
        settings: Optional LLM settings (uses global settings if not provided)

    Returns:
        Configured AsyncLLMProvider instance
    """
    if settings is None:
        settings = get_settings().llm

    name_lower = name.lower()

    if name_lower in ("anthropic", "claude"):
        if not settings.anthropic_api_key:
            raise ProviderError("ANTHROPIC_API_KEY not configured", provider=name)
        return AsyncAnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.anthropic_model,
            timeout=settings.request_timeout,
        )

    elif name_lower in ("openai", "gpt"):
        if not settings.openai_api_key:
            raise ProviderError("OPENAI_API_KEY not configured", provider=name)
        return AsyncOpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            timeout=settings.request_timeout,
        )

    elif name_lower == "ollama":
        return AsyncOllamaProvider(
            base_url=settings.ollama_url or "http://localhost:11434",
            model=settings.ollama_model,
            timeout=settings.request_timeout,
        )

    else:
        raise ProviderError(f"Unknown provider: {name}", provider=name)


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
    "AsyncLLMProvider",
    "AsyncAnthropicProvider",
    "AsyncOpenAIProvider",
    "AsyncOllamaProvider",
    # Factory
    "create_provider",
]
