# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Usage Calculator (v1.3.0)

Unified token counting and usage calculation that provides consistent
billing accuracy across all code paths (chat, stream, tools, etc.).

Problem Solved:
Previously, chat() used calculate_usage() with history, while stream()
used simple count_tokens() without history. This caused inconsistent
billing. UsageCalculator provides a single, accurate method.

Usage:
    calculator = UsageCalculator(model="claude-sonnet-4-20250514")

    # Calculate usage for a request
    usage = calculator.calculate(
        input_message="Hello",
        output_text="Hi there!",
        history=messages,
        system_prompt=system_prompt,
    )

    print(f"Input: {usage.input_tokens}, Output: {usage.output_tokens}")
    print(f"Cost: ${usage.cost_usd}")
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any
from uuid import uuid4

from .settings import DEFAULT_ANTHROPIC_MODEL
from .tokens import (
    TokenCounter,
    TokenUsage,
    extract_usage_from_response,
    get_model_pricing,
    get_token_counter,
)

logger = logging.getLogger(__name__)


# ============================================================
# USAGE RECORD
# ============================================================


@dataclass
class UsageRecord:
    """
    Complete usage record for billing and analytics.

    This is the canonical representation of usage for a single
    request/response cycle. It includes all token counts, costs,
    and metadata needed for accurate billing.
    """

    # Core counts
    input_tokens: int
    output_tokens: int

    # Optional detailed breakdown
    system_tokens: int = 0
    history_tokens: int = 0
    message_tokens: int = 0
    cached_tokens: int = 0

    # Model and pricing
    model: str = ""
    cost_usd: Decimal = Decimal("0")

    # Metadata
    request_id: str = field(default_factory=lambda: str(uuid4())[:8])
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Source of truth indicator
    source: str = "calculated"  # "provider", "calculated", "estimated"

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def billable_input_tokens(self) -> int:
        """Input tokens minus cached (for billing)."""
        return max(0, self.input_tokens - self.cached_tokens)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "system_tokens": self.system_tokens,
            "history_tokens": self.history_tokens,
            "message_tokens": self.message_tokens,
            "cached_tokens": self.cached_tokens,
            "billable_input_tokens": self.billable_input_tokens,
            "cost_usd": str(self.cost_usd),
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_token_usage(self) -> TokenUsage:
        """Convert to legacy TokenUsage for compatibility."""
        return TokenUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model=self.model,
            cached_input_tokens=self.cached_tokens,
            system_tokens=self.system_tokens,
            cost_usd=self.cost_usd,
            request_id=self.request_id,
        )


# ============================================================
# USAGE CALCULATOR
# ============================================================


class UsageCalculator:
    """
    Unified usage calculator for consistent billing.

    This class ensures token counting is done the same way everywhere:
    - chat() method
    - stream() method
    - tool execution
    - batch processing

    It supports multiple sources of token counts with a preference order:
    1. Provider-reported (most accurate when available)
    2. Tokenizer-calculated (accurate)
    3. Estimated (fallback)

    Usage:
        # Create calculator for a model
        calc = UsageCalculator(model="claude-sonnet-4-20250514")

        # Basic calculation
        usage = calc.calculate(
            input_message="Hello",
            output_text="Hi there!",
        )

        # With full context
        usage = calc.calculate(
            input_message="Hello",
            output_text="Hi there!",
            system_prompt="You are a helpful assistant.",
            history=[
                {"role": "user", "content": "Previous question"},
                {"role": "assistant", "content": "Previous answer"},
            ],
            provider_response=raw_api_response,
        )
    """

    def __init__(
        self,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        token_counter: TokenCounter | None = None,
    ):
        """
        Initialize calculator.

        Args:
            model: Model name for tokenization and pricing
            token_counter: Optional custom token counter
        """
        self.model = model
        self._counter = token_counter or get_token_counter()

    def calculate(
        self,
        input_message: str,
        output_text: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
        provider_response: Any = None,
        provider: str = "anthropic",
        request_id: str | None = None,
    ) -> UsageRecord:
        """
        Calculate usage for a request/response cycle.

        This is the primary method for calculating usage. It handles
        all the complexity of token counting and ensures consistency.

        Args:
            input_message: The user's input message
            output_text: The assistant's output text
            system_prompt: Optional system prompt (counted separately)
            history: Optional conversation history
            provider_response: Optional raw API response (for provider-reported tokens)
            provider: Provider name ("anthropic", "openai")
            request_id: Optional request ID for tracking

        Returns:
            UsageRecord with complete usage information
        """
        record = UsageRecord(
            input_tokens=0,
            output_tokens=0,
            model=self.model,
            request_id=request_id or str(uuid4())[:8],
        )

        # Try to extract from provider response first (most accurate)
        if provider_response is not None:
            provider_usage = extract_usage_from_response(provider_response, self.model, provider)
            if provider_usage.input_tokens > 0 or provider_usage.output_tokens > 0:
                record.input_tokens = provider_usage.input_tokens
                record.output_tokens = provider_usage.output_tokens
                record.cached_tokens = provider_usage.cached_input_tokens
                record.source = "provider"

                # Calculate cost
                self._calculate_cost(record)

                logger.debug(
                    f"Usage from provider: in={record.input_tokens}, "
                    f"out={record.output_tokens}, cached={record.cached_tokens}"
                )

                return record

        # Fall back to tokenizer calculation
        record.source = "calculated"

        # Count system prompt
        if system_prompt:
            system_count = self._counter.count_tokens(system_prompt, self.model)
            record.system_tokens = system_count.count

        # Count history
        if history:
            history_count = self._counter.count_messages(history, self.model)
            record.history_tokens = history_count.count

        # Count input message
        message_count = self._counter.count_tokens(input_message, self.model)
        record.message_tokens = message_count.count

        # Total input = system + history + message + overhead
        overhead = self._estimate_overhead(history)
        record.input_tokens = (
            record.system_tokens + record.history_tokens + record.message_tokens + overhead
        )

        # Count output
        output_count = self._counter.count_tokens(output_text, self.model)
        record.output_tokens = output_count.count

        # Calculate cost
        self._calculate_cost(record)

        logger.debug(
            f"Usage calculated: in={record.input_tokens} "
            f"(sys={record.system_tokens}, hist={record.history_tokens}, "
            f"msg={record.message_tokens}), out={record.output_tokens}"
        )

        return record

    def calculate_from_messages(
        self,
        messages: list[dict[str, str]],
        output_text: str,
        system_prompt: str | None = None,
        provider_response: Any = None,
        provider: str = "anthropic",
        request_id: str | None = None,
    ) -> UsageRecord:
        """
        Calculate usage from a full message list.

        Use this when you have the complete message array including
        the latest user message (as used in the API call).

        Args:
            messages: Full message list sent to API
            output_text: The assistant's response
            system_prompt: System prompt (if not in messages)
            provider_response: Optional raw API response
            provider: Provider name
            request_id: Optional request ID

        Returns:
            UsageRecord with usage information
        """
        # Extract the last user message
        input_message = ""
        history = []

        for i, msg in enumerate(messages):
            if i == len(messages) - 1 and msg.get("role") == "user":
                input_message = msg.get("content", "")
            else:
                history.append(msg)

        return self.calculate(
            input_message=input_message,
            output_text=output_text,
            system_prompt=system_prompt,
            history=history if history else None,
            provider_response=provider_response,
            provider=provider,
            request_id=request_id,
        )

    def estimate_input(
        self,
        message: str,
        system_prompt: str | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> int:
        """
        Estimate input tokens before making an API call.

        Useful for pre-flight checks and quota estimation.

        Args:
            message: User message
            system_prompt: System prompt
            history: Conversation history

        Returns:
            Estimated input token count
        """
        total = 0

        if system_prompt:
            total += self._counter.count_tokens(system_prompt, self.model).count

        if history:
            total += self._counter.count_messages(history, self.model).count

        total += self._counter.count_tokens(message, self.model).count
        total += self._estimate_overhead(history)

        return total

    def _estimate_overhead(self, history: list[dict[str, str]] | None) -> int:
        """
        Estimate token overhead from message formatting.

        Different providers have different overhead per message
        for role markers, separators, etc.
        """
        base_overhead = 10  # Base overhead for request

        if not history:
            return base_overhead

        # Add ~4 tokens per message for formatting
        message_overhead = len(history) * 4

        return base_overhead + message_overhead

    def _calculate_cost(self, record: UsageRecord) -> None:
        """Calculate and set the cost on a usage record."""
        pricing = get_model_pricing(self.model)

        if pricing is None:
            record.cost_usd = Decimal("0")
            return

        record.cost_usd = pricing.calculate_cost(
            input_tokens=record.billable_input_tokens,
            output_tokens=record.output_tokens,
            cached_input_tokens=record.cached_tokens,
        )


# ============================================================
# USAGE AGGREGATOR
# ============================================================


@dataclass
class AggregatedUsage:
    """Aggregated usage over multiple requests."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_cost_usd: Decimal = Decimal("0")

    # Per-model breakdown
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    def add(self, record: UsageRecord) -> None:
        """Add a usage record to the aggregate."""
        self.total_requests += 1
        self.total_input_tokens += record.input_tokens
        self.total_output_tokens += record.output_tokens
        self.total_cached_tokens += record.cached_tokens
        self.total_cost_usd += record.cost_usd

        # Update per-model breakdown
        if record.model not in self.by_model:
            self.by_model[record.model] = {
                "requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": Decimal("0"),
            }

        self.by_model[record.model]["requests"] += 1
        self.by_model[record.model]["input_tokens"] += record.input_tokens
        self.by_model[record.model]["output_tokens"] += record.output_tokens
        self.by_model[record.model]["cost_usd"] += record.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/reporting."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "total_cost_usd": str(self.total_cost_usd),
            "by_model": {
                model: {
                    **data,
                    "cost_usd": str(data["cost_usd"]),
                }
                for model, data in self.by_model.items()
            },
        }


# ============================================================
# GLOBAL CALCULATOR
# ============================================================

_default_calculator: UsageCalculator | None = None


def get_usage_calculator(model: str = DEFAULT_ANTHROPIC_MODEL) -> UsageCalculator:
    """
    Get a usage calculator for a model.

    Caches the default calculator for the most common model.
    """
    global _default_calculator

    if model == DEFAULT_ANTHROPIC_MODEL:
        if _default_calculator is None:
            _default_calculator = UsageCalculator(model=model)
        return _default_calculator

    return UsageCalculator(model=model)


def reset_usage_calculator():
    """Reset the cached calculator (for testing)."""
    global _default_calculator
    _default_calculator = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "UsageCalculator",
    "UsageRecord",
    "AggregatedUsage",
    "get_usage_calculator",
    "reset_usage_calculator",
]
