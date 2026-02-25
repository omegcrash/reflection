# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Token Counting and Cost Tracking Module

Provides:
- Accurate token counting using tiktoken (OpenAI) and anthropic tokenizers
- Model pricing database with current rates
- Cost calculation for billing
- Usage extraction from provider responses

This replaces the inaccurate `len(text) // 4` estimation with proper tokenization.
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# TOKEN COUNTING
# ============================================================


class TokenizerType(StrEnum):
    """Tokenizer types for different model families."""

    TIKTOKEN_CL100K = "cl100k_base"  # GPT-4, GPT-3.5-turbo
    TIKTOKEN_O200K = "o200k_base"  # GPT-4o
    ANTHROPIC = "anthropic"  # Claude models
    APPROXIMATE = "approximate"  # Fallback ~4 chars per token


@dataclass
class TokenCount:
    """Result of token counting."""

    count: int
    tokenizer: TokenizerType
    cached: bool = False

    def __add__(self, other: "TokenCount") -> "TokenCount":
        return TokenCount(
            count=self.count + other.count,
            tokenizer=self.tokenizer,
            cached=False,
        )


class TokenCounter:
    """
    Accurate token counter supporting multiple model families.

    Uses:
    - tiktoken for OpenAI models (exact)
    - anthropic tokenizer for Claude models (exact when available)
    - Approximation as fallback

    Usage:
        counter = TokenCounter()

        # Count tokens for a model
        count = counter.count_tokens("Hello, world!", model="gpt-4o")
        print(f"Tokens: {count.count}")

        # Count message list (chat format)
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        count = counter.count_messages(messages, model="claude-sonnet-4-20250514")
    """

    # Model family to tokenizer mapping
    MODEL_TOKENIZERS: dict[str, TokenizerType] = {
        # OpenAI GPT-4o family
        "gpt-4o": TokenizerType.TIKTOKEN_O200K,
        "gpt-4o-mini": TokenizerType.TIKTOKEN_O200K,
        "gpt-4o-2024": TokenizerType.TIKTOKEN_O200K,
        # OpenAI GPT-4 family
        "gpt-4": TokenizerType.TIKTOKEN_CL100K,
        "gpt-4-turbo": TokenizerType.TIKTOKEN_CL100K,
        "gpt-4-32k": TokenizerType.TIKTOKEN_CL100K,
        # OpenAI GPT-3.5 family
        "gpt-3.5-turbo": TokenizerType.TIKTOKEN_CL100K,
        "gpt-3.5-turbo-16k": TokenizerType.TIKTOKEN_CL100K,
        # Anthropic Claude family
        "claude-3-opus": TokenizerType.ANTHROPIC,
        "claude-3-sonnet": TokenizerType.ANTHROPIC,
        "claude-3-haiku": TokenizerType.ANTHROPIC,
        "claude-3.5-sonnet": TokenizerType.ANTHROPIC,
        "claude-3.5-haiku": TokenizerType.ANTHROPIC,
        "claude-sonnet-4": TokenizerType.ANTHROPIC,
        "claude-opus-4": TokenizerType.ANTHROPIC,
        # Full model IDs
        "claude-3-opus-20240229": TokenizerType.ANTHROPIC,
        "claude-3-sonnet-20240229": TokenizerType.ANTHROPIC,
        "claude-3-haiku-20240307": TokenizerType.ANTHROPIC,
        "claude-3-5-sonnet-20241022": TokenizerType.ANTHROPIC,
        "claude-3-5-haiku-20241022": TokenizerType.ANTHROPIC,
        "claude-sonnet-4-20250514": TokenizerType.ANTHROPIC,
        "claude-opus-4-20250514": TokenizerType.ANTHROPIC,
    }

    def __init__(self):
        self._tiktoken_encoders: dict[str, Any] = {}
        self._anthropic_tokenizer = None
        self._tiktoken_available = False
        self._anthropic_available = False

        # Try to import tiktoken
        try:
            import tiktoken

            self._tiktoken = tiktoken
            self._tiktoken_available = True
            logger.debug("tiktoken available for OpenAI token counting")
        except ImportError:
            logger.warning(
                "tiktoken not installed. OpenAI token counting will use approximation. "
                "Install with: pip install tiktoken"
            )

        # Try to import anthropic tokenizer
        try:
            from anthropic import Anthropic

            # Anthropic client has a count_tokens method
            self._anthropic_client = Anthropic()
            self._anthropic_available = True
            logger.debug("anthropic tokenizer available for Claude token counting")
        except ImportError:
            logger.warning(
                "anthropic not installed. Claude token counting will use approximation. "
                "Install with: pip install anthropic"
            )
        except Exception as e:
            logger.warning(f"anthropic tokenizer initialization failed: {e}")

    def _get_tokenizer_type(self, model: str) -> TokenizerType:
        """Get the tokenizer type for a model."""
        # Direct match
        if model in self.MODEL_TOKENIZERS:
            return self.MODEL_TOKENIZERS[model]

        # Partial match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
        model_lower = model.lower()
        for prefix, tokenizer in self.MODEL_TOKENIZERS.items():
            if model_lower.startswith(prefix.lower()):
                return tokenizer

        # Detect by provider prefix
        if "claude" in model_lower:
            return TokenizerType.ANTHROPIC
        if "gpt" in model_lower:
            return TokenizerType.TIKTOKEN_CL100K

        # Fallback to approximation
        return TokenizerType.APPROXIMATE

    @lru_cache(maxsize=4)  # noqa: B019 — intentional: few encoders, instance is long-lived
    def _get_tiktoken_encoder(self, encoding_name: str):
        """Get or create a tiktoken encoder."""
        if not self._tiktoken_available:
            return None
        return self._tiktoken.get_encoding(encoding_name)

    def _count_tiktoken(self, text: str, tokenizer_type: TokenizerType) -> int:
        """Count tokens using tiktoken."""
        encoding_name = tokenizer_type.value
        encoder = self._get_tiktoken_encoder(encoding_name)
        if encoder is None:
            return self._count_approximate(text)
        return len(encoder.encode(text))

    def _count_anthropic(self, text: str) -> int:
        """Count tokens using Anthropic's tokenizer."""
        if not self._anthropic_available:
            return self._count_approximate(text)

        try:
            # Use Anthropic's token counting
            # Note: This requires API key but doesn't make API calls
            result = self._anthropic_client.count_tokens(text)
            return result
        except Exception as e:
            logger.debug(f"Anthropic tokenizer failed, using approximation: {e}")
            return self._count_approximate(text)

    def _count_approximate(self, text: str) -> int:
        """
        Approximate token count.

        Uses ~4 characters per token as a rough estimate.
        This is less accurate but works without dependencies.
        """
        # More sophisticated approximation:
        # - English text: ~4 chars/token
        # - Code: ~3.5 chars/token
        # - Mixed: ~3.8 chars/token
        # We use 3.8 to slightly overestimate (safer for billing)
        return max(1, int(len(text) / 3.8))

    def count_tokens(self, text: str, model: str = "gpt-4o") -> TokenCount:
        """
        Count tokens in a text string.

        Args:
            text: The text to tokenize
            model: Model name to determine tokenizer

        Returns:
            TokenCount with count and tokenizer info
        """
        if not text:
            return TokenCount(count=0, tokenizer=TokenizerType.APPROXIMATE)

        tokenizer_type = self._get_tokenizer_type(model)

        if tokenizer_type == TokenizerType.APPROXIMATE:
            count = self._count_approximate(text)
        elif tokenizer_type == TokenizerType.ANTHROPIC:
            count = self._count_anthropic(text)
        else:
            # tiktoken
            count = self._count_tiktoken(text, tokenizer_type)

        return TokenCount(count=count, tokenizer=tokenizer_type)

    def count_messages(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o",
    ) -> TokenCount:
        """
        Count tokens in a list of chat messages.

        Accounts for message formatting overhead.

        Args:
            messages: List of {"role": str, "content": str} dicts
            model: Model name

        Returns:
            TokenCount for the entire conversation
        """
        tokenizer_type = self._get_tokenizer_type(model)
        total = 0

        # Message format overhead varies by model
        # OpenAI: ~4 tokens per message for formatting
        # Claude: ~3 tokens per message
        overhead_per_message = 3 if "claude" in model.lower() else 4

        for message in messages:
            content = message.get("content", "")
            role = message.get("role", "")

            # Count content tokens
            content_count = self.count_tokens(content, model)
            total += content_count.count

            # Add role name tokens (usually 1-2)
            total += len(role) // 4 + 1

            # Add message formatting overhead
            total += overhead_per_message

        # Add conversation start/end overhead
        total += 3

        return TokenCount(count=total, tokenizer=tokenizer_type)


# ============================================================
# MODEL PRICING
# ============================================================


@dataclass
class ModelPricing:
    """
    Pricing for a model.

    Prices are in USD per 1M tokens.
    """

    input_price_per_million: Decimal
    output_price_per_million: Decimal

    # Optional cached pricing (may differ)
    cached_input_price_per_million: Decimal | None = None

    @property
    def input_price_per_token(self) -> Decimal:
        return self.input_price_per_million / Decimal(1_000_000)

    @property
    def output_price_per_token(self) -> Decimal:
        return self.output_price_per_million / Decimal(1_000_000)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_input_tokens: int = 0,
    ) -> Decimal:
        """Calculate total cost for token usage."""
        input_cost = Decimal(input_tokens) * self.input_price_per_token
        output_cost = Decimal(output_tokens) * self.output_price_per_token

        # Cached tokens (if applicable)
        cached_cost = Decimal(0)
        if cached_input_tokens > 0 and self.cached_input_price_per_million:
            cached_price = self.cached_input_price_per_million / Decimal(1_000_000)
            cached_cost = Decimal(cached_input_tokens) * cached_price

        return input_cost + output_cost + cached_cost


# Current pricing as of early 2025 (USD per 1M tokens)
MODEL_PRICING: dict[str, ModelPricing] = {
    # Anthropic Claude 4 family
    "claude-opus-4-20250514": ModelPricing(
        input_price_per_million=Decimal("15.00"),
        output_price_per_million=Decimal("75.00"),
        cached_input_price_per_million=Decimal("1.50"),
    ),
    "claude-sonnet-4-20250514": ModelPricing(
        input_price_per_million=Decimal("3.00"),
        output_price_per_million=Decimal("15.00"),
        cached_input_price_per_million=Decimal("0.30"),
    ),
    # Anthropic Claude 3.5 family
    "claude-3-5-sonnet-20241022": ModelPricing(
        input_price_per_million=Decimal("3.00"),
        output_price_per_million=Decimal("15.00"),
        cached_input_price_per_million=Decimal("0.30"),
    ),
    "claude-3-5-haiku-20241022": ModelPricing(
        input_price_per_million=Decimal("0.80"),
        output_price_per_million=Decimal("4.00"),
        cached_input_price_per_million=Decimal("0.08"),
    ),
    # Anthropic Claude 3 family
    "claude-3-opus-20240229": ModelPricing(
        input_price_per_million=Decimal("15.00"),
        output_price_per_million=Decimal("75.00"),
    ),
    "claude-3-sonnet-20240229": ModelPricing(
        input_price_per_million=Decimal("3.00"),
        output_price_per_million=Decimal("15.00"),
    ),
    "claude-3-haiku-20240307": ModelPricing(
        input_price_per_million=Decimal("0.25"),
        output_price_per_million=Decimal("1.25"),
    ),
    # OpenAI GPT-4o family
    "gpt-4o": ModelPricing(
        input_price_per_million=Decimal("2.50"),
        output_price_per_million=Decimal("10.00"),
        cached_input_price_per_million=Decimal("1.25"),
    ),
    "gpt-4o-mini": ModelPricing(
        input_price_per_million=Decimal("0.15"),
        output_price_per_million=Decimal("0.60"),
        cached_input_price_per_million=Decimal("0.075"),
    ),
    # OpenAI GPT-4 family
    "gpt-4-turbo": ModelPricing(
        input_price_per_million=Decimal("10.00"),
        output_price_per_million=Decimal("30.00"),
    ),
    "gpt-4": ModelPricing(
        input_price_per_million=Decimal("30.00"),
        output_price_per_million=Decimal("60.00"),
    ),
    # OpenAI GPT-3.5 family
    "gpt-3.5-turbo": ModelPricing(
        input_price_per_million=Decimal("0.50"),
        output_price_per_million=Decimal("1.50"),
    ),
}


def get_model_pricing(model: str) -> ModelPricing | None:
    """
    Get pricing for a model.

    Handles both exact matches and prefix matches.
    """
    # Direct match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Prefix match (e.g., "gpt-4o-2024-08-06" matches "gpt-4o")
    model_lower = model.lower()
    for prefix, pricing in MODEL_PRICING.items():
        if model_lower.startswith(prefix.lower().split("-202")[0]):
            return pricing

    logger.warning(f"No pricing found for model: {model}")
    return None


# ============================================================
# USAGE TRACKING
# ============================================================


@dataclass
class TokenUsage:
    """
    Complete token usage record.

    Includes counts, cost, and metadata for billing.
    """

    input_tokens: int
    output_tokens: int
    model: str

    # Optional detailed breakdown
    cached_input_tokens: int = 0
    system_tokens: int = 0

    # Cost (calculated)
    cost_usd: Decimal | None = None

    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    request_id: str | None = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def calculate_cost(self) -> Decimal:
        """Calculate and cache the cost."""
        if self.cost_usd is not None:
            return self.cost_usd

        pricing = get_model_pricing(self.model)
        if pricing is None:
            self.cost_usd = Decimal(0)
            return self.cost_usd

        self.cost_usd = pricing.calculate_cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
        )
        return self.cost_usd

    def to_dict(self) -> dict[str, Any]:
        """Serialize for storage/logging."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "model": self.model,
            "cost_usd": str(self.calculate_cost()),
            "timestamp": self.timestamp.isoformat(),
            "request_id": self.request_id,
        }


def extract_usage_from_response(
    response: Any,
    model: str,
    provider: str = "anthropic",
) -> TokenUsage:
    """
    Extract token usage from a provider response.

    Handles both Anthropic and OpenAI response formats.

    Args:
        response: Raw response from provider
        model: Model name
        provider: Provider name ("anthropic" or "openai")

    Returns:
        TokenUsage with extracted counts
    """
    input_tokens = 0
    output_tokens = 0
    cached_tokens = 0

    if provider == "anthropic":
        # Anthropic response.usage structure
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
            cached_tokens = getattr(usage, "cache_read_input_tokens", 0)
        elif isinstance(response, dict):
            usage = response.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("cache_read_input_tokens", 0)

    elif provider == "openai":
        # OpenAI response.usage structure
        usage = getattr(response, "usage", None)
        if usage:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            # OpenAI uses different field for cached
            cached_tokens = getattr(usage, "prompt_tokens_details", {}).get("cached_tokens", 0)
        elif isinstance(response, dict):
            usage = response.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_tokens,
        model=model,
    )


def calculate_usage(
    input_text: str,
    output_text: str,
    model: str,
    messages: list[dict[str, str]] | None = None,
    response: Any = None,
    provider: str = "anthropic",
) -> TokenUsage:
    """
    Calculate token usage with the most accurate method available.

    Priority:
    1. Extract from provider response (most accurate)
    2. Count using appropriate tokenizer
    3. Approximate as fallback

    Args:
        input_text: The input/prompt text
        output_text: The output/completion text
        model: Model name
        messages: Optional message history for chat format
        response: Optional raw provider response
        provider: Provider name

    Returns:
        TokenUsage with counts and cost
    """
    # Try to extract from response first (most accurate)
    if response is not None:
        usage = extract_usage_from_response(response, model, provider)
        if usage.input_tokens > 0 or usage.output_tokens > 0:
            usage.calculate_cost()
            return usage

    # Fall back to tokenizer counting
    counter = get_token_counter()

    # Count input tokens
    if messages:
        input_count = counter.count_messages(messages, model)
    else:
        input_count = counter.count_tokens(input_text, model)

    # Count output tokens
    output_count = counter.count_tokens(output_text, model)

    usage = TokenUsage(
        input_tokens=input_count.count,
        output_tokens=output_count.count,
        model=model,
    )
    usage.calculate_cost()

    return usage


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_token_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Get or create the global token counter."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter()
    return _token_counter


def reset_token_counter():
    """Reset the global token counter (for testing)."""
    global _token_counter
    _token_counter = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Token counting
    "TokenCounter",
    "TokenCount",
    "TokenizerType",
    "get_token_counter",
    # Pricing
    "ModelPricing",
    "MODEL_PRICING",
    "get_model_pricing",
    # Usage
    "TokenUsage",
    "extract_usage_from_response",
    "calculate_usage",
]
