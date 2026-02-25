# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Exception Hierarchy

Structured exceptions for both Familiar and Reflection.
All exceptions include context via `details` dict.
"""

from typing import Any


class FamiliarError(Exception):
    """
    Base exception for all Familiar errors.

    Attributes:
        message: Human-readable error message
        details: Additional context as key-value pairs
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "error": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ============================================================
# SECURITY ERRORS
# ============================================================


class SecurityError(FamiliarError):
    """Base class for security-related errors."""

    pass


class AuthenticationError(SecurityError):
    """Failed to authenticate user or API key."""

    pass


class AuthorizationError(SecurityError):
    """User lacks required permissions."""

    def __init__(
        self,
        message: str,
        required_capability: str | None = None,
        user_trust_level: str | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if required_capability:
            details["required_capability"] = required_capability
        if user_trust_level:
            details["user_trust_level"] = user_trust_level
        super().__init__(message, details)


class RateLimitError(SecurityError):
    """Rate limit exceeded."""

    def __init__(
        self,
        message: str,
        limit_type: str | None = None,
        current_value: int | None = None,
        limit_value: int | None = None,
        retry_after: int | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if limit_type:
            details["limit_type"] = limit_type
        if current_value is not None:
            details["current_value"] = current_value
        if limit_value is not None:
            details["limit_value"] = limit_value
        if retry_after is not None:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, details)


class QuotaExceededError(SecurityError):
    """Resource quota exceeded."""

    def __init__(
        self,
        message: str,
        quota_type: str | None = None,
        current: int | None = None,
        limit: int | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if quota_type:
            details["quota_type"] = quota_type
        if current is not None:
            details["current"] = current
        if limit is not None:
            details["limit"] = limit
        super().__init__(message, details)


class InputValidationError(SecurityError):
    """Input failed validation."""

    def __init__(
        self,
        message: str,
        input_type: str | None = None,
        matched_pattern: str | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if input_type:
            details["input_type"] = input_type
        if matched_pattern:
            details["matched_pattern"] = matched_pattern
        super().__init__(message, details)


# ============================================================
# PROVIDER ERRORS
# ============================================================


class ProviderError(FamiliarError):
    """Base class for LLM provider errors."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        original_error: Exception | None = None,
        **kwargs,
    ):
        details = kwargs.get("details", {})
        if provider:
            details["provider"] = provider
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(message, details)


class ProviderConnectionError(ProviderError):
    """Failed to connect to provider."""

    pass


class ProviderAuthenticationError(ProviderError):
    """Provider authentication failed (invalid API key)."""

    pass


class ProviderRateLimitError(ProviderError):
    """Provider rate limit exceeded."""

    def __init__(self, message: str, retry_after: int | None = None, **kwargs):
        super().__init__(message, **kwargs)
        if retry_after is not None:
            self.details["retry_after_seconds"] = retry_after


class ProviderResponseError(ProviderError):
    """Invalid or unexpected response from provider."""

    pass


class ProviderContentFilterError(ProviderError):
    """Content blocked by provider's safety filters."""

    pass


# ============================================================
# TOOL ERRORS
# ============================================================


class ToolError(FamiliarError):
    """Base class for tool execution errors."""

    pass


class ToolNotFoundError(ToolError):
    """Requested tool does not exist."""

    def __init__(self, tool_name: str, **kwargs):
        super().__init__(
            f"Tool not found: {tool_name}",
            details={"tool_name": tool_name, **kwargs.get("details", {})},
        )


class ToolExecutionError(ToolError):
    """Tool execution failed."""

    def __init__(self, message: str, tool_name: str | None = None, **kwargs):
        details = kwargs.get("details", {})
        if tool_name:
            details["tool_name"] = tool_name
        super().__init__(message, details)


class ToolTimeoutError(ToolError):
    """Tool execution timed out."""

    def __init__(self, tool_name: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Tool {tool_name} timed out after {timeout_seconds}s",
            details={
                "tool_name": tool_name,
                "timeout_seconds": timeout_seconds,
                **kwargs.get("details", {}),
            },
        )


class ToolPermissionError(ToolError):
    """User lacks permission to execute tool."""

    def __init__(self, tool_name: str, required_capability: str | None = None, **kwargs):
        details = {"tool_name": tool_name, **kwargs.get("details", {})}
        if required_capability:
            details["required_capability"] = required_capability
        super().__init__(f"Permission denied for tool: {tool_name}", details)


# ============================================================
# TENANT ERRORS (Reflection specific)
# ============================================================


class TenantError(FamiliarError):
    """Base class for tenant-related errors."""

    pass


class TenantNotFoundError(TenantError):
    """Tenant does not exist."""

    def __init__(self, tenant_id: str, **kwargs):
        super().__init__(
            f"Tenant not found: {tenant_id}",
            details={"tenant_id": tenant_id, **kwargs.get("details", {})},
        )


class TenantSuspendedError(TenantError):
    """Tenant is suspended."""

    def __init__(self, tenant_id: str, reason: str | None = None, **kwargs):
        details = {"tenant_id": tenant_id, **kwargs.get("details", {})}
        if reason:
            details["suspension_reason"] = reason
        super().__init__(f"Tenant suspended: {tenant_id}", details)


class TenantQuotaError(TenantError, QuotaExceededError):
    """Tenant-specific quota exceeded."""

    pass


# ============================================================
# CONFIGURATION ERRORS
# ============================================================


class ConfigurationError(FamiliarError):
    """Configuration is invalid or missing."""

    pass


class SkillConfigError(ConfigurationError):
    """Skill configuration error."""

    def __init__(self, skill_name: str, message: str, **kwargs):
        super().__init__(
            f"Skill '{skill_name}' configuration error: {message}",
            details={"skill_name": skill_name, **kwargs.get("details", {})},
        )


class ProviderConfigError(ConfigurationError):
    """Provider configuration error."""

    def __init__(self, provider: str, message: str, **kwargs):
        super().__init__(
            f"Provider '{provider}' configuration error: {message}",
            details={"provider": provider, **kwargs.get("details", {})},
        )


# ============================================================
# SESSION ERRORS
# ============================================================


class SessionError(FamiliarError):
    """Base class for session errors."""

    pass


class SessionExpiredError(SessionError):
    """Session has expired."""

    pass


class SessionValidationError(SessionError):
    """Session validation failed."""

    pass


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Base
    "FamiliarError",
    # Security
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "QuotaExceededError",
    "InputValidationError",
    # Provider
    "ProviderError",
    "ProviderConnectionError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderContentFilterError",
    # Tool
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolPermissionError",
    # Tenant
    "TenantError",
    "TenantNotFoundError",
    "TenantSuspendedError",
    "TenantQuotaError",
    # Configuration
    "ConfigurationError",
    "SkillConfigError",
    "ProviderConfigError",
    # Session
    "SessionError",
    "SessionExpiredError",
    "SessionValidationError",
]
