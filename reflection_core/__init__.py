# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Familiar Core - Shared Security Primitives

This package contains security primitives shared between:
- Familiar (personal AI assistant)
- Reflection (enterprise multi-tenant platform)

Modules:
    security: Trust levels, capabilities, encryption, sanitization
    exceptions: Structured exception hierarchy
    types: Common type definitions
"""

__version__ = "1.0.0"

# Re-export security primitives
# Re-export exceptions
from .exceptions.hierarchy import (
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    FamiliarError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderContentFilterError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    QuotaExceededError,
    RateLimitError,
    SecurityError,
    TenantError,
    ToolError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionError,
    ToolTimeoutError,
)
from .security.encryption import (
    CRYPTO_AVAILABLE,
    DecryptionError,
    EncryptionError,
    Encryptor,
    derive_key_from_password,
    generate_key,
)
from .security.sanitization import (
    SanitizationLevel,
    check_path_safety,
    check_shell_safety,
    detect_prompt_injection,
    sanitize,
    sanitize_tool_output,
)
from .security.trust import (
    DEFAULT_CAPABILITIES,
    Capability,
    TrustLevel,
    check_capability,
    get_capabilities_for_trust,
)

__all__ = [
    # Version
    "__version__",
    # Trust
    "TrustLevel",
    "Capability",
    "DEFAULT_CAPABILITIES",
    "get_capabilities_for_trust",
    "check_capability",
    # Encryption
    "CRYPTO_AVAILABLE",
    "generate_key",
    "derive_key_from_password",
    "Encryptor",
    "EncryptionError",
    "DecryptionError",
    # Sanitization
    "SanitizationLevel",
    "sanitize",
    "check_shell_safety",
    "check_path_safety",
    "sanitize_tool_output",
    "detect_prompt_injection",
    # Exceptions
    "FamiliarError",
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "QuotaExceededError",
    "ProviderError",
    "ProviderConnectionError",
    "ProviderAuthenticationError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "ProviderContentFilterError",
    "ToolError",
    "ToolNotFoundError",
    "ToolExecutionError",
    "ToolTimeoutError",
    "ToolPermissionError",
    "TenantError",
    "ConfigurationError",
]
