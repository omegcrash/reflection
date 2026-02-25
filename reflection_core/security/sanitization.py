# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Input Sanitization

Multi-level input sanitization to prevent:
- Shell injection
- Path traversal
- Prompt injection
- ReDoS attacks

Levels:
    NONE: No sanitization (dangerous, for trusted input only)
    BASIC: Remove null bytes, control characters
    STRICT: Basic + length limits, character restrictions
    PARANOID: Strict + aggressive filtering
"""

import html
import logging
import re
from enum import StrEnum
from pathlib import Path

logger = logging.getLogger(__name__)


class SanitizationLevel(StrEnum):
    """Sanitization strictness levels."""

    NONE = "none"
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


# Dangerous shell patterns (ReDoS-safe)
DANGEROUS_SHELL_PATTERNS: list[tuple[str, str]] = [
    (r"\$\(", "Command substitution $()"),
    (r"`[^`]*`", "Backtick command substitution"),
    (r"\|", "Pipe operator"),
    (r";", "Command separator"),
    (r"&&", "AND operator"),
    (r"\|\|", "OR operator"),
    (r">", "Output redirection"),
    (r"<", "Input redirection"),
    (r"\bsudo\b", "Sudo command"),
    (r"\brm\s+-rf", "Recursive force delete"),
    (r"\bdd\b", "dd command"),
    (r"\bmkfs\b", "Filesystem creation"),
    (r"\b:\(\)\s*\{", "Fork bomb pattern"),
    (r"\\x[0-9a-fA-F]{2}", "Hex escape"),
    (r"curl\s+.*\|\s*(ba)?sh", "Curl pipe to shell"),
    (r"wget\s+.*\|\s*(ba)?sh", "Wget pipe to shell"),
    (r"\beval\b", "Eval command"),
    (r"\bexec\b", "Exec command"),
]

# Path traversal patterns
PATH_TRAVERSAL_PATTERNS: list[str] = [
    r"\.\.",  # Parent directory
    r"\.\./",  # Parent with slash
    r"/\.\./",  # Slash-enclosed parent
    r"%2e%2e",  # URL-encoded ..
    r"%252e%252e",  # Double URL-encoded ..
    r"\.\.\\",  # Windows-style parent
]

# Default length limits
DEFAULT_MAX_INPUT_LENGTH = 100_000  # 100KB
DEFAULT_MAX_TOOL_OUTPUT_LENGTH = 500_000  # 500KB
DEFAULT_MAX_PATH_LENGTH = 4096


def sanitize_basic(text: str) -> str:
    """
    Basic sanitization: remove dangerous characters.

    Removes:
    - Null bytes
    - Control characters (except newline, tab)
    """
    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove control characters except \n and \t
    text = "".join(
        char for char in text if char in "\n\t" or (ord(char) >= 32 and ord(char) != 127)
    )

    return text


def sanitize_strict(text: str, max_length: int = DEFAULT_MAX_INPUT_LENGTH) -> str:
    """
    Strict sanitization: basic + length limits.
    """
    text = sanitize_basic(text)

    # Enforce length limit
    if len(text) > max_length:
        logger.warning(f"Input truncated from {len(text)} to {max_length}")
        text = text[:max_length]

    return text


def sanitize_paranoid(text: str, max_length: int = DEFAULT_MAX_INPUT_LENGTH) -> str:
    """
    Paranoid sanitization: strict + aggressive filtering.

    Use for untrusted input that will be used in sensitive contexts.
    """
    text = sanitize_strict(text, max_length)

    # HTML escape to prevent injection in web contexts
    text = html.escape(text)

    # Remove any remaining suspicious patterns
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"javascript:", "", text, flags=re.IGNORECASE)

    return text


def sanitize(
    text: str, level: SanitizationLevel = SanitizationLevel.STRICT, max_length: int | None = None
) -> str:
    """
    Sanitize input text at the specified level.

    Args:
        text: Input text to sanitize
        level: Sanitization level
        max_length: Optional length limit override

    Returns:
        Sanitized text
    """
    if level == SanitizationLevel.NONE:
        return text
    elif level == SanitizationLevel.BASIC:
        return sanitize_basic(text)
    elif level == SanitizationLevel.STRICT:
        return sanitize_strict(text, max_length or DEFAULT_MAX_INPUT_LENGTH)
    elif level == SanitizationLevel.PARANOID:
        return sanitize_paranoid(text, max_length or DEFAULT_MAX_INPUT_LENGTH)
    else:
        return sanitize_strict(text)


def check_shell_safety(command: str) -> tuple[bool, str | None]:
    """
    Check if a shell command appears safe to execute.

    Args:
        command: Shell command to check

    Returns:
        (is_safe, error_message) tuple
    """
    # Check for dangerous patterns
    for pattern, description in DANGEROUS_SHELL_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Blocked: {description}"

    return True, None


def check_path_safety(path: str, allowed_roots: list[str] | None = None) -> tuple[bool, str | None]:
    """
    Check if a file path is safe.

    Args:
        path: File path to check
        allowed_roots: Optional list of allowed root directories

    Returns:
        (is_safe, error_message) tuple
    """
    # Check for traversal patterns
    for pattern in PATH_TRAVERSAL_PATTERNS:
        if re.search(pattern, path, re.IGNORECASE):
            return False, "Path traversal detected"

    # Length check
    if len(path) > DEFAULT_MAX_PATH_LENGTH:
        return False, f"Path too long (max {DEFAULT_MAX_PATH_LENGTH})"

    # Resolve and check against allowed roots
    if allowed_roots:
        try:
            resolved = Path(path).resolve()
            if not any(
                str(resolved).startswith(str(Path(root).resolve())) for root in allowed_roots
            ):
                return False, "Path outside allowed directories"
        except (ValueError, OSError) as e:
            return False, f"Invalid path: {e}"

    return True, None


def sanitize_tool_output(output: str, max_length: int = DEFAULT_MAX_TOOL_OUTPUT_LENGTH) -> str:
    """
    Sanitize tool output before returning to LLM.

    Prevents prompt injection from tool results.
    """
    # Basic sanitization
    output = sanitize_basic(output)

    # Truncate if too long
    if len(output) > max_length:
        truncated = output[:max_length]
        output = f"{truncated}\n\n[OUTPUT TRUNCATED - {len(output) - max_length} bytes omitted]"

    # Wrap in markers to help detect injection attempts
    # The LLM should be instructed to ignore content outside these markers
    output = f"<tool_output>\n{output}\n</tool_output>"

    return output


def detect_prompt_injection(text: str) -> tuple[bool, str | None]:
    """
    Detect potential prompt injection attempts.

    Args:
        text: Input text to check

    Returns:
        (is_suspicious, reason) tuple
    """
    suspicious_patterns = [
        (r"ignore\s+(all\s+)?(previous|prior|above)", "Instruction override attempt"),
        (r"disregard\s+(all\s+)?(previous|prior|above)", "Instruction override attempt"),
        (r"forget\s+(all\s+)?(previous|prior|above)", "Instruction override attempt"),
        (r"you\s+are\s+now\s+", "Role reassignment attempt"),
        (r"new\s+instructions?:", "New instruction injection"),
        (r"system\s*:\s*", "System prompt injection"),
        (r"\[INST\]|\[/INST\]", "Instruction tag injection"),
        (r"<\|im_start\|>|<\|im_end\|>", "Chat template injection"),
    ]

    for pattern, reason in suspicious_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True, reason

    return False, None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "SanitizationLevel",
    "sanitize",
    "sanitize_basic",
    "sanitize_strict",
    "sanitize_paranoid",
    "check_shell_safety",
    "check_path_safety",
    "sanitize_tool_output",
    "detect_prompt_injection",
    "DANGEROUS_SHELL_PATTERNS",
]
