# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Phase 7: Observability - Structured Logging

JSON-structured logging for:
- Request tracing
- Error tracking
- Audit trails
- Performance monitoring

Features:
- Correlation IDs (request_id, trace_id)
- Tenant context injection
- Sensitive data masking
- Multiple output formats (JSON, human-readable)
"""

import json
import logging
import sys
import traceback
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

# ============================================================
# CONTEXT VARIABLES
# ============================================================

# Request context (set per-request)
request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
trace_id_var: ContextVar[str | None] = ContextVar("trace_id", default=None)
tenant_id_var: ContextVar[str | None] = ContextVar("tenant_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)


def set_request_context(
    request_id: str | None = None,
    trace_id: str | None = None,
    tenant_id: str | None = None,
    user_id: str | None = None,
):
    """Set request context variables."""
    if request_id:
        request_id_var.set(request_id)
    if trace_id:
        trace_id_var.set(trace_id)
    if tenant_id:
        tenant_id_var.set(tenant_id)
    if user_id:
        user_id_var.set(user_id)


def clear_request_context():
    """Clear request context variables."""
    request_id_var.set(None)
    trace_id_var.set(None)
    tenant_id_var.set(None)
    user_id_var.set(None)


def get_request_context() -> dict[str, str | None]:
    """Get current request context."""
    return {
        "request_id": request_id_var.get(),
        "trace_id": trace_id_var.get(),
        "tenant_id": tenant_id_var.get(),
        "user_id": user_id_var.get(),
    }


# ============================================================
# SENSITIVE DATA MASKING
# ============================================================

# Fields that should be masked in logs
SENSITIVE_FIELDS = {
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "auth",
    "credential",
    "private_key",
    "access_token",
    "refresh_token",
    "jwt",
    "bearer",
    "ssn",
    "social_security",
    "credit_card",
    "card_number",
    "cvv",
    "pin",
}


def mask_sensitive_data(data: Any, depth: int = 0, max_depth: int = 10) -> Any:
    """
    Recursively mask sensitive data in dictionaries and lists.

    Args:
        data: Data to mask
        depth: Current recursion depth
        max_depth: Maximum recursion depth

    Returns:
        Data with sensitive fields masked
    """
    if depth > max_depth:
        return "[MAX_DEPTH_EXCEEDED]"

    if isinstance(data, dict):
        masked = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(s in key_lower for s in SENSITIVE_FIELDS):
                masked[key] = "[REDACTED]"
            else:
                masked[key] = mask_sensitive_data(value, depth + 1, max_depth)
        return masked

    elif isinstance(data, list):
        return [mask_sensitive_data(item, depth + 1, max_depth) for item in data]

    elif isinstance(data, str):
        # Mask potential API keys or tokens in strings
        if len(data) > 20 and data.startswith(("sk-", "pk-", "Bearer ", "eyJ")):
            return f"{data[:8]}...[REDACTED]"
        return data

    return data


# ============================================================
# LOG RECORD STRUCTURE
# ============================================================


@dataclass
class StructuredLogRecord:
    """Structured log record for JSON output."""

    timestamp: str
    level: str
    logger: str
    message: str

    # Context
    request_id: str | None = None
    trace_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None

    # Location
    module: str | None = None
    function: str | None = None
    line: int | None = None

    # Error info
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None

    # Extra data
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {}}

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# ============================================================
# JSON FORMATTER
# ============================================================


class JSONFormatter(logging.Formatter):
    """
    JSON log formatter for structured logging.

    Outputs logs as single-line JSON for easy parsing by log aggregators.
    """

    def __init__(self, mask_sensitive: bool = True):
        super().__init__()
        self.mask_sensitive = mask_sensitive

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get request context
        ctx = get_request_context()

        # Build structured record
        log_record = StructuredLogRecord(
            timestamp=datetime.now(UTC).isoformat(),
            level=record.levelname,
            logger=record.name,
            message=record.getMessage(),
            request_id=ctx.get("request_id"),
            trace_id=ctx.get("trace_id"),
            tenant_id=ctx.get("tenant_id"),
            user_id=ctx.get("user_id"),
            module=record.module,
            function=record.funcName,
            line=record.lineno,
        )

        # Add exception info if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            if exc_type is not None:
                log_record.error_type = exc_type.__name__
                log_record.error_message = str(exc_value)
                log_record.stack_trace = "".join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                )

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "created",
                "filename",
                "funcName",
                "levelname",
                "levelno",
                "lineno",
                "module",
                "msecs",
                "pathname",
                "process",
                "processName",
                "relativeCreated",
                "stack_info",
                "exc_info",
                "exc_text",
                "thread",
                "threadName",
                "message",
                "taskName",
            }:
                extra_fields[key] = value

        if extra_fields:
            if self.mask_sensitive:
                extra_fields = mask_sensitive_data(extra_fields)
            log_record.extra = extra_fields

        return log_record.to_json()


# ============================================================
# HUMAN-READABLE FORMATTER
# ============================================================


class HumanFormatter(logging.Formatter):
    """
    Human-readable formatter with color support.

    Includes request context inline for easy debugging.
    """

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_colors: bool = True, mask_sensitive: bool = True):
        super().__init__()
        self.use_colors = use_colors
        self.mask_sensitive = mask_sensitive

    def format(self, record: logging.LogRecord) -> str:
        """Format log record for human reading."""
        # Get request context
        ctx = get_request_context()

        # Build context string
        ctx_parts = []
        if ctx.get("request_id"):
            ctx_parts.append(f"req={ctx['request_id'][:8]}")
        if ctx.get("tenant_id"):
            ctx_parts.append(f"tenant={ctx['tenant_id'][:8]}")
        ctx_str = f"[{' '.join(ctx_parts)}] " if ctx_parts else ""

        # Format timestamp
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Format level with color
        level = record.levelname
        if self.use_colors and level in self.COLORS:
            level = f"{self.COLORS[level]}{level}{self.RESET}"

        # Build message
        message = record.getMessage()

        # Format base line
        line = f"{timestamp} {level:8} {record.name}:{record.lineno} {ctx_str}{message}"

        # Add exception if present
        if record.exc_info:
            exc_text = "".join(traceback.format_exception(*record.exc_info))
            line = f"{line}\n{exc_text}"

        return line


# ============================================================
# LOGGING CONFIGURATION
# ============================================================


def configure_logging(
    level: str = "INFO",
    format: str = "json",  # "json" or "human"
    mask_sensitive: bool = True,
    use_colors: bool = True,
):
    """
    Configure logging for Reflection.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format: Output format ("json" for production, "human" for development)
        mask_sensitive: Whether to mask sensitive data
        use_colors: Whether to use colors (only for human format)
    """
    # Create formatter
    if format == "json":
        formatter = JSONFormatter(mask_sensitive=mask_sensitive)
    else:
        formatter = HumanFormatter(use_colors=use_colors, mask_sensitive=mask_sensitive)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add stdout handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Set levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)

    return root_logger


# ============================================================
# LOGGER ADAPTER WITH CONTEXT
# ============================================================


class ContextLogger(logging.LoggerAdapter):
    """
    Logger adapter that automatically includes context in log messages.

    Usage:
        logger = ContextLogger(logging.getLogger(__name__))
        logger.info("Processing request", extra={"operation": "chat"})
    """

    def __init__(self, logger: logging.Logger, extra: dict | None = None):
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: dict[str, Any]) -> tuple:
        """Process log message with context."""
        # Merge adapter extra with call extra
        extra = {**self.extra, **kwargs.get("extra", {})}

        # Add request context
        ctx = get_request_context()
        for key, value in ctx.items():
            if value is not None and key not in extra:
                extra[key] = value

        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> ContextLogger:
    """Get a context-aware logger."""
    return ContextLogger(logging.getLogger(name))


# ============================================================
# AUDIT LOGGING
# ============================================================


class AuditLogger:
    """
    Specialized logger for audit events.

    Audit events are always logged at INFO level with specific structure.
    """

    def __init__(self, name: str = "audit"):
        self._logger = logging.getLogger(name)

    def log(
        self,
        action: str,
        resource_type: str,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
        success: bool = True,
    ):
        """
        Log an audit event.

        Args:
            action: Action performed (e.g., "create", "delete", "access")
            resource_type: Type of resource (e.g., "conversation", "agent")
            resource_id: ID of the resource
            details: Additional details
            success: Whether the action succeeded
        """
        ctx = get_request_context()

        self._logger.info(
            f"AUDIT: {action} {resource_type}",
            extra={
                "audit_event": True,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "success": success,
                "details": mask_sensitive_data(details) if details else None,
                "tenant_id": ctx.get("tenant_id"),
                "user_id": ctx.get("user_id"),
                "request_id": ctx.get("request_id"),
            },
        )

    def access(self, resource_type: str, resource_id: str, details: dict | None = None):
        """Log an access event."""
        self.log("access", resource_type, resource_id, details)

    def create(self, resource_type: str, resource_id: str, details: dict | None = None):
        """Log a create event."""
        self.log("create", resource_type, resource_id, details)

    def update(self, resource_type: str, resource_id: str, details: dict | None = None):
        """Log an update event."""
        self.log("update", resource_type, resource_id, details)

    def delete(self, resource_type: str, resource_id: str, details: dict | None = None):
        """Log a delete event."""
        self.log("delete", resource_type, resource_id, details)

    def auth(self, method: str, success: bool, details: dict | None = None):
        """Log an authentication event."""
        self.log("auth", method, success=success, details=details)


# Global audit logger instance
audit_logger = AuditLogger()


# ============================================================
# REQUEST LOGGING HELPERS
# ============================================================


def log_request_start(
    method: str,
    path: str,
    request_id: str,
    tenant_id: str | None = None,
):
    """Log request start."""
    logger = get_logger("request")
    logger.info(
        f"Request started: {method} {path}",
        extra={
            "event": "request_start",
            "method": method,
            "path": path,
            "request_id": request_id,
            "tenant_id": tenant_id,
        },
    )


def log_request_end(
    method: str,
    path: str,
    request_id: str,
    status_code: int,
    duration_ms: float,
    tenant_id: str | None = None,
):
    """Log request end."""
    logger = get_logger("request")

    level = logging.INFO
    if status_code >= 500:
        level = logging.ERROR
    elif status_code >= 400:
        level = logging.WARNING

    logger.log(
        level,
        f"Request completed: {method} {path} -> {status_code} ({duration_ms:.1f}ms)",
        extra={
            "event": "request_end",
            "method": method,
            "path": path,
            "request_id": request_id,
            "status_code": status_code,
            "duration_ms": duration_ms,
            "tenant_id": tenant_id,
        },
    )


def log_llm_request(
    provider: str,
    model: str,
    tokens_input: int,
    tokens_output: int,
    duration_ms: float,
    success: bool = True,
    error: str | None = None,
):
    """Log LLM request."""
    logger = get_logger("llm")

    level = logging.INFO if success else logging.ERROR
    status = "success" if success else "error"

    logger.log(
        level,
        f"LLM request: {provider}/{model} -> {status} ({duration_ms:.1f}ms, {tokens_input}+{tokens_output} tokens)",
        extra={
            "event": "llm_request",
            "provider": provider,
            "model": model,
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "duration_ms": duration_ms,
            "success": success,
            "error": error,
        },
    )


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Context
    "set_request_context",
    "clear_request_context",
    "get_request_context",
    # Logging
    "configure_logging",
    "get_logger",
    "ContextLogger",
    "JSONFormatter",
    "HumanFormatter",
    # Audit
    "AuditLogger",
    "audit_logger",
    # Helpers
    "log_request_start",
    "log_request_end",
    "log_llm_request",
    "mask_sensitive_data",
    # Context vars
    "request_id_var",
    "trace_id_var",
    "tenant_id_var",
    "user_id_var",
]
