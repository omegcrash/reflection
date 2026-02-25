# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tenant Tools

Multi-tenant wrapper around Familiar's ToolRegistry.
Adds:
- Per-tenant tool configuration
- Tenant-specific tool permissions
- Usage tracking per tool
- Tool sandboxing per tenant
"""

import logging
import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

# Setup familiar path
from .._familiar_setup import FAMILIAR_AVAILABLE

if not FAMILIAR_AVAILABLE:
    raise ImportError("Familiar core package required")

# Import from Familiar core
from familiar.core.tools import Tool, ToolRegistry, get_tool_registry

logger = logging.getLogger(__name__)


class TenantToolRegistry(ToolRegistry):
    """
    Multi-tenant tool registry extending Familiar's ToolRegistry.

    Adds:
    - Per-tenant tool enable/disable
    - Tenant-specific sandboxed directories
    - Tool usage tracking for billing
    - Cross-tenant isolation
    """

    def __init__(
        self,
        tenant_id: UUID,
        base_registry: ToolRegistry | None = None,
        sandboxed_dirs: list[str] | None = None,
        **kwargs,
    ):
        self.tenant_id = tenant_id

        # Per-tenant tool configuration
        self._enabled_tools: set[str] = set()
        self._disabled_tools: set[str] = set()
        self._tool_configs: dict[str, dict[str, Any]] = {}
        self._usage_callback: Callable | None = None

        # Tenant-specific sandbox
        self.tenant_sandbox = f"/data/tenants/{tenant_id}/sandbox"
        tenant_sandboxed = sandboxed_dirs or [self.tenant_sandbox]

        # Initialize parent
        super().__init__(sandboxed_dirs=tenant_sandboxed)

        # Copy tools from base registry if provided
        if base_registry:
            for name, tool in base_registry.tools.items():
                self.tools[name] = tool

        logger.debug(f"TenantToolRegistry initialized for tenant {tenant_id}")

    def set_usage_callback(self, callback: Callable):
        """Set callback for tool usage tracking."""
        self._usage_callback = callback

    def enable_tool(self, tool_name: str):
        """Enable a tool for this tenant."""
        self._enabled_tools.add(tool_name)
        self._disabled_tools.discard(tool_name)
        logger.info(f"Tenant {self.tenant_id}: Enabled tool {tool_name}")

    def disable_tool(self, tool_name: str):
        """Disable a tool for this tenant."""
        self._disabled_tools.add(tool_name)
        self._enabled_tools.discard(tool_name)
        logger.info(f"Tenant {self.tenant_id}: Disabled tool {tool_name}")

    def set_tool_config(self, tool_name: str, config: dict[str, Any]):
        """Set tenant-specific configuration for a tool."""
        self._tool_configs[tool_name] = config

    def is_tool_enabled(self, tool_name: str) -> bool:
        """Check if tool is enabled for tenant."""
        if tool_name in self._disabled_tools:
            return False
        if tool_name in self._enabled_tools:
            return True
        return True  # Default enabled

    def get_all(self) -> dict[str, Tool]:
        """Get all tools available to tenant."""
        all_tools = super().get_all()
        return {k: v for k, v in all_tools.items() if self.is_tool_enabled(k)}

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas filtered by tenant settings."""
        all_schemas = super().get_schemas()
        return [s for s in all_schemas if self.is_tool_enabled(s.get("name", ""))]

    def execute(
        self,
        name: str,
        input_data: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> str:
        """Execute a tool with tenant context."""
        if not self.is_tool_enabled(name):
            raise PermissionError(f"Tool '{name}' is not enabled for tenant {self.tenant_id}")

        # Apply tenant-specific config
        if name in self._tool_configs:
            input_data = {**input_data, **self._tool_configs[name]}

        # Add tenant to context
        context = context or {}
        context["tenant_id"] = str(self.tenant_id)

        start = time.time()

        result = super().execute(name, input_data, context)

        execution_time_ms = int((time.time() - start) * 1000)

        if self._usage_callback:
            self._usage_callback(
                tenant_id=self.tenant_id,
                tool_name=name,
                execution_time_ms=execution_time_ms,
                success=True,
            )

        return result


# Cache of tenant registries
_tenant_registries: dict[UUID, TenantToolRegistry] = {}


def get_tenant_tool_registry(
    tenant_id: UUID,
    base_registry: ToolRegistry | None = None,
) -> TenantToolRegistry:
    """Get or create tool registry for tenant."""
    if tenant_id not in _tenant_registries:
        _tenant_registries[tenant_id] = TenantToolRegistry(
            tenant_id=tenant_id,
            base_registry=base_registry or get_tool_registry(),
        )
    return _tenant_registries[tenant_id]


def clear_tenant_registry(tenant_id: UUID):
    """Clear cached registry for tenant."""
    if tenant_id in _tenant_registries:
        del _tenant_registries[tenant_id]


__all__ = [
    "TenantToolRegistry",
    "get_tenant_tool_registry",
    "clear_tenant_registry",
]
