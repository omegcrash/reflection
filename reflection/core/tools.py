# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Async Tool Registry

Provides:
- Tool registration and discovery
- Parallel tool execution
- Tenant-scoped tool permissions
- Execution timeouts and quotas
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC
from typing import Any, TypeVar

from reflection_core.security.trust import Capability, TrustLevel, check_capability

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ============================================================
# TOOL DEFINITION
# ============================================================


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""

    name: str
    type: str  # JSON Schema type
    description: str
    required: bool = True
    default: Any = None
    enum: list[Any] | None = None


@dataclass
class Tool:
    """
    Definition of an executable tool.

    Tools can be sync or async functions. Sync functions are
    automatically wrapped for async execution.
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: list[ToolParameter] = field(default_factory=list)

    # Permissions
    required_capability: Capability | None = None
    required_trust: TrustLevel = TrustLevel.KNOWN
    requires_confirmation: bool = False

    # Execution
    timeout_seconds: float = 30.0
    is_async: bool = False

    # Metadata
    category: str = "general"
    tags: set[str] = field(default_factory=set)

    def to_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema for LLM tool use."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default

            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


@dataclass
class ToolResult:
    """Result of a tool execution."""

    tool_name: str
    success: bool
    output: Any
    error: str | None = None
    execution_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "success": self.success,
            "output": self.output if self.success else None,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


# ============================================================
# ASYNC TOOL BASE CLASS
# ============================================================


class AsyncTool:
    """
    Base class for class-based async tools.

    Subclass this to create tools with state, cleanup, etc.

    Usage:
        class MyTool(AsyncTool):
            name = "my_tool"
            description = "Does something"
            parameters = {
                "type": "object",
                "properties": {...}
            }

            async def execute(self, **kwargs) -> Any:
                ...
    """

    # Override these in subclasses
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = {}

    # Permissions
    required_trust: TrustLevel = TrustLevel.KNOWN
    required_capabilities: set[Capability] = set()
    requires_confirmation: bool = False

    # Execution
    timeout_seconds: float = 30.0
    category: str = "general"
    tags: set[str] = set()

    async def execute(self, **kwargs) -> Any:
        """Execute the tool. Override in subclasses."""
        raise NotImplementedError

    async def cleanup(self) -> None:
        """Clean up resources. Override if needed."""
        pass

    def to_tool(self) -> "Tool":
        """Convert to a Tool instance for the registry."""
        # Convert parameters dict to ToolParameter list
        params = []
        if self.parameters and "properties" in self.parameters:
            required = self.parameters.get("required", [])
            for name, spec in self.parameters["properties"].items():
                params.append(
                    ToolParameter(
                        name=name,
                        type=spec.get("type", "string"),
                        description=spec.get("description", ""),
                        required=name in required,
                        default=spec.get("default"),
                        enum=spec.get("enum"),
                    )
                )

        # Get first required capability if any
        req_cap = None
        if self.required_capabilities:
            req_cap = next(iter(self.required_capabilities))

        return Tool(
            name=self.name,
            description=self.description,
            func=self.execute,
            parameters=params,
            required_capability=req_cap,
            required_trust=self.required_trust,
            requires_confirmation=self.requires_confirmation,
            timeout_seconds=self.timeout_seconds,
            is_async=True,
            category=self.category,
            tags=self.tags if isinstance(self.tags, set) else set(self.tags),
        )


# ============================================================
# TOOL REGISTRY
# ============================================================


class AsyncToolRegistry:
    """
    Async tool registry with tenant-scoped execution.

    Usage:
        registry = AsyncToolRegistry()

        @registry.register(
            description="Search the web",
            required_capability=Capability.READ_WEB
        )
        async def web_search(query: str) -> str:
            ...

        result = await registry.execute("web_search", {"query": "hello"})
    """

    def __init__(
        self,
        tenant_id: str | None = None,
        max_parallel: int = 5,
    ):
        self.tenant_id = tenant_id
        self.max_parallel = max_parallel
        self._tools: dict[str, Tool] = {}
        self._semaphore = asyncio.Semaphore(max_parallel)

    def register(
        self,
        name: str | None = None,
        description: str = "",
        parameters: list[ToolParameter] | None = None,
        required_capability: Capability | None = None,
        required_trust: TrustLevel = TrustLevel.KNOWN,
        requires_confirmation: bool = False,
        timeout_seconds: float = 30.0,
        category: str = "general",
        tags: set[str] | None = None,
    ):
        """
        Decorator to register a tool.

        Usage:
            @registry.register(description="Get current time")
            async def get_time() -> str:
                return datetime.now().isoformat()
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            is_async = asyncio.iscoroutinefunction(func)

            # Extract parameters from type hints if not provided
            tool_params = parameters or self._extract_parameters(func)

            tool = Tool(
                name=tool_name,
                description=description or func.__doc__ or "",
                func=func,
                parameters=tool_params,
                required_capability=required_capability,
                required_trust=required_trust,
                requires_confirmation=requires_confirmation,
                timeout_seconds=timeout_seconds,
                is_async=is_async,
                category=category,
                tags=tags or set(),
            )

            self._tools[tool_name] = tool
            logger.debug(f"Registered tool: {tool_name}")

            return func

        return decorator

    def add_tool(self, tool: Tool) -> None:
        """Add a pre-defined tool to the registry."""
        self._tools[tool.name] = tool

    def register_async_tool(self, async_tool: AsyncTool) -> None:
        """Register a class-based AsyncTool."""
        tool = async_tool.to_tool()
        self._tools[tool.name] = tool
        logger.debug(f"Registered async tool: {tool.name}")

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(
        self,
        category: str | None = None,
        trust_level: TrustLevel | None = None,
    ) -> list[Tool]:
        """List available tools, optionally filtered."""
        tools = list(self._tools.values())

        if category:
            tools = [t for t in tools if t.category == category]

        if trust_level:
            tools = [t for t in tools if trust_level >= t.required_trust]

        return tools

    def get_schemas(
        self,
        trust_level: TrustLevel = TrustLevel.KNOWN,
        capabilities: set[Capability] | None = None,
    ) -> list[dict[str, Any]]:
        """Get tool schemas for LLM, filtered by permissions."""
        schemas = []

        for tool in self._tools.values():
            # Check trust level
            if trust_level < tool.required_trust:
                continue

            # Check capability if required
            if (
                tool.required_capability
                and capabilities
                and not check_capability(tool.required_capability, capabilities, trust_level)
            ):
                continue

            schemas.append(tool.to_schema())

        return schemas

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        trust_level: TrustLevel = TrustLevel.KNOWN,
        capabilities: set[Capability] | None = None,
    ) -> ToolResult:
        """
        Execute a single tool.

        Args:
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            trust_level: User's trust level
            capabilities: User's explicit capabilities

        Returns:
            ToolResult with output or error
        """
        start_time = time.monotonic()

        # Get tool
        tool = self._tools.get(tool_name)
        if not tool:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool not found: {tool_name}",
            )

        # Check permissions
        if trust_level < tool.required_trust:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Insufficient trust level for {tool_name}",
            )

        if tool.required_capability and not check_capability(
            tool.required_capability, capabilities or set(), trust_level
        ):
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Missing capability: {tool.required_capability}",
            )

        # Execute with semaphore and timeout
        try:
            async with self._semaphore:
                if tool.is_async:
                    result = await asyncio.wait_for(
                        tool.func(**arguments), timeout=tool.timeout_seconds
                    )
                else:
                    # Run sync function in thread pool
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: tool.func(**arguments)),
                        timeout=tool.timeout_seconds,
                    )

            execution_time = int((time.monotonic() - start_time) * 1000)

            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=result,
                execution_time_ms=execution_time,
            )

        except TimeoutError:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=f"Tool timed out after {tool.timeout_seconds}s",
                execution_time_ms=int(tool.timeout_seconds * 1000),
            )

        except Exception as e:
            execution_time = int((time.monotonic() - start_time) * 1000)
            logger.exception(f"Tool {tool_name} failed: {e}")

            return ToolResult(
                tool_name=tool_name,
                success=False,
                output=None,
                error=str(e),
                execution_time_ms=execution_time,
            )

    async def execute_parallel(
        self,
        tool_calls: list[dict[str, Any]],
        trust_level: TrustLevel = TrustLevel.KNOWN,
        capabilities: set[Capability] | None = None,
    ) -> list[ToolResult]:
        """
        Execute multiple tools in parallel.

        Args:
            tool_calls: List of {"name": str, "arguments": dict}
            trust_level: User's trust level
            capabilities: User's capabilities

        Returns:
            List of ToolResults in same order as input
        """
        tasks = [
            self.execute(tc["name"], tc.get("arguments", {}), trust_level, capabilities)
            for tc in tool_calls
        ]

        return await asyncio.gather(*tasks)

    def _extract_parameters(self, func: Callable) -> list[ToolParameter]:
        """Extract parameters from function signature."""
        sig = inspect.signature(func)
        hints = func.__annotations__ if hasattr(func, "__annotations__") else {}

        params = []
        for name, param in sig.parameters.items():
            if name in ("self", "cls"):
                continue

            # Determine type
            hint = hints.get(name, Any)
            json_type = self._python_type_to_json(hint)

            # Determine if required
            has_default = param.default != inspect.Parameter.empty

            params.append(
                ToolParameter(
                    name=name,
                    type=json_type,
                    description=f"Parameter: {name}",
                    required=not has_default,
                    default=param.default if has_default else None,
                )
            )

        return params

    def _python_type_to_json(self, python_type: type) -> str:
        """Convert Python type hint to JSON Schema type."""
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }
        return type_map.get(python_type, "string")


# ============================================================
# BUILT-IN TOOLS
# ============================================================


def create_default_tools() -> AsyncToolRegistry:
    """Create a registry with default tools."""
    registry = AsyncToolRegistry()

    @registry.register(
        description="Get the current date and time",
        required_trust=TrustLevel.STRANGER,
        category="utility",
    )
    async def get_current_time() -> str:
        """Returns the current date and time in ISO format."""
        from datetime import datetime

        return datetime.now(UTC).isoformat()

    @registry.register(
        name="echo",
        description="Echo back a message (for testing)",
        parameters=[
            ToolParameter(
                name="message",
                type="string",
                description="The message to echo back",
            )
        ],
        required_trust=TrustLevel.STRANGER,
        category="utility",
    )
    async def echo(message: str) -> str:
        """Echo back the input message."""
        return f"Echo: {message}"

    @registry.register(
        name="calculate",
        description="Perform a mathematical calculation",
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate (e.g., '2 + 2')",
            )
        ],
        required_trust=TrustLevel.KNOWN,
        category="utility",
    )
    async def calculate(expression: str) -> str:
        """Safely evaluate a mathematical expression."""
        import ast
        import operator

        # Safe operators
        ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_node(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                left = eval_node(node.left)
                right = eval_node(node.right)
                return ops[type(node.op)](left, right)
            elif isinstance(node, ast.UnaryOp):
                operand = eval_node(node.operand)
                return ops[type(node.op)](operand)
            else:
                raise ValueError(f"Unsupported operation: {type(node)}")

        try:
            tree = ast.parse(expression, mode="eval")
            result = eval_node(tree.body)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    return registry


def create_all_tools() -> AsyncToolRegistry:
    """
    Create a registry with all tools (default + enterprise).

    Use this for production deployments.
    """
    from .enterprise_tools import create_enterprise_tools

    # Start with enterprise tools (they include more features)
    registry = create_enterprise_tools()

    # Add default tools that aren't duplicates
    default = create_default_tools()
    for name, tool in default._tools.items():
        if name not in registry._tools:
            registry._tools[name] = tool

    return registry


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "Tool",
    "ToolParameter",
    "ToolResult",
    "AsyncTool",
    "AsyncToolRegistry",
    "create_default_tools",
    "create_all_tools",
]
