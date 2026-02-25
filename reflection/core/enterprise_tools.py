# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Enterprise Tools

Additional tools for enterprise use cases:
- Web search
- JSON manipulation
- DateTime operations
- Code execution (sandboxed)
"""

import asyncio
import contextlib
import json
import logging
import re
import tempfile
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx

from reflection_core.security.trust import Capability, TrustLevel

from .tools import AsyncToolRegistry, ToolParameter

logger = logging.getLogger(__name__)


# ============================================================
# WEB SEARCH
# ============================================================


async def web_search(query: str, num_results: int = 5) -> dict[str, Any]:
    """Search the web for information using DuckDuckGo."""
    if not query:
        raise ValueError("Search query is required")

    num_results = min(num_results, 10)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(
                "https://api.duckduckgo.com/",
                params={
                    "q": query,
                    "format": "json",
                    "no_html": 1,
                    "skip_disambig": 1,
                },
            )
            response.raise_for_status()
            data = response.json()

            results = []

            if data.get("Abstract"):
                results.append(
                    {
                        "title": data.get("Heading", ""),
                        "snippet": data.get("Abstract", ""),
                        "url": data.get("AbstractURL", ""),
                        "source": data.get("AbstractSource", ""),
                    }
                )

            for topic in data.get("RelatedTopics", [])[:num_results]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(
                        {
                            "title": topic.get("Text", "")[:100],
                            "snippet": topic.get("Text", ""),
                            "url": topic.get("FirstURL", ""),
                        }
                    )

            if not results:
                return {"status": "no_results", "query": query, "message": "No results found"}

            return {"status": "success", "query": query, "results": results[:num_results]}

        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return {"status": "error", "query": query, "message": f"Search failed: {str(e)}"}


# ============================================================
# HTTP REQUEST
# ============================================================

BLOCKED_URL_PATTERNS = [
    r"localhost",
    r"127\.0\.0\.",
    r"192\.168\.",
    r"10\.\d+\.",
    r"172\.(1[6-9]|2\d|3[01])\.",
    r"::1",
    r"0\.0\.0\.0",
]


def _is_blocked_url(url: str) -> bool:
    """Check if URL is blocked for security."""
    return any(re.search(pattern, url, re.IGNORECASE) for pattern in BLOCKED_URL_PATTERNS)


async def http_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> dict[str, Any]:
    """Make an HTTP request to an external API."""
    method = method.upper()
    if not url:
        raise ValueError("URL is required")

    if _is_blocked_url(url):
        raise ValueError("URL is blocked for security reasons")

    timeout = min(timeout, 60)

    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=headers or {},
                json=body if body else None,
            )

            try:
                response_body = response.json()
            except Exception:
                response_body = response.text[:5000]

            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response_body,
            }

        except httpx.TimeoutException:
            return {"status": "error", "message": "Request timed out"}
        except Exception as e:
            return {"status": "error", "message": f"Request failed: {str(e)}"}


# ============================================================
# PYTHON EXECUTION (SANDBOXED)
# ============================================================

BLOCKED_IMPORTS = [
    "os",
    "subprocess",
    "sys",
    "shutil",
    "socket",
    "urllib",
    "requests",
    "httpx",
    "aiohttp",
    "ftplib",
    "telnetlib",
    "pickle",
    "marshal",
    "shelve",
    "ctypes",
    "multiprocessing",
]


def _check_code_safety(code: str) -> str | None:
    """Check code for potentially dangerous patterns."""
    code_lower = code.lower()

    for blocked in BLOCKED_IMPORTS:
        if f"import {blocked}" in code_lower or f"from {blocked}" in code_lower:
            return f"Import of '{blocked}' is not allowed"

    dangerous_patterns = [
        ("exec(", "exec() is not allowed"),
        ("eval(", "eval() is not allowed"),
        ("compile(", "compile() is not allowed"),
        ("__import__", "__import__ is not allowed"),
        ("open(", "File operations are not allowed"),
        ("globals()", "globals() is not allowed"),
        ("locals()", "locals() is not allowed"),
    ]

    for pattern, message in dangerous_patterns:
        if pattern in code:
            return message

    return None


def _indent_code(code: str, spaces: int) -> str:
    """Indent code by specified spaces."""
    indent = " " * spaces
    return "\n".join(indent + line for line in code.split("\n"))


async def python_execute(code: str, timeout: int = 10) -> dict[str, Any]:
    """Execute Python code in a sandboxed environment."""
    if not code:
        raise ValueError("Code is required")

    timeout = min(timeout, 30)

    safety_error = _check_code_safety(code)
    if safety_error:
        return {"status": "error", "message": f"Code rejected: {safety_error}"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        wrapped_code = f"""
import json
import math
import random
import datetime
import re
import collections
import itertools
import functools
import sys

_original_stdout = sys.stdout
_original_stderr = sys.stderr
_result = None
_output = []
_errors = []

class _OutputCapture:
    def __init__(self, target):
        self.target = target
    def write(self, text):
        self.target.append(str(text))
    def flush(self):
        pass

sys.stdout = _OutputCapture(_output)
sys.stderr = _OutputCapture(_errors)

try:
{_indent_code(code, 4)}
except Exception as e:
    _errors.append(f"Error: {{e}}")

# Restore stdout for final output
sys.stdout = _original_stdout
sys.stderr = _original_stderr

result = {{
    "output": "".join(_output),
    "errors": "".join(_errors),
}}
print("__RESULT__" + json.dumps(result))
"""
        f.write(wrapped_code)
        temp_path = f.name

    try:
        process = await asyncio.create_subprocess_exec(
            "python3",
            temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            return {"status": "timeout", "message": f"Execution timed out after {timeout}s"}

        output = stdout.decode()

        if "__RESULT__" in output:
            result_json = output.split("__RESULT__")[1].strip()
            try:
                result = json.loads(result_json)
                return {
                    "status": "success",
                    "output": result.get("output", ""),
                    "errors": result.get("errors", ""),
                }
            except Exception:
                pass

        return {
            "status": "success",
            "output": output,
            "stderr": stderr.decode() if stderr else None,
        }

    finally:
        import os

        with contextlib.suppress(BaseException):
            os.unlink(temp_path)


# ============================================================
# JSON MANIPULATION
# ============================================================


def _jsonpath_query(data: Any, path: str) -> Any:
    """Simple JSONPath query implementation."""
    if path == "$":
        return data

    parts = path.replace("$.", "").split(".")
    result = data

    for part in parts:
        if not part:
            continue

        if "[" in part:
            key, idx = part.split("[")
            idx = int(idx.rstrip("]"))

            if key:
                result = result.get(key, [])

            if isinstance(result, list) and 0 <= idx < len(result):
                result = result[idx]
            else:
                return None

        elif isinstance(result, dict):
            result = result.get(part)

        else:
            return None

    return result


async def json_tool(
    operation: str,
    data: str,
    query: str | None = None,
) -> dict[str, Any]:
    """Parse, query, and transform JSON data."""
    try:
        parsed_data = json.loads(data)
    except json.JSONDecodeError as e:
        return {"status": "error", "message": f"Invalid JSON: {e}"}

    if operation == "parse":
        return {"status": "success", "data": parsed_data, "type": type(parsed_data).__name__}

    elif operation == "query":
        result = _jsonpath_query(parsed_data, query or "$")
        return {"status": "success", "result": result}

    elif operation == "validate":
        return {
            "status": "success",
            "valid": True,
            "type": type(parsed_data).__name__,
            "size": len(parsed_data) if isinstance(parsed_data, (list, dict)) else 1,
        }

    else:
        return {"status": "error", "message": f"Unknown operation: {operation}"}


# ============================================================
# DATETIME OPERATIONS
# ============================================================


async def datetime_tool(
    operation: str,
    date: str | None = None,
    date2: str | None = None,
    format: str | None = None,
    timezone_name: str | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    """Perform date and time operations."""
    if operation == "now":
        now = datetime.now(UTC)
        return {
            "iso": now.isoformat(),
            "unix": int(now.timestamp()),
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S UTC"),
        }

    elif operation == "parse":
        if not date:
            return {"status": "error", "message": "Date is required"}

        try:
            if format:
                dt = datetime.strptime(date, format)
            else:
                for fmt in [
                    "%Y-%m-%d",
                    "%Y-%m-%dT%H:%M:%S",
                    "%Y-%m-%d %H:%M:%S",
                    "%d/%m/%Y",
                    "%m/%d/%Y",
                ]:
                    try:
                        dt = datetime.strptime(date, fmt)
                        break
                    except Exception:
                        continue
                else:
                    raise ValueError(f"Could not parse date: {date}")

            return {
                "iso": dt.isoformat(),
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "weekday": dt.strftime("%A"),
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif operation == "diff":
        if not date or not date2:
            return {"status": "error", "message": "Both dates are required"}

        try:
            dt1 = datetime.fromisoformat(date.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))
            diff = dt2 - dt1
            return {
                "days": diff.days,
                "seconds": diff.seconds,
                "total_seconds": diff.total_seconds(),
            }

        except Exception as e:
            return {"status": "error", "message": str(e)}

    elif operation == "add":
        if not date:
            return {"status": "error", "message": "Date is required"}

        try:
            dt = datetime.fromisoformat(date.replace("Z", "+00:00"))
            result = dt + timedelta(days=days or 0)
            return {"iso": result.isoformat(), "formatted": result.strftime("%Y-%m-%d %H:%M:%S")}

        except Exception as e:
            return {"status": "error", "message": str(e)}

    else:
        return {"status": "error", "message": f"Unknown operation: {operation}"}


# ============================================================
# REGISTER ALL TOOLS
# ============================================================


def create_enterprise_tools() -> AsyncToolRegistry:
    """Create a registry with all enterprise tools."""
    registry = AsyncToolRegistry()

    # Web Search
    registry.register(
        name="web_search",
        description="Search the web for information. Returns relevant snippets and URLs.",
        parameters=[
            ToolParameter(name="query", type="string", description="Search query", required=True),
            ToolParameter(
                name="num_results",
                type="integer",
                description="Number of results (1-10)",
                required=False,
                default=5,
            ),
        ],
        required_capability=Capability.READ_WEB,
        required_trust=TrustLevel.KNOWN,
    )(web_search)

    # HTTP Request
    registry.register(
        name="http_request",
        description="Make an HTTP request to an external API.",
        parameters=[
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method",
                required=True,
                enum=["GET", "POST", "PUT", "DELETE"],
            ),
            ToolParameter(name="url", type="string", description="URL to request", required=True),
            ToolParameter(
                name="headers", type="object", description="Request headers", required=False
            ),
            ToolParameter(name="body", type="object", description="Request body", required=False),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds",
                required=False,
                default=30,
            ),
        ],
        required_capability=Capability.EXECUTE_HTTP,
        required_trust=TrustLevel.TRUSTED,
    )(http_request)

    # Python Execute
    registry.register(
        name="python_execute",
        description="Execute Python code in a sandboxed environment.",
        parameters=[
            ToolParameter(
                name="code", type="string", description="Python code to execute", required=True
            ),
            ToolParameter(
                name="timeout",
                type="integer",
                description="Timeout in seconds",
                required=False,
                default=10,
            ),
        ],
        required_capability=Capability.EXECUTE_CODE,
        required_trust=TrustLevel.TRUSTED,
    )(python_execute)

    # JSON Tool
    registry.register(
        name="json_tool",
        description="Parse, query, and transform JSON data.",
        parameters=[
            ToolParameter(
                name="operation",
                type="string",
                description="Operation",
                required=True,
                enum=["parse", "query", "validate"],
            ),
            ToolParameter(name="data", type="string", description="JSON string", required=True),
            ToolParameter(
                name="query", type="string", description="JSONPath query", required=False
            ),
        ],
        required_trust=TrustLevel.KNOWN,
    )(json_tool)

    # DateTime Tool
    registry.register(
        name="datetime_tool",
        description="Perform date and time operations.",
        parameters=[
            ToolParameter(
                name="operation",
                type="string",
                description="Operation",
                required=True,
                enum=["now", "parse", "diff", "add"],
            ),
            ToolParameter(name="date", type="string", description="Date string", required=False),
            ToolParameter(name="date2", type="string", description="Second date", required=False),
            ToolParameter(name="format", type="string", description="Date format", required=False),
            ToolParameter(name="days", type="integer", description="Days to add", required=False),
        ],
        required_trust=TrustLevel.STRANGER,
    )(datetime_tool)

    return registry


__all__ = [
    "web_search",
    "http_request",
    "python_execute",
    "json_tool",
    "datetime_tool",
    "create_enterprise_tools",
    "_check_code_safety",
    "_is_blocked_url",
]
