# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Extended Tools

Additional tools beyond the defaults for richer agent capabilities.
"""

import ast
import json
import logging
import math
import operator
from datetime import UTC, datetime
from typing import Any

from reflection_core.security.trust import TrustLevel

from .tools import AsyncTool, AsyncToolRegistry

logger = logging.getLogger(__name__)


# ============================================================
# SAFE EXPRESSION EVALUATOR
# ============================================================


class SafeExpressionEvaluator:
    """
    AST-based safe expression evaluator.

    Only allows mathematical operations - no attribute access,
    no arbitrary code execution, no sandbox escapes.

    Security: This replaces eval() to prevent:
    - Attribute access attacks (e.g., "".__class__.__mro__)
    - Code object creation
    - Import statements
    - Any non-mathematical operations
    """

    # Allowed binary operators
    BINARY_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.LShift: operator.lshift,
        ast.RShift: operator.rshift,
        ast.BitOr: operator.or_,
        ast.BitXor: operator.xor,
        ast.BitAnd: operator.and_,
    }

    # Allowed unary operators
    UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
        ast.Invert: operator.invert,
    }

    # Allowed comparison operators
    COMPARE_OPS = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
    }

    # Safe functions (no side effects, no system access)
    SAFE_FUNCTIONS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "sinh": math.sinh,
        "cosh": math.cosh,
        "tanh": math.tanh,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "floor": math.floor,
        "ceil": math.ceil,
        "trunc": math.trunc,
        "factorial": math.factorial,
        "gcd": math.gcd,
        "degrees": math.degrees,
        "radians": math.radians,
        "hypot": math.hypot,
        "isfinite": math.isfinite,
        "isinf": math.isinf,
        "isnan": math.isnan,
    }

    # Safe constants
    SAFE_CONSTANTS = {
        "pi": math.pi,
        "e": math.e,
        "tau": math.tau,
        "inf": math.inf,
        "nan": math.nan,
        "true": True,
        "false": False,
        "True": True,
        "False": False,
    }

    # Maximum allowed depth to prevent stack overflow
    MAX_DEPTH = 50

    def __init__(self):
        self._depth = 0

    def evaluate(self, expression: str) -> int | float | bool:
        """
        Safely evaluate a mathematical expression.

        Args:
            expression: Mathematical expression string

        Returns:
            Numeric result

        Raises:
            ValueError: If expression contains unsafe operations
            SyntaxError: If expression is malformed
        """
        self._depth = 0

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}") from e

        return self._eval_node(tree.body)

    def _eval_node(self, node: ast.AST) -> int | float | bool:
        """Recursively evaluate an AST node."""
        self._depth += 1
        if self._depth > self.MAX_DEPTH:
            raise ValueError("Expression too complex (depth limit exceeded)")

        try:
            # Numeric literals
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (int, float, bool)):
                    return node.value
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

            # Legacy Num node (Python < 3.8 compatibility)
            if isinstance(node, ast.Num):
                return node.n

            # Variable names (constants only)
            if isinstance(node, ast.Name):
                name = node.id
                if name in self.SAFE_CONSTANTS:
                    return self.SAFE_CONSTANTS[name]
                if name in self.SAFE_FUNCTIONS:
                    raise ValueError(f"'{name}' is a function, not a value. Use {name}(...)")
                raise ValueError(f"Unknown variable: '{name}'")

            # Binary operations (a + b, a * b, etc.)
            if isinstance(node, ast.BinOp):
                op_type = type(node.op)
                if op_type not in self.BINARY_OPS:
                    raise ValueError(f"Unsupported operator: {op_type.__name__}")
                left = self._eval_node(node.left)
                right = self._eval_node(node.right)
                return self.BINARY_OPS[op_type](left, right)

            # Unary operations (-a, +a)
            if isinstance(node, ast.UnaryOp):
                op_type = type(node.op)
                if op_type not in self.UNARY_OPS:
                    raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
                operand = self._eval_node(node.operand)
                return self.UNARY_OPS[op_type](operand)

            # Comparisons (a < b, a == b)
            if isinstance(node, ast.Compare):
                left = self._eval_node(node.left)
                for op, comparator in zip(node.ops, node.comparators, strict=False):
                    op_type = type(op)
                    if op_type not in self.COMPARE_OPS:
                        raise ValueError(f"Unsupported comparison: {op_type.__name__}")
                    right = self._eval_node(comparator)
                    if not self.COMPARE_OPS[op_type](left, right):
                        return False
                    left = right
                return True

            # Function calls (sqrt(x), sin(x))
            if isinstance(node, ast.Call):
                # Only allow simple function names, no methods
                if not isinstance(node.func, ast.Name):
                    raise ValueError("Only simple function calls allowed (no methods)")

                func_name = node.func.id
                if func_name not in self.SAFE_FUNCTIONS:
                    raise ValueError(f"Unknown function: '{func_name}'")

                # Evaluate arguments
                args = [self._eval_node(arg) for arg in node.args]

                # No keyword arguments allowed
                if node.keywords:
                    raise ValueError("Keyword arguments not supported")

                return self.SAFE_FUNCTIONS[func_name](*args)

            # Tuple/List for multi-argument functions
            if isinstance(node, (ast.Tuple, ast.List)):
                return [self._eval_node(elt) for elt in node.elts]

            # Ternary expression (a if condition else b)
            if isinstance(node, ast.IfExp):
                condition = self._eval_node(node.test)
                if condition:
                    return self._eval_node(node.body)
                else:
                    return self._eval_node(node.orelse)

            # SECURITY: Explicitly reject dangerous node types
            dangerous_types = (
                ast.Attribute,  # obj.attr - sandbox escape vector
                ast.Subscript,  # obj[key] - can access __class__ etc
                ast.Lambda,  # lambda expressions
                ast.ListComp,  # list comprehensions
                ast.SetComp,  # set comprehensions
                ast.DictComp,  # dict comprehensions
                ast.GeneratorExp,  # generator expressions
                ast.Await,  # async operations
                ast.Yield,  # generator yield
                ast.YieldFrom,  # generator delegation
                ast.FormattedValue,  # f-string components
                ast.JoinedStr,  # f-strings
            )

            if isinstance(node, dangerous_types):
                raise ValueError(f"Operation not allowed: {type(node).__name__}")

            raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        finally:
            self._depth -= 1


# Module-level evaluator instance
_safe_evaluator = SafeExpressionEvaluator()


# ============================================================
# CALCULATION TOOLS
# ============================================================


class CalculatorTool(AsyncTool):
    """
    Safe mathematical calculator.

    Evaluates mathematical expressions using AST-based parsing.
    No eval() - immune to sandbox escape attacks.
    """

    name = "calculator"
    description = "Evaluate mathematical expressions safely. Supports basic arithmetic, powers, roots, and common functions."

    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 3 * 4', 'sqrt(16)', 'sin(pi/2)')",
            }
        },
        "required": ["expression"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        expression = arguments.get("expression", "")

        try:
            # Use safe AST-based evaluator (no eval!)
            result = _safe_evaluator.evaluate(expression)

            # Format result
            if isinstance(result, float):
                result = int(result) if result.is_integer() else round(result, 10)

            return {
                "expression": expression,
                "result": result,
            }

        except ValueError as e:
            return {
                "expression": expression,
                "error": f"Invalid expression: {e}",
            }
        except (TypeError, ZeroDivisionError, OverflowError) as e:
            return {
                "expression": expression,
                "error": str(e),
            }
        except Exception as e:
            logger.warning(f"Calculator unexpected error: {e}")
            return {
                "expression": expression,
                "error": "Evaluation failed",
            }


class UnitConverterTool(AsyncTool):
    """Convert between common units."""

    name = "unit_converter"
    description = "Convert values between different units of measurement."

    parameters = {
        "type": "object",
        "properties": {
            "value": {"type": "number", "description": "The value to convert"},
            "from_unit": {
                "type": "string",
                "description": "Source unit (e.g., 'km', 'miles', 'celsius', 'fahrenheit')",
            },
            "to_unit": {"type": "string", "description": "Target unit"},
        },
        "required": ["value", "from_unit", "to_unit"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    CONVERSIONS = {
        # Length
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "inches"): lambda x: x * 0.393701,
        ("inches", "cm"): lambda x: x * 2.54,
        # Weight
        ("kg", "lbs"): lambda x: x * 2.20462,
        ("lbs", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,
        # Temperature
        ("celsius", "fahrenheit"): lambda x: x * 9 / 5 + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5 / 9,
        ("celsius", "kelvin"): lambda x: x + 273.15,
        ("kelvin", "celsius"): lambda x: x - 273.15,
        # Volume
        ("liters", "gallons"): lambda x: x * 0.264172,
        ("gallons", "liters"): lambda x: x * 3.78541,
        # Time
        ("hours", "minutes"): lambda x: x * 60,
        ("minutes", "hours"): lambda x: x / 60,
        ("days", "hours"): lambda x: x * 24,
        ("hours", "days"): lambda x: x / 24,
    }

    async def execute(self, arguments: dict[str, Any]) -> Any:
        value = arguments.get("value", 0)
        from_unit = arguments.get("from_unit", "").lower()
        to_unit = arguments.get("to_unit", "").lower()

        key = (from_unit, to_unit)

        if key in self.CONVERSIONS:
            result = self.CONVERSIONS[key](value)
            return {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": round(result, 6),
            }

        # Check if same unit
        if from_unit == to_unit:
            return {
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "result": value,
            }

        return {
            "error": f"Unknown conversion: {from_unit} to {to_unit}",
            "supported_conversions": list(self.CONVERSIONS.keys()),
        }


# ============================================================
# DATA TOOLS
# ============================================================


class JSONParserTool(AsyncTool):
    """Parse and query JSON data."""

    name = "json_parser"
    description = "Parse JSON strings and extract data using JSONPath-like queries."

    parameters = {
        "type": "object",
        "properties": {
            "json_string": {"type": "string", "description": "JSON string to parse"},
            "path": {
                "type": "string",
                "description": "Optional path to extract (e.g., 'data.items[0].name')",
            },
        },
        "required": ["json_string"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        json_string = arguments.get("json_string", "")
        path = arguments.get("path", "")

        try:
            data = json.loads(json_string)

            if not path:
                return {"data": data}

            # Navigate path
            current = data
            for part in path.replace("[", ".").replace("]", "").split("."):
                if not part:
                    continue

                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list):
                    try:
                        idx = int(part)
                        current = current[idx]
                    except (ValueError, IndexError):
                        return {"error": f"Invalid index: {part}"}
                else:
                    return {"error": f"Cannot traverse: {part}"}

                if current is None:
                    break

            return {"path": path, "value": current}

        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}"}


class DataAnalysisTool(AsyncTool):
    """Perform basic statistical analysis on numeric data."""

    name = "data_analysis"
    description = "Calculate statistics for a list of numbers (mean, median, std dev, etc.)"

    parameters = {
        "type": "object",
        "properties": {
            "values": {
                "type": "array",
                "items": {"type": "number"},
                "description": "List of numeric values to analyze",
            }
        },
        "required": ["values"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        values = arguments.get("values", [])

        if not values:
            return {"error": "No values provided"}

        if not all(isinstance(v, (int, float)) for v in values):
            return {"error": "All values must be numbers"}

        n = len(values)
        sorted_values = sorted(values)

        # Calculate statistics
        total = sum(values)
        mean = total / n

        # Median
        if n % 2 == 0:
            median = (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2
        else:
            median = sorted_values[n // 2]

        # Variance and std dev
        variance = sum((x - mean) ** 2 for x in values) / n
        std_dev = math.sqrt(variance)

        return {
            "count": n,
            "sum": round(total, 6),
            "mean": round(mean, 6),
            "median": round(median, 6),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
            "variance": round(variance, 6),
            "std_dev": round(std_dev, 6),
        }


# ============================================================
# TEXT TOOLS
# ============================================================


class TextAnalysisTool(AsyncTool):
    """Analyze text for basic metrics."""

    name = "text_analysis"
    description = "Analyze text for word count, character count, and other metrics."

    parameters = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to analyze"}},
        "required": ["text"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        text = arguments.get("text", "")

        words = text.split()
        sentences = [
            s.strip() for s in text.replace("!", ".").replace("?", ".").split(".") if s.strip()
        ]
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        return {
            "character_count": len(text),
            "character_count_no_spaces": len(text.replace(" ", "")),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "paragraph_count": len(paragraphs),
            "average_word_length": round(sum(len(w) for w in words) / max(len(words), 1), 2),
            "average_sentence_length": round(len(words) / max(len(sentences), 1), 2),
        }


class TextTransformTool(AsyncTool):
    """Transform text (case, encoding, etc.)."""

    name = "text_transform"
    description = "Transform text: change case, reverse, encode/decode, etc."

    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to transform"},
            "operation": {
                "type": "string",
                "enum": [
                    "uppercase",
                    "lowercase",
                    "title",
                    "reverse",
                    "base64_encode",
                    "base64_decode",
                    "count_chars",
                ],
                "description": "Transformation to apply",
            },
        },
        "required": ["text", "operation"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        text = arguments.get("text", "")
        operation = arguments.get("operation", "")

        import base64

        operations = {
            "uppercase": lambda t: t.upper(),
            "lowercase": lambda t: t.lower(),
            "title": lambda t: t.title(),
            "reverse": lambda t: t[::-1],
            "base64_encode": lambda t: base64.b64encode(t.encode()).decode(),
            "base64_decode": lambda t: base64.b64decode(t.encode()).decode(),
            "count_chars": lambda t: dict(
                sorted(
                    {c: t.count(c) for c in set(t) if c.strip()}.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),
        }

        if operation not in operations:
            return {"error": f"Unknown operation: {operation}"}

        try:
            result = operations[operation](text)
            return {
                "input": text[:100] + ("..." if len(text) > 100 else ""),
                "operation": operation,
                "result": result,
            }
        except Exception as e:
            return {"error": str(e)}


# ============================================================
# UTILITY TOOLS
# ============================================================


class RandomGeneratorTool(AsyncTool):
    """Generate random values."""

    name = "random_generator"
    description = "Generate random numbers, strings, or UUIDs."

    parameters = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": ["integer", "float", "uuid", "string", "choice"],
                "description": "Type of random value to generate",
            },
            "min": {"type": "number", "description": "Minimum value (for integer/float)"},
            "max": {"type": "number", "description": "Maximum value (for integer/float)"},
            "length": {"type": "integer", "description": "Length for string generation"},
            "choices": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Options to choose from (for choice type)",
            },
        },
        "required": ["type"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        import random
        import string
        import uuid

        gen_type = arguments.get("type", "integer")

        if gen_type == "integer":
            min_val = int(arguments.get("min", 0))
            max_val = int(arguments.get("max", 100))
            return {"type": "integer", "value": random.randint(min_val, max_val)}

        elif gen_type == "float":
            min_val = float(arguments.get("min", 0))
            max_val = float(arguments.get("max", 1))
            return {"type": "float", "value": round(random.uniform(min_val, max_val), 6)}

        elif gen_type == "uuid":
            return {"type": "uuid", "value": str(uuid.uuid4())}

        elif gen_type == "string":
            length = min(arguments.get("length", 16), 100)
            chars = string.ascii_letters + string.digits
            return {"type": "string", "value": "".join(random.choices(chars, k=length))}

        elif gen_type == "choice":
            choices = arguments.get("choices", ["a", "b", "c"])
            return {"type": "choice", "value": random.choice(choices)}

        return {"error": f"Unknown type: {gen_type}"}


class TimeTool(AsyncTool):
    """Get current time and do time calculations."""

    name = "time_tool"
    description = "Get current time, convert timezones, or calculate time differences."

    parameters = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["now", "parse", "diff", "add"],
                "description": "Operation to perform",
            },
            "format": {
                "type": "string",
                "description": "Output format (e.g., 'iso', 'unix', 'human')",
            },
            "datetime1": {"type": "string", "description": "First datetime (ISO format)"},
            "datetime2": {"type": "string", "description": "Second datetime (ISO format)"},
            "days": {"type": "integer", "description": "Days to add/subtract"},
            "hours": {"type": "integer", "description": "Hours to add/subtract"},
        },
        "required": ["operation"],
    }

    required_trust = TrustLevel.STRANGER
    required_capabilities = set()

    async def execute(self, arguments: dict[str, Any]) -> Any:
        from datetime import timedelta

        operation = arguments.get("operation", "now")
        fmt = arguments.get("format", "iso")

        def format_dt(dt: datetime) -> str:
            if fmt == "unix":
                return str(int(dt.timestamp()))
            elif fmt == "human":
                return dt.strftime("%B %d, %Y at %I:%M %p")
            else:
                return dt.isoformat()

        if operation == "now":
            now = datetime.now(UTC)
            return {
                "operation": "now",
                "utc": format_dt(now),
                "unix": int(now.timestamp()),
            }

        elif operation == "parse":
            dt_str = arguments.get("datetime1", "")
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                return {
                    "operation": "parse",
                    "input": dt_str,
                    "parsed": format_dt(dt),
                    "unix": int(dt.timestamp()),
                }
            except Exception as e:
                return {"error": f"Parse failed: {e}"}

        elif operation == "diff":
            dt1_str = arguments.get("datetime1", "")
            dt2_str = arguments.get("datetime2", "")
            try:
                dt1 = datetime.fromisoformat(dt1_str.replace("Z", "+00:00"))
                dt2 = datetime.fromisoformat(dt2_str.replace("Z", "+00:00"))
                diff = dt2 - dt1
                return {
                    "operation": "diff",
                    "datetime1": dt1_str,
                    "datetime2": dt2_str,
                    "difference": {
                        "days": diff.days,
                        "seconds": diff.seconds,
                        "total_seconds": diff.total_seconds(),
                        "total_hours": diff.total_seconds() / 3600,
                    },
                }
            except Exception as e:
                return {"error": f"Diff failed: {e}"}

        elif operation == "add":
            dt_str = arguments.get("datetime1", "")
            days = arguments.get("days", 0)
            hours = arguments.get("hours", 0)
            try:
                dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
                result = dt + timedelta(days=days, hours=hours)
                return {
                    "operation": "add",
                    "input": dt_str,
                    "added": f"{days} days, {hours} hours",
                    "result": format_dt(result),
                }
            except Exception as e:
                return {"error": f"Add failed: {e}"}

        return {"error": f"Unknown operation: {operation}"}


# ============================================================
# REGISTRY SETUP
# ============================================================


def create_extended_tools() -> AsyncToolRegistry:
    """Create a registry with all extended tools."""
    from .tools import create_default_tools

    registry = create_default_tools()

    # Add extended tools
    extended_tools = [
        CalculatorTool(),
        UnitConverterTool(),
        JSONParserTool(),
        DataAnalysisTool(),
        TextAnalysisTool(),
        TextTransformTool(),
        RandomGeneratorTool(),
        TimeTool(),
    ]

    for tool in extended_tools:
        registry.register(tool)

    return registry


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "CalculatorTool",
    "UnitConverterTool",
    "JSONParserTool",
    "DataAnalysisTool",
    "TextAnalysisTool",
    "TextTransformTool",
    "RandomGeneratorTool",
    "TimeTool",
    "create_extended_tools",
]
