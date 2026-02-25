"""
Reflection v2.0.0 Integration Tests
====================================

Verifies that:
1. Familiar v1.4.0 imports resolve correctly
2. TenantAgent constructor works with v1.4.0 Agent
3. Security enums are unified (same identity)
4. Chat flow delegates to Familiar's Agent.chat()
5. stream() delegates to Familiar's Agent.chat_stream()
6. agent_enhanced.py is removed
7. Version strings are updated
"""

import importlib
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Stub out heavy enterprise dependencies so the reflection package can be
# imported in a lightweight test environment (no sqlalchemy, pydantic-settings,
# redis, etc.).
#
# We track which stubs we install so teardown_module() can clean them up,
# preventing stub leakage into other test files in the same pytest session.
# ---------------------------------------------------------------------------

_STUBS = [
    "sqlalchemy", "sqlalchemy.ext", "sqlalchemy.ext.asyncio",
    "sqlalchemy.orm", "sqlalchemy.orm.attributes",
    "sqlalchemy.dialects", "sqlalchemy.dialects.postgresql",
    "sqlalchemy.schema",
    "pydantic_settings",
    "redis", "redis.asyncio",
    "asyncpg",
    "opentelemetry", "opentelemetry.trace", "opentelemetry.sdk",
    "opentelemetry.sdk.trace", "opentelemetry.sdk.resources",
    "opentelemetry.instrumentation",
    "opentelemetry.instrumentation.fastapi",
    "opentelemetry.exporter", "opentelemetry.exporter.otlp",
    "structlog",
    "prometheus_client",
    "fastapi", "fastapi.responses", "fastapi.middleware",
    "uvicorn",
    "httpx",
    "aiohttp",
]

_installed_stubs: list[str] = []

for _mod_name in _STUBS:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()
        _installed_stubs.append(_mod_name)


def teardown_module():
    """Remove enterprise dep stubs and cached reflection modules.

    This prevents MagicMock stubs from leaking into other test files
    when the full test suite runs together.
    """
    for mod_name in _installed_stubs:
        sys.modules.pop(mod_name, None)
    # Clear cached reflection modules that were imported with stubs
    poisoned = [k for k in sys.modules if k.startswith("reflection.")]
    for k in poisoned:
        sys.modules.pop(k, None)


# ---------------------------------------------------------------------------
# 1. Import smoke tests — Familiar v1.4.0
# ---------------------------------------------------------------------------


class TestImportSmoke:
    """Verify all Familiar v1.4.0 imports resolve."""

    def test_core_agent_import(self):
        from familiar.core.agent import Agent
        assert Agent is not None

    def test_config_import(self):
        from familiar.core.config import Config, load_config
        assert Config is not None
        assert callable(load_config)

    def test_providers_import(self):
        from familiar.core.providers import StreamEvent
        assert StreamEvent is not None

    def test_paths_import(self):
        from familiar.core.paths import set_tenant_data_root
        assert callable(set_tenant_data_root)

    def test_security_import(self):
        from familiar.core.security import TRUST_CAPABILITIES, Capability, TrustLevel
        assert TrustLevel is not None
        assert Capability is not None
        assert isinstance(TRUST_CAPABILITIES, dict)

    def test_tool_registry_import(self):
        from familiar.core.tool_registry import get_tool_registry
        assert callable(get_tool_registry)

    def test_memory_import(self):
        from familiar.core.memory import ConversationHistory, Memory, MemoryEntry
        assert Memory is not None
        assert MemoryEntry is not None
        assert ConversationHistory is not None

    def test_channels_import(self):
        from familiar.channels import CLIChannel
        assert CLIChannel is not None

    def test_tenant_agent_module_import(self):
        from reflection.tenant_wrappers.agent import TenantAgent
        assert TenantAgent is not None

    def test_familiar_setup(self):
        from reflection._familiar_setup import (
            FAMILIAR_AVAILABLE,
            FAMILIAR_VERSION,
            _MIN_FAMILIAR_VERSION,
        )
        assert FAMILIAR_AVAILABLE is True
        assert FAMILIAR_VERSION is not None
        assert _MIN_FAMILIAR_VERSION == "1.4.0"


# ---------------------------------------------------------------------------
# 2. TenantAgent constructor
# ---------------------------------------------------------------------------


class TestTenantAgentConstructor:
    """Verify TenantAgent initializes without error."""

    @patch("familiar.core.agent.Agent._init_provider", return_value=MagicMock())
    @patch("reflection.tenant_wrappers.agent.SmartLLMRouter")
    def test_basic_init(self, mock_router, mock_provider, isolate_filesystem):
        from reflection.tenant_wrappers.agent import TenantAgent

        tid = uuid4()
        agent = TenantAgent(tenant_id=tid)
        assert agent.tenant_id == tid
        assert agent.tenant_config == {}

    @patch("familiar.core.agent.Agent._init_provider", return_value=MagicMock())
    @patch("reflection.tenant_wrappers.agent.SmartLLMRouter")
    def test_init_with_security_mode(self, mock_router, mock_provider, isolate_filesystem):
        from familiar.core.security import SecurityMode
        from reflection.tenant_wrappers.agent import TenantAgent

        tid = uuid4()
        agent = TenantAgent(tenant_id=tid, security_mode=SecurityMode.PARANOID)
        assert agent.security_mode == SecurityMode.PARANOID

    @patch("familiar.core.agent.Agent._init_provider", return_value=MagicMock())
    @patch("reflection.tenant_wrappers.agent.SmartLLMRouter")
    def test_init_with_tenant_config(self, mock_router, mock_provider, isolate_filesystem):
        from reflection.tenant_wrappers.agent import TenantAgent

        tid = uuid4()
        tc = {"agent_name": "TestBot", "default_model": "test-model"}
        agent = TenantAgent(tenant_id=tid, tenant_config=tc)
        assert agent.config.agent.name == "TestBot"


# ---------------------------------------------------------------------------
# 3. Security enum unification
# ---------------------------------------------------------------------------


class TestSecurityEnumUnification:
    """Verify reflection_core re-exports Familiar's enums (same object identity)."""

    def test_trust_level_identity(self):
        from familiar.core.security import TrustLevel as FamiliarTL
        from reflection_core.security.trust import TrustLevel as ReflectionTL
        assert ReflectionTL is FamiliarTL

    def test_capability_identity(self):
        from familiar.core.security import Capability as FamiliarCap
        from reflection_core.security.trust import Capability as ReflectionCap
        assert ReflectionCap is FamiliarCap

    def test_trust_capabilities_mapping(self):
        from familiar.core.security import TRUST_CAPABILITIES
        from reflection_core.security.trust import DEFAULT_CAPABILITIES
        assert DEFAULT_CAPABILITIES is TRUST_CAPABILITIES


# ---------------------------------------------------------------------------
# 4. Chat flow (mocked LLM)
# ---------------------------------------------------------------------------


class TestChatFlow:
    """Verify TenantAgent.chat() delegates to Agent.chat()."""

    @patch("familiar.core.agent.Agent._init_provider", return_value=MagicMock())
    @patch("reflection.tenant_wrappers.agent.SmartLLMRouter")
    @patch("reflection.tenant_wrappers.agent.check_quota")
    @patch("reflection.tenant_wrappers.agent.get_token_counter")
    def test_chat_calls_super(
        self, mock_counter, mock_quota, mock_router, mock_provider, isolate_filesystem
    ):
        from reflection.tenant_wrappers.agent import TenantAgent

        mock_count_result = MagicMock()
        mock_count_result.count = 10
        mock_counter.return_value.count_tokens.return_value = mock_count_result

        tid = uuid4()
        agent = TenantAgent(tenant_id=tid)

        # Mock the parent Agent.chat to return a canned response
        with patch.object(
            type(agent).__mro__[1], "chat", return_value="Hello from Familiar!"
        ) as mock_chat:
            response = agent.chat("Hi there")

        assert response == "Hello from Familiar!"
        mock_chat.assert_called_once()


# ---------------------------------------------------------------------------
# 5. stream() → chat_stream() delegation
# ---------------------------------------------------------------------------


class TestStreamDelegation:
    """Verify TenantAgent.stream() calls Agent.chat_stream()."""

    @patch("familiar.core.agent.Agent._init_provider", return_value=MagicMock())
    @patch("reflection.tenant_wrappers.agent.SmartLLMRouter")
    @patch("reflection.tenant_wrappers.agent.check_quota")
    def test_stream_calls_chat_stream(
        self, mock_quota, mock_router, mock_provider, isolate_filesystem
    ):
        from familiar.core.providers import StreamEvent
        from reflection.tenant_wrappers.agent import TenantAgent

        tid = uuid4()
        agent = TenantAgent(tenant_id=tid)

        fake_event = StreamEvent(type="text", content="chunk")

        with patch.object(
            type(agent).__mro__[1],
            "chat_stream",
            return_value=iter([fake_event]),
        ) as mock_stream:
            events = list(agent.stream("Hello"))

        mock_stream.assert_called_once()
        assert len(events) == 1
        assert events[0].content == "chunk"


# ---------------------------------------------------------------------------
# 6. Agent enhanced module is gone
# ---------------------------------------------------------------------------


class TestAgentEnhancedRemoved:
    """Verify agent_enhanced.py is deleted."""

    def test_no_agent_enhanced_file(self):
        wrappers_dir = Path(__file__).parent.parent / "reflection" / "tenant_wrappers"
        assert not (wrappers_dir / "agent_enhanced.py").exists()

    def test_no_has_enhanced_in_init(self):
        init_path = (
            Path(__file__).parent.parent
            / "reflection"
            / "tenant_wrappers"
            / "__init__.py"
        )
        content = init_path.read_text()
        assert "agent_enhanced" not in content
        assert "HAS_ENHANCED" not in content
        assert "EnhancedTenantAgent" not in content


# ---------------------------------------------------------------------------
# 7. Version checks
# ---------------------------------------------------------------------------


class TestVersions:
    """Verify version strings are updated."""

    def test_version_file(self):
        version_path = Path(__file__).parent.parent / "VERSION"
        assert version_path.read_text().strip() == "2.0.0"

    def test_pyproject_version(self):
        import tomllib

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
        assert data["project"]["version"] == "2.0.0"
