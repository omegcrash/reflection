"""
Test Suite: MCP Server (Skills as MCP Servers)
================================================

Verifies that Familiar skills can be exposed as MCP-compliant
servers for Claude Desktop, VS Code, and other MCP clients.
"""

import json
import pytest
from familiar.core.paths import set_tenant_data_root
from familiar.core.mcp.server import (
    SkillMCPServer,
    load_skill_tools,
    get_available_skills,
    generate_claude_desktop_config,
    generate_vscode_config,
    MCPToolDef,
    _jsonrpc_result,
    _jsonrpc_error,
    PROTOCOL_VERSION,
    PARSE_ERROR,
    METHOD_NOT_FOUND,
)


@pytest.fixture
def mcp_tenant(tenant_id, tmp_path):
    """Set up tenant for MCP server tests."""
    set_tenant_data_root(str(tenant_id))
    from familiar.core import paths
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return tenant_id


@pytest.fixture
def nonprofit_server(mcp_tenant):
    return SkillMCPServer.from_skill_name("nonprofit", tenant_id=str(mcp_tenant))


@pytest.fixture
def multi_server(mcp_tenant):
    return SkillMCPServer.from_all_skills(
        tenant_id=str(mcp_tenant),
        skill_names=["nonprofit", "bookkeeping", "contacts"],
    )


# ── Skill Loading ──────────────────────────────────────────

class TestSkillLoading:

    def test_load_nonprofit_tools(self, mcp_tenant):
        tools, desc = load_skill_tools("nonprofit")
        assert len(tools) == 9
        assert all(isinstance(t, MCPToolDef) for t in tools)

    def test_load_bookkeeping_tools(self, mcp_tenant):
        tools, desc = load_skill_tools("bookkeeping")
        assert len(tools) == 8

    def test_tools_have_handlers(self, mcp_tenant):
        tools, _ = load_skill_tools("nonprofit")
        for t in tools:
            assert callable(t.handler)

    def test_tools_have_schemas(self, mcp_tenant):
        tools, _ = load_skill_tools("nonprofit")
        for t in tools:
            assert "type" in t.input_schema
            assert t.input_schema["type"] == "object"

    def test_readonly_annotation(self, mcp_tenant):
        tools, _ = load_skill_tools("nonprofit")
        names = {t.name: t for t in tools}
        assert names["search_donors"].annotations.get("readOnlyHint") is True
        assert names["log_donor"].annotations.get("readOnlyHint", False) is False

    def test_skill_description_from_md(self, mcp_tenant):
        tools, desc = load_skill_tools("nonprofit")
        assert len(desc) > 0
        assert desc != "Familiar nonprofit skill"  # Should have loaded from SKILL.md

    def test_get_available_skills(self, mcp_tenant):
        skills = get_available_skills()
        names = [s["name"] for s in skills]
        assert "nonprofit" in names
        assert "bookkeeping" in names
        assert len(skills) >= 30


# ── Server Construction ────────────────────────────────────

class TestServerConstruction:

    def test_single_skill_server(self, nonprofit_server):
        assert len(nonprofit_server.tools) == 9
        assert "log_donor" in nonprofit_server.tools
        assert "search_donors" in nonprofit_server.tools

    def test_multi_skill_server(self, multi_server):
        assert len(multi_server.tools) > 20
        assert "nonprofit:log_donor" in multi_server.tools
        assert "bookkeeping:log_income" in multi_server.tools
        assert "contacts:add_contact" in multi_server.tools

    def test_multi_skill_no_name_collision(self, multi_server):
        # All names should be prefixed
        for name in multi_server.tools:
            assert ":" in name


# ── MCP Protocol ───────────────────────────────────────────

class TestMCPProtocol:

    def test_initialize(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            },
        })
        assert resp["id"] == 1
        result = resp["result"]
        assert result["protocolVersion"] == PROTOCOL_VERSION
        assert "tools" in result["capabilities"]
        assert "serverInfo" in result

    def test_initialized_notification(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "method": "notifications/initialized",
        })
        assert resp is None  # Notifications return None

    def test_tools_list(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        tools = resp["result"]["tools"]
        assert len(tools) == 9
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t

    def test_tools_call_success(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "log_donor",
                "arguments": {"name": "Protocol Test", "type": "gift", "amount": 100},
            },
        })
        result = resp["result"]
        assert result["isError"] is False
        assert len(result["content"]) > 0
        assert "Protocol Test" in result["content"][0]["text"]

    def test_tools_call_unknown_tool(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert resp["result"]["isError"] is True

    def test_unknown_method(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 5, "method": "bogus/method", "params": {},
        })
        assert "error" in resp
        assert resp["error"]["code"] == METHOD_NOT_FOUND

    def test_ping(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 6, "method": "ping", "params": {},
        })
        assert resp["id"] == 6
        assert "result" in resp

    def test_resources_list(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 7, "method": "resources/list", "params": {},
        })
        assert resp["result"]["resources"] == []

    def test_prompts_list(self, nonprofit_server):
        resp = nonprofit_server.handle_message({
            "jsonrpc": "2.0", "id": 8, "method": "prompts/list", "params": {},
        })
        assert resp["result"]["prompts"] == []


class TestMCPProtocolEndToEnd:
    """Simulate a complete MCP client session."""

    def test_full_session(self, nonprofit_server):
        s = nonprofit_server

        # 1. Initialize
        r = s.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-test", "version": "1.0"},
        }})
        assert r["result"]["protocolVersion"] == PROTOCOL_VERSION

        # 2. Initialized notification
        r = s.handle_message({"jsonrpc": "2.0", "method": "notifications/initialized"})
        assert r is None

        # 3. List tools
        r = s.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        tool_names = [t["name"] for t in r["result"]["tools"]]
        assert "log_donor" in tool_names

        # 4. Call tool — log donor
        r = s.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "log_donor",
            "arguments": {"name": "E2E Donor", "type": "gift", "amount": 250},
        }})
        assert r["result"]["isError"] is False
        assert "E2E Donor" in r["result"]["content"][0]["text"]

        # 5. Call tool — search for the donor
        r = s.handle_message({"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
            "name": "search_donors",
            "arguments": {"query": "E2E"},
        }})
        assert r["result"]["isError"] is False
        assert "E2E Donor" in r["result"]["content"][0]["text"]

    def test_multi_skill_session(self, multi_server):
        s = multi_server

        # Initialize
        s.handle_message({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "multi-test", "version": "1.0"},
        }})

        # List tools — should have prefixed names
        r = s.handle_message({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        names = [t["name"] for t in r["result"]["tools"]]
        assert any(n.startswith("nonprofit:") for n in names)
        assert any(n.startswith("bookkeeping:") for n in names)

        # Call prefixed tool
        r = s.handle_message({"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
            "name": "bookkeeping:log_income",
            "arguments": {"amount": 1000, "source": "MCP Grant", "category": "grants"},
        }})
        assert r["result"]["isError"] is False


# ── Config Generation ──────────────────────────────────────

class TestConfigGeneration:

    def test_claude_desktop_config_all(self):
        config = generate_claude_desktop_config()
        assert "familiar" in config
        assert config["familiar"]["command"] == "python3"
        assert "--all" in config["familiar"]["args"]

    def test_claude_desktop_config_specific(self):
        config = generate_claude_desktop_config(skill_names=["nonprofit", "bookkeeping"])
        assert "familiar-nonprofit" in config
        assert "familiar-bookkeeping" in config
        assert "--skill" in config["familiar-nonprofit"]["args"]
        assert "nonprofit" in config["familiar-nonprofit"]["args"]

    def test_vscode_config(self):
        config = generate_vscode_config()
        assert "familiar" in config
        assert config["familiar"]["type"] == "stdio"

    def test_config_is_valid_json(self):
        config = generate_claude_desktop_config()
        # Should be JSON-serializable
        json_str = json.dumps(config)
        parsed = json.loads(json_str)
        assert parsed == config
