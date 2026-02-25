# Familiar - Self-Hosted AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# Licensed under the MIT License

"""
Tests for Orchestrated Delegation — Phase 4 (v2.7.3)

Tests: RemoteAgent, MeshSkillEndpoint, auto-registration,
       orchestration strategies with remote agents.
"""

import asyncio
import time
import threading
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from familiar.core.mesh.remote_agent import RemoteAgent
from familiar.core.mesh.mesh_skill_endpoint import (
    MeshSkillEndpoint,
    register_remote_skills,
    unregister_remote_skills,
)
from familiar.core.orchestration import (
    AgentBase,
    AgentRole,
    AgentStatus,
    AgentRegistry,
    Task,
    TaskResult,
    TaskStatus,
    Orchestrator,
    OrchestrationStrategy,
    FunctionAgent,
)


# ============================================================
# Helpers
# ============================================================

def _make_mock_gateway(connected_peers=None):
    """Create a mock MeshGateway."""
    gw = MagicMock()
    gw.peer_gateways = {pid: MagicMock() for pid in (connected_peers or [])}
    gw.request_peer_tool = AsyncMock(return_value="Remote result")
    return gw


def _make_mock_trust(permissions=None):
    """Create a mock trust manager."""
    tm = MagicMock()
    perms = permissions or {}
    tm.check_permission = lambda nid, perm: perms.get(nid, {}).get(perm, False)
    return tm


# ============================================================
# RemoteAgent tests
# ============================================================

class TestRemoteAgent:

    def test_creation(self):
        ra = RemoteAgent(
            node_id="peer1",
            node_name="Kitchen Pi",
            skills=["meal_planner", "recipe_search"],
        )
        assert ra.id == "remote:peer1"
        assert ra.name == "Kitchen Pi (remote)"
        assert ra.role == AgentRole.SPECIALIST
        assert ra.capabilities == ["meal_planner", "recipe_search"]
        assert ra.node_id == "peer1"
        assert ra.status == AgentStatus.IDLE

    def test_is_connected_true(self):
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", gateway=gw)
        assert ra.is_connected is True

    def test_is_connected_false_no_gateway(self):
        ra = RemoteAgent("peer1", "Pi")
        assert ra.is_connected is False

    def test_is_connected_false_peer_gone(self):
        gw = _make_mock_gateway(["peer2"])  # peer1 not in list
        ra = RemoteAgent("peer1", "Pi", gateway=gw)
        assert ra.is_connected is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="Meal plan: Monday pasta, Tuesday salad")

        ra = RemoteAgent("peer1", "Pi", skills=["meal_planner"], gateway=gw)
        task = Task(description="Plan meals", input_data={"skill": "meal_planner"})

        result = await ra.execute(task)
        assert result.success is True
        assert "Meal plan" in result.output
        assert result.agent_id == "remote:peer1"
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio
    async def test_execute_disconnected(self):
        gw = _make_mock_gateway([])  # Empty — peer not connected
        ra = RemoteAgent("peer1", "Pi", gateway=gw)
        task = Task(description="Do something")

        result = await ra.execute(task)
        assert result.success is False
        assert "disconnected" in result.error

    @pytest.mark.asyncio
    async def test_execute_permission_denied(self):
        gw = _make_mock_gateway(["peer1"])
        tm = _make_mock_trust({"peer1": {"delegate_tasks": False}})
        ra = RemoteAgent("peer1", "Pi", gateway=gw, trust_manager=tm)
        task = Task(description="Do something")

        result = await ra.execute(task)
        assert result.success is False
        assert "permission" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_permission(self):
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="Done")
        tm = _make_mock_trust({"peer1": {"delegate_tasks": True}})
        ra = RemoteAgent("peer1", "Pi", skills=["task"], gateway=gw, trust_manager=tm)
        task = Task(description="Do task", input_data={"skill": "task"})

        result = await ra.execute(task)
        assert result.success is True

    @pytest.mark.asyncio
    async def test_execute_resets_status(self):
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="ok")
        ra = RemoteAgent("peer1", "Pi", skills=["x"], gateway=gw)

        assert ra.status == AgentStatus.IDLE
        task = Task(description="test")
        await ra.execute(task)
        assert ra.status == AgentStatus.IDLE

    @pytest.mark.asyncio
    async def test_execute_exception(self):
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(side_effect=RuntimeError("Network error"))
        ra = RemoteAgent("peer1", "Pi", skills=["x"], gateway=gw)
        task = Task(description="test")

        result = await ra.execute(task)
        assert result.success is False
        assert "Network error" in result.error

    def test_can_handle_connected(self):
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        task = Task(description="cook", metadata={"required_capabilities": ["cook"]})
        assert ra.can_handle(task) is True

    def test_can_handle_disconnected(self):
        gw = _make_mock_gateway([])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        task = Task(description="cook")
        assert ra.can_handle(task) is False

    def test_can_handle_missing_capability(self):
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        task = Task(description="fly", metadata={"required_capabilities": ["fly"]})
        assert ra.can_handle(task) is False

    def test_skill_matching(self):
        ra = RemoteAgent("p1", "Pi", skills=["meal_planner", "recipe_search"])
        assert ra._match_skill("plan meals using meal_planner") == "meal_planner"
        assert ra._match_skill("search for recipe_search results") == "recipe_search"
        assert ra._match_skill("something generic") == "meal_planner"  # First skill

    def test_get_info(self):
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        info = ra.get_info()
        assert info["remote"] is True
        assert info["node_id"] == "peer1"
        assert info["connected"] is True
        assert info["skills"] == ["cook"]


# ============================================================
# MeshSkillEndpoint tests
# ============================================================

class TestMeshSkillEndpoint:

    def test_creation(self):
        ep = MeshSkillEndpoint(
            skill_name="meal_planner",
            node_id="peer1",
            node_name="Kitchen Pi",
        )
        assert ep.skill_name == "meal_planner"
        assert ep.node_id == "peer1"
        assert ep.metadata["remote"] is True
        assert "invoke" in ep.actions

    def test_is_available_connected(self):
        gw = _make_mock_gateway(["peer1"])
        ep = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw)
        assert ep.is_available is True

    def test_is_available_disconnected(self):
        gw = _make_mock_gateway([])
        ep = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw)
        assert ep.is_available is False

    def test_invoke_disconnected_raises(self):
        gw = _make_mock_gateway([])
        ep = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw)
        with pytest.raises(ConnectionError):
            ep.invoke({"input": "test"})

    def test_invoke_permission_denied(self):
        gw = _make_mock_gateway(["peer1"])
        tm = _make_mock_trust({"peer1": {"invoke_skills": False}})
        ep = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw, trust_manager=tm)
        with pytest.raises(PermissionError):
            ep.invoke({"input": "test"})

    def test_get_schema(self):
        ep = MeshSkillEndpoint(
            "meal_planner", "abcdef1234567890", "Kitchen Pi",
            description="Plans meals",
        )
        schema = ep.get_schema()
        assert schema["name"] == "remote:abcdef12:meal_planner"
        assert schema["_remote"] is True
        assert schema["_source"] == "peer:abcdef1234567890"
        assert schema["description"] == "Plans meals"

    def test_repr(self):
        gw = _make_mock_gateway(["peer1"])
        ep = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw)
        assert "connected" in repr(ep)

        gw2 = _make_mock_gateway([])
        ep2 = MeshSkillEndpoint("skill", "peer1", "Pi", gateway=gw2)
        assert "disconnected" in repr(ep2)


# ============================================================
# register/unregister helpers
# ============================================================

class TestRegisterRemoteSkills:

    def test_register(self):
        bus = MagicMock()
        endpoints = register_remote_skills(
            bus=bus,
            node_id="peer1",
            node_name="Kitchen Pi",
            skills=[
                {"name": "meal_planner", "description": "Plans meals"},
                {"name": "recipe_search", "description": "Search recipes"},
            ],
            gateway=MagicMock(),
        )
        assert len(endpoints) == 2
        assert bus.register_skill.call_count == 2

    def test_register_skips_empty_names(self):
        bus = MagicMock()
        endpoints = register_remote_skills(
            bus=bus, node_id="p", node_name="P",
            skills=[{"name": ""}, {"name": "valid"}],
            gateway=MagicMock(),
        )
        assert len(endpoints) == 1

    def test_unregister(self):
        bus = MagicMock()
        bus._endpoints = {
            "remote:peer1234:skill_a": MagicMock(),
            "remote:peer1234:skill_b": MagicMock(),
            "local:other_skill": MagicMock(),
        }
        unregister_remote_skills(bus, "peer12345678")
        assert "local:other_skill" in bus._endpoints
        assert "remote:peer1234:skill_a" not in bus._endpoints
        assert "remote:peer1234:skill_b" not in bus._endpoints


# ============================================================
# AgentRegistry integration
# ============================================================

class TestRegistryIntegration:

    def test_register_remote_in_registry(self):
        registry = AgentRegistry()
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        agent_id = registry.register(ra)

        assert agent_id == "remote:peer1"
        assert registry.get("remote:peer1") is not None
        assert ra in registry.get_by_role(AgentRole.SPECIALIST)

    def test_unregister_remote(self):
        registry = AgentRegistry()
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        registry.register(ra)
        assert registry.unregister("remote:peer1") is True
        assert registry.get("remote:peer1") is None

    def test_find_agent_for_task(self):
        registry = AgentRegistry()
        gw = _make_mock_gateway(["peer1"])
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        registry.register(ra)

        task = Task(description="cook dinner", metadata={"required_capabilities": ["cook"]})
        found = registry.find_agent_for_task(task)
        assert found is not None
        assert found.id == "remote:peer1"

    def test_mixed_local_remote_agents(self):
        registry = AgentRegistry()
        gw = _make_mock_gateway(["peer1"])

        local = FunctionAgent(
            fn=lambda t: "local result",
            agent_id="local_agent",
            name="Local",
            role=AgentRole.GENERALIST,
            capabilities=["general"],
        )
        remote = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)

        registry.register(local)
        registry.register(remote)

        assert len(registry) == 2
        all_agents = registry.get_all()
        assert len(all_agents) == 2

        specialists = registry.get_by_role(AgentRole.SPECIALIST)
        assert len(specialists) == 1
        assert specialists[0].id == "remote:peer1"


# ============================================================
# Orchestrator integration (async tests)
# ============================================================

class TestOrchestratorIntegration:

    @pytest.mark.asyncio
    async def test_delegate_to_remote(self):
        """Orchestrator can delegate a task to a RemoteAgent."""
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="Remote cooked dinner")

        orchestrator = Orchestrator()
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        orchestrator.registry.register(ra)

        task = Task(description="Cook dinner", input_data={"skill": "cook"})
        result = await orchestrator.delegate(task, "remote:peer1")

        assert result.success is True
        assert result.output == "Remote cooked dinner"

    @pytest.mark.asyncio
    async def test_hierarchical_mixed_agents(self):
        """HIERARCHICAL with local supervisor + remote workers."""
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="Sub-result from remote")

        orchestrator = Orchestrator()

        # Local supervisor
        supervisor = FunctionAgent(
            fn=lambda t: f"Plan: delegate cooking. Aggregated: {t.input_data}",
            agent_id="supervisor",
            name="Boss",
            role=AgentRole.SUPERVISOR,
            capabilities=["planning"],
        )

        # Remote worker
        worker = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)

        orchestrator.registry.register(supervisor)
        orchestrator.registry.register(worker)

        task = Task(description="Make a meal plan")
        result = await orchestrator.execute(
            task,
            strategy=OrchestrationStrategy.HIERARCHICAL,
        )
        # Should complete (supervisor plans, worker executes, supervisor aggregates)
        assert result is not None

    @pytest.mark.asyncio
    async def test_parallel_mixed_agents(self):
        """PARALLEL with mixed local + remote agents."""
        gw = _make_mock_gateway(["peer1"])
        gw.request_peer_tool = AsyncMock(return_value="Remote parallel result")

        orchestrator = Orchestrator()

        local = FunctionAgent(
            fn=lambda t: "Local parallel result",
            agent_id="local",
            name="Local",
            role=AgentRole.GENERALIST,
        )
        remote = RemoteAgent("peer1", "Pi", skills=["task"], gateway=gw)

        orchestrator.registry.register(local)
        orchestrator.registry.register(remote)

        task = Task(description="Do parallel work")
        result = await orchestrator.execute(
            task,
            strategy=OrchestrationStrategy.PARALLEL,
        )
        assert result is not None
        assert result.tasks_completed >= 1

    @pytest.mark.asyncio
    async def test_delegate_handles_disconnect(self):
        """Delegation to disconnected remote should fail gracefully."""
        gw = _make_mock_gateway([])  # peer1 NOT connected

        orchestrator = Orchestrator()
        ra = RemoteAgent("peer1", "Pi", skills=["cook"], gateway=gw)
        orchestrator.registry.register(ra)

        task = Task(description="Cook")
        result = await orchestrator.delegate(task, "remote:peer1")
        assert result.success is False
        assert "disconnected" in result.error


# ============================================================
# Gateway registration wiring
# ============================================================

class TestGatewayRegistration:

    def test_gateway_has_orchestrator_ref(self):
        """MeshGateway can hold an orchestrator reference."""
        from familiar.core.mesh.gateway import MeshGateway
        # Can't fully instantiate without agent, but we can test the attribute
        assert hasattr(MeshGateway, '_orchestrator')

    def test_register_remote_agent_method(self):
        """MeshGateway._register_remote_agent creates RemoteAgent."""
        from familiar.core.mesh.gateway import MeshGateway

        gw = MagicMock(spec=MeshGateway)
        gw.config = MagicMock()
        gw.config.get = MagicMock(return_value="local_id")
        gw.trust_manager = None
        gw.peer_gateways = {"peer1": MagicMock()}

        orchestrator = Orchestrator()
        gw._orchestrator = orchestrator

        # Call the real method
        MeshGateway._register_remote_agent(gw, "peer1", "Kitchen Pi", [
            {"name": "meal_planner", "description": "Plans meals"},
        ])

        agent = orchestrator.registry.get("remote:peer1")
        assert agent is not None
        assert agent.name == "Kitchen Pi (remote)"
        assert "meal_planner" in agent.capabilities
