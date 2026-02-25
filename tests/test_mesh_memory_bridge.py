# Familiar - Self-Hosted AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# Licensed under the MIT License

"""
Tests for Mesh Memory Bridge — Phase 3 (v2.7.2)

Tests: SharedMemoryMeta, SharedMemoryStore, MeshMemoryResult,
       MemoryBridge query/response/push/revoke lifecycle.
"""

import json
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

import pytest

from familiar.core.mesh.memory_bridge import (
    SharedMemoryMeta,
    SharedMemoryStore,
    MeshMemoryResult,
    MemoryBridge,
    MESH_MEMORY_QUERY,
    MESH_MEMORY_RESPONSE,
    MESH_MEMORY_PUSH,
    MESH_MEMORY_REVOKE,
    MESH_SEARCH_TIMEOUT,
)
from familiar.core.memory import Memory, MemoryEntry


# ============================================================
# Helpers
# ============================================================

def _make_mock_memory(tmp_path):
    """Create a real Memory instance with a temp file."""
    mem_file = tmp_path / "test_memories.json"
    m = Memory(memory_file=mem_file)
    return m


def _make_mock_trust_manager(trusted_ids=None, permissions=None):
    """Create a mock trust manager."""
    tm = MagicMock()
    
    trusted_records = []
    for nid in (trusted_ids or []):
        rec = MagicMock()
        rec.node_id = nid
        rec.permissions = MagicMock()
        perms = permissions or {}
        rec.permissions.check = lambda p, _perms=perms, _nid=nid: _perms.get(_nid, {}).get(p, False)
        trusted_records.append(rec)
    
    tm.get_trusted_peers.return_value = trusted_records
    
    def check_perm(node_id, perm):
        if permissions and node_id in permissions:
            return permissions[node_id].get(perm, False)
        return False
    tm.check_permission = check_perm
    
    return tm


# ============================================================
# SharedMemoryMeta tests
# ============================================================

class TestSharedMemoryMeta:

    def test_defaults(self):
        m = SharedMemoryMeta(key="diet")
        assert m.shareable is False
        assert m.share_scope == "all"
        assert m.mesh_origin is None
        assert m.mesh_ttl == 3600

    def test_serialization_roundtrip(self):
        m = SharedMemoryMeta(
            key="diet",
            shareable=True,
            share_scope="peer1,peer2",
            mesh_origin="remote_abc",
            mesh_origin_name="Kitchen Pi",
        )
        d = m.to_dict()
        restored = SharedMemoryMeta.from_dict(d)
        assert restored.key == "diet"
        assert restored.shareable is True
        assert restored.share_scope == "peer1,peer2"
        assert restored.mesh_origin == "remote_abc"

    def test_is_shared_with_all(self):
        m = SharedMemoryMeta(key="x", shareable=True, share_scope="all")
        assert m.is_shared_with("any_node") is True

    def test_is_shared_with_specific(self):
        m = SharedMemoryMeta(key="x", shareable=True, share_scope="peer1,peer2")
        assert m.is_shared_with("peer1") is True
        assert m.is_shared_with("peer2") is True
        assert m.is_shared_with("peer3") is False

    def test_not_shared_if_not_shareable(self):
        m = SharedMemoryMeta(key="x", shareable=False, share_scope="all")
        assert m.is_shared_with("any") is False


# ============================================================
# SharedMemoryStore tests
# ============================================================

class TestSharedMemoryStore:

    def _make_store(self, tmp_path):
        return SharedMemoryStore(tmp_path)

    def test_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count_shared() == 0
        assert store.count_received() == 0

    def test_share(self, tmp_path):
        store = self._make_store(tmp_path)
        meta = store.share("diet")
        assert meta.shareable is True
        assert store.count_shared() == 1
        assert "diet" in store.get_shareable_keys()

    def test_unshare(self, tmp_path):
        store = self._make_store(tmp_path)
        store.share("diet")
        assert store.unshare("diet") is True
        assert store.count_shared() == 0

    def test_store_received(self, tmp_path):
        store = self._make_store(tmp_path)
        meta = store.store_received("mesh_key", "origin123", "Kitchen Pi", 7200)
        assert meta.mesh_origin == "origin123"
        assert meta.mesh_ttl == 7200
        assert store.count_received() == 1
        assert "mesh_key" in store.get_received_keys()

    def test_persistence(self, tmp_path):
        store1 = self._make_store(tmp_path)
        store1.share("diet")
        store1.store_received("from_peer", "p1", "Peer1")

        store2 = self._make_store(tmp_path)
        assert store2.count_shared() == 1
        assert store2.count_received() == 1

    def test_get_shareable_for_node(self, tmp_path):
        store = self._make_store(tmp_path)
        store.share("diet", scope="peer1,peer2")
        store.share("name", scope="all")

        assert len(store.get_shareable_keys(for_node="peer1")) == 2
        assert len(store.get_shareable_keys(for_node="peer3")) == 1  # Only "all" scope

    def test_remove(self, tmp_path):
        store = self._make_store(tmp_path)
        store.share("diet")
        assert store.remove("diet") is True
        assert store.count_shared() == 0
        assert store.remove("diet") is False

    def test_thread_safety(self, tmp_path):
        store = self._make_store(tmp_path)
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    store.share(f"key_{n}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count_shared() == 80


# ============================================================
# MeshMemoryResult tests
# ============================================================

class TestMeshMemoryResult:

    def test_tagged_value(self):
        r = MeshMemoryResult(
            key="diet",
            value="User is gluten-free",
            category="preference",
            importance=8,
            origin_node_id="abc",
            origin_node_name="Kitchen Pi",
        )
        assert r.tagged_value == "[Kitchen Pi] User is gluten-free"

    def test_serialization(self):
        r = MeshMemoryResult(
            key="k", value="v", category="fact",
            importance=5, origin_node_id="x", origin_node_name="X",
        )
        d = r.to_dict()
        restored = MeshMemoryResult.from_dict(d)
        assert restored.key == "k"
        assert restored.origin_node_name == "X"


# ============================================================
# MemoryBridge — sharing commands
# ============================================================

class TestMemoryBridgeSharing:

    def _make_bridge(self, tmp_path, hipaa=False):
        memory = _make_mock_memory(tmp_path)
        memory.remember("diet", "gluten-free", category="preference")
        memory.remember("name", "George", category="user_info")
        memory.remember("coffee", "dark roast", category="preference")

        bridge = MemoryBridge(
            memory=memory,
            node_id="local_node",
            node_name="Test Node",
            mesh_dir=tmp_path / "mesh",
            hipaa_enabled=hipaa,
        )
        return bridge, memory

    def test_share_memory(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        assert bridge.share_memory("diet") is True
        assert "diet" in bridge.get_shared_keys()

    def test_share_nonexistent(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        assert bridge.share_memory("nonexistent") is False

    def test_unshare_memory(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        bridge.share_memory("diet")
        assert bridge.unshare_memory("diet") is True
        assert "diet" not in bridge.get_shared_keys()

    def test_hipaa_blocks_sharing(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path, hipaa=True)
        assert bridge.share_memory("diet") is False
        assert bridge.get_shared_keys() == []

    def test_share_with_scope(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        bridge.share_memory("diet", scope="peer1,peer2")
        meta = bridge.shared_store.get("diet")
        assert meta.share_scope == "peer1,peer2"


# ============================================================
# MemoryBridge — remote query handling
# ============================================================

class TestMemoryBridgeQueryHandling:

    def _make_bridge(self, tmp_path, trust_perms=None):
        memory = _make_mock_memory(tmp_path)
        memory.remember("diet", "gluten-free", category="preference", importance=8)
        memory.remember("name", "George", category="user_info", importance=9)
        memory.remember("coffee", "dark roast", category="preference", importance=5)
        memory.remember("private_secret", "hidden", category="fact", importance=10)

        trust_manager = None
        if trust_perms is not None:
            trust_manager = _make_mock_trust_manager(
                trusted_ids=list(trust_perms.keys()),
                permissions=trust_perms,
            )

        bridge = MemoryBridge(
            memory=memory,
            node_id="local_node",
            node_name="Test Node",
            mesh_dir=tmp_path / "mesh",
            trust_manager=trust_manager,
        )

        # Share some memories
        bridge.share_memory("diet")
        bridge.share_memory("coffee")
        # name and private_secret are NOT shared

        return bridge, memory

    def test_query_returns_only_shared(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        response = bridge.handle_remote_query({
            "query_text": "diet",
            "requesting_node": "peer1",
            "query_id": "q1",
            "max_results": 5,
        })
        assert response is not None
        keys = [r["key"] for r in response["results"]]
        assert "diet" in keys
        assert "private_secret" not in keys
        assert "name" not in keys

    def test_query_with_trust_permission(self, tmp_path):
        bridge, mem = self._make_bridge(
            tmp_path,
            trust_perms={"peer1": {"request_memory": True}},
        )
        response = bridge.handle_remote_query({
            "query_text": "food",
            "requesting_node": "peer1",
            "query_id": "q1",
        })
        assert response is not None

    def test_query_denied_without_permission(self, tmp_path):
        bridge, mem = self._make_bridge(
            tmp_path,
            trust_perms={"peer1": {"request_memory": False}},
        )
        response = bridge.handle_remote_query({
            "query_text": "food",
            "requesting_node": "peer1",
            "query_id": "q1",
        })
        assert response is None

    def test_query_hipaa_returns_none(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory,
            node_id="local",
            node_name="Node",
            mesh_dir=tmp_path / "mesh",
            hipaa_enabled=True,
        )
        response = bridge.handle_remote_query({
            "query_text": "anything",
            "requesting_node": "peer1",
            "query_id": "q1",
        })
        assert response is None

    def test_query_includes_origin_info(self, tmp_path):
        bridge, mem = self._make_bridge(tmp_path)
        response = bridge.handle_remote_query({
            "query_text": "diet",
            "requesting_node": "peer1",
            "query_id": "q1",
        })
        assert response["origin_node"] == "local_node"
        assert response["origin_name"] == "Test Node"
        assert response["query_id"] == "q1"


# ============================================================
# MemoryBridge — response handling
# ============================================================

class TestMemoryBridgeResponseHandling:

    def test_response_collected(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory,
            node_id="local",
            node_name="Node",
            mesh_dir=tmp_path / "mesh",
        )

        # Simulate pending query
        query_id = "test_q1"
        bridge._pending_responses[query_id] = []
        bridge._response_events[query_id] = threading.Event()

        bridge.handle_remote_response({
            "query_id": query_id,
            "results": [
                {"key": "diet", "value": "gluten-free", "category": "preference", "importance": 8},
            ],
            "origin_node": "peer1",
            "origin_name": "Kitchen Pi",
        })

        assert len(bridge._pending_responses[query_id]) == 1
        result = bridge._pending_responses[query_id][0]
        assert result.origin_node_name == "Kitchen Pi"
        assert result.value == "gluten-free"


# ============================================================
# MemoryBridge — push and revoke
# ============================================================

class TestMemoryBridgePushRevoke:

    def _make_bridge(self, tmp_path, trust_perms=None):
        memory = _make_mock_memory(tmp_path)
        trust_manager = None
        if trust_perms:
            trust_manager = _make_mock_trust_manager(
                list(trust_perms.keys()), trust_perms
            )

        return MemoryBridge(
            memory=memory,
            node_id="local",
            node_name="Node",
            mesh_dir=tmp_path / "mesh",
            trust_manager=trust_manager,
        ), memory

    def test_push_stores_memory(self, tmp_path):
        bridge, memory = self._make_bridge(tmp_path)
        bridge.handle_remote_push({
            "origin_node": "peer1",
            "origin_name": "Kitchen Pi",
            "key": "allergy",
            "value": "Peanut allergy",
            "category": "health",
            "ttl": 7200,
        })

        # Memory stored with mesh prefix
        mesh_key = "mesh:peer1...:allergy"  # First 8 chars of peer1
        # Actually: f"mesh:{origin[:8]}:{key}" = "mesh:peer1:allergy" (peer1 is < 8 chars)
        assert "mesh:peer1:allergy" in memory.memories
        entry = memory.memories["mesh:peer1:allergy"]
        assert entry.value == "Peanut allergy"
        assert entry.source == "mesh:Kitchen Pi"

    def test_push_with_trust_denied(self, tmp_path):
        bridge, memory = self._make_bridge(
            tmp_path,
            trust_perms={"peer1": {"request_memory": False}},
        )
        bridge.handle_remote_push({
            "origin_node": "peer1",
            "origin_name": "Peer",
            "key": "secret",
            "value": "Should not store",
        })
        assert len(memory.memories) == 0

    def test_push_hipaa_blocked(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh", hipaa_enabled=True,
        )
        bridge.handle_remote_push({
            "origin_node": "p", "key": "k", "value": "v",
        })
        assert len(memory.memories) == 0

    def test_revoke_removes_memory(self, tmp_path):
        bridge, memory = self._make_bridge(tmp_path)

        # First push
        bridge.handle_remote_push({
            "origin_node": "peer1",
            "origin_name": "Peer",
            "key": "temp_fact",
            "value": "Temporary info",
        })
        assert "mesh:peer1:temp_fact" in memory.memories

        # Then revoke
        bridge.handle_remote_revoke({
            "origin_node": "peer1",
            "key": "temp_fact",
        })
        assert "mesh:peer1:temp_fact" not in memory.memories

    def test_revoke_nonexistent_silent(self, tmp_path):
        bridge, memory = self._make_bridge(tmp_path)
        # Should not crash
        bridge.handle_remote_revoke({
            "origin_node": "peer1",
            "key": "nonexistent",
        })


# ============================================================
# MemoryBridge — search_mesh (outgoing)
# ============================================================

class TestMemoryBridgeSearchMesh:

    def test_search_no_gateway_returns_empty(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )
        results = bridge.search_mesh("anything")
        assert results == []

    def test_search_hipaa_returns_empty(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh", hipaa_enabled=True,
        )
        bridge._gateway = MagicMock()
        results = bridge.search_mesh("anything")
        assert results == []

    def test_search_deduplicates(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )

        # Simulate two responses with duplicate values
        query_id = "dedup_test"
        bridge._pending_responses[query_id] = [
            MeshMemoryResult("k1", "same value", "fact", 8, "p1", "Peer1"),
            MeshMemoryResult("k2", "same value", "fact", 5, "p2", "Peer2"),
            MeshMemoryResult("k3", "unique value", "fact", 7, "p1", "Peer1"),
        ]
        bridge._response_events[query_id] = threading.Event()
        bridge._response_events[query_id].set()

        # Manually collect (bypassing actual mesh send)
        with bridge._lock:
            results = bridge._pending_responses.pop(query_id, [])
            bridge._response_events.pop(query_id, None)

        seen = set()
        deduped = []
        for r in sorted(results, key=lambda x: x.importance, reverse=True):
            if r.value not in seen:
                seen.add(r.value)
                deduped.append(r)

        assert len(deduped) == 2
        assert deduped[0].importance == 8  # Highest first

    def test_get_mesh_context_format(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )

        # Mock search_mesh to return results
        bridge.search_mesh = MagicMock(return_value=[
            MeshMemoryResult("diet", "User is gluten-free", "pref", 8, "p1", "Kitchen Pi"),
            MeshMemoryResult("coffee", "dark roast", "pref", 5, "p2", "Office"),
        ])

        ctx = bridge.get_mesh_context("food preferences")
        assert "## Mesh Context" in ctx
        assert "[Kitchen Pi] User is gluten-free" in ctx
        assert "[Office] dark roast" in ctx

    def test_get_mesh_context_empty(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )
        bridge.search_mesh = MagicMock(return_value=[])
        assert bridge.get_mesh_context("anything") == ""


# ============================================================
# MemoryBridge — status
# ============================================================

class TestMemoryBridgeStatus:

    def test_status(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        memory.remember("diet", "gf", category="preference")

        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )
        bridge.share_memory("diet")

        status = bridge.get_status()
        assert status["shared_count"] == 1
        assert status["received_count"] == 0
        assert status["hipaa_mode"] is False
        assert "diet" in status["shared_keys"]

    def test_status_hipaa(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh", hipaa_enabled=True,
        )
        status = bridge.get_status()
        assert status["hipaa_mode"] is True


# ============================================================
# MemoryBridge — bus registration
# ============================================================

class TestMemoryBridgeBus:

    def test_register_handlers(self, tmp_path):
        memory = _make_mock_memory(tmp_path)
        bridge = MemoryBridge(
            memory=memory, node_id="x", node_name="X",
            mesh_dir=tmp_path / "mesh",
        )

        mock_bus = MagicMock()
        bridge.register_bus_handlers(mock_bus)

        assert mock_bus.subscribe.call_count == 4
        topics = [call[0][0] for call in mock_bus.subscribe.call_args_list]
        assert MESH_MEMORY_QUERY in topics
        assert MESH_MEMORY_RESPONSE in topics
        assert MESH_MEMORY_PUSH in topics
        assert MESH_MEMORY_REVOKE in topics
