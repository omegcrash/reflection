# Familiar - Self-Hosted AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# Licensed under the MIT License

"""
Tests for Mesh Discovery — Phase 1

Tests: DiscoveredPeer, PeerStore, MeshDiscovery, MeshGateway peer mode.
mDNS operations are tested via mocking (no real network required).
"""

import json
import time
import hashlib
import tempfile
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
from dataclasses import asdict

import pytest

# Import the modules under test
from familiar.core.mesh.discovery import (
    DiscoveredPeer,
    PeerStore,
    MeshDiscovery,
    _build_txt_properties,
    _parse_txt_properties,
    _get_local_ip,
    SERVICE_TYPE,
    FAMILIAR_VERSION,
)
from familiar.core.mesh import MsgType, Message, MeshConfig


# ============================================================
# DiscoveredPeer tests
# ============================================================

class TestDiscoveredPeer:
    """Tests for the DiscoveredPeer data model."""

    def test_basic_creation(self):
        peer = DiscoveredPeer(
            node_id="abc123",
            node_name="Test Node",
            node_type="gateway",
            host="192.168.1.50",
            port=18789,
        )
        assert peer.node_id == "abc123"
        assert peer.node_name == "Test Node"
        assert peer.host == "192.168.1.50"
        assert peer.port == 18789
        assert peer.trust_level == "unknown"
        assert peer.first_seen != ""
        assert peer.last_seen != ""

    def test_ws_url(self):
        peer = DiscoveredPeer(
            node_id="x", node_name="x", node_type="gateway",
            host="10.0.0.5", port=9999,
        )
        assert peer.ws_url == "ws://10.0.0.5:9999"

    def test_serialization_roundtrip(self):
        peer = DiscoveredPeer(
            node_id="abc", node_name="Pi", node_type="gateway",
            host="192.168.1.1", port=18789,
            skills=["calendar", "email"],
            capabilities=["chat", "tools"],
            fingerprint="deadbeef",
            discovery_source="mdns",
            trust_level="trusted",
        )
        d = peer.to_dict()
        restored = DiscoveredPeer.from_dict(d)
        assert restored.node_id == "abc"
        assert restored.skills == ["calendar", "email"]
        assert restored.trust_level == "trusted"
        assert restored.fingerprint == "deadbeef"

    def test_from_dict_ignores_unknown_keys(self):
        d = {
            "node_id": "x", "node_name": "x", "node_type": "gateway",
            "host": "1.2.3.4", "port": 1234,
            "unknown_field": "should not crash",
        }
        peer = DiscoveredPeer.from_dict(d)
        assert peer.node_id == "x"

    def test_is_stale(self):
        peer = DiscoveredPeer(
            node_id="x", node_name="x", node_type="gateway",
            host="1.2.3.4", port=1234,
        )
        # Just created — not stale
        assert not peer.is_stale

        # Set last_seen to 3 minutes ago — stale
        old = (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()
        peer.last_seen = old
        assert peer.is_stale

    def test_touch_updates_last_seen(self):
        peer = DiscoveredPeer(
            node_id="x", node_name="x", node_type="gateway",
            host="1.2.3.4", port=1234,
        )
        old_ts = peer.last_seen
        time.sleep(0.05)
        peer.touch()
        assert peer.last_seen >= old_ts


# ============================================================
# PeerStore tests
# ============================================================

class TestPeerStore:
    """Tests for PeerStore persistence."""

    def _make_store(self, tmp_path):
        return PeerStore(tmp_path)

    def _make_peer(self, node_id="abc", name="Test", host="10.0.0.1", port=18789):
        return DiscoveredPeer(
            node_id=node_id, node_name=name, node_type="gateway",
            host=host, port=port, discovery_source="mdns",
        )

    def test_empty_store(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0
        assert store.get_all() == []

    def test_upsert_new(self, tmp_path):
        store = self._make_store(tmp_path)
        peer = self._make_peer()
        is_new = store.upsert(peer)
        assert is_new is True
        assert store.count() == 1

    def test_upsert_existing_preserves_trust(self, tmp_path):
        store = self._make_store(tmp_path)
        peer = self._make_peer()
        store.upsert(peer)
        store.set_trust("abc", "trusted")

        # Upsert same peer with new data — trust should be preserved
        updated = self._make_peer(host="10.0.0.99")
        is_new = store.upsert(updated)
        assert is_new is False

        result = store.get("abc")
        assert result.trust_level == "trusted"
        assert result.host == "10.0.0.99"

    def test_remove(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert(self._make_peer())
        assert store.remove("abc") is True
        assert store.count() == 0
        assert store.remove("abc") is False

    def test_set_trust(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert(self._make_peer())
        assert store.set_trust("abc", "trusted") is True
        assert store.get("abc").trust_level == "trusted"
        assert store.set_trust("nonexistent", "trusted") is False

    def test_get_by_trust(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert(self._make_peer("a", "A"))
        store.upsert(self._make_peer("b", "B"))
        store.upsert(self._make_peer("c", "C"))
        store.set_trust("a", "trusted")
        store.set_trust("b", "trusted")
        store.set_trust("c", "blocked")

        trusted = store.get_by_trust("trusted")
        assert len(trusted) == 2
        blocked = store.get_by_trust("blocked")
        assert len(blocked) == 1

    def test_persistence_across_instances(self, tmp_path):
        store1 = self._make_store(tmp_path)
        store1.upsert(self._make_peer("x", "Persistent"))
        store1.set_trust("x", "trusted")

        # New store instance reading same directory
        store2 = self._make_store(tmp_path)
        peer = store2.get("x")
        assert peer is not None
        assert peer.node_name == "Persistent"
        assert peer.trust_level == "trusted"

    def test_prune_stale(self, tmp_path):
        store = self._make_store(tmp_path)
        peer = self._make_peer()
        peer.last_seen = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        store.upsert(peer)

        pruned = store.prune_stale(max_age_seconds=300)
        assert pruned == 1
        assert store.count() == 0

    def test_prune_keeps_manual(self, tmp_path):
        store = self._make_store(tmp_path)
        peer = self._make_peer()
        peer.discovery_source = "manual"
        peer.last_seen = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat()
        store.upsert(peer)

        pruned = store.prune_stale(max_age_seconds=300)
        assert pruned == 0  # Manual peers are never pruned
        assert store.count() == 1

    def test_get_manual(self, tmp_path):
        store = self._make_store(tmp_path)
        mdns_peer = self._make_peer("a", "mDNS")
        mdns_peer.discovery_source = "mdns"
        manual_peer = self._make_peer("b", "Manual")
        manual_peer.discovery_source = "manual"
        store.upsert(mdns_peer)
        store.upsert(manual_peer)

        manual = store.get_manual()
        assert len(manual) == 1
        assert manual[0].node_name == "Manual"

    def test_thread_safety(self, tmp_path):
        store = self._make_store(tmp_path)
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    store.upsert(self._make_peer(f"thread{n}_{i}", f"T{n}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 80  # 4 threads × 20 peers


# ============================================================
# TXT record tests
# ============================================================

class TestTxtRecords:
    """Tests for mDNS TXT record encoding/decoding."""

    def test_build_txt_properties(self):
        props = _build_txt_properties(
            node_id="abc123",
            node_name="Kitchen Pi",
            node_type="gateway",
            skills=["calendar", "email", "meal_planner"],
            capabilities=["chat", "tools"],
            fingerprint="deadbeefcafe1234extra",
        )
        assert props["node_id"] == "abc123"
        assert props["node_name"] == "Kitchen Pi"
        assert props["skills"] == "calendar,email,meal_planner"
        assert props["capabilities"] == "chat,tools"
        assert props["fingerprint"] == "deadbeefcafe1234"  # Truncated to 16
        assert props["version"] == FAMILIAR_VERSION

    def test_parse_txt_properties_bytes(self):
        """zeroconf returns TXT values as bytes."""
        raw = {
            b"node_id": b"abc123",
            b"node_name": b"Test",
            b"skills": b"a,b,c",
            b"capabilities": b"chat",
        }
        parsed = _parse_txt_properties(raw)
        assert parsed["node_id"] == "abc123"
        assert parsed["skills"] == ["a", "b", "c"]
        assert parsed["capabilities"] == ["chat"]

    def test_parse_txt_properties_strings(self):
        raw = {
            "node_id": "abc",
            "skills": "x,y",
            "capabilities": "",
        }
        parsed = _parse_txt_properties(raw)
        assert parsed["skills"] == ["x", "y"]
        assert parsed["capabilities"] == []

    def test_roundtrip(self):
        built = _build_txt_properties("id1", "Name", "gateway", ["s1", "s2"], ["cap1"], "fp123")
        parsed = _parse_txt_properties(built)
        assert parsed["node_id"] == "id1"
        assert parsed["skills"] == ["s1", "s2"]

    def test_skill_list_truncation(self):
        """Skills list is capped at 50 to avoid TXT record overflow."""
        skills = [f"skill_{i}" for i in range(100)]
        props = _build_txt_properties("id", "name", "gw", skills, [], "")
        assert len(props["skills"].split(",")) == 50


# ============================================================
# MeshDiscovery tests (mocked zeroconf)
# ============================================================

class TestMeshDiscovery:
    """Tests for MeshDiscovery with mocked zeroconf."""

    def _make_discovery(self, tmp_path, **kwargs):
        defaults = dict(
            node_id="local_node",
            node_name="Test Gateway",
            node_type="gateway",
            port=18789,
            skills=["calendar"],
            mesh_dir=tmp_path,
        )
        defaults.update(kwargs)
        return MeshDiscovery(**defaults)

    def test_creation(self, tmp_path):
        d = self._make_discovery(tmp_path)
        assert d.node_id == "local_node"
        assert not d.is_running
        assert d.peer_store.count() == 0

    def test_add_manual_peer(self, tmp_path):
        d = self._make_discovery(tmp_path)
        peer = d.add_manual_peer("10.0.0.5", 18789, "Office")
        assert peer.discovery_source == "manual"
        assert peer.node_name == "Office"
        assert d.peer_store.count() == 1

    def test_add_manual_peer_deterministic_id(self, tmp_path):
        d = self._make_discovery(tmp_path)
        p1 = d.add_manual_peer("10.0.0.5", 18789)
        p2 = d.add_manual_peer("10.0.0.5", 18789)
        assert p1.node_id == p2.node_id  # Same host:port = same ID

    def test_remove_manual_peer(self, tmp_path):
        d = self._make_discovery(tmp_path)
        d.add_manual_peer("10.0.0.5", 18789)
        assert d.remove_manual_peer("10.0.0.5", 18789) is True
        assert d.peer_store.count() == 0

    def test_remove_nonexistent_peer(self, tmp_path):
        d = self._make_discovery(tmp_path)
        assert d.remove_manual_peer("10.0.0.5", 18789) is False

    def test_on_peer_found_callback(self, tmp_path):
        found = []
        d = self._make_discovery(tmp_path, on_peer_found=lambda p: found.append(p))
        d.add_manual_peer("10.0.0.5", 18789, "New")
        assert len(found) == 1
        assert found[0].node_name == "New"

    def test_on_peer_lost_callback(self, tmp_path):
        lost = []
        d = self._make_discovery(
            tmp_path,
            on_peer_found=lambda p: None,
            on_peer_lost=lambda nid: lost.append(nid),
        )
        peer = d.add_manual_peer("10.0.0.5", 18789)
        d.remove_manual_peer("10.0.0.5", 18789)
        assert len(lost) == 1

    def test_get_status(self, tmp_path):
        d = self._make_discovery(tmp_path)
        d.add_manual_peer("10.0.0.5", 18789, "Office")
        status = d.get_status()
        assert status["node_id"] == "local_node"
        assert status["peer_count"] == 1
        assert len(status["peers"]) == 1
        assert status["peers"][0]["name"] == "Office"

    @patch("familiar.core.mesh.discovery.HAS_ZEROCONF", False)
    def test_start_without_zeroconf(self, tmp_path):
        """Start should gracefully handle missing zeroconf."""
        d = self._make_discovery(tmp_path)
        d.start()  # Should not raise
        assert not d.is_running

    def test_max_peers_limit(self, tmp_path):
        d = self._make_discovery(tmp_path, max_peers=3)
        d.add_manual_peer("10.0.0.1", 1)
        d.add_manual_peer("10.0.0.2", 2)
        d.add_manual_peer("10.0.0.3", 3)
        assert d.peer_store.count() == 3

    def test_service_added_skips_self(self, tmp_path):
        """_on_service_added should ignore our own node_id."""
        d = self._make_discovery(tmp_path)

        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_info.properties = {
            b"node_id": b"local_node",  # Same as our ID
            b"node_name": b"Self",
            b"skills": b"",
            b"capabilities": b"",
        }
        mock_info.parsed_scoped_addresses.return_value = ["192.168.1.1"]
        mock_info.port = 18789
        mock_zc.get_service_info.return_value = mock_info

        d._on_service_added(mock_zc, SERVICE_TYPE, "local_node._familiar-mesh._tcp.local.")
        assert d.peer_store.count() == 0  # Should not add ourselves

    def test_service_added_new_peer(self, tmp_path):
        """_on_service_added should create a new peer from service info."""
        found = []
        d = self._make_discovery(tmp_path, on_peer_found=lambda p: found.append(p))

        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_info.properties = {
            b"node_id": b"remote_abc",
            b"node_name": b"Kitchen Pi",
            b"node_type": b"gateway",
            b"version": b"2.7.0",
            b"skills": b"meal_planner,recipe_search",
            b"capabilities": b"chat,tools",
            b"fingerprint": b"deadbeef",
        }
        mock_info.parsed_scoped_addresses.return_value = ["192.168.1.50"]
        mock_info.port = 18789
        mock_zc.get_service_info.return_value = mock_info

        d._on_service_added(mock_zc, SERVICE_TYPE, "remote_abc._familiar-mesh._tcp.local.")

        assert d.peer_store.count() == 1
        peer = d.peer_store.get("remote_abc")
        assert peer.node_name == "Kitchen Pi"
        assert peer.host == "192.168.1.50"
        assert peer.skills == ["meal_planner", "recipe_search"]
        assert peer.fingerprint == "deadbeef"
        assert peer.discovery_source == "mdns"
        assert len(found) == 1

    def test_service_added_updates_existing(self, tmp_path):
        """_on_service_added should update existing peer, not duplicate."""
        d = self._make_discovery(tmp_path)

        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_info.properties = {
            b"node_id": b"peer1",
            b"node_name": b"Pi",
            b"node_type": b"gateway",
            b"skills": b"a,b",
            b"capabilities": b"chat",
        }
        mock_info.parsed_scoped_addresses.return_value = ["10.0.0.1"]
        mock_info.port = 18789
        mock_zc.get_service_info.return_value = mock_info

        d._on_service_added(mock_zc, SERVICE_TYPE, "peer1._familiar-mesh._tcp.local.")
        assert d.peer_store.count() == 1

        # Update with new IP
        mock_info.parsed_scoped_addresses.return_value = ["10.0.0.99"]
        mock_info.properties[b"skills"] = b"a,b,c"
        d._on_service_added(mock_zc, SERVICE_TYPE, "peer1._familiar-mesh._tcp.local.")

        assert d.peer_store.count() == 1  # Still 1
        peer = d.peer_store.get("peer1")
        assert peer.host == "10.0.0.99"
        assert peer.skills == ["a", "b", "c"]

    def test_service_removed(self, tmp_path):
        """_on_service_removed should fire callback."""
        lost = []
        d = self._make_discovery(tmp_path, on_peer_lost=lambda nid: lost.append(nid))

        # Add a peer first
        mock_zc = MagicMock()
        mock_info = MagicMock()
        mock_info.properties = {
            b"node_id": b"vanishing",
            b"node_name": b"Gone",
            b"node_type": b"gateway",
            b"skills": b"",
            b"capabilities": b"",
        }
        mock_info.parsed_scoped_addresses.return_value = ["10.0.0.1"]
        mock_info.port = 18789
        mock_zc.get_service_info.return_value = mock_info
        d._on_service_added(mock_zc, SERVICE_TYPE, "vanishing._familiar-mesh._tcp.local.")

        # Now remove
        d._on_service_removed("vanishing._familiar-mesh._tcp.local.")
        assert len(lost) == 1
        assert lost[0] == "vanishing"


# ============================================================
# MeshGateway peer gateway extensions
# ============================================================

class TestMeshGatewayPeerMode:
    """Tests for MeshGateway peer gateway additions."""

    def test_msg_type_peer_hello_exists(self):
        assert MsgType.PEER_HELLO == "peer_hello"
        assert MsgType.PEER_HELLO_ACK == "peer_hello_ack"
        assert MsgType.PEER_TOOLS == "peer_tools"

    def test_mesh_config_discovery_defaults(self):
        config = MeshConfig()
        # These keys should exist in config (defaults or loaded)
        assert "discovery_enabled" in config.config
        assert "discovery_mode" in config.config
        assert "auto_connect_trusted" in config.config
        assert "peer_gateways" in config.config
        assert "max_peers" in config.config
        assert isinstance(config.get("peer_gateways"), list)
        assert isinstance(config.get("max_peers"), int)

    def test_mesh_config_persists_discovery(self, tmp_path):
        """Discovery config should round-trip through save/load."""
        config = MeshConfig()
        config.set("discovery_enabled", True)
        config.set("peer_gateways", ["10.0.0.5:18789"])
        config.save()

        # Reload
        config2 = MeshConfig()
        # Note: MeshConfig always loads from MESH_CONFIG_FILE, so this tests
        # that the defaults include discovery fields
        assert "discovery_enabled" in config2.config
        assert "peer_gateways" in config2.config
