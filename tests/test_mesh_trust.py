# Familiar - Self-Hosted AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# Licensed under the MIT License

"""
Tests for Mesh Trust — Phase 2

Tests: PermissionMatrix, TrustRecord, TrustStore,
       verification codes, MeshTrustManager lifecycle.
Crypto tests skipped if pynacl not installed.
"""

import json
import hashlib
import threading
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from familiar.core.mesh.trust import (
    PermissionMatrix,
    TrustRecord,
    TrustStore,
    MeshTrustManager,
    compute_verification_code,
    compute_fingerprint,
    format_verification_code,
    HAS_SECURE_TRANSPORT,
)


# ============================================================
# PermissionMatrix tests
# ============================================================

class TestPermissionMatrix:
    """Tests for the PermissionMatrix data model."""

    def test_defaults(self):
        pm = PermissionMatrix()
        assert pm.list_skills is True
        assert pm.invoke_skills is False
        assert pm.request_memory is False
        assert pm.delegate_tasks is False
        assert pm.share_memory_to is False

    def test_all_deny(self):
        pm = PermissionMatrix.all_deny()
        assert pm.list_skills is False
        assert pm.invoke_skills is False

    def test_all_grant(self):
        pm = PermissionMatrix.all_grant()
        assert pm.list_skills is True
        assert pm.invoke_skills is True
        assert pm.delegate_tasks is True

    def test_grant_valid(self):
        pm = PermissionMatrix()
        assert pm.grant("invoke_skills") is True
        assert pm.invoke_skills is True

    def test_grant_invalid(self):
        pm = PermissionMatrix()
        assert pm.grant("nonexistent_perm") is False

    def test_revoke(self):
        pm = PermissionMatrix.all_grant()
        assert pm.revoke("delegate_tasks") is True
        assert pm.delegate_tasks is False

    def test_check(self):
        pm = PermissionMatrix()
        assert pm.check("list_skills") is True
        assert pm.check("invoke_skills") is False
        assert pm.check("nonexistent") is False

    def test_granted_list(self):
        pm = PermissionMatrix()
        pm.grant("invoke_skills")
        granted = pm.granted_list
        assert "list_skills" in granted
        assert "invoke_skills" in granted
        assert "delegate_tasks" not in granted

    def test_serialization_roundtrip(self):
        pm = PermissionMatrix()
        pm.grant("invoke_skills")
        pm.grant("delegate_tasks")
        d = pm.to_dict()
        restored = PermissionMatrix.from_dict(d)
        assert restored.invoke_skills is True
        assert restored.delegate_tasks is True
        assert restored.request_memory is False

    def test_from_dict_ignores_unknown(self):
        d = {"list_skills": True, "invoke_skills": True, "unknown_field": True}
        pm = PermissionMatrix.from_dict(d)
        assert pm.list_skills is True
        assert pm.invoke_skills is True


# ============================================================
# TrustRecord tests
# ============================================================

class TestTrustRecord:

    def test_creation(self):
        rec = TrustRecord(
            node_id="abc",
            node_name="Test",
            identity_fingerprint="deadbeef" * 8,
        )
        assert rec.trust_level == "unknown"
        assert rec.permissions.list_skills is True
        assert rec.trusted_at is None

    def test_serialization_roundtrip(self):
        rec = TrustRecord(
            node_id="abc",
            node_name="Pi",
            identity_fingerprint="ff" * 32,
            trust_level="trusted",
            verification_code="847291",
        )
        rec.permissions.grant("invoke_skills")
        d = rec.to_dict()
        restored = TrustRecord.from_dict(d)
        assert restored.node_id == "abc"
        assert restored.trust_level == "trusted"
        assert restored.permissions.invoke_skills is True
        assert restored.verification_code == "847291"

    def test_from_dict_ignores_unknown(self):
        d = {
            "node_id": "x", "node_name": "x",
            "identity_fingerprint": "aa" * 32,
            "extra_field": "ignored",
        }
        rec = TrustRecord.from_dict(d)
        assert rec.node_id == "x"


# ============================================================
# Verification code tests
# ============================================================

class TestVerificationCode:

    def test_deterministic(self):
        key_a = b"a" * 32
        key_b = b"b" * 32
        code1 = compute_verification_code(key_a, key_b)
        code2 = compute_verification_code(key_a, key_b)
        assert code1 == code2

    def test_order_independent(self):
        key_a = b"a" * 32
        key_b = b"b" * 32
        code_ab = compute_verification_code(key_a, key_b)
        code_ba = compute_verification_code(key_b, key_a)
        assert code_ab == code_ba

    def test_six_digits(self):
        code = compute_verification_code(b"x" * 32, b"y" * 32)
        assert len(code) == 6
        assert code.isdigit()

    def test_different_keys_different_codes(self):
        code1 = compute_verification_code(b"a" * 32, b"b" * 32)
        code2 = compute_verification_code(b"a" * 32, b"c" * 32)
        # Extremely unlikely to be equal (1 in 1M chance)
        assert code1 != code2

    def test_format(self):
        assert format_verification_code("847291") == "847 291"
        assert format_verification_code("000000") == "000 000"


class TestFingerprint:

    def test_deterministic(self):
        key = b"test_key_bytes" * 3
        fp1 = compute_fingerprint(key)
        fp2 = compute_fingerprint(key)
        assert fp1 == fp2

    def test_is_sha256_hex(self):
        fp = compute_fingerprint(b"key")
        assert len(fp) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in fp)


# ============================================================
# TrustStore tests
# ============================================================

class TestTrustStore:

    def _make_store(self, tmp_path):
        return TrustStore(tmp_path)

    def _make_record(self, node_id="abc", name="Test"):
        return TrustRecord(
            node_id=node_id,
            node_name=name,
            identity_fingerprint=hashlib.sha256(node_id.encode()).hexdigest(),
        )

    def test_empty(self, tmp_path):
        store = self._make_store(tmp_path)
        assert store.count() == 0

    def test_upsert_and_get(self, tmp_path):
        store = self._make_store(tmp_path)
        rec = self._make_record()
        store.upsert(rec)
        assert store.count() == 1
        assert store.get("abc").node_name == "Test"

    def test_remove(self, tmp_path):
        store = self._make_store(tmp_path)
        store.upsert(self._make_record())
        assert store.remove("abc") is True
        assert store.count() == 0
        assert store.remove("abc") is False

    def test_persistence(self, tmp_path):
        store1 = self._make_store(tmp_path)
        rec = self._make_record()
        rec.trust_level = "trusted"
        rec.permissions.grant("invoke_skills")
        store1.upsert(rec)

        store2 = self._make_store(tmp_path)
        loaded = store2.get("abc")
        assert loaded.trust_level == "trusted"
        assert loaded.permissions.invoke_skills is True

    def test_get_trusted(self, tmp_path):
        store = self._make_store(tmp_path)
        r1 = self._make_record("a", "A")
        r1.trust_level = "trusted"
        r2 = self._make_record("b", "B")
        r2.trust_level = "pending"
        r3 = self._make_record("c", "C")
        r3.trust_level = "trusted"
        store.upsert(r1)
        store.upsert(r2)
        store.upsert(r3)
        assert len(store.get_trusted()) == 2

    def test_thread_safety(self, tmp_path):
        store = self._make_store(tmp_path)
        errors = []

        def writer(n):
            try:
                for i in range(20):
                    store.upsert(self._make_record(f"t{n}_{i}", f"Thread{n}_{i}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert store.count() == 80


# ============================================================
# MeshTrustManager tests (without crypto)
# ============================================================

class TestMeshTrustManagerNoCrypto:
    """Tests that work without pynacl."""

    def _make_manager(self, tmp_path):
        # Force no crypto for these tests
        with patch("familiar.core.mesh.trust.HAS_SECURE_TRANSPORT", False):
            return MeshTrustManager(mesh_dir=tmp_path)

    def test_creation(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert not mgr.has_crypto
        assert mgr.get_our_fingerprint() == ""

    def test_initiate_handshake_no_crypto(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        result = mgr.initiate_handshake("peer1", "Test Peer", {"ik": "aa"})
        assert result is None
        # Should still create a trust record
        rec = mgr.get_trust_record("peer1")
        assert rec is not None
        assert rec.trust_level == "pending"

    def test_approve(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        assert mgr.approve("peer1") is True
        assert mgr.is_trusted("peer1") is True

    def test_approve_nonexistent(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.approve("nobody") is False

    def test_block(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.approve("peer1")
        assert mgr.block("peer1") is True
        assert mgr.is_blocked("peer1") is True
        assert mgr.is_trusted("peer1") is False

    def test_cannot_approve_blocked(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.block("peer1")
        assert mgr.approve("peer1") is False

    def test_unblock(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.block("peer1")
        assert mgr.unblock("peer1") is True
        rec = mgr.get_trust_record("peer1")
        assert rec.trust_level == "unknown"

    def test_grant_permission_trusted(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.approve("peer1")
        assert mgr.grant_permission("peer1", "invoke_skills") is True
        assert mgr.check_permission("peer1", "invoke_skills") is True

    def test_grant_permission_untrusted(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        # Still pending, not trusted
        assert mgr.grant_permission("peer1", "invoke_skills") is False

    def test_revoke_permission(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.approve("peer1")
        mgr.grant_permission("peer1", "invoke_skills")
        assert mgr.revoke_permission("peer1", "invoke_skills") is True
        assert mgr.check_permission("peer1", "invoke_skills") is False

    def test_check_permission_untrusted(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        # Not trusted — even if permission was somehow set, check returns False
        assert mgr.check_permission("peer1", "list_skills") is False

    def test_key_change_revokes_trust(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        mgr.approve("peer1")
        assert mgr.is_trusted("peer1") is True

        # Simulate key change by calling _create_trust_record with different fingerprint
        mgr._create_trust_record("peer1", "Peer", "new_fingerprint_" + "0" * 50, "123456")
        rec = mgr.get_trust_record("peer1")
        assert rec.trust_level == "unknown"  # Trust revoked

    def test_get_trusted_peers(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("a", "A", {})
        mgr.initiate_handshake("b", "B", {})
        mgr.initiate_handshake("c", "C", {})
        mgr.approve("a")
        mgr.approve("c")
        trusted = mgr.get_trusted_peers()
        assert len(trusted) == 2

    def test_get_status(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("a", "Alpha", {})
        mgr.approve("a")
        mgr.initiate_handshake("b", "Beta", {})
        mgr.block("b")
        status = mgr.get_status()
        assert status["total_peers"] == 2
        assert status["trusted"] == 1
        assert status["blocked"] == 1
        assert not status["has_crypto"]

    def test_encrypt_no_crypto(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.encrypt_for_peer("x", b"hello") is None
        assert mgr.decrypt_from_peer("x", b"data") is None
        assert mgr.has_session("x") is False

    def test_get_verification_code(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        mgr.initiate_handshake("peer1", "Peer", {})
        # Without crypto, verification code is empty
        code = mgr.get_verification_code("peer1")
        # Should not crash
        assert code is None or isinstance(code, str)

    def test_persistence_across_instances(self, tmp_path):
        with patch("familiar.core.mesh.trust.HAS_SECURE_TRANSPORT", False):
            mgr1 = MeshTrustManager(mesh_dir=tmp_path)
            mgr1.initiate_handshake("peer1", "Persistent", {})
            mgr1.approve("peer1")
            mgr1.grant_permission("peer1", "invoke_skills")

        with patch("familiar.core.mesh.trust.HAS_SECURE_TRANSPORT", False):
            mgr2 = MeshTrustManager(mesh_dir=tmp_path)
            assert mgr2.is_trusted("peer1") is True
            assert mgr2.check_permission("peer1", "invoke_skills") is True


# ============================================================
# MeshTrustManager tests WITH crypto (skipped if no pynacl)
# ============================================================

@pytest.mark.skipif(not HAS_SECURE_TRANSPORT, reason="pynacl not installed")
class TestMeshTrustManagerWithCrypto:
    """Tests requiring pynacl for real crypto operations."""

    def _make_manager(self, tmp_path, subdir="node"):
        return MeshTrustManager(mesh_dir=tmp_path / subdir)

    def test_has_crypto(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        assert mgr.has_crypto is True

    def test_fingerprint(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        fp = mgr.get_our_fingerprint()
        assert len(fp) == 16  # Truncated to 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_prekey_bundle_dict(self, tmp_path):
        mgr = self._make_manager(tmp_path)
        bundle = mgr.get_prekey_bundle_dict()
        assert bundle is not None
        assert "ik" in bundle
        assert "spk" in bundle
        assert "spk_sig" in bundle

    def test_full_handshake(self, tmp_path):
        """Two managers perform X3DH handshake."""
        alice = self._make_manager(tmp_path, "alice")
        bob = self._make_manager(tmp_path, "bob")

        # Alice gets Bob's bundle
        bob_bundle = bob.get_prekey_bundle_dict()

        # Alice initiates
        init_hex = alice.initiate_handshake("bob_id", "Bob", bob_bundle)
        assert init_hex is not None

        # Bob accepts
        alice_bundle = alice.get_prekey_bundle_dict()
        bob.accept_handshake("alice_id", "Alice", init_hex, alice_bundle["ik"])

        # Both should have sessions
        assert alice.has_session("bob_id")
        assert bob.has_session("alice_id")

    def test_verification_codes_match(self, tmp_path):
        """After handshake, both sides see the same verification code."""
        alice = self._make_manager(tmp_path, "alice")
        bob = self._make_manager(tmp_path, "bob")

        bob_bundle = bob.get_prekey_bundle_dict()
        init_hex = alice.initiate_handshake("bob_id", "Bob", bob_bundle)

        alice_bundle = alice.get_prekey_bundle_dict()
        bob.accept_handshake("alice_id", "Alice", init_hex, alice_bundle["ik"])

        code_alice = alice.get_verification_code("bob_id")
        code_bob = bob.get_verification_code("alice_id")
        assert code_alice is not None
        assert code_alice == code_bob

    def test_encrypt_decrypt_after_handshake(self, tmp_path):
        """Messages can be encrypted and decrypted after trust."""
        alice = self._make_manager(tmp_path, "alice")
        bob = self._make_manager(tmp_path, "bob")

        bob_bundle = bob.get_prekey_bundle_dict()
        init_hex = alice.initiate_handshake("bob_id", "Bob", bob_bundle)
        alice_bundle = alice.get_prekey_bundle_dict()
        bob.accept_handshake("alice_id", "Alice", init_hex, alice_bundle["ik"])

        # Alice encrypts, Bob decrypts
        plaintext = b"Hello from Alice!"
        ciphertext = alice.encrypt_for_peer("bob_id", plaintext)
        assert ciphertext is not None
        assert ciphertext != plaintext

        decrypted = bob.decrypt_from_peer("alice_id", ciphertext)
        assert decrypted == plaintext

    def test_block_removes_session(self, tmp_path):
        """Blocking a peer removes the encrypted session."""
        alice = self._make_manager(tmp_path, "alice")
        bob = self._make_manager(tmp_path, "bob")

        bob_bundle = bob.get_prekey_bundle_dict()
        init_hex = alice.initiate_handshake("bob_id", "Bob", bob_bundle)
        assert alice.has_session("bob_id")

        alice.approve("bob_id")
        alice.block("bob_id")
        assert not alice.has_session("bob_id")

    def test_key_change_revokes_trust_with_crypto(self, tmp_path):
        """If a peer's identity key changes, trust is revoked."""
        alice = self._make_manager(tmp_path, "alice")
        bob = self._make_manager(tmp_path, "bob")

        bob_bundle = bob.get_prekey_bundle_dict()
        alice.initiate_handshake("bob_id", "Bob", bob_bundle)
        alice.approve("bob_id")
        assert alice.is_trusted("bob_id")

        # Simulate Bob with new keys
        bob2 = self._make_manager(tmp_path, "bob2")
        bob2_bundle = bob2.get_prekey_bundle_dict()

        # Alice sees new bundle — fingerprint changes
        alice.initiate_handshake("bob_id", "Bob", bob2_bundle)
        assert not alice.is_trusted("bob_id")  # Trust revoked
