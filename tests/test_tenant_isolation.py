"""
Test Suite: Tenant Data Isolation
=================================

Verifies that the tenant-scoped data directory system works correctly.
This is the most critical safety property of Reflection: tenant A's
data must NEVER appear in tenant B's queries.
"""

import json
import pytest
from pathlib import Path
from uuid import UUID

from familiar.core import paths
from familiar.core.paths import set_tenant_data_root, get_tenant_id, get_data_dir


class TestTenantPaths:
    """Verify the path scoping mechanism itself."""

    def test_default_paths_are_standalone(self):
        """Without set_tenant_data_root, paths are in ~/.familiar/data/"""
        assert get_tenant_id() is None
        assert ".familiar/data" in str(paths.DATA_DIR)
        assert "tenants" not in str(paths.DATA_DIR)

    def test_set_tenant_scopes_data_dir(self, tenant_id):
        tid = str(tenant_id)
        set_tenant_data_root(tid)

        assert get_tenant_id() == tid
        assert f"tenants/{tid}/data" in str(paths.DATA_DIR)

    def test_derived_paths_follow_tenant(self, tenant_id):
        tid = str(tenant_id)
        set_tenant_data_root(tid)

        assert f"tenants/{tid}" in str(paths.MEMORY_FILE)
        assert f"tenants/{tid}" in str(paths.TASKS_FILE)
        assert f"tenants/{tid}" in str(paths.LOGS_DIR)

    def test_tenant_switch(self, tenant_id, second_tenant_id):
        tid_a = str(tenant_id)
        tid_b = str(second_tenant_id)

        set_tenant_data_root(tid_a)
        assert f"tenants/{tid_a}" in str(paths.DATA_DIR)

        set_tenant_data_root(tid_b)
        assert f"tenants/{tid_b}" in str(paths.DATA_DIR)
        assert tid_a not in str(paths.DATA_DIR)

    def test_data_dir_is_created(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        assert paths.DATA_DIR.exists()


class TestSkillDataIsolation:
    """Verify that skills read/write to the correct tenant directory."""

    def test_nonprofit_data_file_scoped(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        from familiar.skills.nonprofit.skill import _get_data_file

        df = _get_data_file()
        assert f"tenants/{tenant_id}" in str(df)
        assert df.name == "nonprofit.json"

    def test_bookkeeping_data_file_scoped(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        from familiar.skills.bookkeeping.skill import _get_data_file

        df = _get_data_file()
        assert f"tenants/{tenant_id}" in str(df)
        assert df.name == "bookkeeping.json"

    def test_workflows_data_file_scoped(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        from familiar.skills.workflows.skill import _get_data_file

        df = _get_data_file()
        assert f"tenants/{tenant_id}" in str(df)
        assert df.name == "workflows.json"

    def test_contacts_data_file_scoped(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        from familiar.skills.contacts.skill import _get_data_file

        df = _get_data_file()
        assert f"tenants/{tenant_id}" in str(df)
        assert df.name == "contacts.json"

    def test_meetings_data_file_scoped(self, tenant_id):
        set_tenant_data_root(str(tenant_id))
        from familiar.skills.meetings.skill import _get_data_file

        df = _get_data_file()
        assert f"tenants/{tenant_id}" in str(df)
        assert df.name == "meetings.json"

    def test_data_dir_switches_between_tenants(self, tenant_id, second_tenant_id):
        """The critical isolation test: switching tenants changes where skills write."""
        from familiar.skills.nonprofit.skill import _get_data_file

        set_tenant_data_root(str(tenant_id))
        path_a = _get_data_file()

        set_tenant_data_root(str(second_tenant_id))
        path_b = _get_data_file()

        assert path_a != path_b
        assert str(tenant_id) in str(path_a)
        assert str(second_tenant_id) in str(path_b)


class TestCrossTenantLeakage:
    """
    The most important tests: verify that data written by tenant A
    is NOT visible to tenant B.
    """

    def test_nonprofit_donor_isolation(self, tenant_id, second_tenant_id):
        """Donor logged by tenant A must not appear in tenant B's data."""
        from familiar.skills.nonprofit.skill import log_donor, search_donors, _get_data_file

        # Tenant A logs a donor
        set_tenant_data_root(str(tenant_id))
        _get_data_file().parent.mkdir(parents=True, exist_ok=True)
        result_a = log_donor({
            "name": "Alice Tenant-A",
            "type": "gift",
            "amount": 500,
            "email": "alice@tenant-a.org"
        })
        assert "Alice Tenant-A" in result_a

        # Switch to tenant B
        set_tenant_data_root(str(second_tenant_id))
        _get_data_file().parent.mkdir(parents=True, exist_ok=True)

        # Tenant B should NOT see Alice
        result_b = search_donors({"query": "Alice"})
        assert "Alice Tenant-A" not in result_b

        # Tenant B logs their own donor
        result_b2 = log_donor({
            "name": "Bob Tenant-B",
            "type": "gift",
            "amount": 1000,
            "email": "bob@tenant-b.org"
        })
        assert "Bob Tenant-B" in result_b2

        # Switch back to tenant A — should NOT see Bob
        set_tenant_data_root(str(tenant_id))
        result_a2 = search_donors({"query": "Bob"})
        assert "Bob Tenant-B" not in result_a2

        # But should still see Alice
        result_a3 = search_donors({"query": "Alice"})
        assert "Alice" in result_a3

    def test_bookkeeping_transaction_isolation(self, tenant_id, second_tenant_id):
        """Financial transactions must be strictly tenant-isolated."""
        from familiar.skills.bookkeeping.skill import log_income, list_transactions, _get_data_file

        # Tenant A records income
        set_tenant_data_root(str(tenant_id))
        _get_data_file().parent.mkdir(parents=True, exist_ok=True)
        result_a = log_income({
            "amount": 5000,
            "source": "Foundation Grant Alpha",
            "category": "grants",
            "fund": "general"
        })
        assert "5000" in result_a or "5,000" in result_a

        # Tenant B should have no transactions
        set_tenant_data_root(str(second_tenant_id))
        _get_data_file().parent.mkdir(parents=True, exist_ok=True)
        result_b = list_transactions({})
        # Should be empty or show no transactions
        assert "Foundation Grant Alpha" not in result_b

    def test_data_files_are_physically_separate(self, tenant_id, second_tenant_id):
        """The actual JSON files should be in different directories."""
        from familiar.skills.nonprofit.skill import log_donor, _get_data_file

        set_tenant_data_root(str(tenant_id))
        _get_data_file().parent.mkdir(parents=True, exist_ok=True)
        log_donor({"name": "Test Donor", "type": "gift", "amount": 100})
        file_a = _get_data_file()

        set_tenant_data_root(str(second_tenant_id))
        file_b = _get_data_file()

        assert file_a.exists()
        assert not file_b.exists()  # Tenant B never wrote data
        assert file_a.parent != file_b.parent
