"""
Test Suite: Skill Handler Integration
======================================

Tests the actual skill handler functions end-to-end.
These are the functions that TenantAgent calls when processing
tool use requests from the LLM.
"""

import json
import pytest
from pathlib import Path
from familiar.core.paths import set_tenant_data_root


@pytest.fixture
def nonprofit_tenant(tenant_id, tmp_path):
    """Set up a nonprofit tenant with clean data directory."""
    set_tenant_data_root(str(tenant_id))
    from familiar.core import paths
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return tenant_id


class TestNonprofitSkill:

    def test_log_donor(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import log_donor
        result = log_donor({
            "name": "Jane Smith",
            "type": "gift",
            "amount": 500,
            "email": "jane@example.org"
        })
        assert "Jane Smith" in result
        assert "500" in result

    def test_search_donors_found(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import log_donor, search_donors
        log_donor({"name": "Jane Smith", "type": "gift", "amount": 500})
        result = search_donors({"query": "Jane"})
        assert "Jane Smith" in result

    def test_search_donors_not_found(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import search_donors
        result = search_donors({"query": "Nonexistent Person"})
        # Should return empty or "no donors found" — not crash
        assert isinstance(result, str)

    def test_add_grant(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import add_grant
        result = add_grant({
            "name": "Community Foundation Grant",
            "funder": "Community Foundation",
            "amount": 25000,
            "deadline": "2026-06-01"
        })
        assert "Community Foundation" in result

    def test_grant_deadlines(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import add_grant, grant_deadlines
        from datetime import datetime, timedelta
        # Deadline must be within 60 days to appear
        near_deadline = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
        add_grant({
            "name": "Test Grant",
            "funder": "Test Foundation",
            "amount": 10000,
            "deadline": near_deadline
        })
        result = grant_deadlines({})
        assert "Test Grant" in result or "Test Foundation" in result

    def test_donor_details_by_name(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import log_donor, donor_details
        log_donor({"name": "Specific Donor", "type": "gift", "amount": 1000})
        result = donor_details({"name": "Specific Donor"})
        assert "Specific Donor" in result

    def test_add_note(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import add_note
        result = add_note({
            "title": "Board Meeting Notes",
            "content": "Approved the annual budget."
        })
        assert "Board Meeting" in result or "note" in result.lower()

    def test_multiple_donors(self, nonprofit_tenant):
        from familiar.skills.nonprofit.skill import log_donor, search_donors
        for name in ["Alice Adams", "Bob Brown", "Carol Chen"]:
            log_donor({"name": name, "type": "gift", "amount": 100})

        result = search_donors({"query": "Bob"})
        assert "Bob Brown" in result
        assert "Alice" not in result  # Should only match Bob


class TestBookkeepingSkill:

    def test_log_income(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import log_income
        result = log_income({
            "amount": 5000,
            "source": "Foundation Grant",
            "category": "grants",
            "fund": "general"
        })
        assert "5000" in result or "5,000" in result

    def test_log_expense(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import log_expense
        result = log_expense({
            "amount": 200,
            "vendor": "Office Depot",
            "category": "supplies",
            "fund": "general"
        })
        assert "200" in result

    def test_list_transactions(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import log_income, log_expense, list_transactions
        log_income({"amount": 1000, "source": "Donation", "category": "donations"})
        log_expense({"amount": 50, "vendor": "Stamps.com", "category": "postage"})
        result = list_transactions({})
        assert "Donation" in result or "1000" in result or "1,000" in result

    def test_financial_summary(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import log_income, log_expense, financial_summary
        log_income({"amount": 10000, "source": "Grant", "category": "grants"})
        log_expense({"amount": 3000, "vendor": "Rent", "category": "occupancy"})
        result = financial_summary({})
        assert isinstance(result, str)
        # Should mention income/expense or total
        assert any(word in result.lower() for word in ["income", "expense", "total", "net", "balance"])

    def test_budget_status(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import budget_status
        result = budget_status({})
        assert isinstance(result, str)

    def test_fund_balances(self, nonprofit_tenant):
        from familiar.skills.bookkeeping.skill import log_income, fund_balances
        log_income({"amount": 5000, "source": "Restricted Gift", "category": "donations", "fund": "program_a"})
        result = fund_balances({})
        assert isinstance(result, str)


class TestWorkflowsSkill:

    def test_list_workflows(self, nonprofit_tenant):
        from familiar.skills.workflows.skill import list_workflows
        result = list_workflows({})
        assert isinstance(result, str)
        # Should list available workflow templates
        assert "workflow" in result.lower() or "template" in result.lower() or "available" in result.lower() or "none" in result.lower()

    def test_create_and_run_workflow(self, nonprofit_tenant):
        from familiar.skills.workflows.skill import create_workflow, list_workflows
        # Steps must be {tool, params} objects
        create_result = create_workflow({
            "name": "Test Workflow",
            "steps": [
                {"tool": "log_donor", "params": {"name": "WF Donor"}},
                {"tool": "search_donors", "params": {"query": "WF"}}
            ]
        })
        assert "Test Workflow" in create_result or "workflow" in create_result.lower() or "created" in create_result.lower()


class TestContactsSkill:

    def test_add_contact(self, nonprofit_tenant):
        from familiar.skills.contacts.skill import TOOLS
        # Find the add_contact handler
        handlers = {t["name"]: t["handler"] for t in TOOLS}
        assert "add_contact" in handlers

        result = handlers["add_contact"]({
            "name": "Test Contact",
            "email": "test@example.org",
            "phone": "555-0100"
        })
        assert "Test Contact" in result

    def test_search_contacts(self, nonprofit_tenant):
        from familiar.skills.contacts.skill import TOOLS
        handlers = {t["name"]: t["handler"] for t in TOOLS}

        handlers["add_contact"]({"name": "Alice Wonderland", "email": "alice@wonder.org"})
        result = handlers["search_contacts"]({"query": "Alice"})
        assert "Alice" in result


class TestMeetingsSkill:

    def test_schedule_meeting(self, nonprofit_tenant):
        from familiar.skills.meetings.skill import TOOLS
        handlers = {t["name"]: t["handler"] for t in TOOLS}
        assert "schedule_meeting" in handlers

        result = handlers["schedule_meeting"]({
            "title": "Board Meeting",
            "start": "2026-03-15T14:00",
            "duration_minutes": 60,
            "attendees": ["Board Members"]
        })
        # Familiar v1.4.0 may return a ConfirmationRequired object
        # instead of a plain string for destructive/scheduling actions
        if isinstance(result, str):
            assert "Board Meeting" in result or "scheduled" in result.lower()
        else:
            # ConfirmationRequired — check preview text
            preview = getattr(result, "preview", str(result))
            assert "Board Meeting" in preview or "meeting" in preview.lower()
