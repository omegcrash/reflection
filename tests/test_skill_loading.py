"""
Test Suite: Skill Loading
=========================

Verifies that all 39 skills load without errors and expose
the expected TOOLS interface.
"""

import os
import importlib
import pytest
import familiar
from familiar.core.paths import set_tenant_data_root


# Resolve skills directory from the installed familiar package
SKILLS_DIR = os.path.join(os.path.dirname(familiar.__file__), "skills")

# Skills with skill.py as entry point
SKILL_PY_SKILLS = sorted([
    d for d in os.listdir(SKILLS_DIR)
    if os.path.isfile(os.path.join(SKILLS_DIR, d, "skill.py"))
])

# Skills with __init__.py as entry point (complex multi-file skills)
# These require aiohttp which may not be installed in test environments
INIT_PY_SKILLS = sorted([
    d for d in os.listdir(SKILLS_DIR)
    if (os.path.isfile(os.path.join(SKILLS_DIR, d, "__init__.py"))
        and not os.path.isfile(os.path.join(SKILLS_DIR, d, "skill.py")))
])

# Skills that may have empty TOOLS (server-style skills without tool interface)
EMPTY_TOOLS_OK = {"email_server"}


@pytest.fixture(autouse=True)
def set_tenant(tenant_id):
    set_tenant_data_root(str(tenant_id))


@pytest.mark.parametrize("skill_name", SKILL_PY_SKILLS)
def test_skill_loads(skill_name):
    """Each skill.py module loads without ImportError."""
    mod = importlib.import_module(f"familiar.skills.{skill_name}.skill")
    assert mod is not None


@pytest.mark.parametrize("skill_name", SKILL_PY_SKILLS)
def test_skill_has_tools(skill_name):
    """Each skill exposes a TOOLS list."""
    mod = importlib.import_module(f"familiar.skills.{skill_name}.skill")
    tools = getattr(mod, "TOOLS", None)
    if skill_name in EMPTY_TOOLS_OK:
        pytest.skip(f"{skill_name} is a server-style skill (no TOOLS interface)")
    assert tools is not None, f"{skill_name} has no TOOLS"
    assert isinstance(tools, list)
    assert len(tools) > 0, f"{skill_name} TOOLS is empty"


@pytest.mark.parametrize("skill_name", [s for s in SKILL_PY_SKILLS if s not in EMPTY_TOOLS_OK])
def test_skill_tools_have_required_keys(skill_name):
    """Each tool has name, description, handler."""
    mod = importlib.import_module(f"familiar.skills.{skill_name}.skill")
    for tool in mod.TOOLS:
        assert "name" in tool, f"{skill_name}: tool missing 'name'"
        assert "description" in tool, f"{skill_name}/{tool.get('name', '?')}: missing 'description'"
        assert "handler" in tool, f"{skill_name}/{tool['name']}: missing 'handler'"
        assert callable(tool["handler"]), f"{skill_name}/{tool['name']}: handler not callable"


@pytest.mark.parametrize("skill_name", [s for s in SKILL_PY_SKILLS if s not in EMPTY_TOOLS_OK])
def test_skill_tools_have_input_schema(skill_name):
    """Each tool defines its input schema."""
    mod = importlib.import_module(f"familiar.skills.{skill_name}.skill")
    for tool in mod.TOOLS:
        assert "input_schema" in tool, f"{skill_name}/{tool['name']}: missing input_schema"
        schema = tool["input_schema"]
        assert isinstance(schema, dict)
        assert "type" in schema


@pytest.mark.parametrize("skill_name", INIT_PY_SKILLS)
def test_init_skill_loads(skill_name):
    """Multi-file skills (using __init__.py) load without error.
    May fail if aiohttp is not installed (browser_proxy, private_browser, video)."""
    try:
        mod = importlib.import_module(f"familiar.skills.{skill_name}")
        assert mod is not None
    except (NameError, ImportError) as e:
        if "aiohttp" in str(e):
            pytest.skip(f"{skill_name} requires aiohttp: {e}")
        raise


# ── Skill inventory assertions ─────────────────────────────────

def test_total_skill_count():
    """We should have exactly 39 skills."""
    all_skills = [
        d for d in os.listdir(SKILLS_DIR)
        if os.path.isdir(os.path.join(SKILLS_DIR, d))
        and not d.startswith("_")
    ]
    assert len(all_skills) == 40, f"Expected 40 skills, got {len(all_skills)}: {all_skills}"


def test_nonprofit_suite_present():
    """All nonprofit skills must be present."""
    for skill in ["nonprofit", "bookkeeping", "contacts", "documents",
                   "reports", "workflows", "meetings"]:
        assert skill in SKILL_PY_SKILLS, f"Missing nonprofit skill: {skill}"


def test_enterprise_suite_present():
    """All enterprise security skills must be present."""
    for skill in ["audit", "rbac", "user_management", "phi_detection",
                   "sessions", "encryption"]:
        assert skill in SKILL_PY_SKILLS, f"Missing enterprise skill: {skill}"


def test_utility_suite_present():
    """Utility skills must be present."""
    for skill in ["websearch", "smart_search", "filereader",
                   "knowledge_base", "notifications", "transcription"]:
        assert skill in SKILL_PY_SKILLS, f"Missing utility skill: {skill}"
