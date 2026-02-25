"""
Test Suite: Skill Presets
=========================

Verifies that skill presets correctly configure tenant agents with
the right skill bundles, personas, and HIPAA settings.
"""

import types
import pytest
from pathlib import Path

# Direct import of TenantConfig to avoid the full Reflection init chain
# (which needs SQLAlchemy, Redis, etc.).
# models.py only uses stdlib imports so we can exec it directly without
# polluting sys.modules with stub parent packages.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_models_path = PROJECT_ROOT / "reflection" / "tenants" / "models.py"
_models_mod = types.ModuleType("_tenant_models")
_models_mod.__file__ = str(_models_path)

# Pre-read the source and replace the relative import that exec() can't resolve.
# The constant is defined in reflection/core/settings.py; we inline it here.
_models_src = _models_path.read_text()
_models_src = _models_src.replace(
    "from ..core.settings import DEFAULT_ANTHROPIC_MODEL",
    'DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"',
)
exec(compile(_models_src, _models_path, "exec"), _models_mod.__dict__)
TenantConfig = _models_mod.TenantConfig


class TestGeneralPreset:

    def test_default_is_general(self):
        tc = TenantConfig()
        assert tc.skill_preset == "general"

    def test_general_allows_all_skills(self):
        tc = TenantConfig()
        assert tc.get_allowed_skills() is None  # None means no filtering

    def test_general_default_persona(self):
        tc = TenantConfig()
        assert tc.default_agent_persona == "a helpful AI assistant"


class TestNonprofitPreset:

    def test_nonprofit_activates_correct_skills(self):
        tc = TenantConfig(skill_preset="nonprofit")
        skills = tc.get_allowed_skills()
        assert skills is not None

        # Core nonprofit skills
        for required in ["nonprofit", "bookkeeping", "contacts", "documents",
                         "reports", "workflows", "meetings"]:
            assert required in skills, f"Missing required skill: {required}"

    def test_nonprofit_skill_count(self):
        tc = TenantConfig(skill_preset="nonprofit")
        skills = tc.get_allowed_skills()
        assert len(skills) == 14

    def test_nonprofit_persona(self):
        tc = TenantConfig(skill_preset="nonprofit")
        assert "501(c)(3)" in tc.default_agent_persona
        assert "donor" in tc.default_agent_persona.lower()

    def test_nonprofit_persona_can_be_overridden(self):
        tc = TenantConfig(skill_preset="nonprofit")
        tc.default_agent_persona = "custom nonprofit bot"
        assert tc.default_agent_persona == "custom nonprofit bot"

    def test_nonprofit_includes_utility_skills(self):
        tc = TenantConfig(skill_preset="nonprofit")
        skills = tc.get_allowed_skills()
        assert "tasks" in skills
        assert "calendar" in skills
        assert "websearch" in skills


class TestHealthcarePreset:

    def test_healthcare_activates_security_skills(self):
        tc = TenantConfig(skill_preset="healthcare")
        skills = tc.get_allowed_skills()

        for required in ["phi_detection", "audit", "rbac", "encryption"]:
            assert required in skills, f"Missing security skill: {required}"

    def test_healthcare_skill_count(self):
        tc = TenantConfig(skill_preset="healthcare")
        skills = tc.get_allowed_skills()
        assert len(skills) == 13

    def test_healthcare_persona_mentions_hipaa(self):
        tc = TenantConfig(skill_preset="healthcare")
        assert "HIPAA" in tc.default_agent_persona

    def test_healthcare_with_hipaa_flag(self):
        tc = TenantConfig(skill_preset="healthcare", hipaa_compliant=True)
        assert tc.audit_log_retention_days >= 2190  # 6 years
        assert tc.security_mode != "permissive"

    def test_hipaa_overrides_retention(self):
        tc = TenantConfig(
            skill_preset="healthcare",
            hipaa_compliant=True,
            audit_log_retention_days=30  # Too low
        )
        assert tc.audit_log_retention_days == 2190  # Forced up

    def test_hipaa_overrides_permissive_mode(self):
        tc = TenantConfig(
            skill_preset="healthcare",
            hipaa_compliant=True,
            security_mode="permissive"
        )
        assert tc.security_mode == "balanced"


class TestEnterprisePreset:

    def test_enterprise_activates_full_suite(self):
        tc = TenantConfig(skill_preset="enterprise")
        skills = tc.get_allowed_skills()

        for required in ["audit", "rbac", "user_management", "websearch",
                         "knowledge_base", "workflows", "notifications"]:
            assert required in skills, f"Missing enterprise skill: {required}"

    def test_enterprise_skill_count(self):
        tc = TenantConfig(skill_preset="enterprise")
        skills = tc.get_allowed_skills()
        assert len(skills) == 18

    def test_enterprise_persona(self):
        tc = TenantConfig(skill_preset="enterprise")
        assert "enterprise" in tc.default_agent_persona.lower()


class TestPresetEdgeCases:

    def test_unknown_preset_warns(self):
        """Unknown preset should not crash, just warn and use defaults."""
        tc = TenantConfig(skill_preset="banana")
        # Should still function — get_allowed_skills returns None (no filter)
        assert tc.get_allowed_skills() is None

    def test_preset_serialization(self):
        tc = TenantConfig(skill_preset="nonprofit")
        d = tc.to_dict()
        assert d["skill_preset"] == "nonprofit"

    def test_preset_from_dict(self):
        tc = TenantConfig.from_dict({"skill_preset": "healthcare"})
        assert tc.skill_preset == "healthcare"
        assert "HIPAA" in tc.default_agent_persona
