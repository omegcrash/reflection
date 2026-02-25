# Familiar - Self-Hosted AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Tests for the Constitutional Personality Layer (Phase 1).

Verifies:
- Relationship stage progression
- Channel-specific prompt formatting
- Article injection per stage
- Interaction tracking persistence
- Constitutional presets for Reflection tenants
- Legacy fallback when constitution module unavailable
"""

import json
import tempfile
from pathlib import Path

import pytest

from familiar.core.constitution import (
    ConstitutionalIdentity,
    RelationshipStage,
    ChannelPersonality,
    InteractionTracker,
    get_relationship_stage,
    get_constitutional_prompt,
    get_constitutional_preset,
    STAGE_THRESHOLDS,
    CHANNEL_GUIDANCE,
    ARTICLE_0,
    CONSTITUTIONAL_PRESETS,
)


# ============================================================
# RELATIONSHIP STAGE TESTS (Articles 3, 5)
# ============================================================

class TestRelationshipStage:
    """Verify trust calibration through interaction count."""
    
    def test_stage_1_default(self):
        """New users start at Stage 1."""
        assert get_relationship_stage(0) == RelationshipStage.STAGE_1
        assert get_relationship_stage(5) == RelationshipStage.STAGE_1
        assert get_relationship_stage(9) == RelationshipStage.STAGE_1
    
    def test_stage_2_threshold(self):
        """Stage 2 begins at 10 interactions."""
        assert get_relationship_stage(10) == RelationshipStage.STAGE_2
        assert get_relationship_stage(25) == RelationshipStage.STAGE_2
        assert get_relationship_stage(49) == RelationshipStage.STAGE_2
    
    def test_stage_3_threshold(self):
        """Stage 3 begins at 50 interactions."""
        assert get_relationship_stage(50) == RelationshipStage.STAGE_3
        assert get_relationship_stage(100) == RelationshipStage.STAGE_3
        assert get_relationship_stage(1000) == RelationshipStage.STAGE_3
    
    def test_custom_thresholds(self):
        """Custom thresholds override defaults."""
        custom = {
            RelationshipStage.STAGE_1: 0,
            RelationshipStage.STAGE_2: 5,
            RelationshipStage.STAGE_3: 20,
        }
        assert get_relationship_stage(4, custom) == RelationshipStage.STAGE_1
        assert get_relationship_stage(5, custom) == RelationshipStage.STAGE_2
        assert get_relationship_stage(20, custom) == RelationshipStage.STAGE_3


# ============================================================
# INTERACTION TRACKER TESTS
# ============================================================

class TestInteractionTracker:
    """Verify per-user tracking with persistence."""
    
    def test_initial_count_is_zero(self, tmp_path):
        tracker = InteractionTracker(data_dir=tmp_path)
        assert tracker.get_count("user_123") == 0
    
    def test_record_increments(self, tmp_path):
        tracker = InteractionTracker(data_dir=tmp_path)
        assert tracker.record_interaction("user_123") == 1
        assert tracker.record_interaction("user_123") == 2
        assert tracker.record_interaction("user_123") == 3
    
    def test_separate_users(self, tmp_path):
        tracker = InteractionTracker(data_dir=tmp_path)
        tracker.record_interaction("alice")
        tracker.record_interaction("alice")
        tracker.record_interaction("bob")
        assert tracker.get_count("alice") == 2
        assert tracker.get_count("bob") == 1
    
    def test_persistence(self, tmp_path):
        """Counts survive tracker re-creation."""
        tracker1 = InteractionTracker(data_dir=tmp_path)
        tracker1.record_interaction("alice")
        tracker1.record_interaction("alice")
        
        tracker2 = InteractionTracker(data_dir=tmp_path)
        assert tracker2.get_count("alice") == 2
    
    def test_get_stage(self, tmp_path):
        tracker = InteractionTracker(data_dir=tmp_path)
        assert tracker.get_stage("new_user") == RelationshipStage.STAGE_1
        
        for _ in range(10):
            tracker.record_interaction("growing_user")
        assert tracker.get_stage("growing_user") == RelationshipStage.STAGE_2
        
        for _ in range(40):
            tracker.record_interaction("growing_user")
        assert tracker.get_stage("growing_user") == RelationshipStage.STAGE_3


# ============================================================
# CONSTITUTIONAL IDENTITY TESTS (Articles 0-7)
# ============================================================

class TestConstitutionalIdentity:
    """Verify constitutional prompt generation."""
    
    def test_contains_article_0(self):
        """Sacred Exchange ethic always present."""
        identity = ConstitutionalIdentity(agent_name="Willow")
        prompt = identity.build()
        assert "sacred exchange" in prompt.lower()
        assert "transform" in prompt.lower()
        assert "extract" in prompt.lower()
    
    def test_contains_agent_name(self):
        """Agent name appears in prompt."""
        identity = ConstitutionalIdentity(agent_name="Willow")
        prompt = identity.build()
        assert "Willow" in prompt
    
    def test_stage_1_professional_tone(self):
        """Stage 1 is professional, not familiar."""
        identity = ConstitutionalIdentity(
            relationship_stage=RelationshipStage.STAGE_1,
        )
        prompt = identity.build()
        assert "professional" in prompt.lower()
        assert "Do not presume familiarity" in prompt
    
    def test_stage_2_warmer_tone(self):
        """Stage 2 has earned warmth."""
        identity = ConstitutionalIdentity(
            relationship_stage=RelationshipStage.STAGE_2,
        )
        prompt = identity.build()
        assert "earned warmth" in prompt.lower() or "warm" in prompt.lower()
        assert "shared context" in prompt.lower() or "shared history" in prompt.lower()
    
    def test_stage_3_colleague_tone(self):
        """Stage 3 is trusted colleague."""
        identity = ConstitutionalIdentity(
            relationship_stage=RelationshipStage.STAGE_3,
        )
        prompt = identity.build()
        assert "trusted colleague" in prompt.lower()
        assert "anticipate" in prompt.lower()
    
    def test_channel_telegram(self):
        """Telegram channel guidance included."""
        identity = ConstitutionalIdentity(channel="telegram")
        prompt = identity.build()
        assert "Telegram" in prompt
    
    def test_channel_whatsapp(self):
        """WhatsApp channel guidance included."""
        identity = ConstitutionalIdentity(channel="whatsapp")
        prompt = identity.build()
        assert "WhatsApp" in prompt
    
    def test_proactive_includes_silence_guidance(self):
        """Proactive messages include silence guidance."""
        identity = ConstitutionalIdentity(is_proactive=True)
        prompt = identity.build()
        assert "Silence is also service" in prompt
    
    def test_non_proactive_omits_silence_guidance(self):
        """Regular messages omit silence guidance."""
        identity = ConstitutionalIdentity(is_proactive=False)
        prompt = identity.build()
        assert "Silence is also service" not in prompt
    
    def test_persona_overlay(self):
        """Tenant persona overlay injected."""
        identity = ConstitutionalIdentity(
            persona_overlay="You serve a 501(c)(3) nonprofit."
        )
        prompt = identity.build()
        assert "501(c)(3)" in prompt
    
    def test_hot_dog_principle_present(self):
        """Article 4 (Hot Dog) always present."""
        identity = ConstitutionalIdentity()
        prompt = identity.build()
        assert "hot dog" in prompt.lower()
    
    def test_silent_protection_present(self):
        """Article 6 (Silent Protection) always present."""
        identity = ConstitutionalIdentity()
        prompt = identity.build()
        assert "silently" in prompt.lower() or "silent" in prompt.lower()
        assert "Never narrate your security measures" in prompt
    
    def test_to_dict_serialization(self):
        """Identity serializes for logging."""
        identity = ConstitutionalIdentity(
            agent_name="Willow",
            relationship_stage=RelationshipStage.STAGE_2,
            channel="telegram",
        )
        d = identity.to_dict()
        assert d["agent_name"] == "Willow"
        assert d["relationship_stage"] == "stage_2"
        assert d["channel"] == "telegram"


# ============================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================

class TestGetConstitutionalPrompt:
    """Verify the one-call convenience function."""
    
    def test_returns_string(self, tmp_path):
        prompt = get_constitutional_prompt(
            agent_name="Test",
            user_id="test_user",
            channel="cli",
            interaction_tracker=InteractionTracker(data_dir=tmp_path),
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 100
    
    def test_sacred_exchange_always_present(self, tmp_path):
        prompt = get_constitutional_prompt(
            interaction_tracker=InteractionTracker(data_dir=tmp_path),
        )
        assert "sacred exchange" in prompt.lower()


# ============================================================
# REFLECTION PRESET TESTS
# ============================================================

class TestConstitutionalPresets:
    """Verify presets for tenant skill bundles."""
    
    def test_general_preset_empty(self):
        overlay = get_constitutional_preset("general")
        assert overlay == ""
    
    def test_nonprofit_preset(self):
        overlay = get_constitutional_preset("nonprofit")
        assert "501(c)(3)" in overlay
        assert "donation" in overlay.lower() or "dollar" in overlay.lower()
    
    def test_healthcare_preset(self):
        overlay = get_constitutional_preset("healthcare")
        assert "HIPAA" in overlay
        assert "Article 6" in overlay
    
    def test_enterprise_preset(self):
        overlay = get_constitutional_preset("enterprise")
        assert "enterprise" in overlay.lower()
    
    def test_unknown_preset_falls_back(self):
        overlay = get_constitutional_preset("nonexistent")
        assert overlay == ""
    
    def test_all_presets_match_tenant_models(self):
        """Every preset in constitution matches tenant model presets."""
        for preset_name in ["general", "nonprofit", "healthcare", "enterprise"]:
            assert preset_name in CONSTITUTIONAL_PRESETS


# ============================================================
# CHANNEL GUIDANCE TESTS (Article 7)
# ============================================================

class TestChannelGuidance:
    """Verify channel-specific delivery guidance."""
    
    def test_all_channels_have_guidance(self):
        for channel in ChannelPersonality:
            assert channel in CHANNEL_GUIDANCE
    
    def test_telegram_mentions_voice(self):
        """Telegram guidance mentions voice notes."""
        assert "voice" in CHANNEL_GUIDANCE[ChannelPersonality.TELEGRAM].lower()
    
    def test_signal_mentions_privacy(self):
        """Signal guidance respects privacy choice."""
        assert "privacy" in CHANNEL_GUIDANCE[ChannelPersonality.SIGNAL].lower()
    
    def test_whatsapp_no_markdown(self):
        """WhatsApp guidance says no markdown."""
        assert "no markdown" in CHANNEL_GUIDANCE[ChannelPersonality.WHATSAPP].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
