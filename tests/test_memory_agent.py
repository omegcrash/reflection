"""
Test Suite: Agentic Memory Management (Phase 4)
=================================================

Tests conversation filtering, memory extraction parsing,
supersedes logic, memory decay, and archival.
"""

import json
import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from familiar.core.memory_agent import (
    MemoryExtractor,
    ExtractedMemory,
    should_extract,
    decay_memories,
    _build_extraction_message,
    EXTRACTION_SYSTEM,
)
from familiar.core.memory import Memory


@pytest.fixture
def tmp_memory(tmp_path):
    return Memory(memory_file=tmp_path / "memory.json")


@pytest.fixture
def extractor(tmp_memory):
    return MemoryExtractor(provider=None, memory=tmp_memory)


# ── Conversation Filtering ─────────────────────────────────

class TestShouldExtract:

    def test_greeting_skipped(self):
        assert should_extract("hi", "Hello!") is False

    def test_yes_no_skipped(self):
        assert should_extract("yes", "Great, proceeding.") is False

    def test_short_user_message_skipped(self):
        assert should_extract("ok", "I'll do that for you right away. Here are the results.") is False

    def test_short_assistant_response_skipped(self):
        assert should_extract("Tell me about our grant deadlines and donor status", "Sure.") is False

    def test_time_question_skipped(self):
        assert should_extract("what time is it", "It is 3pm EST.") is False

    def test_substantive_conversation_passes(self):
        assert should_extract(
            "I need to prepare a grant report for the Community Foundation by March 15",
            "I will help you prepare the grant report. Let me pull the financial data "
            "first and then compile the donor summary."
        ) is True

    def test_preference_mention_passes(self):
        assert should_extract(
            "My email is george@example.com and I prefer reports in PDF format",
            "Got it, I have noted your email and PDF preference. I will use those going forward."
        ) is True

    def test_help_command_skipped(self):
        assert should_extract("help", "Here are the available commands...") is False


# ── Extraction Parsing ─────────────────────────────────────

class TestExtractionParsing:

    def test_parse_valid_json(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "user_email", "value": "george@example.com", '
            '"category": "user_info", "importance": 7}]',
            []
        )
        assert len(parsed) == 1
        assert parsed[0].key == "user_email"
        assert parsed[0].value == "george@example.com"
        assert parsed[0].category == "user_info"
        assert parsed[0].importance == 7

    def test_parse_markdown_fences(self, extractor):
        parsed = extractor._parse_extractions(
            '```json\n[{"key": "test", "value": "val", "category": "fact"}]\n```',
            []
        )
        assert len(parsed) == 1

    def test_parse_empty_array(self, extractor):
        assert extractor._parse_extractions("[]", []) == []

    def test_parse_invalid_json(self, extractor):
        assert extractor._parse_extractions("not json at all", []) == []

    def test_parse_empty_string(self, extractor):
        assert extractor._parse_extractions("", []) == []

    def test_key_sanitized(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "User Email!!", "value": "test", "category": "fact"}]',
            []
        )
        assert parsed[0].key == "user_email__"

    def test_key_truncated(self, extractor):
        long_key = "a" * 100
        parsed = extractor._parse_extractions(
            f'[{{"key": "{long_key}", "value": "test", "category": "fact"}}]',
            []
        )
        assert len(parsed[0].key) <= 50

    def test_invalid_category_normalized(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "test", "value": "test", "category": "bogus"}]',
            []
        )
        assert parsed[0].category == "fact"

    def test_valid_categories_accepted(self, extractor):
        for cat in ["user_info", "org_info", "deadline", "preference", "correction"]:
            parsed = extractor._parse_extractions(
                f'[{{"key": "test_{cat}", "value": "v", "category": "{cat}"}}]',
                []
            )
            assert parsed[0].category == cat

    def test_importance_clamped_high(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "test", "value": "test", "category": "fact", "importance": 99}]',
            []
        )
        assert parsed[0].importance == 10

    def test_importance_clamped_low(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "test", "value": "test", "category": "fact", "importance": -5}]',
            []
        )
        assert parsed[0].importance == 1

    def test_importance_default(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "test", "value": "test", "category": "fact"}]',
            []
        )
        assert parsed[0].importance == 5

    def test_missing_key_skipped(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"value": "no key", "category": "fact"}]',
            []
        )
        assert len(parsed) == 0

    def test_missing_value_skipped(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "no_value", "category": "fact"}]',
            []
        )
        assert len(parsed) == 0

    def test_capped_at_five(self, extractor):
        items = [{"key": f"k{i}", "value": f"v{i}", "category": "fact"} for i in range(10)]
        parsed = extractor._parse_extractions(json.dumps(items), [])
        assert len(parsed) <= 5


class TestSupersedes:

    def test_valid_supersedes_kept(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "new_email", "value": "new@example.com", '
            '"category": "correction", "supersedes": "old_email"}]',
            ["old_email", "other"]
        )
        assert parsed[0].supersedes == "old_email"

    def test_invalid_supersedes_removed(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "new_email", "value": "new@example.com", '
            '"category": "correction", "supersedes": "nonexistent"}]',
            ["old_email"]
        )
        assert parsed[0].supersedes is None

    def test_null_supersedes(self, extractor):
        parsed = extractor._parse_extractions(
            '[{"key": "test", "value": "test", "category": "fact", "supersedes": null}]',
            []
        )
        assert parsed[0].supersedes is None

    def test_duplicate_exact_value_skipped(self, tmp_memory):
        """If same key+value already exists, skip it."""
        tmp_memory.remember("existing_key", "existing value", category="fact")
        ext = MemoryExtractor(provider=None, memory=tmp_memory)
        parsed = ext._parse_extractions(
            '[{"key": "existing_key", "value": "existing value", "category": "fact"}]',
            ["existing_key"]
        )
        assert len(parsed) == 0

    def test_same_key_different_value_passes(self, tmp_memory):
        """If same key but different value, allow it (update)."""
        tmp_memory.remember("user_email", "old@example.com", category="user_info")
        ext = MemoryExtractor(provider=None, memory=tmp_memory)
        parsed = ext._parse_extractions(
            '[{"key": "user_email", "value": "new@example.com", "category": "user_info"}]',
            ["user_email"]
        )
        assert len(parsed) == 1
        assert parsed[0].value == "new@example.com"


# ── ExtractedMemory Data Structure ─────────────────────────

class TestExtractedMemory:

    def test_to_dict(self):
        em = ExtractedMemory(key="k", value="v", category="fact", importance=5)
        d = em.to_dict()
        assert d["key"] == "k"
        assert "supersedes" not in d

    def test_to_dict_with_supersedes(self):
        em = ExtractedMemory(key="k", value="v", category="fact", supersedes="old")
        d = em.to_dict()
        assert d["supersedes"] == "old"

    def test_roundtrip(self):
        em = ExtractedMemory(key="k", value="v", category="user_info", importance=7, supersedes="old")
        em2 = ExtractedMemory.from_dict(em.to_dict())
        assert em2.key == em.key
        assert em2.supersedes == "old"
        assert em2.importance == 7


# ── Extraction Message Builder ─────────────────────────────

class TestBuildExtractionMessage:

    def test_includes_user_message(self):
        msg = _build_extraction_message("my request", "response", [])
        assert "my request" in msg

    def test_includes_assistant_response(self):
        msg = _build_extraction_message("request", "my response", [])
        assert "my response" in msg

    def test_includes_existing_keys(self):
        msg = _build_extraction_message("req", "resp", ["key1", "key2"])
        assert "key1" in msg
        assert "key2" in msg

    def test_no_keys_section_when_empty(self):
        msg = _build_extraction_message("req", "resp", [])
        assert "memory keys" not in msg.lower()

    def test_truncates_long_messages(self):
        long_msg = "x" * 5000
        msg = _build_extraction_message(long_msg, "resp", [])
        assert len(msg) < 5000  # Should be truncated


# ── Memory Decay ───────────────────────────────────────────

class TestMemoryDecay:

    def test_fresh_memory_not_decayed(self, tmp_memory):
        tmp_memory.remember("fresh", "just added", category="fact", importance=5)
        stats = decay_memories(tmp_memory, days_idle=30)
        assert stats["decayed"] == 0
        assert tmp_memory.memories["fresh"].importance == 5

    def test_old_memory_decayed(self, tmp_memory):
        tmp_memory.remember("old", "old fact", category="fact", importance=5)
        old_date = (datetime.now() - timedelta(days=45)).isoformat()
        tmp_memory.memories["old"].updated_at = old_date
        
        stats = decay_memories(tmp_memory, days_idle=30, decay_amount=1)
        assert stats["decayed"] == 1
        assert tmp_memory.memories["old"].importance == 4

    def test_importance_floor_at_one(self, tmp_memory):
        tmp_memory.remember("low", "low importance", category="fact", importance=1)
        old_date = (datetime.now() - timedelta(days=45)).isoformat()
        tmp_memory.memories["low"].updated_at = old_date
        
        decay_memories(tmp_memory, days_idle=30, decay_amount=5)
        assert tmp_memory.memories["low"].importance == 1

    def test_multiple_decay_cycles(self, tmp_memory):
        tmp_memory.remember("test", "decaying", category="fact", importance=5)
        old_date = (datetime.now() - timedelta(days=45)).isoformat()
        tmp_memory.memories["test"].updated_at = old_date
        
        for expected in [4, 3, 2, 1, 1]:
            decay_memories(tmp_memory, days_idle=30, decay_amount=1)
            assert tmp_memory.memories["test"].importance == expected

    def test_archive_old_low_importance(self, tmp_memory):
        tmp_memory.remember("ancient", "very old", category="fact", importance=2)
        very_old = (datetime.now() - timedelta(days=100)).isoformat()
        tmp_memory.memories["ancient"].updated_at = very_old
        
        stats = decay_memories(
            tmp_memory, days_idle=30,
            archive_threshold=2, archive_after_days=90
        )
        assert "ancient" in stats["archived_keys"]
        assert "ancient" not in tmp_memory.memories

    def test_old_important_not_archived(self, tmp_memory):
        tmp_memory.remember("important", "critical fact", category="fact", importance=8)
        very_old = (datetime.now() - timedelta(days=100)).isoformat()
        tmp_memory.memories["important"].updated_at = very_old
        
        stats = decay_memories(
            tmp_memory, days_idle=30,
            archive_threshold=2, archive_after_days=90
        )
        assert "important" not in stats["archived_keys"]
        assert "important" in tmp_memory.memories

    def test_not_old_enough_not_archived(self, tmp_memory):
        """Low importance but not old enough for archival."""
        tmp_memory.remember("recent_low", "low but recent", category="fact", importance=2)
        thirty_days = (datetime.now() - timedelta(days=35)).isoformat()
        tmp_memory.memories["recent_low"].updated_at = thirty_days
        
        stats = decay_memories(
            tmp_memory, days_idle=30,
            archive_threshold=2, archive_after_days=90
        )
        assert "recent_low" not in stats["archived_keys"]

    def test_stats_include_total(self, tmp_memory):
        tmp_memory.remember("a", "fact a", category="fact", importance=5)
        tmp_memory.remember("b", "fact b", category="fact", importance=5)
        stats = decay_memories(tmp_memory)
        assert stats["total_memories"] == 2

    def test_empty_memory_no_error(self, tmp_memory):
        stats = decay_memories(tmp_memory)
        assert stats["decayed"] == 0
        assert stats["archived"] == 0


# ── Extractor Stats ────────────────────────────────────────

class TestExtractorStats:

    def test_initial_stats(self):
        ext = MemoryExtractor(provider=None, memory=None)
        stats = ext.get_stats()
        assert stats["extractions_run"] == 0
        assert stats["memories_stored"] == 0

    def test_skipped_counted(self, tmp_memory):
        ext = MemoryExtractor(provider="fake", memory=tmp_memory, enabled=True)
        ext.extract_async("hi", "Hello!")  # Should skip (trivial)
        assert ext.extractions_skipped == 1
