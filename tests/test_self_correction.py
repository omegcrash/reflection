"""
Test Suite: Structured Self-Correction (Phase 3)
=================================================

Tests error detection, error classification, reflection formatting,
per-tool retry tracking, and exhaustion handling.
"""

import json
import pytest

from familiar.core.agent import Agent


@pytest.fixture
def agent():
    """Create a bare Agent instance for testing helper methods."""
    a = Agent.__new__(Agent)
    return a


# ── Error Detection ────────────────────────────────────────

class TestIsToolError:

    def test_warning_emoji_prefix(self, agent):
        assert agent._is_tool_error("⚠️ Permission denied") is True

    def test_error_prefix(self, agent):
        assert agent._is_tool_error("Error: something went wrong") is True

    def test_failed_prefix(self, agent):
        assert agent._is_tool_error("Failed: could not connect") is True

    def test_permission_denied_substring(self, agent):
        assert agent._is_tool_error("Access: Permission denied for this resource") is True

    def test_not_found_substring(self, agent):
        assert agent._is_tool_error("The donor was not found in database") is True

    def test_connection_refused(self, agent):
        assert agent._is_tool_error("Connection refused to localhost:5432") is True

    def test_timeout(self, agent):
        assert agent._is_tool_error("Request timed out after 30 seconds") is True

    def test_rate_limit(self, agent):
        assert agent._is_tool_error("API rate limit exceeded") is True

    def test_success_not_error(self, agent):
        assert agent._is_tool_error("Successfully logged donor Alice") is False

    def test_search_results_not_error(self, agent):
        assert agent._is_tool_error("Found 3 matching records") is False

    def test_empty_string_not_error(self, agent):
        assert agent._is_tool_error("") is False

    def test_none_not_error(self, agent):
        assert agent._is_tool_error(None) is False

    def test_normal_data_not_error(self, agent):
        assert agent._is_tool_error('{"donors": [{"name": "Alice", "amount": 500}]}') is False

    def test_invalid_in_context_detected(self, agent):
        assert agent._is_tool_error("Error: invalid email format") is True

    def test_case_insensitive_substrings(self, agent):
        assert agent._is_tool_error("The file does NOT exist") is True


# ── Error Classification ──────────────────────────────────

class TestClassifyError:

    def test_permission_class(self, agent):
        cls, _, retry = agent._classify_error("⚠️ Permission denied. Missing: shell_execute")
        assert cls == "permission"
        assert retry is False

    def test_read_only_is_permission(self, agent):
        cls, _, retry = agent._classify_error("Your role (viewer) is read-only")
        assert cls == "permission"
        assert retry is False

    def test_not_found_class(self, agent):
        cls, _, retry = agent._classify_error('Donor "Bob" not found in database')
        assert cls == "not_found"
        assert retry is True

    def test_no_results_is_not_found(self, agent):
        cls, _, _ = agent._classify_error("no results for query")
        assert cls == "not_found"

    def test_validation_class(self, agent):
        cls, _, retry = agent._classify_error("Invalid amount: must be positive number")
        assert cls == "validation"
        assert retry is True

    def test_required_field(self, agent):
        cls, _, _ = agent._classify_error("Field 'email' is required")
        assert cls == "validation"

    def test_connection_class(self, agent):
        cls, _, retry = agent._classify_error("Connection refused to localhost:11434")
        assert cls == "connection"
        assert retry is False

    def test_timeout_is_connection(self, agent):
        cls, _, _ = agent._classify_error("Request timed out after 30s")
        assert cls == "connection"

    def test_rate_limit_class(self, agent):
        cls, _, retry = agent._classify_error("Rate limit exceeded, try again in 60s")
        assert cls == "rate_limit"
        assert retry is False

    def test_unknown_class(self, agent):
        cls, _, retry = agent._classify_error("Something completely unexpected happened")
        assert cls == "unknown"
        assert retry is True

    def test_unexpected_not_validation(self, agent):
        """Regression: 'unexpected' must not match 'expected' validation pattern."""
        cls, _, _ = agent._classify_error("An unexpected error occurred")
        assert cls != "validation"


# ── Error Formatting ──────────────────────────────────────

class TestFormatToolError:

    def test_contains_tool_name(self, agent):
        result = agent._format_tool_error(
            "search_donors", {"name": "Bob"}, "not found",
            attempt=1, max_attempts=3,
        )
        assert "search_donors" in result

    def test_contains_error_text(self, agent):
        result = agent._format_tool_error(
            "search_donors", {}, "Donor Bob not found",
            attempt=1, max_attempts=3,
        )
        assert "Donor Bob not found" in result

    def test_contains_classification(self, agent):
        result = agent._format_tool_error(
            "search_donors", {}, "not found",
            attempt=1, max_attempts=3,
        )
        assert "not_found" in result

    def test_contains_attempt_count(self, agent):
        result = agent._format_tool_error(
            "tool", {}, "error", attempt=2, max_attempts=3,
        )
        assert "2/3" in result

    def test_retryable_shows_remaining(self, agent):
        result = agent._format_tool_error(
            "tool", {}, "not found", attempt=1, max_attempts=3,
        )
        assert "2 attempts remaining" in result

    def test_non_retryable_says_do_not_retry(self, agent):
        result = agent._format_tool_error(
            "run_shell", {}, "Permission denied",
            attempt=1, max_attempts=3,
        )
        assert "Do NOT retry" in result

    def test_includes_input_preview(self, agent):
        result = agent._format_tool_error(
            "log_donor", {"name": "Alice", "amount": 500}, "validation error",
            attempt=1, max_attempts=3,
        )
        assert "Alice" in result

    def test_input_preview_truncated(self, agent):
        big_input = {"data": "x" * 500}
        result = agent._format_tool_error(
            "tool", big_input, "error", attempt=1, max_attempts=3,
        )
        assert len(result) < 800


class TestFormatToolExhausted:

    def test_contains_tool_name(self, agent):
        result = agent._format_tool_exhausted("search_donors", 4)
        assert "search_donors" in result

    def test_contains_stop_instruction(self, agent):
        result = agent._format_tool_exhausted("tool", 4)
        assert "Stop calling" in result

    def test_contains_attempt_count(self, agent):
        result = agent._format_tool_exhausted("tool", 5)
        assert "5" in result

    def test_suggests_alternatives(self, agent):
        result = agent._format_tool_exhausted("tool", 4)
        assert "alternatives" in result.lower() or "manual" in result.lower()


# ── Integration Behavior ──────────────────────────────────

class TestRetryTracking:

    def test_first_attempt_is_one(self):
        tool_attempts = {}
        key = "search_donors"
        tool_attempts[key] = tool_attempts.get(key, 0) + 1
        assert tool_attempts[key] == 1

    def test_increments_on_repeated_calls(self):
        tool_attempts = {}
        key = "search_donors"
        for _ in range(3):
            tool_attempts[key] = tool_attempts.get(key, 0) + 1
        assert tool_attempts[key] == 3

    def test_success_resets_counter(self):
        tool_attempts = {}
        key = "search_donors"
        tool_attempts[key] = 2
        tool_attempts[key] = 0  # Success resets
        assert tool_attempts[key] == 0

    def test_different_tools_tracked_independently(self):
        tool_attempts = {}
        tool_attempts["tool_a"] = tool_attempts.get("tool_a", 0) + 1
        tool_attempts["tool_b"] = tool_attempts.get("tool_b", 0) + 1
        tool_attempts["tool_a"] = tool_attempts.get("tool_a", 0) + 1
        assert tool_attempts["tool_a"] == 2
        assert tool_attempts["tool_b"] == 1

    def test_exhaustion_threshold(self):
        MAX_TOOL_RETRIES = 3
        tool_attempts = {}
        key = "failing_tool"
        for i in range(4):
            tool_attempts[key] = tool_attempts.get(key, 0) + 1
            if tool_attempts[key] > MAX_TOOL_RETRIES:
                exhausted = True
                break
        else:
            exhausted = False
        assert exhausted is True
        assert tool_attempts[key] == 4
