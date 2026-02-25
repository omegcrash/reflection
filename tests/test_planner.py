"""
Test Suite: Task Planning & Decomposition (Phase 2)
====================================================

Tests complexity estimation, plan generation, plan parsing,
and agent integration.
"""

import json
import pytest

from familiar.core.planner import (
    TaskPlanner,
    TaskPlan,
    PlanStep,
    estimate_complexity,
    _build_planning_message,
    PLANNING_SYSTEM,
)


# ── Complexity Estimation ──────────────────────────────────

class TestComplexityEstimation:

    def test_greeting_is_simple(self):
        assert estimate_complexity("hi") < 0.3

    def test_question_is_simple(self):
        assert estimate_complexity("what time is it?") < 0.3

    def test_single_action_is_simple(self):
        assert estimate_complexity("list my donors") < 0.3

    def test_search_is_simple(self):
        assert estimate_complexity("search for Alice", tool_count=5) < 0.3

    def test_board_packet_is_complex(self):
        msg = "Prepare a board packet with this quarter financials, donor summary, and grant status"
        assert estimate_complexity(msg, tool_count=10) >= 0.55

    def test_comparison_is_complex(self):
        msg = "Create a report comparing Q1 and Q2 fundraising across all programs"
        assert estimate_complexity(msg, tool_count=10) >= 0.55

    def test_setup_with_list_is_complex(self):
        msg = "Set up the nonprofit with bookkeeping, contacts, and grant tracking"
        assert estimate_complexity(msg, tool_count=10) >= 0.55

    def test_review_all_is_complex(self):
        msg = "Review all grant deadlines and create reminders for each one"
        assert estimate_complexity(msg, tool_count=10) >= 0.55

    def test_single_log_is_simple(self):
        assert estimate_complexity("Log a $500 donation from Alice", tool_count=5) < 0.55

    def test_factual_question_is_simple(self):
        assert estimate_complexity("What is the fiscal year end date?", tool_count=5) < 0.55

    def test_more_tools_increases_complexity(self):
        msg = "Check the financial status"
        c_few = estimate_complexity(msg, tool_count=3)
        c_many = estimate_complexity(msg, tool_count=15)
        assert c_many >= c_few

    def test_complexity_capped_at_one(self):
        msg = "For each and every program, compare, analyze and cross-reference Q1, Q2, Q3, and Q4 across all departments, then generate a comprehensive report, summary, and overview"
        assert estimate_complexity(msg, tool_count=20) <= 1.0

    def test_complexity_floor_at_zero(self):
        assert estimate_complexity("hi") >= 0.0


# ── Plan Data Structures ──────────────────────────────────

class TestPlanStep:

    def test_to_dict(self):
        step = PlanStep(id=1, description="do thing", tool="my_tool")
        d = step.to_dict()
        assert d["id"] == 1
        assert d["description"] == "do thing"
        assert d["tool"] == "my_tool"

    def test_to_dict_no_tool(self):
        step = PlanStep(id=1, description="think about it")
        d = step.to_dict()
        assert "tool" not in d

    def test_to_dict_with_deps(self):
        step = PlanStep(id=3, description="combine", depends_on=[1, 2])
        d = step.to_dict()
        assert d["depends_on"] == [1, 2]


class TestTaskPlan:

    def test_to_prompt_format(self):
        plan = TaskPlan(
            goal="Prepare board packet",
            steps=[
                PlanStep(id=1, description="Get financials", tool="financial_summary"),
                PlanStep(id=2, description="Get donors", tool="search_donors"),
                PlanStep(id=3, description="Compile", depends_on=[1, 2]),
            ],
        )
        prompt = plan.to_prompt()
        assert "## Execution Plan" in prompt
        assert "Prepare board packet" in prompt
        assert "financial_summary" in prompt
        assert "after step 1, 2" in prompt

    def test_serialization_roundtrip(self):
        plan = TaskPlan(
            goal="test goal",
            steps=[
                PlanStep(id=1, description="step one", tool="tool_a"),
                PlanStep(id=2, description="step two", depends_on=[1]),
            ],
            estimated_iterations=4,
        )
        d = plan.to_dict()
        plan2 = TaskPlan.from_dict(d)
        assert plan2.goal == "test goal"
        assert len(plan2.steps) == 2
        assert plan2.steps[0].tool == "tool_a"
        assert plan2.steps[1].depends_on == [1]
        assert plan2.estimated_iterations == 4

    def test_from_dict_empty_steps(self):
        plan = TaskPlan.from_dict({"goal": "empty", "steps": []})
        assert len(plan.steps) == 0


# ── Plan Parsing ──────────────────────────────────────────

class TestPlanParsing:

    @pytest.fixture
    def planner(self):
        return TaskPlanner(provider=None)

    def test_parse_valid_json(self, planner):
        raw = json.dumps({
            "goal": "test",
            "steps": [
                {"id": 1, "description": "step 1", "tool": "log_donor"},
            ],
            "estimated_iterations": 2,
        })
        plan = planner._parse_plan(raw, ["log_donor"])
        assert plan is not None
        assert plan.goal == "test"
        assert len(plan.steps) == 1

    def test_parse_json_in_markdown_fences(self, planner):
        raw = '```json\n{"goal": "test", "steps": [{"id": 1, "description": "x", "tool": null}]}\n```'
        plan = planner._parse_plan(raw, [])
        assert plan is not None

    def test_parse_json_with_surrounding_text(self, planner):
        raw = 'Here is my plan:\n{"goal": "test", "steps": [{"id": 1, "description": "x"}]}\nDone!'
        plan = planner._parse_plan(raw, [])
        assert plan is not None

    def test_invalid_tool_removed(self, planner):
        raw = json.dumps({
            "goal": "test",
            "steps": [
                {"id": 1, "description": "x", "tool": "nonexistent_tool"},
            ],
        })
        plan = planner._parse_plan(raw, ["log_donor"])
        assert plan.steps[0].tool is None

    def test_valid_tool_kept(self, planner):
        raw = json.dumps({
            "goal": "test",
            "steps": [
                {"id": 1, "description": "x", "tool": "log_donor"},
            ],
        })
        plan = planner._parse_plan(raw, ["log_donor", "search_donors"])
        assert plan.steps[0].tool == "log_donor"

    def test_parse_empty_string(self, planner):
        assert planner._parse_plan("", []) is None

    def test_parse_invalid_json(self, planner):
        assert planner._parse_plan("not json at all", []) is None

    def test_parse_missing_steps(self, planner):
        assert planner._parse_plan('{"goal": "no steps"}', []) is None

    def test_steps_capped_at_eight(self, planner):
        steps = [{"id": i, "description": f"step {i}"} for i in range(15)]
        raw = json.dumps({"goal": "many steps", "steps": steps})
        plan = planner._parse_plan(raw, [])
        assert len(plan.steps) <= 8


# ── Should Plan Decision ──────────────────────────────────

class TestShouldPlan:

    def test_disabled_planner_never_plans(self):
        planner = TaskPlanner(provider="fake", enabled=False)
        assert planner.should_plan("complex multi-step task", [{}] * 10) is False

    def test_no_provider_never_plans(self):
        planner = TaskPlanner(provider=None, enabled=True)
        assert planner.should_plan("complex task", [{}] * 10) is False

    def test_few_tools_never_plans(self):
        planner = TaskPlanner(provider="fake", enabled=True)
        assert planner.should_plan("complex task", [{}] * 2) is False

    def test_simple_query_does_not_plan(self):
        planner = TaskPlanner(provider="fake", enabled=True)
        assert planner.should_plan("what time is it?", [{}] * 10) is False

    def test_complex_query_plans(self):
        planner = TaskPlanner(provider="fake", enabled=True)
        msg = "Prepare a board packet with financials, donors, and grants"
        assert planner.should_plan(msg, [{}] * 10) is True

    def test_skipped_plans_counted(self):
        planner = TaskPlanner(provider="fake", enabled=True)
        planner.should_plan("hi", [{}] * 10)
        planner.should_plan("hello", [{}] * 10)
        assert planner.plans_skipped == 2


# ── Planning Message Construction ─────────────────────────

class TestPlanningMessage:

    def test_includes_user_request(self):
        msg = _build_planning_message("test request", ["tool1", "tool2"])
        assert "test request" in msg

    def test_includes_tools(self):
        msg = _build_planning_message("test", ["log_donor", "search_donors"])
        assert "log_donor" in msg
        assert "search_donors" in msg

    def test_includes_memory_context(self):
        msg = _build_planning_message("test", ["tool1"], memory_context="Fiscal year ends June 30")
        assert "Fiscal year ends June 30" in msg

    def test_no_memory_context(self):
        msg = _build_planning_message("test", ["tool1"])
        assert "context" not in msg.lower()


# ── Stats ─────────────────────────────────────────────────

class TestPlannerStats:

    def test_initial_stats(self):
        planner = TaskPlanner(provider=None)
        stats = planner.get_stats()
        assert stats["plans_created"] == 0
        assert stats["plans_skipped"] == 0
        assert stats["total_requests"] == 0

    def test_stats_after_skips(self):
        planner = TaskPlanner(provider="fake", enabled=True)
        planner.should_plan("hi", [{}] * 10)
        planner.should_plan("hello", [{}] * 10)
        stats = planner.get_stats()
        assert stats["plans_skipped"] == 2
        assert stats["plan_rate"] == 0.0
