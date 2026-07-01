"""Pure-helper unit tests for render_tool_budget_preamble (neograph-iyo2).

These pin the rendering contract of the framework-generated tool-budget
preamble: the ONE canonical place tool budgets are turned into prose for the
model. The helper is pure logic (tools + max_iterations -> str), so a direct
unit test is the correct level — there is no I/O or outer surface to drive.

TDD red: src/neograph/_tool_budget_preamble.py does not exist yet, so every
test here fails at import/collection until the helper is implemented.
"""

from __future__ import annotations

from neograph import Tool


class TestRenderToolBudgetPreamble:
    """render_tool_budget_preamble(tools, max_iterations) rendering contract."""

    def test_finite_budget_tools_listed_with_exact_numbers_when_finite(self):
        """Each finite-budget tool appears with its exact Tool.budget number."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("search", budget=3), Tool("read", budget=5)]
        out = render_tool_budget_preamble(tools, max_iterations=20)

        assert "search" in out
        assert "read" in out
        # numbers announced == numbers enforced (Tool.budget)
        assert "3 calls" in out
        assert "5 calls" in out

    def test_unlimited_budget_tool_absent_when_budget_zero(self):
        """budget=0 (unlimited) tools are omitted from the preamble entirely."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("browse", budget=0)]
        out = render_tool_budget_preamble(tools, max_iterations=20)

        assert "browse" not in out

    def test_only_finite_listed_when_mixed_finite_and_unlimited(self):
        """Mixed set: finite tool listed with its number, unlimited tool absent."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("search", budget=4), Tool("browse", budget=0)]
        out = render_tool_budget_preamble(tools, max_iterations=12)

        assert "search" in out
        assert "4 calls" in out
        assert "browse" not in out

    def test_cap_and_directive_present_when_all_unlimited(self):
        """All-unlimited: per-tool list empty but step cap + directive still render."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("browse", budget=0), Tool("crawl", budget=0)]
        out = render_tool_budget_preamble(tools, max_iterations=7)

        assert "browse" not in out
        assert "crawl" not in out
        # step cap == max_iterations still announced
        assert "7" in out
        # plan-ahead / batch directive still present
        assert "plan" in out.lower()
        assert "batch" in out.lower()

    def test_does_not_raise_and_renders_cap_when_tools_empty(self):
        """Empty tools list is total: no raise, cap + directive still render."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        out = render_tool_budget_preamble([], max_iterations=9)

        assert "9" in out
        assert "plan" in out.lower()
        assert "batch" in out.lower()

    def test_singular_call_noun_when_budget_is_one(self):
        """budget=1 renders singular '1 call', never '1 calls' (LOW finding)."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("peek", budget=1)]
        out = render_tool_budget_preamble(tools, max_iterations=20)

        assert "1 call" in out
        assert "1 calls" not in out

    def test_step_cap_is_max_iterations_and_directive_present_when_finite(self):
        """Rendered step cap == max_iterations; directive text is present."""
        from neograph._tool_budget_preamble import render_tool_budget_preamble

        tools = [Tool("search", budget=3)]
        out = render_tool_budget_preamble(tools, max_iterations=15)

        assert "15" in out
        assert "plan" in out.lower()
        assert "batch" in out.lower()
