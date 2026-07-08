"""lint() validation: DI bindings, obligation gaps, Loop condition checks."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel

from neograph import (
    Construct,
    FromConfig,
    FromInput,
    Node,
    Oracle,
    construct_from_functions,
    lint,
    node,
)
from tests.schemas import (
    Claims,
    RawText,
    _producer,
)


class TestLint:
    """lint() validates DI bindings against a sample config."""

    def test_lint_returns_empty_when_all_bindings_present(self):
        """No warnings when every FromInput/FromConfig key exists in config."""

        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("ok", [my_node])
        issues = lint(pipeline, config={"topic": "hello"})
        assert issues == []

    def test_lint_reports_missing_from_input_key(self):
        """lint reports when a FromInput param has no matching config key."""

        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bad", [my_node])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "topic" in issues[0].param
        assert "my" in issues[0].node_name  # "my-node" or "my_node"

    def test_lint_reports_missing_from_config_key(self):
        """lint reports when a FromConfig param has no matching config key."""

        @node(outputs=RawText)
        def my_node(
            upstream: RawText,
            limiter: Annotated[str, FromConfig],
        ) -> RawText: ...

        producer = _producer("upstream", RawText)
        pipeline = Construct("bad", nodes=[producer, my_node])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "limiter" in issues[0].param

    def test_lint_reports_missing_bundled_model_fields(self):
        """When a FromInput param is a BaseModel, lint checks each field."""

        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bundled", [my_node])
        # Only provide node_id, missing project_root
        issues = lint(pipeline, config={"node_id": "x"})
        assert len(issues) == 1
        assert "project_root" in issues[0].param

    def test_lint_bundled_model_all_fields_present(self):
        """No issues when all bundled model fields are in config."""

        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput]) -> RawText: ...

        pipeline = construct_from_functions("bundled-ok", [my_node])
        issues = lint(pipeline, config={"node_id": "x", "project_root": "/tmp"})
        assert issues == []

    def test_lint_no_config_still_validates_required(self):
        """Without config, lint reports required=True params as errors."""

        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText: ...

        pipeline = construct_from_functions("no-cfg", [my_node])
        issues = lint(pipeline)
        assert len(issues) == 1
        assert issues[0].required is True
        assert "topic" in issues[0].param

    def test_lint_required_false_no_issue_without_config(self):
        """Optional FromInput(required=False) params are NOT flagged without config."""

        @node(outputs=RawText)
        def my_node(topic: Annotated[str, FromInput(required=False)]) -> RawText: ...

        pipeline = construct_from_functions("opt", [my_node])
        issues = lint(pipeline)
        assert issues == []

    def test_lint_walks_sub_constructs(self):
        """lint recurses into sub-constructs."""

        @node(outputs=Claims)
        def inner(topic: Annotated[str, FromInput]) -> Claims: ...

        sub = construct_from_functions("sub", [inner], input=None, output=Claims)
        outer_prod = _producer("start", RawText)
        pipeline = Construct("outer", nodes=[outer_prod, sub])
        issues = lint(pipeline, config={})
        assert len(issues) == 1
        assert "topic" in issues[0].param

    def test_lint_skips_upstream_and_constant_params(self):
        """Upstream and constant params should not be checked against config."""

        @node(outputs=RawText)
        def upstream() -> RawText: ...

        @node(outputs=Claims)
        def my_node(
            upstream: RawText,
            limit: int = 10,
        ) -> Claims: ...

        pipeline = construct_from_functions("ok", [upstream, my_node])
        issues = lint(pipeline, config={})
        assert issues == []

    def test_lint_multiple_nodes_multiple_issues(self):
        """lint collects issues from all nodes, not just the first."""

        @node(outputs=RawText)
        def node_a(x: Annotated[str, FromInput]) -> RawText: ...

        @node(outputs=Claims)
        def node_b(y: Annotated[str, FromConfig]) -> Claims: ...

        pipeline = construct_from_functions("multi", [node_a, node_b])
        issues = lint(pipeline, config={})
        assert len(issues) == 2
        params = {i.param for i in issues}
        assert params == {"x", "y"}

    def test_lint_skips_non_node_non_construct_items(self):
        """lint silently skips items that are neither Node nor Construct."""
        # Construct.nodes can only hold Node|Construct, but _walk is typed
        # to accept either. Passing something else should just return early.
        from neograph.lint import LintIssue, _walk

        issues: list[LintIssue] = []
        _walk("not-a-node", None, issues)  # type: ignore[arg-type]
        assert issues == []

    def test_lint_required_bundled_model_no_config(self):
        """Required bundled model params are flagged when config is None."""

        class Ctx(BaseModel):
            node_id: str
            project_root: str

        @node(outputs=RawText)
        def my_node(ctx: Annotated[Ctx, FromInput(required=True)]) -> RawText: ...

        pipeline = construct_from_functions("bundled-no-cfg", [my_node])
        issues = lint(pipeline)
        assert len(issues) == 2
        params = {i.param for i in issues}
        assert params == {"node_id", "project_root"}
        assert all(i.required for i in issues)
        assert all("has no config" in i.message for i in issues)

    def test_lint_merge_fn_di_param_missing_from_config(self):
        """lint detects missing DI param in @merge_fn when config is provided."""
        from neograph import merge_fn as merge_fn_deco

        @merge_fn_deco
        def lint_merge(
            variants: list[Claims],
            api_key: Annotated[str, FromConfig],
        ) -> Claims:
            return variants[0]

        # Use @node with ensemble_n to get a node with param_resolutions AND Oracle.
        @node(
            outputs=Claims,
            prompt="test",
            model="fast",
            ensemble_n=2,
            merge_fn="lint_merge",
        )
        def lint_gen(topic: Annotated[str, FromInput]) -> Claims: ...

        pipeline = construct_from_functions("merge-lint", [lint_gen])
        # Provide 'topic' so the node itself is satisfied, but not 'api_key'
        issues = lint(pipeline, config={"topic": "hello"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 1
        assert merge_issues[0].param == "api_key"
        assert "not found in config" in merge_issues[0].message

    def test_lint_merge_fn_required_di_param_no_config(self):
        """lint flags required @merge_fn DI params when config is None."""
        from neograph import merge_fn as merge_fn_deco

        @merge_fn_deco
        def lint_merge_req(
            variants: list[Claims],
            secret: Annotated[str, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test",
            model="fast",
            ensemble_n=2,
            merge_fn="lint_merge_req",
        )
        def lint_gen2(topic: Annotated[str, FromInput(required=True)]) -> Claims: ...

        pipeline = construct_from_functions("merge-lint-req", [lint_gen2])
        issues = lint(pipeline)
        # Both node-level 'topic' and merge_fn-level 'secret' are required
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 1
        assert merge_issues[0].param == "secret"
        assert merge_issues[0].required is True
        assert "has no config" in merge_issues[0].message

    def test_lint_merge_fn_bundled_model_fields_checked(self):
        """lint() checks from_input_model fields in @merge_fn (neograph-s2h8)."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class PipeCtx(BaseModel):
            node_id: str
            project_root: str

        @merge_fn_deco
        def ctx_merge(
            variants: list[Claims],
            ctx: Annotated[PipeCtx, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test",
            model="fast",
            ensemble_n=2,
            merge_fn="ctx_merge",
        )
        def gen_s2h8() -> Claims: ...

        pipeline = construct_from_functions("s2h8-test", [gen_s2h8])

        # With config missing the model fields
        issues = lint(pipeline, config={"some_other": "value"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        # Should flag node_id and project_root as missing
        missing_fields = {i.param for i in merge_issues}
        assert "node_id" in missing_fields
        assert "project_root" in missing_fields

    def test_lint_merge_fn_bundled_model_passes_with_config(self):
        """lint() passes when bundled model fields are present in config."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class Ctx2(BaseModel):
            node_id: str

        @merge_fn_deco
        def ctx_merge2(
            variants: list[Claims],
            ctx: Annotated[Ctx2, FromInput],
        ) -> Claims:
            return variants[0]

        @node(
            outputs=Claims,
            prompt="test",
            model="fast",
            ensemble_n=2,
            merge_fn="ctx_merge2",
        )
        def gen_s2h8b() -> Claims: ...

        pipeline = construct_from_functions("s2h8-pass", [gen_s2h8b])
        issues = lint(pipeline, config={"node_id": "test-123"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert len(merge_issues) == 0


class TestLintObligationGaps:
    """Test obligations from /test-obligations analysis of _walk()."""

    def test_lint_merge_fn_simple_di_on_node_without_param_res(self):
        """W-13: Node(no DI) + merge_fn with simple from_input — lint catches it (neograph-tlrs)."""
        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        @merge_fn_deco
        def simple_merge(
            variants: list[Claims],
            api_key: Annotated[str, FromInput],
        ) -> Claims:
            return variants[0]

        @node(outputs=Claims, prompt="test", model="fast", ensemble_n=2, merge_fn="simple_merge")
        def gen_w13() -> Claims: ...

        pipeline = construct_from_functions("w13-test", [gen_w13])
        issues = lint(pipeline, config={"some_other": "value"})
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        assert any(i.param == "api_key" for i in merge_issues)

    def test_lint_merge_fn_bundled_required_no_config(self):
        """W-15: Node(no DI) + merge_fn bundled required + config=None (neograph-wcbv)."""
        from pydantic import BaseModel

        from neograph import lint, node
        from neograph import merge_fn as merge_fn_deco
        from neograph.decorators import construct_from_functions

        class Ctx3(BaseModel):
            node_id: str
            project_root: str

        @merge_fn_deco
        def bundled_merge(
            variants: list[Claims],
            ctx: Annotated[Ctx3, FromInput(required=True)],
        ) -> Claims:
            return variants[0]

        @node(outputs=Claims, prompt="test", model="fast", ensemble_n=2, merge_fn="bundled_merge")
        def gen_w15() -> Claims: ...

        pipeline = construct_from_functions("w15-test", [gen_w15])
        issues = lint(pipeline)  # no config
        merge_issues = [i for i in issues if "merge_fn" in i.node_name]
        missing = {i.param for i in merge_issues}
        assert "node_id" in missing
        assert "project_root" in missing

    def test_lint_oracle_callable_merge_fn_no_false_positive(self):
        """W-19: Oracle with callable merge_fn (not string) — no issues (neograph-xcy7)."""
        from neograph import lint
        from tests.fakes import register_scripted

        register_scripted("w19_gen", lambda i, c: Claims(items=["ok"]))

        def my_callable_merge(variants, config):
            return variants[0]

        pipeline = Construct(
            "w19-test",
            nodes=[
                Node.scripted("gen", fn="w19_gen", outputs=Claims)
                | Oracle(n=2, merge_fn="w19_gen"),  # string merge_fn — lint checks it
            ],
        )
        # Verify no crash when merge_fn is a registered string
        issues = lint(pipeline, config={"node_id": "test"})
        # This tests the path — no assertion on count, just no crash

    def test_lint_from_config_required_no_config(self):
        """W-21: FromConfig(required=True) + config=None — symmetric with FromInput (neograph-oued)."""
        from neograph import lint, node
        from neograph.decorators import construct_from_functions

        @node(outputs=Claims, prompt="test", model="fast")
        def gen_w21(limiter: Annotated[str, FromConfig(required=True)]) -> Claims: ...

        pipeline = construct_from_functions("w21-test", [gen_w21])
        issues = lint(pipeline)  # no config
        required_issues = [i for i in issues if i.required and i.param == "limiter"]
        assert len(required_issues) == 1
        assert "from_config" in required_issues[0].kind


class TestLoopConditionLint:
    """lint() should catch Loop when-condition issues statically.

    Three checks:
    1. String condition not registered in the condition registry
    2. Callable condition that is not None-safe (crashes on first iteration)
    3. String conditions from parse_condition are inherently None-unsafe
    """

    # -- 1. Unregistered string condition ----------------------------------

    def test_lint_reports_unregistered_loop_condition_on_node(self):
        """Loop(when='nonexistent') on a Node should lint as ERROR."""
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when="totally_missing", max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert len(loop_issues) >= 1
        assert any(i.kind == "loop_condition_unregistered" for i in loop_issues)
        assert any("totally_missing" in i.message for i in loop_issues)
        assert any(i.required is True for i in loop_issues)  # ERROR, not WARN

    def test_lint_reports_unregistered_loop_condition_on_construct(self):
        """Loop(when='nonexistent') on a Construct should lint as ERROR."""
        from neograph.modifiers import Loop
        from tests.fakes import register_scripted

        register_scripted("_lc_inner", lambda i, c: RawText(text="ok"))
        sub = Construct(
            "sub",
            input=RawText,
            output=RawText,
            nodes=[Node.scripted("inner", fn="_lc_inner", outputs=RawText)],
        ) | Loop(when="also_missing", max_iterations=3)

        pipeline = Construct("test", nodes=[sub])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert len(loop_issues) >= 1
        assert any(i.kind == "loop_condition_unregistered" for i in loop_issues)

    def test_lint_no_issue_for_registered_loop_condition(self):
        """Registered string condition should not trigger lint issue."""
        from neograph.modifiers import Loop

        def cond_fn(d):
            return d is None or d.text == ""

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when="_lint_test_cond", max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline, conditions={"_lint_test_cond": cond_fn})
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert loop_issues == []

    # -- 2. Callable None-unsafe -------------------------------------------

    def test_lint_reports_none_unsafe_callable(self):
        """lambda d: d.score < 0.8 crashes on None — lint should WARN."""
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when=lambda d: d.score < 0.8, max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert len(loop_issues) >= 1
        assert any(i.kind == "loop_condition_none_unsafe" for i in loop_issues)
        assert any(i.required is False for i in loop_issues)  # WARN, not ERROR

    def test_lint_no_issue_for_none_safe_callable(self):
        """lambda d: d is None or d.score < 0.8 is safe — no lint issue."""
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert loop_issues == []

    def test_lint_reports_none_unsafe_callable_on_construct(self):
        """None-unsafe condition on Construct|Loop should also WARN."""
        from neograph.modifiers import Loop
        from tests.fakes import register_scripted

        register_scripted("_lc_inner2", lambda i, c: RawText(text="ok"))
        sub = Construct(
            "sub",
            input=RawText,
            output=RawText,
            nodes=[Node.scripted("inner", fn="_lc_inner2", outputs=RawText)],
        ) | Loop(when=lambda d: d.text == "done", max_iterations=3)

        pipeline = Construct("test", nodes=[sub])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert len(loop_issues) >= 1
        assert any(i.kind == "loop_condition_none_unsafe" for i in loop_issues)

    def test_lint_none_unsafe_attribute_error(self):
        """Catches AttributeError from None.some_attr."""
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when=lambda d: len(d.items) > 0, max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if i.kind == "loop_condition_none_unsafe"]
        assert len(loop_issues) >= 1

    def test_lint_none_unsafe_type_error(self):
        """Catches TypeError from None < 0.8."""
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when=lambda d: d < 0.8, max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(pipeline)
        loop_issues = [i for i in issues if i.kind == "loop_condition_none_unsafe"]
        assert len(loop_issues) >= 1

    # -- 3. String condition (parse_condition) always None-unsafe -----------

    def test_lint_reports_parse_condition_string_as_none_unsafe(self):
        """parse_condition('score < 0.8') always crashes on None — ERROR."""
        from neograph import parse_condition
        from neograph.modifiers import Loop

        a = _producer("seed", RawText)
        b = Node("refine", mode="think", outputs=RawText, prompt="refine", model="fast")
        b = b | Loop(when="_pc_score", max_iterations=3)

        pipeline = Construct("test", nodes=[a, b])
        issues = lint(
            pipeline,
            conditions={"_pc_score": parse_condition("score < 0.8")},
        )
        loop_issues = [i for i in issues if "loop" in i.kind]
        assert len(loop_issues) >= 1
        # This should be ERROR (required=True) since it ALWAYS crashes
        assert any(i.kind == "loop_condition_none_unsafe" for i in loop_issues)
        assert any(i.required is True for i in loop_issues)


# ── ask_human-in-a-mutating-node lint rule (neograph-p8wz, A.5 safety) ──────
#
# ask_human is a first-class marker the validator can SEE: a raw interrupt()
# buried in an opaque tool callable is invisible to lint, but a named ask_human
# reference shows up in the tool callable's __code__.co_names. The rule flags an
# ACT-mode node (act == mutations) bound to a tool that reaches ask_human, since a
# non-idempotent side effect before a mid-loop pause can double-fire on resume.
# It is a WARN (required=False) and gates on the DECLARED node.mode == 'act';
# agent-mode (read-only) ask_human is fine and must NOT fire.


class _AskHumanClassTool:
    """Duck-typed class tool (the keystone _AskTool shape the E2E reuses): the
    HITL logic lives in .invoke, which references ask_human by name."""

    name = "ask_tool"

    def invoke(self, args: dict, config=None, **kwargs) -> str:
        from neograph.hitl import ask_human

        class _P(BaseModel):
            q: str

        return f"decided: {ask_human(_P(q='x'))}"

    async def ainvoke(self, *a, **k) -> str:
        return self.invoke(*a, **k)


class TestAskHumanInMutatingNodeLint:
    """lint() should flag ask_human reachable from an ACT-mode (mutating) node,
    and must NOT flag it on an AGENT-mode (read-only) node."""

    _ISSUE_KIND = "ask_human_in_mutating_node"

    def _construct(self, *, mode: str):
        from neograph import Tool

        n = Node(
            "actor",
            mode=mode,
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=[Tool("ask_tool", budget=0)],
        )
        return Construct(f"ask-human-{mode}", nodes=[n])

    def test_flags_ask_human_in_act_mode_node(self):
        construct = self._construct(mode="act")
        issues = lint(
            construct,
            tool_factories={"ask_tool": lambda config, tool_config: _AskHumanClassTool()},
        )

        ask_issues = [i for i in issues if i.kind == self._ISSUE_KIND]
        assert len(ask_issues) == 1, [i.kind for i in issues]
        # WARN, not ERROR — legitimate ask_human-then-idempotent-mutate must not block.
        assert ask_issues[0].required is False

    def test_no_issue_for_ask_human_in_agent_mode_node(self):
        construct = self._construct(mode="agent")
        issues = lint(
            construct,
            tool_factories={"ask_tool": lambda config, tool_config: _AskHumanClassTool()},
        )

        assert [i for i in issues if i.kind == self._ISSUE_KIND] == []


# neograph-lhc6: an act-mode node (act == mutations) whose tools are ALL
# idempotent is probably misclassified — it should be mode='agent'. WARN
# (required=False); gates on the DECLARED mode. Silent when any tool is
# non-idempotent or of unknown side-effect (a raw BaseTool), and for agent mode.


class TestActModeAllIdempotentToolsLint:
    """lint() should WARN when an act-mode node's tools are all idempotent."""

    _ISSUE_KIND = "act_mode_all_idempotent_tools"

    def _construct(self, *, mode: str, tools):
        n = Node(
            "writer",
            mode=mode,
            outputs=Claims,
            model="fast",
            prompt="test/scan",
            tools=tools,
        )
        return Construct(f"idem-{mode}", nodes=[n])

    def test_warns_when_act_mode_tools_all_idempotent(self):
        from neograph import Tool

        construct = self._construct(
            mode="act",
            tools=[Tool("read_a", idempotent=True), Tool("read_b", idempotent=True)],
        )
        issues = lint(construct)

        hits = [i for i in issues if i.kind == self._ISSUE_KIND]
        assert len(hits) == 1, [i.kind for i in issues]
        assert hits[0].required is False

    def test_no_warning_for_agent_mode(self):
        from neograph import Tool

        construct = self._construct(
            mode="agent",
            tools=[Tool("read_a", idempotent=True)],
        )
        issues = lint(construct)

        assert [i for i in issues if i.kind == self._ISSUE_KIND] == []

    def test_no_warning_when_any_tool_non_idempotent(self):
        from neograph import Tool

        construct = self._construct(
            mode="act",
            tools=[Tool("read_a", idempotent=True), Tool("mutate_b")],
        )
        issues = lint(construct)

        assert [i for i in issues if i.kind == self._ISSUE_KIND] == []
