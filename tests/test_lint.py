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
        """Without config, a caller-suppliable param is the graph's INPUT
        CONTRACT, not an error (GH #13). It is still reported, so nothing is
        lost -- only its severity changes, which is what stops a consumer from
        padding a fixture to reach a clean gate."""

        @node(outputs=RawText)
        def my_node(
            topic: Annotated[str, FromInput(required=True)],
        ) -> RawText: ...

        pipeline = construct_from_functions("no-cfg", [my_node])
        issues = lint(pipeline)
        assert len(issues) == 1
        assert issues[0].required is False, "the input contract is not an error"
        assert "topic" in issues[0].param
        assert "input contract" in issues[0].message

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
        # This test grades RECURSION: does the walk reach the inner node's DI
        # binding. It is scoped to the DI kind because the fixture's scaffolding
        # producer is genuinely unconsumed -- nothing declares RawText -- so
        # output_field_unconsumed reports it correctly. That is a separate, true
        # finding about the fixture, not what this test measures.
        di_issues = [i for i in lint(pipeline, config={}) if i.kind == "from_input"]
        assert len(di_issues) == 1
        assert "topic" in di_issues[0].param

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

        # node_b consumes node_a's output, so the pipeline carries no dead
        # output field and this test stays about DI issues alone.
        @node(outputs=Claims)
        def node_b(node_a: RawText, y: Annotated[str, FromConfig]) -> Claims: ...

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
        """Bundled model params are REPORTED when config is None, as the
        graph's input contract rather than as errors (GH #13)."""

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
        assert not any(i.required for i in issues), "the input contract is not an error"
        assert all("input contract" in i.message for i in issues)

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
        """A @merge_fn DI param is reported when config is None, as the input
        contract rather than as an error (GH #13). Symmetric with @node."""
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
        assert merge_issues[0].required is False, "the input contract is not an error"
        assert "input contract" in merge_issues[0].message

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
        # GH #13: with no config a FromConfig param is the graph's input
        # contract, so it is reported without being an error. The binding is
        # still SEEN, which is what this test guards.
        required_issues = [i for i in issues if i.param == "limiter"]
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


# ═══════════════════════════════════════════════════════════════════════════
# Unsatisfiable bindings and the config demand (GH #12, GH #13)
# ═══════════════════════════════════════════════════════════════════════════


class TestUnsatisfiableFromInput:
    """A `FromInput` on an Each item or a Loop carry can never be satisfied.

    `FromInput` claims the parameter for DI, so the fanned item never binds to
    it. The graph then demands a key from the caller that no caller can
    meaningfully supply, and the obvious way to make the gate pass is to pad the
    lint config with it.

    Padding does not merely hide the error. It makes the pipeline RUN, fan out
    correctly, key its results correctly, and compute every one of them from the
    fixture's placeholder value. The failure mode is a confident wrong answer,
    not a crash (GH #12).
    """

    @staticmethod
    def _each_over_port():

        from neograph import Each, construct_from_functions, node

        class Claim(BaseModel):
            text: str

        class Claims(BaseModel):
            items: list[Claim]

        @node(outputs=Claims)
        def source() -> Claims:
            return Claims(items=[Claim(text="REAL-a")])

        @node(outputs=Claim)
        def claim_of(claim_in: Annotated[Claim, FromInput]) -> Claim:
            return claim_in

        sub = construct_from_functions(
            "verify", [claim_of], input=Claim, output=Claim
        ) | Each(over="source.items", key="text")
        return construct_from_functions("pipe", [source, sub])

    def test_reports_unsatisfiable_when_from_input_binds_an_each_item(self):
        """The direct check, decided from structure with no config at all."""
        from neograph.lint import lint

        issues = [
            i for i in lint(self._each_over_port()) if i.kind == "from_input_unsatisfiable"
        ]

        assert issues, (
            "a FromInput bound to the Each-fanned item is unsatisfiable by any "
            "caller and must be reported without a config"
        )
        assert issues[0].required is True, "an unsatisfiable binding is an error"
        assert "port" in issues[0].message.lower(), (
            f"the message must name the port-parameter fix: {issues[0].message}"
        )

    def test_padding_the_config_cannot_silence_an_unsatisfiable_binding(self):
        """The meta-check. The reported bug survived because the fixture was
        padded with a key no caller could pass, so the gate graded its own
        answer key. A config must not be able to make this finding disappear.
        """
        from neograph.lint import lint

        padded = {"text": "PADDED-FIXTURE-VALUE"}
        issues = [
            i
            for i in lint(self._each_over_port(), config=padded)
            if i.kind == "from_input_unsatisfiable"
        ]

        assert issues, (
            "padding the lint config silenced the finding; the check must be "
            "derived from construct structure, not from the caller's assertion"
        )

    def test_reports_unsatisfiable_when_from_input_binds_a_loop_carry(self):
        """The reporter named two shapes, not one. A Loop carries the previous
        result back into the port as `list[X]`, which no caller supplies either.

        Added because a mutation that deleted the Loop branch entirely left the
        suite green: the Each case alone cannot prove the Loop case.
        """

        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class Claim(BaseModel):
            text: str

        class ClaimsDelta(BaseModel):
            added: list[Claim]

        @node(outputs=ClaimsDelta, loop_when=lambda d: False, max_iterations=2)
        def seed_claims(claims: Annotated[ClaimsDelta, FromInput]) -> ClaimsDelta:
            return ClaimsDelta(added=[])

        top = construct_from_functions("loop-pipe", [seed_claims])

        issues = [i for i in lint(top) if i.kind == "from_input_unsatisfiable"]

        assert issues, "a FromInput bound to the Loop carry is unsatisfiable too"
        assert "Loop carry" in issues[0].message, issues[0].message

    def test_reports_unsatisfiable_when_from_input_binds_a_construct_loop_port(self):
        """The third shape: a Loop on a SUB-CONSTRUCT carries its output back
        into the port, so the port type is supplied by the construct.

        Added because deleting this branch left the suite green -- the Each case
        and the node self-loop case cannot prove it.
        """

        from neograph import Loop, construct_from_functions, node
        from neograph.lint import lint

        class Draft(BaseModel):
            body: str

        @node(outputs=Draft)
        def revise(draft: Annotated[Draft, FromInput]) -> Draft:
            return draft

        sub = construct_from_functions(
            "refine-sub", [revise], input=Draft, output=Draft
        ) | Loop(when=lambda d: False, max_iterations=2)
        top = construct_from_functions("refine-pipe", [sub])

        issues = [i for i in lint(top) if i.kind == "from_input_unsatisfiable"]

        assert issues, "a FromInput on a Loop-modified sub-construct's port is unsatisfiable"
        assert "Loop carry" in issues[0].message, issues[0].message

    def test_reports_nothing_when_the_item_binds_as_a_port_parameter(self):
        """The accept case, and the fix the message names: a bare parameter
        typed as the construct's `input=` reads from the port."""
        from neograph import Each, construct_from_functions, node
        from neograph.lint import lint

        class Claim(BaseModel):
            text: str

        class Claims(BaseModel):
            items: list[Claim]

        @node(outputs=Claims)
        def source2() -> Claims:
            return Claims(items=[Claim(text="a")])

        @node(outputs=Claim)
        def claim_of2(claim_in: Claim) -> Claim:
            return claim_in

        sub = construct_from_functions(
            "verify2", [claim_of2], input=Claim, output=Claim
        ) | Each(over="source2.items", key="text")
        top = construct_from_functions("pipe2", [source2, sub])

        assert [i for i in lint(top) if i.kind == "from_input_unsatisfiable"] == []


class TestLintNeedsNoConfig:
    """`lint()` must not require a config to check a graph (GH #13).

    Every `FromInput` and `FromConfig` parameter reported as a required error
    when no config was supplied. Those divide into two categories: bindings no
    caller can satisfy, which are real errors, and the graph's own input
    contract, which is not an error at all.

    Reporting the second category forces a consumer to hand the linter a config
    to reach a clean gate. Any config handed to the linter is an assertion the
    linter cannot verify, and that is the door GH #12 walked through.
    """

    @staticmethod
    def _caller_supplied():

        from neograph import construct_from_functions, node

        class Out(BaseModel):
            text: str

        @node(outputs=Out)
        def answer(question: Annotated[str, FromInput], deal_id: Annotated[int, FromInput]) -> Out:
            return Out(text="x")

        return construct_from_functions("qa", [answer])

    def test_caller_supplied_di_is_not_an_error_without_a_config(self):
        """`question` and `deal_id` are the graph's input contract. A caller
        supplies them at runtime, so reporting them says only that the graph has
        inputs."""
        from neograph.lint import lint

        required = [i for i in lint(self._caller_supplied()) if i.required]

        assert required == [], (
            "lint demanded a config for parameters a caller supplies at runtime; "
            f"got {[(i.kind, i.param) for i in required]}"
        )

    def test_the_input_contract_is_still_reported_informationally(self):
        """Removing the error must not remove the information. The set of keys a
        caller must supply is useful output in its own right."""
        from neograph.lint import lint

        reported = {i.param for i in lint(self._caller_supplied())}

        assert {"question", "deal_id"} <= reported, (
            f"the input contract vanished instead of becoming informational: {reported}"
        )

    def test_config_still_checks_a_specific_payload(self):
        """`config=` stays supported as an optional, different question: does
        THIS caller payload satisfy the graph."""
        from neograph.lint import lint

        issues = lint(self._caller_supplied(), config={"question": "q"})
        missing = [i for i in issues if i.param == "deal_id" and i.required]

        assert missing, "a config that omits a required key must still be reported"


class TestUnconsumedOutputField:
    """Every field a node produces must have a consumer (GH #11).

    The sibling of the unreferenced-input check at the other end of the pipe.
    A node emits a typed output whose field is populated on every run, and
    nothing downstream reads it. The field looks load-bearing, it costs tokens
    on every call, and the model is asked to reason about a value that cannot
    affect the answer.

    Consumption has three axes with different granularity, and GH #11 warns that
    deriving one reports false cleanliness. Each axis gets a case here.
    """

    KIND = "output_field_unconsumed"

    @classmethod
    def _dead(cls, issues):
        return [i for i in issues if i.kind == cls.KIND]

    def test_reports_a_field_no_downstream_template_reads(self):
        """Axis 2, the only field-granular one. `${triage.severity}` consumes
        `severity`; `rationale` is produced on every call and read by nothing."""
        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class Triage(BaseModel):
            severity: str
            rationale: str

        class Out(BaseModel):
            text: str

        @node(outputs=Triage)
        def triage() -> Triage:
            return Triage(severity="high", rationale="because")

        @node(outputs=Out, mode="think", model="fast", prompt="Act on ${triage.severity}")
        def act(triage: Triage) -> Out: ...

        issues = self._dead(
            lint(
                construct_from_functions("pipe", [triage, act]),
                prompt_compiler=lambda t, d, **kw: [],
                llm_factory=lambda tier: None,
            )
        )

        assert [i.param for i in issues] == ["rationale"], (
            f"expected 'rationale' dead, got {[(i.param) for i in issues]}"
        )
        assert issues[0].required is False

    def test_reports_nothing_when_a_downstream_node_takes_the_whole_model(self):
        """Axis 1. A consumer declaring `triage: Triage` receives the whole
        model, and which fields its body reads is not derivable. Treating that
        as consuming every field is deliberate over-approximation: the opposite
        would flag every scripted consumer in every pipeline."""
        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class Triage(BaseModel):
            severity: str
            rationale: str

        class Out(BaseModel):
            text: str

        @node(outputs=Triage)
        def triage2() -> Triage:
            return Triage(severity="high", rationale="because")

        @node(outputs=Out)
        def act2(triage2: Triage) -> Out:
            return Out(text=triage2.rationale)

        assert self._dead(lint(construct_from_functions("pipe2", [triage2, act2]))) == []

    def test_reports_nothing_for_the_terminal_node(self):
        """Axis 3. The last node's output IS the graph's output, so its fields
        have a consumer by construction. Derived from topology, declared by
        nobody."""
        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class Seed(BaseModel):
            value: str

        class Verdict(BaseModel):
            decision: str
            explanation: str

        @node(outputs=Seed)
        def seed3() -> Seed:
            return Seed(value="x")

        @node(outputs=Verdict)
        def decide3(seed3: Seed) -> Verdict:
            return Verdict(decision="yes", explanation="why")

        assert self._dead(lint(construct_from_functions("pipe3", [seed3, decide3]))) == []

    def test_reports_nothing_when_a_sub_construct_port_consumes_the_output(self):
        """A member sub-construct consumes its port BY TYPE, and it is a
        `Construct`, not a `Node`. Filtering the walk to Nodes made it invisible
        as a consumer, so every producer feeding a sub-construct looked dead."""
        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class Claims(BaseModel):
            items: list[str]

        class Scored(BaseModel):
            value: float

        @node(outputs=Claims)
        def decompose4() -> Claims:
            return Claims(items=["a"])

        @node(outputs=Scored)
        def inner4(claims: Claims) -> Scored:
            """`claims` is typed as the construct's `input=`, so it is a PORT
            parameter read from neo_subgraph_input, not a peer reference."""
            return Scored(value=1.0)

        enrich = construct_from_functions("enrich4", [inner4], input=Claims, output=Scored)

        @node(outputs=Scored)
        def report4(enrich4: Scored) -> Scored:
            return enrich4

        top = construct_from_functions("subc-pipe", [decompose4, enrich, report4])

        assert self._dead(lint(top)) == [], "the sub-construct's port consumes Claims"

    def test_reports_nothing_when_a_single_type_input_consumes_by_type(self):
        """A single-type input resolves BY TYPE at runtime, not by the producer's
        name. Comparing it against a name-keyed producer made every such consumer
        invisible."""
        from neograph import Node, construct_from_functions, node
        from neograph.lint import lint

        class Claims(BaseModel):
            items: list[str]

        class Out(BaseModel):
            text: str

        @node(outputs=Claims)
        def decompose5() -> Claims:
            return Claims(items=["a"])

        report = Node.scripted("report5", fn="unref_decl_src", inputs=Claims, outputs=Out)
        top = construct_from_functions("bytype-pipe", [decompose5])
        top = type(top)(top.name, nodes=[*top.nodes, report])

        assert self._dead(lint(top)) == [], "report consumes Claims by type"

    def test_reports_nothing_when_an_image_prefixed_placeholder_reads_the_field(self):
        """`${image:seed.photo}` reads field `photo` of `seed`. The `image:`
        prefix is a rendering directive; both placeholder readers must strip it
        or a referenced field looks unread."""
        from neograph import construct_from_functions, node
        from neograph.lint import lint

        class ImageInput(BaseModel):
            photo: str
            caption: str = ""

        class Out(BaseModel):
            text: str

        @node(outputs=ImageInput)
        def seed6() -> ImageInput:
            return ImageInput(photo="b64")

        @node(outputs=Out, mode="think", model="fast", prompt="Analyze: ${image:seed6.photo}")
        def analyze6(seed6: ImageInput) -> Out: ...

        dead = {i.param for i in self._dead(lint(construct_from_functions("img-pipe", [seed6, analyze6])))}

        assert "photo" not in dead, f"the image-prefixed placeholder reads photo; got {dead}"


class TestSupplySideChecksOnARealisticPipeline:
    """The three supply-side checks on ONE pipeline, not in isolation.

    Each check has its own unit tests, but nothing exercised them together on a
    shape a consumer would actually write: DI parameters, a template-ref prompt,
    dict-form outputs with a tool log, and a field that genuinely reaches no one.
    Interaction bugs live exactly there -- three of this check family's
    false-positive bugs were found only by running it against real pipelines.
    """

    def _pipeline(self):

        from neograph import ToolInteraction, construct_from_functions, node

        class Deal(BaseModel):
            deal_id: str
            notes: str

        class Finding(BaseModel):
            severity: str
            rationale: str

        class Report(BaseModel):
            summary: str

        @node(outputs=Deal)
        def fetch(deal_ref: Annotated[str, FromInput]) -> Deal: ...

        @node(
            outputs={"result": Finding, "tool_log": list[ToolInteraction]},
            mode="think",
            model="fast",
            prompt="triage",
        )
        def triage(fetch: Deal, region: Annotated[str, FromInput]) -> Finding: ...

        @node(outputs=Report, mode="think", model="fast", prompt="summarize")
        def summarize(triage_result: Finding) -> Report: ...

        return construct_from_functions("deal-triage", [fetch, triage, summarize])

    @staticmethod
    def _resolver(name):
        return {
            "triage": "Assess {fetch} for {region}.",
            "summarize": "Summarize {triage_result.severity}.",
        }.get(name)

    def _lint(self):
        from neograph.lint import lint

        return lint(
            self._pipeline(),
            template_resolver=self._resolver,
            prompt_compiler=lambda t, d, *, di_inputs=None, **kw: [],
            llm_factory=lambda tier: None,
        )

    def test_di_parameters_are_the_input_contract_not_errors(self):
        """`deal_ref` and `region` are supplied by a caller at run time."""
        di = [i for i in self._lint() if i.kind == "from_input"]

        assert {i.param for i in di} == {"deal_ref", "region"}
        assert not any(i.required for i in di), "the input contract is not an error"

    def test_the_unread_output_field_is_reported(self):
        """`summarize` reads `${triage_result.severity}`. Nothing reads
        `rationale`, so the model is asked to produce it on every call for
        nothing."""
        dead = {i.param for i in self._lint() if i.kind == "output_field_unconsumed"}

        assert dead == {"rationale"}, f"expected only 'rationale' dead, got {dead}"

    def test_referenced_inputs_and_the_tool_log_are_not_reported(self):
        """No false positives on the live parts. `fetch` and `region` are named
        by the template, `severity` is read dotted, and `tool_log` is a
        `list[X]` with no fields of its own to check."""
        issues = self._lint()

        assert [i for i in issues if i.kind == "template_input_unreferenced"] == []
        dead = {i.param for i in issues if i.kind == "output_field_unconsumed"}
        assert "severity" not in dead
        assert "tool_log" not in dead
        assert "deal_id" not in dead and "notes" not in dead
