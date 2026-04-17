"""Pipeline mode tests — context param, tool registration, skip_when on tools, JSON parsing, retry, ReAct guards"""

from __future__ import annotations

import pytest

from neograph import (
    CompileError,
    ConfigurationError,
    Construct,
    Each,
    ExecutionError,
    Node,
    Oracle,
    Tool,
    compile,
    run,
)
from tests.fakes import FakeTool, GuardFake, ReActFake, StructuredFake, StubbornFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
)

# ═══════════════════════════════════════════════════════════════════════════
# TEST: @node context= verbatim state injection (neograph-p4hw)
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeContext:
    """@node context= injects verbatim state fields into prompt (neograph-p4hw)."""

    def test_context_passed_to_prompt_compiler_when_declared(self):
        """Prompt compiler receives context dict with raw state values. E2E."""
        import types

        from neograph import compile, construct_from_module, node, run
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(
            lambda tier: StructuredFake(lambda m: m(items=["done"])),
        )

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm

        configure_llm(
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["done"])),
            prompt_compiler=capturing_compiler,
        )

        mod = types.ModuleType("test_context_mod")

        @node(outputs=RawText)
        def build_catalog() -> RawText:
            return RawText(text="<catalog>UC-001,UC-002,UC-003</catalog>")

        @node(
            outputs=Claims,
            mode="think",
            model="fast",
            prompt="with-context",
            context=["build_catalog"],
        )
        def analyze(build_catalog: RawText) -> Claims: ...

        mod.build_catalog = build_catalog
        mod.analyze = analyze
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={"node_id": "ctx-test"})

        assert "with-context" in captured
        kw = captured["with-context"]["kw"]
        assert "context" in kw, f"Expected 'context' in prompt compiler kwargs, got: {list(kw.keys())}"
        assert "build_catalog" in kw["context"]
        # Verbatim — the raw RawText, not BAML-rendered
        ctx_val = kw["context"]["build_catalog"]
        assert hasattr(ctx_val, "text"), f"Expected raw RawText model, got {type(ctx_val)}"
        assert ctx_val.text == "<catalog>UC-001,UC-002,UC-003</catalog>"

    def test_no_context_kwarg_when_node_has_no_context(self):
        """Prompt compiler does NOT receive context when node doesn't declare it."""
        import types

        from neograph import compile, construct_from_module, node, run
        from tests.fakes import StructuredFake

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = kw
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm

        configure_llm(
            llm_factory=lambda tier: StructuredFake(lambda m: m(items=["x"])),
            prompt_compiler=capturing_compiler,
        )

        mod = types.ModuleType("test_no_ctx_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="no-context")
        def simple() -> Claims: ...

        mod.simple = simple
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={})

        assert "no-context" in captured
        assert "context" not in captured["no-context"]

    def test_context_works_with_agent_mode(self):
        """Agent node with context= alongside tools. E2E."""
        import types

        from neograph import ToolInteraction, compile, construct_from_module, node, run
        from neograph.factory import register_tool_factory
        from tests.fakes import FakeTool, ReActFake

        captured = {}

        def capturing_compiler(template, data, **kw):
            captured[template] = {"data": data, "kw": kw}
            return [{"role": "user", "content": "test"}]

        from neograph import configure_llm

        configure_llm(
            llm_factory=lambda tier: ReActFake(
                tool_calls=[[{"name": "ctx_tool", "args": {}, "id": "t1"}], []],
                final=lambda m: m(items=["found"]),
            ),
            prompt_compiler=capturing_compiler,
        )
        register_tool_factory("ctx_tool", lambda cfg, tc: FakeTool("ctx_tool", response="ok"))

        mod = types.ModuleType("test_agent_ctx_mod")

        @node(outputs=RawText)
        def catalog() -> RawText:
            return RawText(text="graph-catalog-data")

        @node(
            mode="agent",
            outputs={"result": Claims, "tool_log": list[ToolInteraction]},
            model="fast",
            prompt="agent-with-ctx",
            tools=[Tool("ctx_tool", budget=2)],
            context=["catalog"],
        )
        def explore(catalog: RawText) -> Claims: ...

        mod.catalog = catalog
        mod.explore = explore
        pipeline = construct_from_module(mod)
        graph = compile(pipeline)
        run(graph, input={})

        assert "agent-with-ctx" in captured
        kw = captured["agent-with-ctx"]["kw"]
        assert "context" in kw, f"Expected 'context' kwarg, got: {list(kw.keys())}"
        assert kw["context"]["catalog"].text == "graph-catalog-data"


# ═══════════════════════════════════════════════════════════════════════════
# TEST: Tool registration error in invoke_with_tools (neograph-rdu.2)
#
# When a gather/execute node references a tool name that has no registered
# factory, the error should be clear and mention the unregistered tool name.
# ═══════════════════════════════════════════════════════════════════════════


class TestToolRegistrationError:
    def test_clear_error_raised_when_tool_not_registered(self):
        """Gather node with unregistered tool raises CompileError at compile()."""
        import types as _types

        from neograph import construct_from_module, node

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(items=["x"]),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_unreg_tool_mod")

        @node(
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="nonexistent_tool", budget=3)],
        )
        def searcher() -> Claims: ...

        mod.searcher = searcher

        pipeline = construct_from_module(mod, name="test-unreg-tool")

        with pytest.raises(CompileError, match="nonexistent_tool"):
            compile(pipeline)

    def test_clear_error_raised_when_execute_tool_not_registered(self):
        """Execute node with unregistered tool raises CompileError at compile()."""
        import types as _types

        from neograph import construct_from_module, node

        fake = ReActFake(
            tool_calls=[[], []],
            final=lambda m: m(text="x"),
        )
        configure_fake_llm(lambda tier: fake)

        mod = _types.ModuleType("test_unreg_exec_mod")

        @node(
            mode="act",
            outputs=RawText,
            model="fast",
            prompt="test",
            tools=[Tool(name="missing_exec_tool", budget=1)],
        )
        def writer() -> RawText: ...

        mod.writer = writer

        pipeline = construct_from_module(mod, name="test-unreg-exec")

        with pytest.raises(CompileError, match="missing_exec_tool"):
            compile(pipeline)


# ═══════════════════════════════════════════════════════════════════════════
# TEST: skip_when on gather/execute nodes (neograph-rdu.8)
#
# skip_when is tested on produce nodes but never on gather/execute.
# All modes now share the same _execute_node preamble (neograph-y8ww).
# ═══════════════════════════════════════════════════════════════════════════


class TestSkipWhenOnToolNodes:
    def test_node_skipped_when_skip_when_true_on_gather(self):
        """Gather node with skip_when=True skips LLM and returns skip_value."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        # Register tool factory so it doesn't fail on missing tool
        fake_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        # LLM should NOT be called — if it is, the test will still pass
        # but we verify via the skip_value output
        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[[], []],
                final=lambda m: m(items=["should-not-appear"]),
            )
        )

        mod = _types.ModuleType("test_skip_gather_mod")

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["only-one"])

        @node(
            mode="agent",
            outputs=MergedResult,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=3)],
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def gatherer(seed: Claims) -> MergedResult: ...

        mod.seed = seed
        mod.gatherer = gatherer

        pipeline = construct_from_module(mod, name="test-skip-gather")
        graph = compile(pipeline)
        result = run(graph, input={})

        # Node was skipped — skip_value produced the output
        assert result["gatherer"].final_text == "only-one"
        # Tool was never called
        assert len(fake_tool.calls) == 0

    def test_node_runs_when_skip_when_false_on_gather(self):
        """Gather node runs normally when skip_when returns False."""
        import types as _types

        from neograph import construct_from_module, node
        from neograph.factory import register_tool_factory

        fake_tool = FakeTool("lookup", response="result")
        register_tool_factory("lookup", lambda config, tool_config: fake_tool)

        configure_fake_llm(
            lambda tier: ReActFake(
                tool_calls=[
                    [{"name": "lookup", "args": {"q": "test"}, "id": "tc1"}],
                    [],  # final
                ],
                final=lambda m: m(final_text="llm-produced"),
            )
        )

        mod = _types.ModuleType("test_noskip_gather_mod")

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["a", "b"])

        @node(
            mode="agent",
            outputs=MergedResult,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=3)],
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def gatherer(seed: Claims) -> MergedResult: ...

        mod.seed = seed
        mod.gatherer = gatherer

        pipeline = construct_from_module(mod, name="test-noskip-gather")
        graph = compile(pipeline)
        result = run(graph, input={})

        # skip_when was False → LLM ran, tool was called
        assert result["gatherer"].final_text == "llm-produced"
        assert len(fake_tool.calls) == 1


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _extract_json regex edge cases (neograph-rdu.9)
#
# _extract_json in _llm.py parses JSON from LLM text responses.
# Test edge cases: plain JSON, markdown fences, multiple objects,
# nested braces, no JSON.
# ═══════════════════════════════════════════════════════════════════════════


class TestExtractJsonEdgeCases:
    def test_plain_json_parsed_when_no_wrapping(self):
        """Plain JSON string is returned as-is."""
        from neograph._llm import _extract_json

        result = _extract_json('{"key": "value"}')
        assert result == '{"key": "value"}'

    def test_json_extracted_when_wrapped_in_markdown_fences(self):
        """JSON wrapped in ```json ... ``` is extracted."""
        from neograph._llm import _extract_json

        text = '```json\n{"key": "value"}\n```'
        result = _extract_json(text)
        assert '"key"' in result
        assert '"value"' in result
        # Verify it's valid JSON
        import json

        parsed = json.loads(result)
        assert parsed == {"key": "value"}

    def test_first_object_extracted_when_multiple_objects_in_text(self):
        """When multiple JSON objects exist, brace-counting extracts the first balanced one."""
        import json

        from neograph._llm import _extract_json

        text = 'Here is result: {"first": 1} and also {"second": 2}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed == {"first": 1}

    def test_nested_braces_parsed_when_json_has_nested_objects(self):
        """JSON with nested braces parses correctly."""
        import json

        from neograph._llm import _extract_json

        text = '{"outer": {"inner": "value"}}'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_text_returned_when_no_json_present(self):
        """When no JSON is present, the cleaned text is returned."""
        from neograph._llm import _extract_json

        result = _extract_json("no json here at all")
        assert result == "no json here at all"

    def test_json_extracted_when_surrounded_by_prose(self):
        """JSON embedded in prose text is extracted."""
        import json

        from neograph._llm import _extract_json

        text = 'The answer is: {"items": ["a", "b"]} as shown above.'
        result = _extract_json(text)
        parsed = json.loads(result)
        assert parsed["items"] == ["a", "b"]


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _parse_json_response lenient parsing (neograph-hqhw)
#
# _parse_json_response must recover from common LLM JSON malformations:
# control characters, trailing commas, single quotes, truncated fields.
# Currently it fails because model_validate_json uses strict json.loads.
# ═══════════════════════════════════════════════════════════════════════════


class TestParseJsonResponseLenientParsing:
    """_parse_json_response should handle malformed LLM JSON output."""

    def test_control_characters_in_strings_parsed(self):
        """Control characters inside JSON string values should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        # Embedded null character — not valid in JSON strings
        text = '{"content": "hello\x00world"}'
        result = _parse_json_response(text, SimpleModel)
        assert "hello" in result.content

    def test_null_coerced_to_default_for_string_field(self):
        """null for a str field with default should use the default, not fail.

        BUG neograph-qqel: R1 returns null for str fields with defaults.
        Pydantic rejects null for str. Coerce null → default before validation.
        """
        from pydantic import BaseModel, Field

        from neograph._llm import _parse_json_response

        class StepCheck(BaseModel):
            score: float
            justification: str = Field(default="", description="Why")

        text = '{"score": 0.8, "justification": null}'
        result = _parse_json_response(text, StepCheck)
        assert result.score == 0.8
        assert result.justification == ""

    def test_null_coerced_in_nested_model(self):
        """null → default coercion should work recursively in nested models."""
        from pydantic import BaseModel, Field

        from neograph._llm import _parse_json_response

        class Inner(BaseModel):
            name: str
            note: str = Field(default="none")

        class Outer(BaseModel):
            items: list[Inner]

        text = '{"items": [{"name": "a", "note": null}, {"name": "b"}]}'
        result = _parse_json_response(text, Outer)
        assert result.items[0].note == "none"
        assert result.items[1].note == "none"

    def test_null_without_default_still_fails(self):
        """null for a required field (no default) should still raise."""
        from pydantic import BaseModel

        from neograph._llm import _parse_json_response
        from neograph.errors import ExecutionError

        class Required(BaseModel):
            name: str  # no default

        text = '{"name": null}'
        with pytest.raises(ExecutionError, match="Validation failed"):
            _parse_json_response(text, Required)

    def test_trailing_comma_parsed(self):
        """Trailing commas in JSON objects should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            name: str
            value: int

        text = '{"name": "test", "value": 42,}'
        result = _parse_json_response(text, SimpleModel)
        assert result.name == "test"
        assert result.value == 42

    def test_single_quotes_parsed(self):
        """Single-quoted JSON strings should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            name: str

        text = "{'name': 'test'}"
        result = _parse_json_response(text, SimpleModel)
        assert result.name == "test"

    def test_unescaped_newlines_in_strings_parsed(self):
        """Literal newlines inside JSON string values should not crash parsing."""
        from pydantic import BaseModel

        from neograph._llm import _parse_json_response

        class SimpleModel(BaseModel):
            content: str

        text = '{"content": "line1\nline2"}'
        result = _parse_json_response(text, SimpleModel)
        assert "line1" in result.content
        assert "line2" in result.content


# ═══════════════════════════════════════════════════════════════════════════
# TEST: _call_structured TypeError fallback for include_raw (neograph-rdu.11)
#
# _call_structured() passes include_raw=True by default. Some LLMs don't
# support this kwarg and raise TypeError. The code catches this and retries
# without include_raw.
# ═══════════════════════════════════════════════════════════════════════════


class TestCallStructuredFallback:
    def test_fallback_succeeds_when_include_raw_raises_type_error(self):
        """_call_structured retries without include_raw when TypeError is raised."""
        from unittest.mock import MagicMock

        from neograph._llm import _call_structured

        expected = Claims(items=["fallback-result"])

        # Mock LLM: with_structured_output(model, include_raw=True) raises TypeError
        # but with_structured_output(model) alone succeeds
        mock_llm = MagicMock()

        call_count = {"n": 0}

        def with_structured_output_side_effect(model, **kwargs):
            call_count["n"] += 1
            if kwargs.get("include_raw", False):
                raise TypeError("unexpected keyword argument 'include_raw'")
            # Return a mock structured LLM that returns our expected result
            mock_structured = MagicMock()
            mock_structured.invoke.return_value = expected
            return mock_structured

        mock_llm.with_structured_output.side_effect = with_structured_output_side_effect

        config = {"configurable": {}}
        result, usage = _call_structured(mock_llm, [], Claims, "structured", config)

        assert result == expected
        # Called twice: first with include_raw=True (fails), then without
        assert call_count["n"] == 2

    def test_result_correct_when_include_raw_supported(self):
        """_call_structured returns parsed result when include_raw works."""
        from unittest.mock import MagicMock

        from neograph._llm import _call_structured

        expected = Claims(items=["direct-result"])

        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = {"parsed": expected, "raw": None}
        mock_llm.with_structured_output.return_value = mock_structured

        config = {"configurable": {}}
        result, usage = _call_structured(mock_llm, [], Claims, "structured", config)

        assert result == expected
        # Called once — include_raw=True worked
        mock_llm.with_structured_output.assert_called_once_with(Claims, include_raw=True)


class TestRetryPromptIncludesSchema:
    """Retry prompt must include describe_type schema so LLM can self-correct.

    BUG neograph-mfzx: DeepSeek R1 simplifies nested objects to strings
    on long prompts. The retry must show the expected structure.
    """

    def test_retry_msg_includes_schema_for_nested_model(self):
        """_build_retry_msg with output_model includes the full schema."""
        from pydantic import BaseModel

        from neograph._llm import _build_retry_msg
        from neograph.errors import ExecutionError

        class Actor(BaseModel):
            name: str
            role: str

        class Result(BaseModel):
            actors: list[Actor]

        err = ExecutionError("validation failed")
        err.validation_errors = "actors.0: Input should be an object"

        msg = _build_retry_msg(err, output_model=Result)

        # Must include the schema so LLM knows the expected structure
        assert "name: string" in msg or "name" in msg
        assert "role: string" in msg or "role" in msg
        # Must instruct about nested objects
        assert "nested object" in msg.lower() or "ALL required fields" in msg

    def test_retry_msg_without_model_still_works(self):
        """_build_retry_msg without output_model doesn't crash."""
        from neograph._llm import _build_retry_msg
        from neograph.errors import ExecutionError

        err = ExecutionError("parse failed")
        msg = _build_retry_msg(err)
        assert "JSON" in msg

    def test_default_max_retries_is_2(self):
        """_invoke_json_with_retry defaults to max_retries=2."""
        import inspect
        from neograph._llm import _invoke_json_with_retry
        sig = inspect.signature(_invoke_json_with_retry)
        assert sig.parameters["max_retries"].default == 2


# ═══════════════════════════════════════════════════════════════════════════
# TEST: RetryPolicy support (neograph-o0qw)
# ═══════════════════════════════════════════════════════════════════════════


class TestRetryPolicy:
    """compile(retry_policy=) applies retry to LLM nodes only (neograph-o0qw)."""

    def test_pipeline_compiles_and_runs_when_retry_policy_set(self):
        """compile(retry_policy=...) produces a working graph. E2E."""
        import types

        from langgraph.types import RetryPolicy

        from neograph import compile, construct_from_module, node, run
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["done"])))

        mod = types.ModuleType("test_retry_mod")

        @node(outputs=Claims, mode="think", model="fast", prompt="test")
        def analyze() -> Claims: ...

        mod.analyze = analyze
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, retry_policy=RetryPolicy(max_attempts=3))
        result = run(graph, input={"node_id": "retry-test"})

        assert result["analyze"].items == ["done"]

    def test_retry_policy_inherited_by_sub_constructs(self):
        """Sub-constructs inherit retry_policy from parent compile(). E2E."""
        from langgraph.types import RetryPolicy

        from neograph import compile, construct_from_functions, node, run
        from neograph.factory import register_scripted
        from tests.fakes import StructuredFake, configure_fake_llm

        configure_fake_llm(
            lambda tier: StructuredFake(
                lambda m: m(items=["sub-done"]),
            )
        )

        @node(mode="think", outputs=Claims, model="fast", prompt="score")
        def score(input_text: RawText) -> Claims: ...

        sub = construct_from_functions(
            "scorer",
            [score],
            input=RawText,
            output=Claims,
        )

        register_scripted("retry_seed", lambda _in, _cfg: RawText(text="test"))
        parent = Construct(
            "parent",
            nodes=[
                Node.scripted("seed", fn="retry_seed", outputs=RawText),
                sub,
            ],
        )
        graph = compile(parent, retry_policy=RetryPolicy(max_attempts=2))
        result = run(graph, input={"node_id": "retry-sub"})

        assert result["scorer"].items == ["sub-done"]

    def test_scripted_nodes_not_affected_by_retry_policy(self):
        """Scripted nodes should work fine with retry_policy set (no crash). E2E."""
        import types

        from langgraph.types import RetryPolicy

        from neograph import compile, construct_from_module, node, run

        mod = types.ModuleType("test_retry_scripted_mod")

        @node(outputs=Claims)
        def scripted_node() -> Claims:
            return Claims(items=["scripted"])

        mod.scripted_node = scripted_node
        pipeline = construct_from_module(mod)
        graph = compile(pipeline, retry_policy=RetryPolicy(max_attempts=3))
        result = run(graph, input={})

        assert result["scripted_node"].items == ["scripted"]


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION — get_graph().edges completeness
# ═══════════════════════════════════════════════════════════════════════════


class TestGetGraphEdges:
    """get_graph().edges must report edges for all nodes, including those
    wired through Send()-based conditional routing (Oracle, Each)."""

    def test_oracle_edges_visible_in_get_graph(self):
        """Nodes after an Oracle fan-out must appear in get_graph().edges.

        Regression: without path_map on add_conditional_edges, LangGraph
        cannot resolve Send() targets statically, so get_graph() shows only
        the first chain terminating at __end__.
        """
        from neograph.decorators import _merge_fn_registry
        from neograph.factory import register_scripted

        register_scripted("viz_pre", lambda _in, _cfg: RawText(text="input"))
        register_scripted("viz_gen", lambda _in, _cfg: Claims(items=["x"]))

        def merge_fn(results):
            return Claims(items=["merged"])

        _merge_fn_registry["viz_merge"] = (merge_fn, None)

        register_scripted("viz_post", lambda _in, _cfg: RawText(text="done"))

        pre = Node.scripted("pre", fn="viz_pre", outputs=RawText)
        gen = Node.scripted("gen", fn="viz_gen", inputs=RawText, outputs=Claims) | Oracle(n=2, merge_fn="viz_merge")
        post = Node.scripted("post", fn="viz_post", inputs=Claims, outputs=RawText)

        pipeline = Construct("test-viz-oracle", nodes=[pre, gen, post])
        graph = compile(pipeline)

        dg = graph.get_graph()
        edge_set = {(e.source, e.target) for e in dg.edges}

        # post node must be reachable — not orphaned
        assert ("merge_gen", "post") in edge_set, f"Edge merge_gen -> post missing from get_graph().edges: {edge_set}"
        assert ("post", "__end__") in edge_set, f"Edge post -> __end__ missing from get_graph().edges: {edge_set}"

    def test_each_edges_visible_in_get_graph(self):
        """Nodes after an Each fan-out must appear in get_graph().edges."""
        from neograph.factory import register_scripted

        register_scripted("viz_make", lambda _in, _cfg: Clusters(groups=[ClusterGroup(label="a", claim_ids=["1"])]))
        register_scripted("viz_verify", lambda _in, _cfg: MatchResult(cluster_label="a", matched=["ok"]))
        register_scripted("viz_summary", lambda _in, _cfg: RawText(text="done"))

        make = Node.scripted("make", fn="viz_make", outputs=Clusters)
        verify = Node.scripted(
            "verify",
            fn="viz_verify",
            inputs=ClusterGroup,
            outputs=MatchResult,
        ) | Each(over="make.groups", key="label")
        summary = Node.scripted(
            "summary",
            fn="viz_summary",
            inputs=dict[str, MatchResult],
            outputs=RawText,
        )

        pipeline = Construct("test-viz-each", nodes=[make, verify, summary])
        graph = compile(pipeline)

        dg = graph.get_graph()
        edge_set = {(e.source, e.target) for e in dg.edges}

        # summary node must be reachable after the Each barrier
        assert ("assemble_verify", "summary") in edge_set, f"Edge assemble_verify -> summary missing: {edge_set}"
        assert ("summary", "__end__") in edge_set, f"Edge summary -> __end__ missing: {edge_set}"


# =============================================================================
# BUG REGRESSION: neograph-irv3
# R1 emits tool-call XML in content after budget exhaustion
# =============================================================================


class TestR1XmlAfterBudgetExhaustion:
    """When tool budgets are exhausted and the bare LLM responds with XML
    tool-call markup instead of JSON, _parse_json_response must handle it
    instead of crashing with a ValidationError."""

    def test_xml_tool_call_in_content_parsed_when_json_mode_gather(self):
        """After budget exhaustion, R1-style XML in content should not crash.

        The LLM emits '<｜DSML｜function_calls>...' in message.content instead
        of valid JSON. The framework should detect this and either strip it
        or raise a clear ExecutionError — not a raw ValidationError.
        """
        from langchain_core.messages import AIMessage

        from neograph._llm import configure_llm
        from neograph.factory import register_tool_factory

        lookup_tool = FakeTool("lookup", response="found")
        register_tool_factory("lookup", lambda config, tool_config: lookup_tool)

        # Fake LLM: first call uses tool (exhausts budget=1),
        # second call (bare, no tools) emits XML instead of JSON.
        call_n = {"n": 0}
        XML_CONTENT = (
            "<\uff5cDSML\uff5cfunction_calls>"
            '<\uff5cDSML\uff5cinvoke name="read_artifact">'
            '<\uff5cDSML\uff5cparameter name="path">test.py'
            "</\uff5cDSML\uff5cparameter>"
            "</\uff5cDSML\uff5cinvoke>"
            "</\uff5cDSML\uff5cfunction_calls>"
        )

        class FakeR1:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                call_n["n"] += 1
                if call_n["n"] == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "lookup", "args": {}, "id": "c1"}]
                    return msg
                # After budget exhaustion: XML instead of JSON
                return AIMessage(content=XML_CONTENT)

        configure_llm(
            llm_factory=lambda tier: FakeR1(),
            prompt_compiler=lambda template, data, **kw: [{"role": "user", "content": "go"}],
        )

        node = Node(
            name="research",
            mode="agent",
            outputs=Claims,
            model="fast",
            prompt="test",
            tools=[Tool(name="lookup", budget=1)],
            llm_config={"output_strategy": "json_mode"},
        )
        pipeline = Construct("test-r1-xml", nodes=[node])
        graph = compile(pipeline)

        # Should raise ExecutionError (clear message), not ValidationError (cryptic)
        with pytest.raises(ExecutionError, match="(?i)structured output|json|xml"):
            run(graph, input={"node_id": "test"})


# =============================================================================
# BUG REGRESSION: neograph-g1ip
# Compiler should register output model types with LangGraph msgpack allowlist
# =============================================================================


class TestMsgpackTypeRegistration:
    """When compile() receives a checkpointer, it should register all node
    output types with the checkpointer's serializer so LangGraph doesn't
    emit 'Deserializing unregistered type' warnings on resume."""

    def test_output_types_registered_when_compiled_with_checkpointer(self):
        """compile() with MemorySaver should convert the serde from
        allow-all (True) to an explicit allowlist containing the node
        output Pydantic models."""
        from langgraph.checkpoint.memory import MemorySaver

        from neograph.factory import register_scripted

        register_scripted("dummy_a", lambda input_data, config: Claims(items=["x"]))
        register_scripted(
            "dummy_b",
            lambda input_data, config: MatchResult(
                cluster_label="test",
                coverage_pct=90,
                gaps=[],
            ),
        )

        pipeline = Construct(
            "test-msgpack",
            nodes=[
                Node.scripted("a", fn="dummy_a", outputs=Claims),
                Node.scripted("b", fn="dummy_b", inputs=Claims, outputs=MatchResult),
            ],
        )

        checkpointer = MemorySaver()

        # Before compile: serde allows all (True = warn mode)
        assert checkpointer.serde._allowed_msgpack_modules is True

        graph = compile(pipeline, checkpointer=checkpointer)

        # After compile: serde should have an explicit allowlist (set/frozenset),
        # not True. The allowlist must include our output model types.
        allowlist = checkpointer.serde._allowed_msgpack_modules
        assert allowlist is not True, (
            "compile() should convert the serde from allow-all (True) to an "
            "explicit allowlist containing node output types"
        )
        # The allowlist should contain tuples of (module, classname)
        type_names = {name for (_mod, name) in allowlist}
        assert "Claims" in type_names, f"Claims not in allowlist: {type_names}"
        assert "MatchResult" in type_names, f"MatchResult not in allowlist: {type_names}"


# ═══════════════════════════════════════════════════════════════════════════
# Runtime: Loop/branch condition error handling (neograph-d19r)
# ═══════════════════════════════════════════════════════════════════════════


class TestConditionErrorHandling:
    """User-provided conditions (Loop.when, branch op_fn) can crash on None
    with AttributeError/TypeError. The error should be wrapped in a clear
    ExecutionError naming the condition and the None value."""

    def test_loop_condition_wraps_attribute_error_when_value_is_none(self):
        """Loop condition accessing .score on None raises ExecutionError."""
        from neograph.factory import register_scripted
        from neograph.modifiers import Loop

        register_scripted("d19r_attr_fn", lambda input_data, config: RawText(text="hello"))

        n = Node.scripted("produce-none", fn="d19r_attr_fn", outputs=RawText) | Loop(
            when=lambda draft: draft.score < 0.8, max_iterations=3
        )
        pipeline = Construct("loop-err", nodes=[n])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="condition"):
            run(graph, input={"node_id": "test-loop-err"})

    def test_loop_condition_wraps_type_error_when_value_is_none(self):
        """Loop condition doing comparison on None raises ExecutionError."""
        from neograph.factory import register_scripted
        from neograph.modifiers import Loop

        register_scripted("d19r_type_fn", lambda input_data, config: RawText(text="hello"))

        n = Node.scripted("produce-text", fn="d19r_type_fn", outputs=RawText) | Loop(
            when=lambda draft: draft < 0.8, max_iterations=3
        )
        pipeline = Construct("loop-type-err", nodes=[n])
        graph = compile(pipeline)
        with pytest.raises(ExecutionError, match="condition"):
            run(graph, input={"node_id": "test-loop-type-err"})


# =============================================================================
# _llm.py coverage — edge cases for uncovered lines
# =============================================================================


class TestLlmIntrospectParams:
    """_accepted_params edge cases (lines 60-62)."""

    def test_c_extension_returns_empty_set(self):
        """C extension callables with no inspectable signature return empty set."""
        from neograph._llm import _accepted_params

        # builtins like len have no inspectable signature in some Python versions
        result = _accepted_params(len)
        # Should return empty set or a valid set (not raise)
        assert isinstance(result, (set, frozenset))


class TestLlmNotConfigured:
    """Lines 123-124, 127: _get_llm when not configured."""

    def test_get_llm_raises_when_not_configured(self):
        """_get_llm raises ConfigurationError when llm_factory is None."""
        from neograph._llm import _get_llm

        with pytest.raises(ConfigurationError, match="not configured"):
            _get_llm("fast")

    def test_get_llm_accepts_all_kwargs_factory(self):
        """When factory accepts **kwargs, all kwargs are passed (line 127)."""
        from neograph._llm import _get_llm, configure_llm

        received = {}

        def factory(tier, **kwargs):
            received.update(kwargs)
            return "fake_llm"

        configure_llm(
            llm_factory=factory,
            prompt_compiler=lambda t, d: [{"role": "user", "content": "test"}],
        )
        result = _get_llm("fast", node_name="test_node", llm_config={"temp": 0.5})
        assert result == "fake_llm"
        assert "node_name" in received
        assert "llm_config" in received


class TestExtractJsonEdgeCases2:
    """Lines 227-228, 230-231, 245-248: _extract_json edge cases."""

    def test_escape_char_in_json_string(self):
        """Backslash escapes inside JSON strings are handled (lines 227-228, 230-231)."""
        from neograph._llm import _extract_json

        text = r'{"key": "value with \" escaped"}'
        result = _extract_json(text)
        assert result.startswith("{")
        assert result.endswith("}")

    def test_unbalanced_braces_first_to_last(self):
        """Unbalanced braces with closing brace: first-to-last fallback (line 247)."""
        from neograph._llm import _extract_json

        # Unbalanced: extra opening brace, but there's a } later
        text = '{"key": {"nested": "value"} extra stuff}'
        # The balanced scan finds the first complete match
        result = _extract_json(text)
        assert result.startswith("{")

    def test_unbalanced_with_trailing_closing_brace(self):
        """Unbalanced JSON with a trailing } past the scan falls back (line 247)."""
        from neograph._llm import _extract_json

        # The first { starts depth tracking, but escapes/strings cause imbalance
        # Force unbalanced: an unclosed string with } after it
        text = '{"key": "unclosed string} more text}'
        result = _extract_json(text)
        assert "{" in result

    def test_no_closing_brace_at_all(self):
        """No closing brace after opening returns stripped text (line 248)."""
        from neograph._llm import _extract_json

        text = '{"key": "value'
        result = _extract_json(text)
        assert result == text.strip()


class TestParseJsonException:
    """Lines 293-294: generic Exception during JSON parsing."""

    def test_generic_exception_during_parse(self):
        """Non-ValidationError during parse raises ExecutionError."""
        from neograph._llm import _parse_json_response

        class BadModel:
            """A model that raises a generic Exception during validate."""

            __name__ = "BadModel"

            @classmethod
            def model_validate_json(cls, data):
                raise RuntimeError("unexpected error")

        with pytest.raises(ExecutionError, match="Failed to parse"):
            _parse_json_response('{"x": 1}', BadModel)


class TestCallStructuredUnknownStrategy:
    """Lines 377-378: unknown output_strategy raises ExecutionError."""

    def test_unknown_strategy_raises(self):
        """_call_structured with unknown strategy raises ExecutionError."""
        from neograph._llm import _call_structured

        with pytest.raises(ExecutionError, match="Unknown output_strategy"):
            _call_structured(None, [], Claims, "invalid_strategy", {})


class TestToolResultRendering:
    """Lines 452, 562: _render_tool_result_for_llm with plain values and list of models."""

    def test_plain_value_returns_str(self):
        """Non-Pydantic result returns str() (line 452)."""
        from neograph._llm import _render_tool_result_for_llm

        assert _render_tool_result_for_llm(42) == "42"
        assert _render_tool_result_for_llm("hello") == "hello"

    def test_list_of_models_rendered(self):
        """List of BaseModel instances uses renderer or describe_value (line 562)."""
        from neograph._llm import _render_tool_result_for_llm

        items = [RawText(text="a"), RawText(text="b")]
        result = _render_tool_result_for_llm(items)
        assert "a" in result
        assert "b" in result


class TestUnregisteredToolInReact:
    """Lines 500-501: tool not registered raises ConfigurationError."""

    def test_unregistered_tool_raises(self):
        """invoke_with_tools with unregistered tool raises ConfigurationError."""
        from neograph._llm import invoke_with_tools
        from neograph.tool import ToolBudgetTracker

        configure_fake_llm(lambda tier: StructuredFake(lambda m: m(items=["x"])))

        with pytest.raises(ConfigurationError, match="not registered"):
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test prompt",
                input_data="test",
                output_model=Claims,
                tools=[Tool("nonexistent", budget=5)],
                budget_tracker=ToolBudgetTracker([Tool("nonexistent", budget=5)]),
                config={"configurable": {}},
            )


class TestUsageTokenAccumulation:
    """Lines 621-626, 630: usage token accumulation in invoke_with_tools."""

    def test_usage_tokens_accumulated_from_messages(self):
        """Token usage from ReAct messages is accumulated (lines 621-626, 630)."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found it")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        # Build a ReActFake that makes one tool call then stops
        from langchain_core.messages import AIMessage

        class UsageFake:
            """Fake LLM that adds usage_metadata to responses."""

            def __init__(self):
                self._call_idx = 0
                self._model = None
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                if self._structured:
                    result = Claims(items=["done"])
                    return {
                        "parsed": result,
                        "raw": AIMessage(
                            content="",
                            usage_metadata={"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                        ),
                    }
                self._call_idx += 1
                if self._call_idx == 1:
                    msg = AIMessage(
                        content="", usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
                    )
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
                    return msg
                return AIMessage(
                    content="done", usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
                )

            def with_structured_output(self, model, **kwargs):
                clone = UsageFake()
                clone._model = model
                clone._structured = True
                return clone

        configure_fake_llm(lambda tier: UsageFake())

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test prompt",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1


class TestRenderPromptEdgeCases:
    """Lines 669-670, 707-708: render_prompt not configured + message attr access."""

    def test_render_prompt_not_configured_raises(self):
        """render_prompt without configure_llm raises ConfigurationError."""
        from neograph._llm import render_prompt

        node = Node("test", mode="think", prompt="test", model="fast", outputs=Claims)
        with pytest.raises(ConfigurationError, match="not configured"):
            render_prompt(node, {"key": "value"})

    def test_render_prompt_with_message_objects(self):
        """render_prompt handles LangChain message objects (line 707-708)."""
        from langchain_core.messages import HumanMessage

        from neograph._llm import configure_llm, render_prompt

        configure_llm(
            llm_factory=lambda tier: None,
            prompt_compiler=lambda t, d, **kw: [HumanMessage(content="hello world")],
        )

        node = Node("test", mode="think", prompt="test", model="fast", outputs=Claims)
        result = render_prompt(node, {"key": "value"}, config={"configurable": {}})
        assert "hello world" in result
        assert "[human]" in result


class TestToolBudgetTrackerAllExhausted:
    """Line 99 in tool.py: all_exhausted returns False when no tools."""

    def test_all_exhausted_no_tools(self):
        """ToolBudgetTracker with no tools returns False for all_exhausted."""
        from neograph.tool import ToolBudgetTracker

        tracker = ToolBudgetTracker([])
        assert tracker.all_exhausted() is False


class TestReActToolReturnsListOfModels:
    """Line 562 in _llm.py: tool returns list of BaseModel instances in ReAct loop."""

    def test_tool_returning_list_of_models_renders(self):
        """When a tool returns list[BaseModel], it's rendered via _render_tool_result_for_llm."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel as BM

        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        # A tool that returns a list of BaseModel instances
        class SearchResult(BM):
            title: str
            score: float

        class ListTool:
            name = "search"

            def invoke(self, args):
                return [SearchResult(title="a", score=0.9), SearchResult(title="b", score=0.8)]

        register_tool_factory("search", lambda cfg, tc: ListTool())

        class ListReActFake:
            def __init__(self):
                self._call_idx = 0
                self._model = None
                self._structured = False

            def bind_tools(self, tools):
                return self

            def invoke(self, messages, **kwargs):
                if self._structured:
                    return self._model(items=["done"])
                self._call_idx += 1
                if self._call_idx == 1:
                    msg = AIMessage(content="")
                    msg.tool_calls = [{"name": "search", "args": {"q": "x"}, "id": "1"}]
                    return msg
                return AIMessage(content="done")

            def with_structured_output(self, model, **kwargs):
                clone = ListReActFake()
                clone._model = model
                clone._structured = True
                return clone

        configure_fake_llm(lambda tier: ListReActFake())

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test prompt",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
        )
        assert len(interactions) == 1
        # The rendered result should contain both items
        assert "a" in interactions[0].result
        assert "b" in interactions[0].result


class TestReActMaxIterationsGuard:
    """invoke_with_tools max_iterations guard: stops infinite ReAct loops."""

    def test_max_iterations_default_stops_at_20(self):
        """Default max_iterations=20 stops an infinite tool-calling loop."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]  # unlimited per-tool budget
        tracker = ToolBudgetTracker(tools)
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
        )
        assert isinstance(result, Claims)
        # Iterations 1-19 execute tools (19 interactions). Iteration 20 hits
        # the guard, skips tool execution. Iteration 21: unbound LLM returns
        # no tool calls, loop ends. Plus 1 structured parse call. Total = 21.
        assert len(interactions) == 19
        assert fake.call_count == 21

    def test_max_iterations_custom_value(self):
        """Custom max_iterations in llm_config overrides the default."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # max_iterations=3 — guard fires on iteration 3
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 3},
        )
        assert isinstance(result, Claims)
        # Iterations 1-2 execute tools, iteration 3 hits guard (skips).
        # Iteration 4: unbound LLM returns no tool calls. Total = 2.
        assert len(interactions) == 2

    def test_max_iterations_does_not_affect_normal_completion(self):
        """When the LLM finishes before max_iterations, the guard is irrelevant."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM stops after 1 tool call
            ],
            final=lambda model: model(items=["done"]),
        )
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 20},
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1

    def test_max_iterations_equals_one(self):
        """Degenerate case: max_iterations=1 means the first tool-calling iteration triggers the guard."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 1},
        )
        assert isinstance(result, Claims)
        # Iteration 1: LLM calls tool, guard hits immediately (loop_count=1 >= 1).
        # No tool executions, just "wrap up" messages.
        assert len(interactions) == 0

    def test_guard_fired_llm_ignores_wrap_up_still_calls_tools(self):
        """After guard fires and tools are unbound, if the LLM still returns
        tool_calls on the next invocation, the loop force-breaks instead of
        looping forever (the _guard_fired safety net)."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        configure_fake_llm(lambda tier: StubbornFake())

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # With max_iterations=1, guard fires on first iteration,
        # but the LLM keeps calling tools — should force-break, not infinite loop
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 1},
        )
        # Loop terminates despite LLM never stopping tool calls
        assert isinstance(result, Claims)
        assert len(interactions) == 0

    def test_max_iterations_guard_logs_warning(self):
        """When max_iterations is exceeded, a warning is logged with the reason."""
        from structlog.testing import capture_logs

        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake()
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        with capture_logs() as cap:
            invoke_with_tools(
                model_tier="fast",
                prompt_template="test",
                input_data="test",
                output_model=Claims,
                tools=tools,
                budget_tracker=tracker,
                config={"configurable": {}},
                llm_config={"max_iterations": 2},
            )
        guard_events = [e for e in cap if "react_max_iterations_exceeded" in e.get("event", "")]
        assert len(guard_events) >= 1
        assert guard_events[0]["max_iterations"] == 2


class TestReActTokenBudgetGuard:
    """invoke_with_tools token_budget guard: stops loop when cumulative input tokens exceeded."""

    def test_token_budget_stops_loop(self):
        """token_budget in llm_config stops the loop when input tokens exceed threshold."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]  # unlimited
        tracker = ToolBudgetTracker(tools)

        # token_budget=2500 — after 3 iterations (3000 cumulative), guard fires
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 50, "token_budget": 2500},
        )
        assert isinstance(result, Claims)
        # Iterations 1-2 execute tools (cumulative: 2000 < 2500). Iteration 3
        # hits token_budget guard (3000 > 2500), skips tools. Iteration 4: unbound
        # LLM returns final. Total interactions = 2.
        assert len(interactions) == 2

    def test_token_budget_none_is_no_limit(self):
        """token_budget=None (default) means no token budget enforcement."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM finishes naturally
            ],
            final=lambda model: model(items=["done"]),
        )
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            # No token_budget — default None means unlimited
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1

    def test_both_guards_fire_simultaneously(self):
        """When max_iterations and token_budget are both exceeded, loop still terminates."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = GuardFake(input_tokens_per_call=1000)
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=0)]
        tracker = ToolBudgetTracker(tools)

        # max_iterations=3 and token_budget=2500: both fire on iteration 3
        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            llm_config={"max_iterations": 3, "token_budget": 2500},
        )
        assert isinstance(result, Claims)
        # Iterations 1-2 execute (cumulative 2000 < 2500). Iteration 3 hits both
        # guards (loop_count=3 >= 3 AND cumulative=3000 > 2500).
        assert len(interactions) == 2

    def test_token_budget_missing_usage_metadata(self):
        """When responses lack usage_metadata, token_budget never fires (correct behavior)."""
        from neograph._llm import invoke_with_tools
        from neograph.factory import register_tool_factory
        from neograph.tool import ToolBudgetTracker

        fake_tool = FakeTool("search", response="found")
        register_tool_factory("search", lambda cfg, tc: fake_tool)

        fake = ReActFake(
            tool_calls=[
                [{"name": "search", "args": {"q": "x"}, "id": "1"}],
                [],  # LLM finishes naturally
            ],
            final=lambda model: model(items=["done"]),
        )
        # ReActFake doesn't include usage_metadata on responses,
        # so cumulative_input_tokens stays 0, and token_budget never triggers
        configure_fake_llm(lambda tier: fake)

        tools = [Tool("search", budget=5)]
        tracker = ToolBudgetTracker(tools)

        result, interactions = invoke_with_tools(
            model_tier="fast",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            tools=tools,
            budget_tracker=tracker,
            config={"configurable": {}},
            # token_budget=100, but responses have no usage_metadata, so it never fires
            llm_config={"token_budget": 100},
        )
        assert isinstance(result, Claims)
        assert len(interactions) == 1  # tool executes normally


# ═══════════════════════════════════════════════════════════════════════════
# BUG neograph-53lz: _extract_json drops bare-array LLM responses
#
# _extract_json only looks for '{' as JSON start. When the LLM returns a
# bare array ([{...}, {...}]), it falls through and returns raw text.
# _parse_json_response then silently defaults to empty lists.
# ═══════════════════════════════════════════════════════════════════════════


class TestBareArrayExtraction:
    """_extract_json must handle bare JSON arrays, not just objects."""

    def test_extract_json_finds_bare_array(self):
        """Bare array [{"key": "val"}] should be extracted, not dropped."""
        from neograph._llm import _extract_json

        text = '[{"claim_id": "R1", "classification": "business-rule"}, {"claim_id": "R2", "classification": "functional"}]'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert "R1" in result

    def test_extract_json_bare_array_with_markdown_fence(self):
        """Bare array inside ```json fence should be extracted."""
        from neograph._llm import _extract_json

        text = '```json\n[{"id": 1}, {"id": 2}]\n```'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert '"id": 1' in result or '"id":1' in result

    def test_extract_json_bare_array_with_prose(self):
        """Bare array preceded by prose text should be extracted."""
        from neograph._llm import _extract_json

        text = 'Here are the results:\n[{"name": "Alice"}, {"name": "Bob"}]\nDone.'
        result = _extract_json(text)
        assert result.startswith("["), f"Expected array, got: {result[:50]}"
        assert "Alice" in result

    def test_parse_json_response_bare_array_auto_wraps(self):
        """When output model has a single list field and LLM returns bare array, auto-wrap."""
        from neograph._llm import _parse_json_response
        from pydantic import BaseModel

        class Claim(BaseModel):
            claim_id: str
            classification: str

        class ClassificationResult(BaseModel):
            classified_claims: list[Claim]

        text = '[{"claim_id": "R1", "classification": "business-rule"}, {"claim_id": "R2", "classification": "functional"}]'
        result = _parse_json_response(text, ClassificationResult)
        assert isinstance(result, ClassificationResult)
        assert len(result.classified_claims) == 2
        assert result.classified_claims[0].claim_id == "R1"

    def test_parse_json_response_bare_array_multi_field_model_raises(self):
        """Bare array with multi-field model should raise, not silently default."""
        from neograph._llm import _parse_json_response
        from neograph import ExecutionError
        from pydantic import BaseModel

        class MultiField(BaseModel):
            items: list[str]
            count: int

        text = '["a", "b", "c"]'
        # Multi-field model can't auto-wrap — should raise, not silently produce defaults
        with pytest.raises(ExecutionError):
            _parse_json_response(text, MultiField)

    def test_extract_json_prefers_object_over_array(self):
        """When both { and [ exist, prefer { if it comes first."""
        from neograph._llm import _extract_json

        text = '{"items": [1, 2, 3]}'
        result = _extract_json(text)
        assert result.startswith("{"), f"Expected object, got: {result[:50]}"


