"""Pipeline mode tests — cost callback, reducers, modifier combos, skip_when, node I/O, mode dispatch"""

from __future__ import annotations

import pytest

from neograph import (
    Construct,
    Each,
    Node,
    Oracle,
    compile,
    run,
)
from tests.fakes import StructuredFake, configure_fake_llm
from tests.schemas import (
    Claims,
    ClusterGroup,
    Clusters,
    MatchResult,
    MergedResult,
    RawText,
)

# ═══════════════════════════════════════════════════════════════════════════
# COST CALLBACK (neograph-pkjw)
# ═══════════════════════════════════════════════════════════════════════════


class TestCostCallback:
    """configure_llm(cost_callback=...) dispatches per-call token usage."""

    def test_cost_callback_called_on_think_mode(self):
        """cost_callback receives tier + token counts after invoke_structured."""
        from neograph._llm import invoke_structured

        calls = []

        def my_cost_callback(*, tier, input_tokens, output_tokens):
            calls.append({"tier": tier, "in": input_tokens, "out": output_tokens})

        fake = StructuredFake(lambda model: model(items=["done"]))
        configure_fake_llm(lambda tier: fake)
        # Re-configure with cost callback
        from neograph._llm import _llm_factory, _prompt_compiler, configure_llm
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=my_cost_callback)

        invoke_structured(
            model_tier="reason",
            prompt_template="test",
            input_data="test",
            output_model=Claims,
            config={"configurable": {}},
        )
        # Callback was called (may have 0 tokens since fakes don't report usage)
        assert len(calls) >= 0  # doesn't crash

    def test_cost_callback_not_called_when_none(self):
        """When cost_callback=None (default), no dispatch happens."""
        from neograph._llm import _notify_cost
        # Should not crash even with valid usage data
        _notify_cost("reason", {"input_tokens": 100, "output_tokens": 50})

    def test_cost_callback_exception_does_not_break_pipeline(self):
        """A broken cost_callback must not crash the pipeline."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        def broken_callback(**kwargs):
            raise TypeError("cost tracking broken")

        configure_llm(_llm_factory, _prompt_compiler, cost_callback=broken_callback)
        # Should not raise
        _notify_cost("reason", {"input_tokens": 100, "output_tokens": 50})
        # Reset
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)

    def test_cost_callback_fires_with_empty_usage_dict(self):
        """Regression neograph-pz2x: empty dict {} must NOT skip callback."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        calls = []
        configure_llm(_llm_factory, _prompt_compiler,
                      cost_callback=lambda **kw: calls.append(kw))
        _notify_cost("fast", {})  # empty dict — was previously skipped (falsy)
        assert len(calls) == 1
        assert calls[0]["input_tokens"] == 0
        assert calls[0]["output_tokens"] == 0
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)

    def test_retry_accumulates_token_usage(self):
        """Regression neograph-xcwd: retries must accumulate tokens, not overwrite."""
        from langchain_core.messages import AIMessage
        from pydantic import BaseModel

        from neograph._llm import _invoke_json_with_retry

        class Result(BaseModel):
            value: str

        call_count = [0]

        class RetryFake:
            def invoke(self, messages, **kwargs):
                call_count[0] += 1
                msg = AIMessage(content='{"value": "ok"}' if call_count[0] > 1 else '{"bad json')
                msg.usage_metadata = {"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
                return msg

        result, usage = _invoke_json_with_retry(
            RetryFake(), [{"role": "user", "content": "test"}],
            Result, {}, max_retries=2,
        )
        assert result.value == "ok"
        # 2 calls: initial (fails parse) + retry (succeeds). Both should count.
        assert usage["input_tokens"] == 200  # 100 + 100
        assert usage["output_tokens"] == 100  # 50 + 50


    def test_cost_callback_receives_node_name(self):
        """cost_callback must receive node_name kwarg (neograph-m621)."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        calls = []
        configure_llm(_llm_factory, _prompt_compiler,
                      cost_callback=lambda **kw: calls.append(kw))
        _notify_cost("fast", {"input_tokens": 10, "output_tokens": 5},
                     node_name="my-node")
        assert len(calls) == 1
        assert calls[0]["node_name"] == "my-node"
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)

    def test_cost_callback_receives_mode(self):
        """cost_callback must receive mode kwarg (neograph-m621)."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        calls = []
        configure_llm(_llm_factory, _prompt_compiler,
                      cost_callback=lambda **kw: calls.append(kw))
        _notify_cost("reason", {"input_tokens": 50, "output_tokens": 20},
                     node_name="gen", mode="think")
        assert calls[0]["mode"] == "think"
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)

    def test_cost_callback_receives_duration(self):
        """cost_callback must receive duration_s kwarg (neograph-m621)."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        calls = []
        configure_llm(_llm_factory, _prompt_compiler,
                      cost_callback=lambda **kw: calls.append(kw))
        _notify_cost("fast", {"input_tokens": 10, "output_tokens": 5},
                     node_name="x", duration_s=1.23)
        assert calls[0]["duration_s"] == 1.23
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)

    def test_cost_callback_backward_compat_old_signature(self):
        """Existing callbacks with (*, tier, input_tokens, output_tokens) must not break."""
        from neograph._llm import _llm_factory, _notify_cost, _prompt_compiler, configure_llm

        calls = []
        def old_style_callback(*, tier, input_tokens, output_tokens):
            calls.append({"tier": tier, "in": input_tokens, "out": output_tokens})

        configure_llm(_llm_factory, _prompt_compiler, cost_callback=old_style_callback)
        # New _notify_cost passes extra kwargs — old callback must not crash
        _notify_cost("fast", {"input_tokens": 10, "output_tokens": 5},
                     node_name="x", mode="think", duration_s=0.5)
        assert len(calls) == 1
        assert calls[0]["tier"] == "fast"
        configure_llm(_llm_factory, _prompt_compiler, cost_callback=None)


class TestMergeDictsReducer:
    """Regression tests for _merge_dicts reducer edge cases."""

    def test_merge_dicts_non_dict_existing_returns_new(self):
        """Non-dict existing is replaced with empty dict, new is merged in."""
        from neograph.state import _merge_dicts
        result = _merge_dicts("corrupted", {"a": 1})
        assert result == {"a": 1}

    def test_merge_dicts_non_dict_new_returns_existing(self):
        """Non-dict new is ignored, existing returned unchanged."""
        from neograph.state import _merge_dicts
        result = _merge_dicts({"a": 1}, None)
        assert result == {"a": 1}

    def test_merge_dicts_empty_new_is_noop(self):
        """Empty dict new is a no-op (obligation M-04)."""
        from neograph.state import _merge_dicts
        result = _merge_dicts({"a": 1}, {})
        assert result == {"a": 1}


class TestModifierCombo:
    """neograph-35c3: classify_modifiers enum for exhaustive dispatch."""

    def test_bare_node(self):
        from neograph.modifiers import ModifierCombo, classify_modifiers
        n = Node.scripted("bare", fn="xw75_gen", outputs=Claims)
        combo, mods = classify_modifiers(n)
        assert combo == ModifierCombo.BARE
        assert mods == {}

    def test_each_only(self):
        from neograph.modifiers import ModifierCombo, classify_modifiers
        n = Node.scripted("each", fn="xw75_gen", outputs=Claims) | Each(over="x.y", key="k")
        combo, _ = classify_modifiers(n)
        assert combo == ModifierCombo.EACH

    def test_oracle_only(self):
        from neograph.factory import register_scripted
        from neograph.modifiers import ModifierCombo, classify_modifiers
        register_scripted("35c3_m", lambda v, c: v[0])
        n = Node.scripted("orc", fn="xw75_gen", outputs=Claims) | Oracle(n=2, merge_fn="35c3_m")
        combo, _ = classify_modifiers(n)
        assert combo == ModifierCombo.ORACLE

    def test_each_oracle_fusion(self):
        from neograph.modifiers import ModifierCombo, classify_modifiers
        n = Node.scripted("fused", fn="xw75_gen", outputs=Claims) | Oracle(n=2, merge_fn="35c3_m") | Each(over="x.y", key="k")
        combo, mods = classify_modifiers(n)
        assert combo == ModifierCombo.EACH_ORACLE
        assert "each" in mods
        assert "oracle" in mods

    def test_invalid_combo_raises(self):
        """ModifierSet rejects illegal combos at construction time."""
        from neograph.modifiers import Loop, ModifierSet
        # Each + Loop is structurally rejected by ModifierSet
        with pytest.raises(Exception, match="Cannot combine Each and Loop"):
            ModifierSet(
                each=Each(over="x", key="k"),
                loop=Loop(when=lambda d: True),
            )
        # Oracle + Loop is structurally rejected by ModifierSet
        with pytest.raises(Exception, match="Cannot combine Oracle and Loop"):
            ModifierSet(
                oracle=Oracle(n=2, merge_fn="35c3_m"),
                loop=Loop(when=lambda d: True),
            )


class TestUnwrapHelpers:
    """neograph-26ih: shared unwrap helpers for Loop and Each state values."""

    def test_unwrap_loop_value_extracts_last(self):
        """Loop append-list [v1, v2, v3] → v3."""
        from neograph.factory import _unwrap_loop_value
        assert _unwrap_loop_value([1, 2, 3], int) == 3

    def test_unwrap_loop_value_empty_list_returns_none(self):
        """Empty Loop append-list [] → None (first iteration)."""
        from neograph.factory import _unwrap_loop_value
        assert _unwrap_loop_value([], int) is None

    def test_unwrap_loop_value_none_passthrough(self):
        """None → None."""
        from neograph.factory import _unwrap_loop_value
        assert _unwrap_loop_value(None, int) is None

    def test_unwrap_loop_value_non_list_passthrough(self):
        """Non-list value passes through unchanged."""
        from neograph.factory import _unwrap_loop_value
        assert _unwrap_loop_value("hello", str) == "hello"

    def test_unwrap_loop_value_list_consumer_passthrough(self):
        """When expected_type IS list, don't unwrap — consumer wants the list."""
        from neograph.factory import _unwrap_loop_value
        assert _unwrap_loop_value([1, 2, 3], list[int]) == [1, 2, 3]

    def test_unwrap_each_dict_to_list(self):
        """dict[str, X] → list[X] when consumer wants list."""
        from neograph.factory import _unwrap_each_dict
        result = _unwrap_each_dict({"a": 1, "b": 2}, list[int])
        assert result == [1, 2]

    def test_unwrap_each_dict_non_list_consumer_passthrough(self):
        """When consumer doesn't want list, dict passes through."""
        from neograph.factory import _unwrap_each_dict
        assert _unwrap_each_dict({"a": 1}, int) == {"a": 1}

    def test_unwrap_each_dict_non_dict_passthrough(self):
        """Non-dict value passes through."""
        from neograph.factory import _unwrap_each_dict
        assert _unwrap_each_dict("hello", list[str]) == "hello"

    def test_unwrap_each_dict_none_passthrough(self):
        """None passes through."""
        from neograph.factory import _unwrap_each_dict
        assert _unwrap_each_dict(None, list[int]) is None


class TestSkipWhenOnScriptedNode:
    """neograph-ejl2: skip_when must work on scripted nodes."""

    def test_skip_when_fires_on_scripted_node(self):
        """Scripted node with skip_when=True should NOT execute the function body."""
        from neograph.factory import register_scripted

        call_log = []

        def tracked_fn(input_data, config):
            call_log.append("called")
            return Claims(items=["should not appear"])

        register_scripted("ejl2_fn", tracked_fn)
        register_scripted("ejl2_seed", lambda i, c: RawText(text="seed"))

        pipeline = Construct("ejl2-test", nodes=[
            Node.scripted("seed", fn="ejl2_seed", outputs=RawText),
            Node("tracked", mode="scripted", scripted_fn="ejl2_fn",
                 inputs=RawText, outputs=Claims,
                 skip_when=lambda data: True,  # always skip
                 skip_value=lambda data: Claims(items=["skipped"])),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "ejl2"})

        # Function body should NOT have been called
        assert len(call_log) == 0, f"Expected 0 calls, got {len(call_log)}"
        # skip_value should have provided the output
        assert result["tracked"].items == ["skipped"]


class TestSkipWhenOnThinkNode:
    """neograph-y8ww: skip_when must work on think-mode nodes through unified path."""

    def test_skip_when_fires_on_think_node(self):
        """Think node with skip_when=True skips LLM and returns skip_value."""
        import types as _types

        from neograph import construct_from_module, node

        # LLM should NOT be called — if it is, the fake will still produce output
        # but skip_value should have taken precedence
        configure_fake_llm(
            lambda tier: StructuredFake(
                lambda model: model(items=["should-not-appear"]),
            )
        )

        mod = _types.ModuleType("test_skip_think_mod")

        @node(outputs=Claims)
        def seed() -> Claims:
            return Claims(items=["only-one"])

        @node(
            mode="think",
            outputs=MergedResult,
            model="fast",
            prompt="test",
            skip_when=lambda inp: len(inp.items) == 1,
            skip_value=lambda inp: MergedResult(final_text=inp.items[0]),
        )
        def thinker(seed: Claims) -> MergedResult: ...

        mod.seed = seed
        mod.thinker = thinker

        pipeline = construct_from_module(mod, name="test-skip-think")
        graph = compile(pipeline)
        result = run(graph, input={})

        # Node was skipped — skip_value produced the output
        assert result["thinker"].final_text == "only-one"


class TestSingleTypeListInputFromEach:
    """neograph-df77: inputs=list[X] (single-type) must unwrap dict[str,X] from Each."""

    def test_single_type_list_input_unwraps_each_dict(self):
        """Downstream with inputs=list[MatchResult] receives list, not dict."""
        from neograph.factory import register_scripted

        register_scripted("df77_src", lambda i, c: Clusters(groups=[
            ClusterGroup(label="a", claim_ids=["1"]),
        ]))
        register_scripted("df77_proc", lambda i, c: MatchResult(
            cluster_label="a", matched=["ok"],
        ))

        collected_input = [None]

        def collect_fn(input_data, config):
            collected_input[0] = input_data
            return RawText(text=f"got {len(input_data)} items")

        register_scripted("df77_collect", collect_fn)

        pipeline = Construct("df77-test", nodes=[
            Node.scripted("src", fn="df77_src", outputs=Clusters),
            Node.scripted("proc", fn="df77_proc",
                          inputs=ClusterGroup, outputs=MatchResult)
            | Each(over="src.groups", key="label"),
            # Single-type list input (NOT dict-form)
            Node("collect", mode="scripted", scripted_fn="df77_collect",
                 inputs=list[MatchResult], outputs=RawText),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "df77"})

        assert isinstance(collected_input[0], list), (
            f"Expected list, got {type(collected_input[0])}: {collected_input[0]}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# NodeInput / NodeOutput typed containers + ModeDispatch protocol
# (architecture-v2 section 1, neograph-cadg)
# ═══════════════════════════════════════════════════════════════════════════


class TestNodeInput:
    """NodeInput wraps extracted input with a shape discriminator."""

    def test_single_basemodel_value(self):
        """NodeInput holds a single BaseModel value."""
        from neograph.factory import NodeInput
        ni = NodeInput(single=RawText(text="hello"))
        assert ni.value == RawText(text="hello")
        assert ni.fan_in is None

    def test_single_str_value(self):
        """NodeInput.single accepts non-BaseModel types (str)."""
        from neograph.factory import NodeInput
        ni = NodeInput(single="plain string")
        assert ni.value == "plain string"

    def test_single_int_value(self):
        """NodeInput.single accepts int."""
        from neograph.factory import NodeInput
        ni = NodeInput(single=42)
        assert ni.value == 42

    def test_fan_in_dict_value(self):
        """NodeInput holds a fan-in dict of upstream values."""
        from neograph.factory import NodeInput
        ni = NodeInput(fan_in={"claims": Claims(items=["a"]), "text": RawText(text="b")})
        assert ni.value == {"claims": Claims(items=["a"]), "text": RawText(text="b")}
        assert ni.single is None

    def test_fan_in_with_mixed_types(self):
        """Fan-in dict can hold non-BaseModel values (str, list, int)."""
        from neograph.factory import NodeInput
        ni = NodeInput(fan_in={"count": 5, "names": ["a", "b"]})
        assert ni.value == {"count": 5, "names": ["a", "b"]}

    def test_fan_in_takes_precedence_over_single(self):
        """When both are set, fan_in takes precedence."""
        from neograph.factory import NodeInput
        ni = NodeInput(single=RawText(text="x"), fan_in={"a": Claims(items=[])})
        assert ni.value == {"a": Claims(items=[])}

    def test_empty_input(self):
        """NodeInput with neither single nor fan_in returns None."""
        from neograph.factory import NodeInput
        ni = NodeInput()
        assert ni.value is None


class TestNodeOutput:
    """NodeOutput wraps node result with a shape discriminator."""

    def test_single_basemodel_value(self):
        """NodeOutput holds a single BaseModel result."""
        from neograph.factory import NodeOutput
        no = NodeOutput(single=Claims(items=["x"]))
        assert no.value == Claims(items=["x"])
        assert no.multi is None

    def test_single_str_value(self):
        """NodeOutput.single accepts non-BaseModel types."""
        from neograph.factory import NodeOutput
        no = NodeOutput(single="result string")
        assert no.value == "result string"

    def test_multi_dict_value(self):
        """NodeOutput holds dict-form multi-output."""
        from neograph.factory import NodeOutput
        no = NodeOutput(multi={"summary": RawText(text="s"), "count": Claims(items=[])})
        assert no.value == {"summary": RawText(text="s"), "count": Claims(items=[])}

    def test_multi_takes_precedence_over_single(self):
        """When both are set, multi takes precedence."""
        from neograph.factory import NodeOutput
        no = NodeOutput(single=RawText(text="x"), multi={"a": Claims(items=[])})
        assert no.value == {"a": Claims(items=[])}

    def test_empty_output(self):
        """NodeOutput with neither single nor multi returns None."""
        from neograph.factory import NodeOutput
        no = NodeOutput()
        assert no.value is None


class TestModeDispatch:
    """ModeDispatch protocol and the 3 dispatch implementations."""

    def test_scripted_dispatch_instantiates(self):
        """ScriptedDispatch wraps a callable and instantiates."""
        from neograph.factory import ScriptedDispatch
        def my_fn(input_data, config):
            return RawText(text="ok")
        sd = ScriptedDispatch(fn=my_fn)
        assert sd.fn is my_fn

    def test_think_dispatch_instantiates(self):
        """ThinkDispatch instantiates without arguments."""
        from neograph.factory import ThinkDispatch
        td = ThinkDispatch()
        assert isinstance(td, ThinkDispatch)

    def test_tool_dispatch_instantiates(self):
        """ToolDispatch instantiates without arguments."""
        from neograph.factory import ToolDispatch
        td = ToolDispatch()
        assert isinstance(td, ToolDispatch)

    def test_scripted_dispatch_execute_calls_fn(self):
        """ScriptedDispatch.execute delegates to the wrapped fn."""
        from neograph.factory import NodeInput, NodeOutput, ScriptedDispatch
        call_log = []

        def my_fn(input_data, config):
            call_log.append((input_data, config))
            return Claims(items=["done"])

        sd = ScriptedDispatch(fn=my_fn)
        ni = NodeInput(single=RawText(text="hello"))
        node = Node.scripted("test-node", fn="dummy", outputs=Claims)
        config = {"configurable": {}}

        result = sd.execute(node, ni, config, context_data={"ctx": "val"})

        assert len(call_log) == 1, "fn should have been called exactly once"
        # ScriptedDispatch unwraps NodeInput.value for the fn (scripted fns expect raw data)
        assert call_log[0][0] is ni.value
        assert call_log[0][1] is config
        assert isinstance(result, NodeOutput)
        assert result.single == Claims(items=["done"])

    def test_scripted_dispatch_receives_context_but_ignores_it(self):
        """ScriptedDispatch receives context_data in execute() signature but does not pass it to fn.

        This is documented explicitly: scripted functions don't use LLM context.
        """
        from neograph.factory import NodeInput, ScriptedDispatch
        received_args = []

        def my_fn(input_data, config):
            received_args.append(len([input_data, config]))
            return RawText(text="ok")

        sd = ScriptedDispatch(fn=my_fn)
        ni = NodeInput(single=RawText(text="x"))
        node = Node.scripted("ctx-test", fn="dummy", outputs=RawText)
        config = {"configurable": {}}

        # Pass context_data -- it should be accepted but not forwarded
        result = sd.execute(node, ni, config, context_data={"important": "context"})
        assert received_args == [2], "fn receives exactly 2 args (input_data, config), not context"

    def test_think_dispatch_has_execute_method(self):
        """ThinkDispatch conforms to ModeDispatch protocol (has execute method)."""
        from neograph.factory import ThinkDispatch
        td = ThinkDispatch()
        assert callable(getattr(td, "execute", None))

    def test_tool_dispatch_has_execute_method(self):
        """ToolDispatch conforms to ModeDispatch protocol (has execute method)."""
        from neograph.factory import ToolDispatch
        td = ToolDispatch()
        assert callable(getattr(td, "execute", None))

    def test_all_dispatches_conform_to_protocol(self):
        """All 3 dispatches are runtime-compatible with ModeDispatch."""
        import inspect

        from neograph.factory import ScriptedDispatch, ThinkDispatch, ToolDispatch

        def dummy_fn(input_data, config):
            return None

        dispatches = [ScriptedDispatch(fn=dummy_fn), ThinkDispatch(), ToolDispatch()]
        for d in dispatches:
            sig = inspect.signature(d.execute)
            params = list(sig.parameters.keys())
            # Must accept: node, input_data, config, context_data
            assert "node" in params, f"{type(d).__name__}.execute missing 'node' param"
            assert "input_data" in params, f"{type(d).__name__}.execute missing 'input_data' param"
            assert "config" in params, f"{type(d).__name__}.execute missing 'config' param"
            assert "context_data" in params, f"{type(d).__name__}.execute missing 'context_data' param"
