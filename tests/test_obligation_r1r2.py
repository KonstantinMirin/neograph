"""Obligation probe tests for R1/R2 fixes — meta-hammer analysis.

Each test probes a specific edge case in recently-fixed code to find
bugs in the fixes themselves.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Each,
    Node,
    compile,
    run,
)
from neograph.factory import register_condition, register_scripted
from neograph.modifiers import Loop, Operator

# ═══════════════════════════════════════════════════════════════════════════
# Shared schemas
# ═══════════════════════════════════════════════════════════════════════════

class Alpha(BaseModel, frozen=True):
    value: str

class Beta(BaseModel, frozen=True):
    value: str
    score: float = 0.0

class Gamma(BaseModel, frozen=True):
    result: str


# ═══════════════════════════════════════════════════════════════════════════
# 1. Loop + dict-form multi-key inputs (factory.py:_extract_input)
# ═══════════════════════════════════════════════════════════════════════════

class TestLoopDictFormEdgeCases:
    """Probe edge cases in the Loop + dict-form fix at factory.py:515-544."""

    def test_loop_with_hyphenated_node_name(self):
        """Node name with hyphens: name.replace('-', '_') must match state key."""
        iteration = [0]

        register_scripted("hyp_ctx", lambda i, c: Alpha(value="context"))

        def refine_fn(input_data, config):
            iteration[0] += 1
            assert isinstance(input_data, dict), (
                f"Expected dict input, got {type(input_data).__name__}"
            )
            ctx = input_data.get("context")
            assert isinstance(ctx, Alpha), (
                f"Iter {iteration[0]}: context={type(ctx).__name__}"
            )
            return Beta(value=f"v{iteration[0]}", score=0.4 * iteration[0])

        register_scripted("hyp_refine", refine_fn)

        pipeline = Construct("hyp-test", nodes=[
            Node.scripted("context", fn="hyp_ctx", outputs=Alpha),
            Node.scripted("my-refiner", fn="hyp_refine",
                          inputs={"context": Alpha, "my_refiner": Beta},
                          outputs=Beta)
            | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "hyp"})
        assert iteration[0] >= 2

    def test_loop_self_ref_is_last_key(self):
        """Self-reference key is the LAST key (not first) in inputs dict."""
        iteration = [0]

        register_scripted("last_ctx", lambda i, c: Alpha(value="ctx"))

        def refine_fn(input_data, config):
            iteration[0] += 1
            assert isinstance(input_data, dict), (
                f"Expected dict input, got {type(input_data).__name__}"
            )
            ctx = input_data.get("context")
            # Context must be Alpha, not Beta
            assert isinstance(ctx, Alpha), (
                f"Iter {iteration[0]}: context={type(ctx).__name__}, val={ctx}"
            )
            return Beta(value=f"v{iteration[0]}", score=0.5 * iteration[0])

        register_scripted("last_refine", refine_fn)

        # Deliberately order inputs so self-ref is LAST
        pipeline = Construct("last-test", nodes=[
            Node.scripted("context", fn="last_ctx", outputs=Alpha),
            Node.scripted("refine-last", fn="last_refine",
                          inputs={"context": Alpha, "refine_last": Beta},
                          outputs=Beta)
            | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "last"})
        assert iteration[0] >= 2

    def test_loop_two_upstreams_same_type_as_output(self):
        """Two upstream fields have the same type as the Loop output.
        The self-reference must still be correctly identified."""
        iteration = [0]

        register_scripted("same_a", lambda i, c: Beta(value="a", score=0.1))
        register_scripted("same_b", lambda i, c: Beta(value="b", score=0.2))

        def refine_fn(input_data, config):
            iteration[0] += 1
            assert isinstance(input_data, dict), (
                f"Expected dict input, got {type(input_data).__name__}"
            )
            source_a = input_data.get("source_a")
            # source_a must come from upstream (state), not be the loop value
            assert isinstance(source_a, Beta), f"source_a={type(source_a).__name__}"
            assert source_a.value == "a", f"source_a.value={source_a.value}"
            return Beta(value=f"merged-v{iteration[0]}", score=0.5 * iteration[0])

        register_scripted("same_merge", refine_fn)

        pipeline = Construct("same-type-test", nodes=[
            Node.scripted("source-a", fn="same_a", outputs=Beta),
            Node.scripted("merge-node", fn="same_merge",
                          inputs={"source_a": Beta, "merge_node": Beta},
                          outputs=Beta)
            | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5),
        ])
        graph = compile(pipeline)
        result = run(graph, input={"node_id": "same"})
        assert iteration[0] >= 2


# ═══════════════════════════════════════════════════════════════════════════
# 2. _resolve_merge_args Loop unwrap (decorators.py:418)
# ═══════════════════════════════════════════════════════════════════════════

class TestResolveMergeArgsEdgeCases:
    """Probe edge cases in the from_state unwrap fix at decorators.py:418-422."""

    def test_from_state_non_list_value(self):
        """Non-list state value should pass through without unwrap attempt."""
        from types import SimpleNamespace

        from neograph.decorators import _resolve_merge_args
        from neograph.di import DIBinding, DIKind

        param_res = {"field": DIBinding(name="field", kind=DIKind.FROM_STATE, inner_type=type(None), required=False)}
        state = SimpleNamespace(field=Alpha(value="direct"))
        config = {"configurable": {}}

        args = _resolve_merge_args(param_res, config, state)
        assert len(args) == 1
        assert isinstance(args[0], Alpha)
        assert args[0].value == "direct"

    def test_from_state_empty_list(self):
        """Empty list in state: val[-1] would IndexError. Should handle gracefully."""
        from types import SimpleNamespace

        from neograph.decorators import _resolve_merge_args
        from neograph.di import DIBinding, DIKind

        param_res = {"field": DIBinding(name="field", kind=DIKind.FROM_STATE, inner_type=type(None), required=False)}
        state = SimpleNamespace(field=[])
        config = {"configurable": {}}

        # Should NOT raise IndexError
        args = _resolve_merge_args(param_res, config, state)
        assert len(args) == 1
        # Empty list = first Loop iteration → None (neograph-26ih unified unwrap)
        assert args[0] is None

    def test_from_state_none_state(self):
        """state=None should return None for from_state params."""
        from neograph.decorators import _resolve_merge_args
        from neograph.di import DIBinding, DIKind

        param_res = {"field": DIBinding(name="field", kind=DIKind.FROM_STATE, inner_type=type(None), required=False)}
        config = {"configurable": {}}

        args = _resolve_merge_args(param_res, config, None)
        assert len(args) == 1
        assert args[0] is None


# ═══════════════════════════════════════════════════════════════════════════
# 3. _merge_dicts guards (state.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestMergeDictsEdgeCases:
    """Probe edge cases in _merge_dicts guards."""

    def test_new_is_list_of_dicts(self):
        """new is a list of dicts (bad reducer chain). Should not crash."""
        from neograph.state import _merge_dicts

        existing = {"a": 1}
        new = [{"b": 2}, {"c": 3}]
        # The guard `not isinstance(new, dict)` should catch this
        result = _merge_dicts(existing, new)
        assert result == {"a": 1}  # returns existing unchanged

    def test_existing_is_truthy_non_dict(self):
        """existing is True/1/non-dict truthy. Should reset to empty dict."""
        from neograph.state import _merge_dicts

        # True
        result = _merge_dicts(True, {"a": 1})
        assert result == {"a": 1}

        # 1
        result = _merge_dicts(1, {"b": 2})
        assert result == {"b": 2}

        # "string"
        result = _merge_dicts("hello", {"c": 3})
        assert result == {"c": 3}

    def test_existing_is_zero_or_false(self):
        """existing is 0 or False (falsy non-None). Should reset to empty dict."""
        from neograph.state import _merge_dicts

        result = _merge_dicts(0, {"a": 1})
        assert result == {"a": 1}

        result = _merge_dicts(False, {"b": 2})
        assert result == {"b": 2}


# ═══════════════════════════════════════════════════════════════════════════
# 5. Each fan-in validation (_construct_validation.py:417-422)
# ═══════════════════════════════════════════════════════════════════════════

class TestEachFanInValidationEdgeCases:
    """Probe: fan_out_param set AND unmatched keys — does skip double-fire?

    Validation runs at Construct() init time (not at compile time).
    Tests that need to pass validation register dummy scripted fns.
    Tests that expect ConstructError just construct — the error fires in __init__.
    """

    def test_each_without_fan_out_param_allows_one_unmatched(self):
        """Programmatic API: Each modifier without fan_out_param set.
        ONE unmatched key should be silently allowed (Each receiver slot)."""
        register_scripted("each_v_claims", lambda i, c: Alpha(value="x"))
        register_scripted("each_v_consumer", lambda i, c: Gamma(result="y"))

        n = Node.scripted(
            "consumer",
            fn="each_v_consumer",
            inputs={"claims": Alpha, "item": Beta},
            outputs=Gamma,
        )
        n = n | Each(over="claims", key="value")
        # fan_out_param NOT set (programmatic API path)

        producer = Node.scripted("claims", fn="each_v_claims", outputs=Alpha)

        # Should NOT raise at Construct() init — the Each skip handles "item"
        pipeline = Construct("test-each-skip", nodes=[producer, n])
        assert isinstance(pipeline, Construct)
        assert pipeline.name == "test-each-skip"

    def test_each_without_fan_out_param_two_unmatched_is_error(self):
        """Programmatic API: Each modifier without fan_out_param.
        TWO unmatched keys should error — only one skip allowed."""
        n = Node.scripted(
            "consumer",
            fn="f",
            inputs={"claims": Alpha, "item": Beta, "extra": Gamma},
            outputs=Gamma,
        )
        n = n | Each(over="claims", key="value")

        producer = Node.scripted("claims", fn="f", outputs=Alpha)

        with pytest.raises(ConstructError):
            Construct("test-each-two-unmatched", nodes=[producer, n])

    def test_fan_out_param_set_plus_each_skip_double_fires(self):
        """BUG PROBE: fan_out_param='item' skips 'item' at line 416,
        but then has_each=True still allows ONE additional unmatched key.
        So 'mystery' gets silently swallowed.

        inputs: {claims: Alpha, item: Beta, mystery: Gamma}
        - 'claims' matches upstream producer
        - 'item' skipped by fan_out_param (line 416)
        - 'mystery' should error BUT each_skip (line 422-424) swallows it

        This is a BUG: when fan_out_param is set, the Each skip should NOT
        also fire, because the fan-out receiver slot is already consumed.
        """
        n = Node.scripted(
            "consumer",
            fn="f",
            inputs={"claims": Alpha, "item": Beta, "mystery": Gamma},
            outputs=Gamma,
        )
        n = n | Each(over="claims", key="value")
        object.__setattr__(n, "fan_out_param", "item")

        producer = Node.scripted("claims", fn="f", outputs=Alpha)

        # EXPECTATION: 'mystery' is a typo and should error.
        # ACTUAL: both fan_out_param skip AND each_skip fire → silent pass
        with pytest.raises(ConstructError, match="mystery"):
            Construct("test-double-fire", nodes=[producer, n])


# ═══════════════════════════════════════════════════════════════════════════
# 6. _extract_json edge cases (_llm.py:236-270)
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractJsonEdgeCases:
    """Probe edge cases in the prose-brace skip fix."""

    def test_empty_string(self):
        """Empty input should not infinite-loop."""
        from neograph._llm import _extract_json

        result = _extract_json("")
        assert result == ""

    def test_no_json_at_all(self):
        """Pure prose with no braces should return stripped text, not loop."""
        from neograph._llm import _extract_json

        result = _extract_json("This is just plain text with no JSON.")
        assert result == "This is just plain text with no JSON."

    def test_prose_with_braces_but_no_json(self):
        """Prose containing {variable} style interpolation."""
        from neograph._llm import _extract_json

        text = "The result of {a + b} is 42. And {x * y} is 100."
        result = _extract_json(text)
        # Should return the whole text since no valid JSON found
        assert "a + b" in result or result == text.strip()

    def test_json_starts_with_bracket_inside(self):
        """JSON object containing array: {"data": [1, 2, 3]}."""
        from neograph._llm import _extract_json

        text = 'Here is the result: {"data": [1, 2, 3]}'
        result = _extract_json(text)
        assert result == '{"data": [1, 2, 3]}'

    def test_prose_double_quote_brace(self):
        """Prose containing {" which looks like JSON start but isn't really.
        The after_brace check sees '"' and tries to parse it as JSON."""
        from neograph._llm import _extract_json

        text = 'She said {"hello"} and left. The real JSON is {"name": "Alice"}'
        result = _extract_json(text)
        # First {"hello"} is a balanced JSON-like object, will be extracted
        # This may be a false positive — let's see what happens
        assert "{" in result

    def test_unbalanced_single_open_brace(self):
        """Text with a single { and no closing }."""
        from neograph._llm import _extract_json

        text = 'Unbalanced { but no close'
        result = _extract_json(text)
        # Should not infinite loop — should return text.strip()
        assert result == text.strip()

    def test_response_with_json_after_prose_braces(self):
        """Prose braces followed by real JSON — must find the JSON."""
        from neograph._llm import _extract_json

        text = 'Calculate {a + b} first. Result: {"value": 42}'
        result = _extract_json(text)
        assert result == '{"value": 42}'

    def test_deeply_nested_json(self):
        """Deeply nested JSON object."""
        from neograph._llm import _extract_json

        text = '{"a": {"b": {"c": {"d": 1}}}}'
        result = _extract_json(text)
        assert result == text

    def test_json_with_escaped_quotes(self):
        """JSON with escaped quotes in string values."""
        from neograph._llm import _extract_json

        text = r'{"key": "value with \"escaped\" quotes"}'
        result = _extract_json(text)
        assert result == text

    def test_multiple_json_objects_returns_first(self):
        """Multiple JSON objects — should return the first balanced one."""
        from neograph._llm import _extract_json

        text = '{"first": 1} {"second": 2}'
        result = _extract_json(text)
        assert result == '{"first": 1}'


# ═══════════════════════════════════════════════════════════════════════════
# 4. Operator + other modifiers on sub-construct (compiler.py)
# ═══════════════════════════════════════════════════════════════════════════

class TestOperatorOnSubConstruct:
    """Probe: Operator chaining with other modifiers on sub-constructs."""

    def test_operator_with_loop_on_sub_construct(self):
        """Sub-construct with both Operator and Loop. The sub-construct has
        an internal Loop. Operator check applied AFTER the sub-construct.
        The Operator is on the outer Construct, not inside the sub-construct."""
        from langgraph.checkpoint.memory import MemorySaver

        register_scripted("inner_seed", lambda i, c: Beta(value="seed", score=0.0))

        iteration = [0]
        def inner_refine(i, c):
            iteration[0] += 1
            return Beta(value=f"v{iteration[0]}", score=0.5 * iteration[0])

        register_scripted("inner_refine", inner_refine)

        inner = Construct(
            "inner-loop",
            nodes=[
                Node.scripted("inner-seed", fn="inner_seed", outputs=Beta),
                Node.scripted("inner-refine", fn="inner_refine", inputs=Beta, outputs=Beta)
                | Loop(when=lambda d: d is None or d.score < 0.8, max_iterations=5),
            ],
            input=Alpha,
            output=Beta,
        )
        # Apply Operator on the sub-construct
        register_condition("always_ok", lambda state: False)  # don't interrupt
        inner_with_op = inner | Operator(when="always_ok")

        register_scripted("parent_seed", lambda i, c: Alpha(value="start"))

        parent = Construct("parent", nodes=[
            Node.scripted("parent-seed", fn="parent_seed", outputs=Alpha),
            inner_with_op,
        ])
        checkpointer = MemorySaver()
        graph = compile(parent, checkpointer=checkpointer)
        result = run(
            graph,
            input={"node_id": "op-loop"},
            config={"configurable": {"thread_id": "test-op-loop"}},
        )
        # Inner sub-construct produces Beta via inner_refine
        assert isinstance(result, dict)
        assert "parent_seed" in result or "inner_loop" in result or len(result) > 0
