"""Sub-construct context-field validation and output-boundary checks."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import (
    Construct,
    ConstructError,
    Node,
)
from tests.schemas import (
    Claims,
    MatchResult,
    RawText,
    _producer,
)


class TestSubConstructOutputBoundary:
    """When a Construct declares output=SomeType, at least one internal node
    must produce a compatible type. Silent None propagation at runtime
    indicates the contract was never satisfied."""

    def test_output_mismatch_raises_when_no_node_produces_output(self):
        """Sub-construct declares output=Claims but only node produces RawText."""
        inner = _producer("inner", RawText)
        with pytest.raises(ConstructError, match="output=Claims"):
            Construct("bad-sub", input=RawText, output=Claims, nodes=[inner])

    def test_output_match_passes_when_node_produces_compatible_type(self):
        """Sub-construct declares output=Claims and inner node produces Claims."""
        inner = Node.scripted("inner", fn="f", inputs=RawText, outputs=Claims)
        sub = Construct("ok-sub", input=RawText, output=Claims, nodes=[inner])
        assert sub.output is Claims

    def test_output_subclass_passes_when_node_produces_subclass(self):
        """Sub-construct output=BaseModel is satisfied by any Pydantic model."""
        inner = _producer("inner", Claims)
        sub = Construct("sub-sub", output=BaseModel, nodes=[inner])
        assert sub.output is BaseModel

    def test_no_output_declared_skips_check(self):
        """Construct without output= declaration skips boundary check."""
        inner = _producer("inner", RawText)
        pipeline = Construct("top-level", nodes=[inner])
        assert pipeline.output is None

    def test_multiple_nodes_passes_when_last_produces_output(self):
        """Only one of several nodes needs to produce the output type."""
        inner_a = _producer("a", RawText)
        inner_b = Node.scripted("b", fn="f", inputs=RawText, outputs=Claims)
        sub = Construct("multi", input=RawText, output=Claims, nodes=[inner_a, inner_b])
        assert sub.output is Claims


# ═══════════════════════════════════════════════════════════════════════════
# lint() — DI binding validation (neograph-no0q)
# ═══════════════════════════════════════════════════════════════════════════


class TestContextFieldProducerValidation:
    """§7 / izo1-C — node.context fields must have an upstream producer.

    Without this check, a typoed context name silently renders as the literal
    string ``"None"`` in LLM prompts (see _execute.py:46). The validator
    catches the typo at compile time and refuses to construct the pipeline.
    """

    def _make_consumer(self, name: str, in_type: type, ctx: list[str]) -> Node:
        return Node(
            name=name,
            mode="scripted",
            scripted_fn="f",
            inputs={in_type.__name__.lower(): in_type},
            outputs=MatchResult,
            context=ctx,
        )

    def test_typoed_context_field_with_no_producer_raises_at_compile(self):
        """consumer.context=['nonexistent'] with no upstream producer raises."""
        producer = _producer("topic", RawText)
        consumer = Node(
            name="summarize",
            mode="scripted",
            scripted_fn="f",
            inputs={"topic": RawText},
            outputs=MatchResult,
            context=["nonexistent_field"],
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("bad-context", nodes=[producer, consumer])
        msg = str(exc_info.value)
        assert "nonexistent_field" in msg
        assert "no upstream node" in msg or "context" in msg.lower()

    def test_error_message_names_node_and_field(self):
        """Error message includes node name and the offending context field."""
        producer = _producer("topic", RawText)
        consumer = Node(
            name="summarize",
            mode="scripted",
            scripted_fn="f",
            inputs={"topic": RawText},
            outputs=MatchResult,
            context=["typoed"],
        )
        with pytest.raises(ConstructError) as exc_info:
            Construct("ctx-msg", nodes=[producer, consumer])
        msg = str(exc_info.value)
        assert "summarize" in msg
        assert "typoed" in msg

    def test_valid_context_field_with_upstream_producer_passes(self):
        """consumer.context=['catalog'] where 'catalog' is an upstream output
        compiles cleanly (regression guard — no false positive)."""
        catalog = _producer("catalog", RawText)
        consumer = Node(
            name="summarize",
            mode="scripted",
            scripted_fn="f",
            inputs={"catalog": RawText},
            outputs=MatchResult,
            context=["catalog"],
        )
        pipeline = Construct("ctx-ok", nodes=[catalog, consumer])
        assert len(pipeline.nodes) == 2

    def test_hyphenated_context_field_with_producer_passes(self):
        """node.context references the unmangled node name; the validator
        must apply field_name_for() consistently when comparing against
        producers. Regression for hyphen handling."""
        producer = _producer("topic-source", RawText)
        consumer = Node(
            name="summarize",
            mode="scripted",
            scripted_fn="f",
            inputs={"topic_source": RawText},
            outputs=MatchResult,
            context=["topic-source"],
        )
        pipeline = Construct("ctx-hyphen", nodes=[producer, consumer])
        assert len(pipeline.nodes) == 2


class TestSubConstructContextFieldValidation:
    """neograph-51m7: a context field declared on a node INSIDE a
    sub-construct must be produced by an UPSTREAM node in the PARENT
    construct. The validator previously skipped this check whenever
    construct.input was set (i.e. for every sub-construct), so typos
    inside sub-constructs only surfaced at runtime — breaking the
    'if it compiles, it runs' positioning."""

    def _build_subconstruct_with_inner_context(self, ctx_field: str) -> Construct:
        """Build a sub-construct whose inner node declares context=[ctx_field].
        The inner node consumes the sub-construct's port input."""
        from neograph._state_keys import StateKeys

        inner = Node(
            name="inner",
            mode="scripted",
            scripted_fn="f",
            inputs={StateKeys.SUBGRAPH_INPUT: RawText},
            outputs=MatchResult,
            context=[ctx_field],
        )
        return Construct("sub", input=RawText, output=MatchResult, nodes=[inner])

    def test_subconstruct_inner_context_typo_rejected_at_compile(self):
        """Inner node's context=['nonexistent'] with no upstream producer in
        the parent must be rejected at parent-construct compile time."""
        parent_seed = _producer("topic", RawText)
        sub = self._build_subconstruct_with_inner_context("nonexistent_field")
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent", nodes=[parent_seed, sub])
        msg = str(exc_info.value)
        assert "nonexistent_field" in msg
        assert "context" in msg.lower() or "no upstream" in msg

    def test_subconstruct_inner_context_with_parent_producer_passes(self):
        """Inner node's context=['catalog'] with 'catalog' produced by an
        upstream parent node compiles cleanly (no false positive)."""
        catalog = _producer("catalog", RawText)
        parent_seed = _producer("topic", RawText)
        sub = self._build_subconstruct_with_inner_context("catalog")
        parent = Construct("parent-ok", nodes=[catalog, parent_seed, sub])
        assert len(parent.nodes) == 3

    def test_subconstruct_inner_context_error_names_inner_node(self):
        """Error message must name the offending INNER node, not the
        sub-construct, so the user can find the typo."""
        parent_seed = _producer("topic", RawText)
        sub = self._build_subconstruct_with_inner_context("typoed")
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent-msg", nodes=[parent_seed, sub])
        msg = str(exc_info.value)
        assert "inner" in msg
        assert "typoed" in msg

    # ── Depth-coverage tests (ta43) ────────────────────────────────────────
    # The validator must be recursive: typo'd contexts at ANY depth of
    # sub-construct nesting must be rejected at compile time.

    def _build_nested_subconstruct(self, ctx_field: str, depth: int) -> Construct:
        """Build a sub-construct with `depth` levels of nesting. The deepest
        inner node declares context=[ctx_field]."""
        from neograph._state_keys import StateKeys

        # Deepest inner node — declares the suspect context.
        leaf = Node(
            name=f"leaf-d{depth}",
            mode="scripted",
            scripted_fn="f",
            inputs={StateKeys.SUBGRAPH_INPUT: RawText},
            outputs=MatchResult,
            context=[ctx_field],
        )
        current: Construct = Construct(
            f"sub-d{depth}",
            input=RawText,
            output=MatchResult,
            nodes=[leaf],
        )
        # Wrap in additional layers up to the target depth.
        for level in range(depth - 1, 0, -1):
            current = Construct(
                f"sub-d{level}",
                input=RawText,
                output=MatchResult,
                nodes=[current],
            )
        return current

    def test_depth_3_typo_rejected_at_compile(self):
        """A context typo at the DEEPEST level of a parent->sub->sub-sub
        topology must be caught at compile time. Today's elif-based fix
        (commit 0330cea) handles depth 1 only — this depth-3 case is the
        ta43 reproducer."""
        parent_seed = _producer("topic", RawText)
        deep = self._build_nested_subconstruct("nonexistent_deep", depth=3)
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent-d3-typo", nodes=[parent_seed, deep])
        msg = str(exc_info.value)
        assert "nonexistent_deep" in msg, msg

    def test_depth_3_happy_path_compiles(self):
        """Depth-3 nesting where the deepest context IS produced by the
        parent must compile cleanly (no false positive from the recursive
        check)."""
        catalog = _producer("catalog", RawText)
        parent_seed = _producer("topic", RawText)
        deep = self._build_nested_subconstruct("catalog", depth=3)
        parent = Construct("parent-d3-ok", nodes=[catalog, parent_seed, deep])
        assert len(parent.nodes) == 3

    def test_depth_4_typo_rejected_at_compile(self):
        """Even-deeper nesting; further guards against off-by-one in any
        recursive implementation."""
        parent_seed = _producer("topic", RawText)
        deep = self._build_nested_subconstruct("nonexistent_4", depth=4)
        with pytest.raises(ConstructError) as exc_info:
            Construct("parent-d4-typo", nodes=[parent_seed, deep])
        msg = str(exc_info.value)
        assert "nonexistent_4" in msg, msg
