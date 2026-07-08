"""Transitive fingerprint invalidation in runner._compute_invalidated_nodes.

Regression guard for neograph-j36u. When an upstream node's fingerprint
changes, every transitive descendant must be returned as invalidated —
otherwise downstream nodes with unchanged signatures are silently stale.

Adjacency is keyed by state-field name (the same key the per-node
fingerprint store uses), and modifier-bearing nodes participate via their
state-field names.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, Node, compile
from neograph.runner import _compute_invalidated_nodes
from tests.fakes import build_test_compile_kwargs, register_scripted

# Dummy shims for compile() — these tests inspect fingerprints only, never
# execute the graph, so any callable will do. Registered per-test via fixture
# because the autouse conftest fixture clears the test registry between tests.
_DUMMY = lambda input_data, config: None  # noqa: E731
_SHIM_NAMES = (
    "tj_a",
    "tj_b",
    "tj_c",
    "tj_each_a_v1",
    "tj_each_a_v2",
    "tj_each_b",
    "tj_each_c",
    "tj_d_a_v1",
    "tj_d_a_v2",
    "tj_d_b",
    "tj_d_c",
    "tj_d_d",
)


@pytest.fixture(autouse=True)
def _register_dummies():
    for name in _SHIM_NAMES:
        register_scripted(name, _DUMMY)


class _A(BaseModel):
    val: str = "a"


class _B(BaseModel):
    val: str = "b"


class _C(BaseModel):
    val: str = "c"


class _A_v2(BaseModel):
    val: str = "a"
    extra: int = 0


def _linear_chain(a_out: type, b_out: type = _B, c_out: type = _C) -> Construct:

    return Construct(
        "tj-pipe",
        nodes=[
            Node.scripted("a", fn="tj_a", outputs=a_out),
            Node.scripted("b", fn="tj_b", inputs={"a": a_out}, outputs=b_out),
            Node.scripted("c", fn="tj_c", inputs={"b": b_out}, outputs=c_out),
        ],
    )


def test_compute_invalidated_nodes_returns_transitive_closure():
    """A→B→C: change A's output type. B and C signatures unchanged.
    Returned set must contain {a, b, c} (transitive closure).
    """
    pipe_v1 = _linear_chain(_A)
    graph_v1 = compile(pipe_v1, **build_test_compile_kwargs())
    stored_fps = dict(graph_v1.node_fingerprints)

    pipe_v2 = _linear_chain(_A_v2)
    graph_v2 = compile(pipe_v2, **build_test_compile_kwargs())

    channel_values = {"neo_node_fingerprints": stored_fps}
    invalidated = _compute_invalidated_nodes(graph_v2, channel_values)

    assert invalidated == {"a", "b", "c"}, f"Expected full transitive closure {{a, b, c}}, got {sorted(invalidated)}"


def test_compute_invalidated_nodes_empty_when_unchanged():
    pipe = _linear_chain(_A)
    graph = compile(pipe, **build_test_compile_kwargs())
    channel_values = {"neo_node_fingerprints": dict(graph.node_fingerprints)}

    invalidated = _compute_invalidated_nodes(graph, channel_values)
    assert invalidated == set()


def test_compute_invalidated_nodes_through_each_modifier():
    """Modifier-bearing intermediates participate in adjacency via state-field name.

    A produces list[_A]. B has Each(over='a') and consumes each _A → _B.
    C reads B as a list and produces _C.
    Change A's element type. B (Each) and C must both invalidate.
    """

    class _AList(BaseModel):
        items: list

    def build(a_elem: type, a_fn: str) -> Construct:
        return Construct(
            "tj-each-pipe",
            nodes=[
                Node.scripted("a", fn=a_fn, outputs=list[a_elem]),  # type: ignore[valid-type]
                Node.scripted(
                    "b",
                    fn="tj_each_b",
                    inputs={"a": list[a_elem], "item": a_elem},  # type: ignore[valid-type]
                    outputs=_B,
                )
                | Each(over="a", key="val"),
                Node.scripted("c", fn="tj_each_c", inputs={"b": list[_B]}, outputs=_C),
            ],
        )

    pipe_v1 = build(_A, "tj_each_a_v1")
    graph_v1 = compile(pipe_v1, **build_test_compile_kwargs())
    stored_fps = dict(graph_v1.node_fingerprints)

    pipe_v2 = build(_A_v2, "tj_each_a_v2")
    graph_v2 = compile(pipe_v2, **build_test_compile_kwargs())

    channel_values = {"neo_node_fingerprints": stored_fps}
    invalidated = _compute_invalidated_nodes(graph_v2, channel_values)

    assert invalidated == {"a", "b", "c"}, (
        f"Each-bearing intermediate must propagate invalidation. Expected {{a, b, c}}, got {sorted(invalidated)}"
    )


def test_compute_invalidated_nodes_handles_diamond():
    """Diamond A→{B,C}→D. Change A. All four must invalidate."""

    class _D(BaseModel):
        val: str = "d"

    def build(a_out: type, a_fn: str) -> Construct:
        return Construct(
            "tj-diamond",
            nodes=[
                Node.scripted("a", fn=a_fn, outputs=a_out),
                Node.scripted("b", fn="tj_d_b", inputs={"a": a_out}, outputs=_B),
                Node.scripted("c", fn="tj_d_c", inputs={"a": a_out}, outputs=_C),
                Node.scripted(
                    "d",
                    fn="tj_d_d",
                    inputs={"b": _B, "c": _C},
                    outputs=_D,
                ),
            ],
        )

    pipe_v1 = build(_A, "tj_d_a_v1")
    graph_v1 = compile(pipe_v1, **build_test_compile_kwargs())
    stored_fps = dict(graph_v1.node_fingerprints)

    pipe_v2 = build(_A_v2, "tj_d_a_v2")
    graph_v2 = compile(pipe_v2, **build_test_compile_kwargs())

    invalidated = _compute_invalidated_nodes(graph_v2, {"neo_node_fingerprints": stored_fps})
    assert invalidated == {"a", "b", "c", "d"}, f"Diamond: expected all four nodes, got {sorted(invalidated)}"
