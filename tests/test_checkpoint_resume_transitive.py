"""Transitive fingerprint invalidation in runner._compute_invalidated_nodes.

Regression guard for neograph-j36u. When an upstream node's fingerprint
changes, every transitive descendant must be returned as invalidated —
otherwise downstream nodes with unchanged signatures are silently stale.

Adjacency is keyed by state-field name (the same key the per-node
fingerprint store uses), and modifier-bearing nodes participate via their
state-field names.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import BaseModel

from neograph import Construct, Each, Node, compile
from neograph.runner import _compute_invalidated_nodes
from neograph.state import compute_node_fingerprints
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


# ---------------------------------------------------------------------------
# Fingerprint coarseness (neograph-v63o / review 080726 PAT-03, LOW-01).
# The per-node fingerprint used to hash ONLY ``type.__qualname__``, so a
# field-level change to an output model that keeps the SAME class name did not
# change the fingerprint -> the node was never marked invalidated and the
# auto-rewind never triggered. These pin the structural (field-aware) fingerprint.
# ---------------------------------------------------------------------------


def _make_same_qualname_model(with_extra: bool) -> type:
    """Two models with an IDENTICAL ``__qualname__`` (``_make...<locals>.Payload``)
    differing only by a field. This is the exact false-negative the qualname-only
    fingerprint missed."""
    if with_extra:

        class Payload(BaseModel):
            val: str = "x"
            extra: int = 0

    else:

        class Payload(BaseModel):
            val: str = "x"

    return Payload


def test_node_fingerprint_distinguishes_same_qualname_different_fields():
    """Same class name, changed field set -> the node fingerprint must differ."""
    p1 = _make_same_qualname_model(with_extra=False)
    p2 = _make_same_qualname_model(with_extra=True)
    assert p1.__qualname__ == p2.__qualname__, "precondition: identical qualname"

    c1 = Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p1)])
    c2 = Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p2)])

    fp1 = compute_node_fingerprints(c1)
    fp2 = compute_node_fingerprints(c2)
    assert fp1["a"] != fp2["a"], (
        "qualname-only fingerprint collided two structurally-different models; "
        "a same-name field change must change the per-node fingerprint"
    )


def test_schema_fingerprint_distinguishes_same_qualname_different_fields():
    """The schema fingerprint GATES the whole auto-rewind check (a match returns
    early). It too must open on a same-qualname field change, or the enriched
    node fingerprint is never reached."""
    p1 = _make_same_qualname_model(with_extra=False)
    p2 = _make_same_qualname_model(with_extra=True)

    g1 = compile(Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p1)]), **build_test_compile_kwargs())
    g2 = compile(Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p2)]), **build_test_compile_kwargs())

    assert g1.schema_fingerprint != g2.schema_fingerprint, (
        "schema fingerprint gate stayed closed on a same-qualname field change; auto-rewind would never trigger"
    )


def test_same_qualname_field_change_invalidates_node():
    """End-to-end at the invalidation layer: recompiling with a same-qualname
    output model that gained a field must flag the node as invalidated."""
    p1 = _make_same_qualname_model(with_extra=False)
    p2 = _make_same_qualname_model(with_extra=True)

    g1 = compile(Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p1)]), **build_test_compile_kwargs())
    stored = dict(g1.node_fingerprints)

    g2 = compile(Construct("fp-pipe", nodes=[Node.scripted("a", fn="tj_a", outputs=p2)]), **build_test_compile_kwargs())
    invalidated = _compute_invalidated_nodes(g2, {"neo_node_fingerprints": stored})
    assert "a" in invalidated


def test_dict_form_output_keys_get_distinct_structural_fingerprints():
    """Regression pin for the metaclass-hash bug fixed in neograph-v63o: the old
    dict-form branch hashed type(typ).__name__ — 'ModelMetaclass' for EVERY
    BaseModel — collapsing all dict-form output keys of all models to one
    identical hash, so a changed dict-form value type never invalidated. The
    structural _type_signature must give distinct models distinct per-key
    fingerprints, and a field change on ONE key must change ONLY that key."""
    from pydantic import create_model

    alpha = create_model("TjAlpha", a=(str, ...))
    beta = create_model("TjBeta", b=(int, ...))
    node = Node.scripted("d", fn="tj_a", outputs={"one": alpha, "two": beta})
    fps = compute_node_fingerprints(Construct("fp-dict", nodes=[node]))
    assert fps["d_one"] != fps["d_two"], (
        "distinct dict-form value models must not share a fingerprint (the ModelMetaclass collapse)"
    )

    alpha2 = create_model("TjAlpha", a=(int, ...))  # same qualname, changed field type
    node2 = Node.scripted("d", fn="tj_a", outputs={"one": alpha2, "two": beta})
    fps2 = compute_node_fingerprints(Construct("fp-dict", nodes=[node2]))
    assert fps2["d_one"] != fps["d_one"], "changed field type must change that key"
    assert fps2["d_two"] == fps["d_two"], "untouched key must keep its fingerprint"


def test_old_format_node_fingerprint_invalidates_on_upgrade():
    """Migration behavior (DESIRED + pinned): checkpoints written before v63o
    carry qualname-only node fingerprints. On the first resume after upgrade the
    new structural fingerprint differs from the stored one, so every node is
    invalidated -> one full re-run. Better than silently trusting a stale, coarser
    signature."""
    graph = compile(_linear_chain(_A), **build_test_compile_kwargs())

    # Reconstruct the OLD (pre-v63o) formula: sha256(f"{fname}:{qualname}")[:12].
    old_a = hashlib.sha256(f"a:{_A.__qualname__}".encode()).hexdigest()[:12]
    stored = dict(graph.node_fingerprints)
    stored["a"] = old_a

    invalidated = _compute_invalidated_nodes(graph, {"neo_node_fingerprints": stored})
    assert "a" in invalidated, "old-format stored fingerprint must be seen as changed after upgrade"
