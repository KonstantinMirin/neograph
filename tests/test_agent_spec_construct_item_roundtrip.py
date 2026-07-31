"""neograph-s7zt3.11 (Phase 8) -- a MODIFIED sub-Construct must survive the Agent
Spec round trip as a usable sub-construct, not merely as a correctly-labelled one.

THE CORE INVARIANT this module grades (from the ticket's DESIGN field):

    A Construct used as ONE ITEM inside a parent Construct must survive the Agent
    Spec round trip with EXACTLY the modifier combo it was written with AND its
    declared boundary port (input/output) intact, such that the reimported parent
    STILL COMPILES.

The second half is not decoration. Both halves fail today, in two different ways,
and only the four assertions together can see both:

  (a) the reimported item is still a ``Construct``
  (b) ``classify_modifiers(item)[0]`` is EXACTLY the combo it was written with
  (c) its ``input`` / ``output`` boundary matches what was declared
  (d) THE REIMPORTED PARENT COMPILES  -- the outermost acceptance

Assertion (d) is the only one that can catch a seam which reconstructs an item's
SHAPE but not its PORT. Verified at develop @ d1420db, all seven rows:

    row              export  import                                       (a)(b)  (c)(d)
    BARE             OK      OK                                           green   RED
    OPERATOR         OK      OK                                           green   RED
    EACH             OK      ConfigurationError unsupported type FlowNode  RED     RED
    ORACLE           OK      ConfigurationError Oracle group variant node  RED     RED
    EACH_OPERATOR    OK      ConfigurationError unsupported type FlowNode  RED     RED
    ORACLE_OPERATOR  OK      ConfigurationError Oracle group variant node  RED     RED
    LOOP (matching)  OK      ConfigurationError unsupported type FlowNode  RED     RED

BARE and OPERATOR are NOT passive controls. They import cleanly and classify
correctly, and they STILL fail (c) and (d): a sub-Construct declared
``input=Plain, output=Result`` round-trips to ``input=None, output=None`` and the
reimported parent then dies with::

    CompileError: [Construct 'sub'] sub-construct has no output type

That is the defect the ticket's HIGH-1 finding named -- the de-facto seam (the
BARE arm's inlined ``from_agent_spec(spec_node.subflow)`` recursion) drops the
boundary, and hoisting it unchanged would replicate the drop at five call sites.
So BARE/OPERATOR being red here is the point: a version of this suite in which
they pass on arrival is a suite that is not really asserting (c) and (d).

THE MATCHING-BOUNDARY LOOP ROW -- READ THIS BEFORE CITING IT AS COVERAGE.
The LOOP row below is pinned at the MATCHING boundary ONLY: ``input == output``
(both ``Plain``). That shape exports cleanly and then dies on import with the
seam's own error, so it is a genuine in-scope red row. The DIFFERING-boundary
shape (``input=A, output=B``) is explicitly OUT OF SCOPE for Phase 8: it never
reaches the import seam because it fails earlier, at EXPORT, with::

    ValidationError: Flow data connection named sub__loop_self_<out> is connected
    to a property named <out> of the destination node sub, but the node does not
    have any property with that name

-- the self-loop DataFlowEdge is built from the loop body's OUTPUT field and
pointed at the node's own input port. That is filed as **neograph-rh5fb** and is
a separate defect in the export lowering this ticket deliberately leaves alone.
A GREEN matching-boundary LOOP row therefore says NOTHING about the
differing-boundary case. Do not read it as covering rh5fb. (The differing shape
is legal at the compiler level -- ``tests/test_loop.py::
TestLoopInputNotEqualOutput::test_construct_loop_input_neq_output_compiles``
pins that it compiles and runs -- so this is an Agent Spec export gap, not an
invalid pipeline.)

SCOPE, restated so nothing here drifts:
  IN  -- the Agent Spec IMPORT path for a modified sub-Construct, and restoring
         the sub-Construct's boundary inside that seam.
  OUT -- LOOP / LOOP_OPERATOR at a DIFFERING boundary (neograph-rh5fb).
  OUT -- any export-side lowering change, including ``_lower_oracle``'s second
         FlowNode construction site.
  OUT -- EACH_ORACLE / EACH_ORACLE_OPERATOR on a sub-Construct, which stay
         permanently fail-loud (``SUB_CONSTRUCT_UNSUPPORTED_COMBOS``, pinned by
         ``test_agent_spec_export.py`` and a should_fail check fixture).

OBJECT-LEVEL ROUND TRIP ONLY -- ``to_agent_spec`` then ``from_agent_spec``, with
NO ``Flow.from_dict`` hop. Dict-form fan-in exports dotted Property titles that
pyagentspec's validator rejects on read-back (**neograph-8zvd1**, unfixed), so a
JSON hop here would surface an unrelated P1 and invite misdiagnosing it as a
Phase 8 bug. Add the JSON hop when 8zvd1 lands -- it strictly widens what these
prove.

WHY (c) COMPARES STRUCTURE, NOT CLASS IDENTITY. An imported pipeline's types are
freshly synthesized from the spec's Property lists, which carry no back-reference
to the original class -- a round-tripped ``Plain`` comes back as
``AgentSpecType_<hash>``. That is pre-existing, deliberate, and orthogonal to
this ticket (see ``test_agent_spec_roundtrip.py``, which asserts the
``AgentSpecType_`` prefix directly). So ``_shape()`` below compares field names
and annotations recursively. This is the strongest available spelling of "the
boundary equals what was declared", not a softened one: it still fails loudly on
the live defect, where the boundary is ``None``.

This module complements ``test_agent_spec_export.py::
TestConstructItemModifierExport``, which covers the same rows in the EXPORT
direction only -- it contains no ``from_agent_spec`` call anywhere, which is how
this gap survived Phase 4.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple, get_args, get_origin

import pytest
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel

from neograph import Construct, Each, Loop, Node, Operator, Oracle, compile
from neograph._agent_spec import to_agent_spec
from neograph.loader import from_agent_spec
from neograph.modifiers import ModifierCombo, classify_modifiers
from tests.fakes import build_test_compile_kwargs, register_condition, register_scripted

# ── fixture types ────────────────────────────────────────────────────────────
#
# Every SUB-CONSTRUCT BOUNDARY type is FLAT (primitive fields only). A boundary
# carrying a direct nested-model field would drag in neograph-p7dyq (export
# overrides a field's title with the nested model's title), which is a separate
# unfixed P1 and would stop these fixtures from being able to fail for the
# reason they exist. Only the Each rows' UPSTREAM producer is nested, and only
# as ``list[Tagged]`` -- the shape ``test_agent_spec_each_operator.py`` already
# round-trips green today.


class Plain(BaseModel, frozen=True):
    token: str


class Tagged(BaseModel, frozen=True):
    label: str


class Bag(BaseModel, frozen=True):
    items: list[Tagged]


class Result(BaseModel, frozen=True):
    value: str


def _register() -> None:
    register_scripted("p8_seed_plain", lambda i, c: Plain(token="t"))
    register_scripted("p8_seed_bag", lambda i, c: Bag(items=[Tagged(label="a"), Tagged(label="b")]))
    register_scripted("p8_inner", lambda i, c: Result(value="ok"))
    register_scripted("p8_inner_same", lambda i, c: Plain(token="t"))
    register_scripted("p8_merge", lambda variants, c: type(variants[0])(**dict(variants[0])))
    # Deliberately FALSE for both the Operator gate and the Loop condition: the
    # rows claim only that the modifier SURVIVES, never pause/resume or
    # iteration semantics.
    register_condition("p8_never", lambda state: None)


def _sub(input_type: type, output_type: type, fn: str = "p8_inner") -> Construct:
    return Construct(
        "sub",
        input=input_type,
        output=output_type,
        nodes=[Node.scripted("inner", fn=fn, inputs=input_type, outputs=output_type)],
    )


def _parent(sub: Construct, seed_fn: str, seed_out: type) -> Construct:
    """One producer feeding the sub-Construct's input port, plus the item."""
    return Construct("parent", nodes=[Node.scripted("seed", fn=seed_fn, outputs=seed_out), sub])


# ── the row table ────────────────────────────────────────────────────────────


class Row(NamedTuple):
    """One (combo, boundary) cell of the Construct-item round-trip matrix."""

    combo: ModifierCombo
    build: Any  # () -> Construct  (the parent, already assembled)
    declared_input: type
    declared_output: type


def _bare() -> Construct:
    return _parent(_sub(Plain, Result), "p8_seed_plain", Plain)


def _operator() -> Construct:
    return _parent(_sub(Plain, Result) | Operator(when="p8_never"), "p8_seed_plain", Plain)


def _each() -> Construct:
    return _parent(_sub(Tagged, Result) | Each(over="seed.items", key="label"), "p8_seed_bag", Bag)


def _oracle() -> Construct:
    return _parent(_sub(Plain, Result) | Oracle(n=2, merge_fn="p8_merge"), "p8_seed_plain", Plain)


def _each_operator() -> Construct:
    return _parent(
        _sub(Tagged, Result) | Each(over="seed.items", key="label") | Operator(when="p8_never"),
        "p8_seed_bag",
        Bag,
    )


def _oracle_operator() -> Construct:
    return _parent(
        _sub(Plain, Result) | Oracle(n=2, merge_fn="p8_merge") | Operator(when="p8_never"),
        "p8_seed_plain",
        Plain,
    )


def _loop_matching_boundary() -> Construct:
    """LOOP at input == output == Plain. See the module docstring: the DIFFERING
    boundary is out of scope (neograph-rh5fb, fails at EXPORT)."""
    return _parent(
        _sub(Plain, Plain, fn="p8_inner_same") | Loop(when="p8_never", max_iterations=3),
        "p8_seed_plain",
        Plain,
    )


ROWS = [
    pytest.param(Row(ModifierCombo.BARE, _bare, Plain, Result), id="BARE"),
    pytest.param(Row(ModifierCombo.OPERATOR, _operator, Plain, Result), id="OPERATOR"),
    pytest.param(Row(ModifierCombo.EACH, _each, Tagged, Result), id="EACH"),
    pytest.param(Row(ModifierCombo.ORACLE, _oracle, Plain, Result), id="ORACLE"),
    pytest.param(Row(ModifierCombo.EACH_OPERATOR, _each_operator, Tagged, Result), id="EACH_OPERATOR"),
    pytest.param(Row(ModifierCombo.ORACLE_OPERATOR, _oracle_operator, Plain, Result), id="ORACLE_OPERATOR"),
    pytest.param(
        Row(ModifierCombo.LOOP, _loop_matching_boundary, Plain, Plain),
        # The id itself carries the boundary condition, so a green run of this
        # row can never be quoted as differing-boundary coverage (rh5fb).
        id="LOOP-matching-boundary-input-eq-output",
    ),
]


# ── helpers ──────────────────────────────────────────────────────────────────


def _round_trip(parent: Construct) -> Construct:
    """Object-level only -- no ``Flow.from_dict`` hop (see module docstring)."""
    flow = to_agent_spec(parent)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return from_agent_spec(flow)


def _item(construct: Construct) -> Any:
    return {n.name: n for n in construct.nodes}["sub"]


def _shape(annotation: Any) -> Any:
    """A recursive, class-identity-free signature of a type annotation.

    A round-tripped model is a freshly-synthesized ``AgentSpecType_<hash>``
    class, so ``is``/``==`` on the class cannot express "the boundary equals
    what was declared". Field names and (recursively) field annotations can.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {name: _shape(f.annotation) for name, f in annotation.model_fields.items()}
    origin = get_origin(annotation)
    if origin is not None:
        return (origin, tuple(_shape(a) for a in get_args(annotation)))
    return annotation


def _compile_kwargs() -> dict[str, Any]:
    # A checkpointer is mandatory for the Operator rows (the pause composite) and
    # harmless for the rest -- one uniform spelling keeps the rows comparable.
    return build_test_compile_kwargs(checkpointer=MemorySaver())


# ── the four assertions, one class each, in (a)(b) -> (c) -> (d) order ───────


class TestConstructItemRoundTripFixturesAreSane:
    """CONTROL, green today. Every row's ORIGINAL parent compiles.

    Without this, a red assertion (d) below could just as easily mean "the
    fixture was never valid" as "the round trip broke it". This is what makes
    (d)'s failure attributable to the round trip.

    Note the LOOP row here is the MATCHING-boundary shape only -- see the module
    docstring; the differing-boundary shape is neograph-rh5fb and out of scope.
    """

    @pytest.mark.parametrize("row", ROWS)
    def test_original_parent_compiles_before_any_round_trip(self, row: Row):
        _register()
        compile(row.build(), **_compile_kwargs())


class TestConstructItemRoundTripPreservesKindAndCombo:
    """(a) + (b): the reimported item is a Construct, carrying EXACTLY its combo.

    A silent downgrade -- EACH_OPERATOR arriving back as plain EACH, or a
    Construct arriving back as a Node -- satisfies every structure-shape
    assertion while dropping a modifier, so (b) is spelled as an identity check
    against the combo the row was WRITTEN with.

    RED TODAY for EACH / ORACLE / EACH_OPERATOR / ORACLE_OPERATOR / LOOP: import
    raises before the item exists, because the FlowNode-to-Construct recursion
    is inlined in the BARE arm instead of being a seam the other reconstructors
    can reach. GREEN for BARE / OPERATOR, which take that inlined arm.

    LOOP row = matching boundary only (module docstring; rh5fb).
    """

    @pytest.mark.parametrize("row", ROWS)
    def test_reimported_item_is_a_construct_with_the_declared_combo(self, row: Row):
        _register()
        item = _item(_round_trip(row.build()))

        assert isinstance(item, Construct), f"(a) the sub-Construct came back as {type(item).__name__}, not a Construct"
        assert classify_modifiers(item)[0] is row.combo, (
            f"(b) expected {row.combo}, got {classify_modifiers(item)[0]} -- "
            "a silent downgrade keeps the shape and drops the modifier"
        )


class TestConstructItemRoundTripPreservesBoundary:
    """(c): the reimported sub-Construct still declares its input/output ports.

    RED FOR EVERY ROW, INCLUDING BARE AND OPERATOR. This is the live defect the
    ticket's HIGH-1 finding named: the de-facto seam recurses into the sub-flow
    and renames the result but never restores ``input=`` / ``output=``, so a
    sub-Construct declared ``input=Plain, output=Result`` comes back with both
    set to ``None``. The exported FlowNode still CARRIES the boundary (its
    ``.inputs`` / ``.outputs`` Property lists, and the sub-flow's own
    Start/End Properties) -- the data is present and simply unused.

    Compared structurally, not by class identity -- see ``_shape``. That is not
    a softening: on the live defect the boundary is ``None``, which no
    comparison spelling could mistake for a match.

    LOOP row = matching boundary only (module docstring; rh5fb).
    """

    @pytest.mark.parametrize("row", ROWS)
    def test_reimported_item_keeps_its_declared_input_port(self, row: Row):
        _register()
        item = _item(_round_trip(row.build()))

        assert _shape(item.input) == _shape(row.declared_input), (
            f"(c) input port: expected the shape of {row.declared_input.__name__}, got {item.input!r}"
        )

    @pytest.mark.parametrize("row", ROWS)
    def test_reimported_item_keeps_its_declared_output_port(self, row: Row):
        _register()
        item = _item(_round_trip(row.build()))

        assert _shape(item.output) == _shape(row.declared_output), (
            f"(c) output port: expected the shape of {row.declared_output.__name__}, got {item.output!r}"
        )


class TestReimportedParentCompiles:
    """(d): THE OUTERMOST ACCEPTANCE -- the reimported parent still compiles.

    This is the only assertion that can catch a seam which reconstructs an
    item's SHAPE but not its PORT. (a) and (b) both pass today on a BARE
    sub-Construct that has been gutted of its ports; (d) does not::

        CompileError: [Construct 'sub'] sub-construct has no output type
          hint: declare output=SomeModel on the sub-construct

    An item that keeps its combo but stops being a usable sub-construct is not
    what this ticket is for, so this assertion -- not the combo check -- is the
    one that grades the Core Invariant.

    RED FOR EVERY ROW. The controls above prove each original parent compiles,
    so every failure here is attributable to the round trip.

    LOOP row = matching boundary only (module docstring; rh5fb).
    """

    @pytest.mark.parametrize("row", ROWS)
    def test_round_tripped_parent_still_compiles(self, row: Row):
        _register()
        reimported = _round_trip(row.build())

        compile(reimported, **_compile_kwargs())
