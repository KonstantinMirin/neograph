"""Systematic Agent Spec export + round-trip COVERAGE MATRIX (neograph-sdfgz).

The Agent Spec export/round-trip surface produced a sibling bug on every fix
(ozxqw -> qtfof.1/.2/.3 -> hf505 -> 3lk2l). Every one hid behind the same blind
spot: the existing export/round-trip tests exercise the SINGLE-TYPE-INPUT
workaround shape (``Node.scripted(inputs=SomeType)``), never the PRIMARY
``@node`` shapes real users write -- ``map_over=`` fan-out, dict-form fan-in via
parameter names, and Each nodes that also take non-fan-out context inputs.

This module is the permanent regression guard that the surface stays converged.
It parametrizes:

    {no modifier, Each(map_over=), Oracle(ensemble), Loop, Operator}
      x {single-type input, dict-form fan-in, +non-fan-out context input}
      x {export, round-trip}

using ONLY the primary ``@node`` shapes. Every cell that has a meaningful
combination is built by a ``@node`` builder below; the two operations are
separate parametrized tests so a failure names the exact (cell, operation)
that regressed.

The two root-cause families this matrix pins:
  (A) dict-form title/property mapping in ``to_agent_spec``'s DataFlowEdge
      wiring -- an Each-modified (MapNode) destination needs ``iterated_``-
      prefixed inputs, and its sub-flow StartNode must declare them
      (neograph-hf505).
  (B) type-identity loss on round-trip -- the fan-out receiver of a primary
      ``@node(map_over=)`` node must reconstruct to the SAME structural type as
      the producer's list element, or Each's assembly-time element-type check
      rejects the imported Construct (neograph-3lk2l / qtfof.4 / wqb5t).

Run with::

    uv run --extra dev --extra agent-spec pytest tests/test_agent_spec_matrix.py
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

pytest.importorskip("pyagentspec")

from neograph import Construct, node  # noqa: E402
from neograph._agent_spec import to_agent_spec  # noqa: E402
from neograph.decorators import construct_from_functions  # noqa: E402
from neograph.loader import from_agent_spec  # noqa: E402
from tests.fakes import register_scripted  # noqa: E402

# -- Shared models -----------------------------------------------------------


class Alpha(BaseModel, frozen=True):
    a: str


class Beta(BaseModel, frozen=True):
    b: str


class Ctx(BaseModel, frozen=True):
    c: str


class Elem(BaseModel, frozen=True):
    v: str


class Coll(BaseModel, frozen=True):
    groups: list[Elem]


class Out(BaseModel, frozen=True):
    ok: str


# -- Cell builders (PRIMARY @node shapes only) -------------------------------
#
# Each builder returns a fresh Construct. Any condition / merge referenced by a
# modifier is registered inside the builder because conftest's autouse fixture
# resets every registry between tests.


def build_none_single() -> Construct:
    @node(outputs=Alpha)
    def prod() -> Alpha: ...

    @node(outputs=Out)
    def cons(prod: Alpha) -> Out: ...

    return construct_from_functions("m-none-single", [prod, cons])


def build_none_dict() -> Construct:
    @node(outputs=Alpha)
    def pa() -> Alpha: ...

    @node(outputs=Beta)
    def pb() -> Beta: ...

    @node(outputs=Out)
    def cons(pa: Alpha, pb: Beta) -> Out: ...

    return construct_from_functions("m-none-dict", [pa, pb, cons])


def build_each_single() -> Construct:
    """PRIMARY @node map_over= with ONLY the fan-out receiver (neograph-3lk2l)."""

    @node(outputs=Coll)
    def prod() -> Coll: ...

    @node(outputs=Out, map_over="prod.groups", map_key="v")
    def ver(item: Elem) -> Out: ...

    return construct_from_functions("m-each-single", [prod, ver])


def build_each_context() -> Construct:
    """Each map_over= node with a non-fan-out CONTEXT input (neograph-hf505)."""

    @node(outputs=Ctx)
    def source() -> Ctx: ...

    @node(outputs=Coll)
    def clusters() -> Coll: ...

    @node(outputs=Out, map_over="clusters.groups", map_key="v")
    def verify(source: Ctx, cluster: Elem) -> Out: ...

    return construct_from_functions("m-each-context", [source, clusters, verify])


def build_oracle_single() -> Construct:
    register_scripted("m_combine", lambda variants, config: variants[0])

    @node(outputs=Alpha)
    def prod() -> Alpha: ...

    @node(outputs=Out, ensemble_n=2, merge_fn="m_combine")
    def gen(prod: Alpha) -> Out: ...

    return construct_from_functions("m-oracle-single", [prod, gen])


def build_oracle_dict() -> Construct:
    register_scripted("m_combine", lambda variants, config: variants[0])

    @node(outputs=Alpha)
    def pa() -> Alpha: ...

    @node(outputs=Beta)
    def pb() -> Beta: ...

    @node(outputs=Out, ensemble_n=2, merge_fn="m_combine")
    def gen(pa: Alpha, pb: Beta) -> Out: ...

    return construct_from_functions("m-oracle-dict", [pa, pb, gen])


def build_loop_single() -> Construct:
    # Expression condition (the documented serialization target + the shape the
    # existing passing round-trip tests use); registered-NAME-condition
    # round-trip is an orthogonal concern to the two agent-spec families here.
    @node(outputs=Alpha)
    def prod() -> Alpha: ...

    @node(outputs=Alpha, loop_when='a == "x"', max_iterations=3)
    def refine(prod: Alpha) -> Alpha: ...

    return construct_from_functions("m-loop-single", [prod, refine])


def build_operator_single() -> Construct:
    @node(outputs=Alpha)
    def prod() -> Alpha: ...

    @node(outputs=Alpha, interrupt_when='a == "x"')
    def review(prod: Alpha) -> Alpha: ...

    return construct_from_functions("m-operator-single", [prod, review])


CELLS: dict[str, callable] = {
    "none-single": build_none_single,
    "none-dict": build_none_dict,
    "each-single": build_each_single,
    "each-context": build_each_context,
    "oracle-single": build_oracle_single,
    "oracle-dict": build_oracle_dict,
    "loop-single": build_loop_single,
    "operator-single": build_operator_single,
}

# neograph-m57mn (fixed): an Oracle(ensemble, scripted-mode) node's variants
# used to unconditionally lower to LlmNode regardless of node.mode, so ANY
# external input failed pyagentspec's {{prompt placeholder}} inference. Fixed
# by dispatching variant construction per node.mode (scripted -> ToolNode,
# which has zero placeholder coupling) -- see _agent_spec.py's
# _check_placeholder_inputs and _lower_oracle's per-mode dispatch.
_XFAIL_EXPORT: set[str] = set()
_XFAIL_ROUND_TRIP: set[str] = set()
_ORACLE_INPUT_BLOCKER = "neograph-m57mn: Oracle+external-input export (LlmNode placeholder coupling)"


# -- The matrix --------------------------------------------------------------


class TestAgentSpecExportMatrix:
    """Every primary-@node cell must EXPORT to a valid Agent Spec Flow."""

    @pytest.mark.parametrize("cell_id", list(CELLS))
    def test_export(self, cell_id: str, request: pytest.FixtureRequest) -> None:
        if cell_id in _XFAIL_EXPORT:
            request.node.add_marker(pytest.mark.xfail(reason=_ORACLE_INPUT_BLOCKER, strict=True))
        construct = CELLS[cell_id]()
        # Must not raise (hf505: Each + context input raised a pydantic
        # DataFlowEdge ValidationError here).
        flow = to_agent_spec(construct)
        assert flow is not None
        assert flow.nodes


class TestAgentSpecRoundTripMatrix:
    """Every primary-@node cell must ROUND-TRIP: export then re-import into a
    valid Construct (from_agent_spec runs Construct assembly validation, which
    is where 3lk2l's Each element-type mismatch was raised)."""

    @pytest.mark.parametrize("cell_id", list(CELLS))
    def test_round_trip(self, cell_id: str, request: pytest.FixtureRequest) -> None:
        if cell_id in _XFAIL_ROUND_TRIP:
            request.node.add_marker(pytest.mark.xfail(reason=_ORACLE_INPUT_BLOCKER, strict=True))
        construct = CELLS[cell_id]()
        flow = to_agent_spec(construct)
        imported = from_agent_spec(flow)
        assert isinstance(imported, Construct)
        assert imported.nodes
