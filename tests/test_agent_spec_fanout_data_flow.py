"""An ``Each``'s fan-out collection must arrive over a REAL data path (neograph-qtfof.7).

Core Invariant (from the ticket): the exported Flow must carry every fact a
METADATA-BLIND Agent Spec runtime needs to execute it, as real topology
(``DataFlowEdge`` + Property SHAPE). ``neograph/*`` metadata is the round-trip
channel only, never the sole carrier of a runtime-load-bearing value.

Today an ``Each``-modified node lowers to a ``MapNode`` whose ``iterated_item:v``
input is fed by NO ``DataFlowEdge`` -- the collection source lives only in
``metadata['neograph/each_spec']['over']``. A third-party runtime that never reads
neograph metadata therefore cannot determine what to iterate over, and
``AgentSpecLoader`` raises "Expected node ``target`` to have a value for property
``iterated_item:v``, but none was found."

**Why this is asserted by EXECUTION and by ARGS, not by "an edge exists".** The
receiver Property shape decides what the edge MEANS. pyagentspec's loader treats a
MapNode input as ITERATED only when ``json_schemas_have_same_type(source,
List[inner_input])`` holds; any other incoming edge is BROADCAST -- the whole list
bound where one element belongs. So an edge pointed at today's FLATTENED
``iterated_item:v`` receiver validates (``property_is_castable_to`` is lenient),
loads, and runs -- while silently handing the inner tool the entire collection
once. That is strictly WORSE than the current honest gap, and a test asserting
only "a DataFlowEdge exists", or only "the load does not raise", or only an
invocation COUNT, would certify it:

  * count alone is defeated by a 3-superstep broadcast that happens to total 3
    calls, so every invocation's recorded ARGS are checked -- exactly one distinct
    element per call, and the three together covering the collection;
  * "does not raise" alone is the stand-in this whole conformance epic exists to
    retire.

**Why the ``dict`` and ``context`` cells are exercised, not just the single one.**
Wiring the fan-out edge adds the FIRST explicit ``DataFlowEdge`` to the
``scripted-each-single`` outer Flow, and the loader auto-wires same-titled
output->input pairs ONLY while ``data_flow_connections is None``. That flip is the
same trap neograph-qtfof.6 hit one level down. ``scripted-each-dict`` and
``scripted-each-context`` carry an ADDITIONAL non-fan-out input whose value must
still reach every iteration (broadcast is the intended neograph semantics for a
context input), so they are the cells where a broken flip is observable.

SCOPE: scripted Each only, deliberately. Per the ticket's own carve-outs, an
LLM-mode (translation-eligible) Each declares flat ``item_v`` scalars with no
object to bind and a single-type-inputs Each has no receiver param NAME to title
the object Property with -- both keep shipping NO fan-out edge and keep the
``modifier_metadata_only_fanout`` conformance finding firing, which is the honest
outcome rather than a broadcast-shaped lie.

NOT CLAIMED: that the exported Flow surfaces a RESULT. The outermost Construct's
EndNode declares no outputs (neograph-qtfof.9, tracked separately), so these tests
assert what the inner tool was INVOKED with, never what ``invoke()`` returns.

Gated on ``pyagentspec`` (the ``[agent-spec]`` extra keeps ``src/neograph``
dependency-light)::

    uv run pytest tests/test_agent_spec_fanout_data_flow.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import to_agent_spec  # noqa: E402
from tests.agent_spec_flow_walk import (  # noqa: E402
    declared_input_titles,
    fed_inputs,
    wired_edges,
)
from tests.agent_spec_loader_harness import (  # noqa: E402
    run_via_agent_spec_loader,
    server_tools,
)
from tests.test_agent_spec_matrix import CELLS, build_cell  # noqa: E402

#: The scripted Each cells this ticket must close. Read from the matrix's own cell
#: table rather than rebuilt here, so these tests grade the SAME shapes
#: ``test_agent_spec_execute.py``'s EXEC_EXEMPT tracks -- a locally hand-rolled
#: Each construct could drift into a shape the matrix never exports.
_EACH_CELLS: tuple[str, ...] = ("scripted-each-single", "scripted-each-dict", "scripted-each-context")

#: Cells carrying an ADDITIONAL non-fan-out input alongside the receiver.
_EACH_CELLS_WITH_CONTEXT: tuple[str, ...] = ("scripted-each-dict", "scripted-each-context")

#: The name of the node producing the collection, and of the Each-modified node
#: (``build_cell``'s scripted Each shape: ``prod -> target``, ``map_over="prod.groups"``).
_PRODUCER = "prod"
_INNER = "target"

#: Three DISTINCT elements. Three, not one, so real iteration is distinguishable
#: from a broadcast; distinct, so a per-call assertion can name WHICH element
#: arrived. The values are deliberately unlike the context values below so a leaf
#: scan of the recorded kwargs can tell the two apart.
_ELEMENTS: tuple[str, ...] = ("elem-a", "elem-b", "elem-c")

#: The non-fan-out producers the dict/context cells add, and the payload each
#: returns. ``pb`` outputs ``Beta(b=...)``; ``source`` outputs ``Ctx(c=...)``.
_CONTEXT_PAYLOADS: dict[str, dict[str, str]] = {
    "pb": {"b": "ctx-pb"},
    "source": {"c": "ctx-source"},
}


def _leaf_strings(value: Any) -> Iterator[str]:
    """Every string leaf of a nested tool-call payload.

    The recorded kwargs' SHAPE is what this ticket changes (the flat
    ``item:v='elem-a'`` becomes an object-typed ``item``), so the assertions must
    read through the shape rather than pin it -- pinning the kwarg spelling would
    pin the lowering's private choice, not the behavior a metadata-blind runtime
    depends on.

    An object-typed receiver arrives as an INSTANCE of the model pyagentspec
    generates from the declared ObjectProperty, not as a plain dict, so the walk
    unwraps ``model_dump()``/``vars()`` too.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for inner in value.values():
            yield from _leaf_strings(inner)
    elif isinstance(value, (list, tuple, set)):
        for inner in value:
            yield from _leaf_strings(inner)
    elif callable(getattr(value, "model_dump", None)):
        yield from _leaf_strings(value.model_dump())
    elif hasattr(value, "__dict__"):
        yield from _leaf_strings(vars(value))


def _elements_in(call: dict[str, Any]) -> set[str]:
    """Which collection elements one recorded invocation was handed."""
    return {leaf for leaf in _leaf_strings(call) if leaf in _ELEMENTS}


def _recording_registry(flow: Any, calls: list[dict[str, Any]]) -> dict[str, Any]:
    """A ``tool_registry`` that feeds a 3-element collection in and records every
    invocation of the Each body.

    Every tool the Flow declares must be recognised: an unrecognised name means the
    cell's shape moved out from under these assertions, and a pass-through default
    (``agent_spec_loader_harness.compare_registry``'s behavior, correct for its own
    purpose) would let that drift go silent here.
    """
    registry: dict[str, Any] = {}
    for name in server_tools(flow):
        if name == _PRODUCER:
            registry[name] = lambda **_kwargs: {"groups": [{"v": v} for v in _ELEMENTS]}
        elif name == _INNER:

            def _record(**kwargs: Any) -> dict[str, str]:
                calls.append(kwargs)
                return {"ok": "done"}

            registry[name] = _record
        elif name in _CONTEXT_PAYLOADS:
            registry[name] = (lambda payload: lambda **_kwargs: dict(payload))(_CONTEXT_PAYLOADS[name])
        else:
            raise AssertionError(
                f"unrecognised exported tool {name!r} (known: {_PRODUCER}, {_INNER}, "
                f"{sorted(_CONTEXT_PAYLOADS)}) -- the cell's shape changed and these "
                "assertions no longer grade what they claim to"
            )
    return registry


def _run_and_record(cell_id: str) -> tuple[list[dict[str, Any]], Exception | None]:
    """Export one Each cell, run it through the third-party runtime, and return
    every recorded body invocation plus any failure.

    The failure is RETURNED rather than raised so each test can lead with its own
    behavioral assertion and fold the loader error into that message -- the RED
    signal should read "the body was never invoked, because <loader error>", not a
    bare traceback that says nothing about iteration.
    """
    flow = to_agent_spec(build_cell(*CELLS[cell_id]))
    calls: list[dict[str, Any]] = []
    try:
        run_via_agent_spec_loader(flow, cell_id, _recording_registry(flow, calls))
    except Exception as exc:  # noqa: BLE001 -- the failure IS part of the finding
        return calls, exc
    return calls, None


def _map_node(flow: Any) -> Any:
    nodes = [n for n in flow.nodes if type(n).__name__ == "MapNode"]
    assert len(nodes) == 1, f"expected exactly one MapNode, got {[n.name for n in nodes]}"
    return nodes[0]


class TestEachFanOutDrivesRealIteration:
    """A metadata-blind runtime must iterate the exported ``MapNode`` over the real
    collection -- once per element, one element per invocation."""

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_the_each_body_is_invoked_once_per_collection_element(self, cell_id: str) -> None:
        calls, error = _run_and_record(cell_id)

        assert len(calls) == len(_ELEMENTS), (
            f"{cell_id}: a third-party Agent Spec runtime invoked the Each body {len(calls)} time(s) "
            f"for a {len(_ELEMENTS)}-element collection. The MapNode's iterated input is fed by no "
            f"DataFlowEdge -- the collection source lives only in metadata['neograph/each_spec']"
            f"['over'], which a metadata-blind runtime never reads.\n"
            f"  recorded invocations: {calls!r}\n"
            f"  loader outcome      : {error!r}"
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_each_invocation_receives_exactly_one_distinct_element(self, cell_id: str) -> None:
        """The anti-BROADCAST assertion -- the half a call COUNT cannot make.

        A wrongly-shaped edge (source ``list[Elem]`` landing on the flattened
        ``iterated_item:v`` receiver) passes pyagentspec's lenient edge validator
        and is then BROADCAST: the whole list is bound where one element belongs.
        Three supersteps that each receive all three elements would satisfy a
        count-only test while computing something entirely different.
        """
        calls, error = _run_and_record(cell_id)

        assert calls, (
            f"{cell_id}: the Each body was never invoked, so there are no args to grade -- "
            f"close the invocation-count assertion first. loader outcome: {error!r}"
        )

        overfed = [call for call in calls if len(_elements_in(call)) != 1]
        assert not overfed, (
            f"{cell_id}: {len(overfed)} of {len(calls)} invocation(s) did NOT receive exactly one "
            f"collection element: {overfed!r}. An invocation holding the whole collection is a "
            "BROADCAST, not an iteration -- the edge exists but a metadata-blind runtime reads it "
            "as binding the entire list to the fan-out receiver."
        )

        delivered = {element for call in calls for element in _elements_in(call)}
        assert delivered == set(_ELEMENTS), (
            f"{cell_id}: the invocations collectively covered {sorted(delivered)}, not the whole "
            f"collection {sorted(_ELEMENTS)} -- some element was never iterated. "
            f"recorded invocations: {calls!r}"
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS_WITH_CONTEXT)
    def test_non_fan_out_inputs_still_reach_every_iteration(self, cell_id: str) -> None:
        """The outer Flow's auto-wiring must survive the new explicit edge.

        pyagentspec auto-wires same-titled output->input pairs only while a Flow's
        ``data_flow_connections`` is None, so adding the fan-out edge can flip a
        Flow from implicit to explicit wiring and strand every input that was
        riding on the implicit path. A non-fan-out input is meant to be BROADCAST
        to every iteration (that IS neograph's Each context semantics), so its
        value must appear in all three invocations, not one.
        """
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        expected_context = {
            value
            for name, payload in _CONTEXT_PAYLOADS.items()
            if name in server_tools(flow)
            for value in payload.values()
        }
        assert expected_context, (
            f"{cell_id}: no non-fan-out producer found among {sorted(server_tools(flow))} -- this "
            "test would be vacuous; the cell no longer carries a context input"
        )

        calls, error = _run_and_record(cell_id)
        assert calls, (
            f"{cell_id}: the Each body was never invoked, so broadcast cannot be graded -- "
            f"close the invocation-count assertion first. loader outcome: {error!r}"
        )

        starved = [call for call in calls if not expected_context <= set(_leaf_strings(call))]
        assert not starved, (
            f"{cell_id}: {len(starved)} of {len(calls)} invocation(s) did not receive the "
            f"non-fan-out input(s) {sorted(expected_context)}: {starved!r}. Declaring the first "
            "explicit DataFlowEdge on this Flow disables the loader's title-matching auto-wiring "
            "for the WHOLE Flow -- every remaining input must then be declared explicitly."
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_the_exported_run_completes_without_raising(self, cell_id: str) -> None:
        """Loud confirmation that the iteration assertions were graded on a COMPLETE
        run, not on a prefix of one that later blew up.

        Deliberately NOT the ticket's acceptance criterion on its own -- "does not
        raise" is satisfied by a broadcast, which is why it lands last and the args
        assertions lead.
        """
        _calls, error = _run_and_record(cell_id)

        assert error is None, (
            f"{cell_id}: a third-party Agent Spec runtime cannot run the exported Each Flow -- "
            f"{type(error).__name__}: {error}"
        )


class TestMapNodeDeclaresNoInputThatNothingFeeds:
    """Structural companion: the loader hard-errors on ANY declared MapNode input it
    cannot fill, so an ADDITIVE fix -- leaving the old flattened ``iterated_item:v``
    Property in place beside a new object-typed one -- is not merely untidy, it is
    broken. Asserted against the node's OWN declared titles rather than the literal
    ``iterated_item``, because the title is a CHOICE the lowering makes."""

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_every_declared_map_node_input_has_an_incoming_data_flow_edge(self, cell_id: str) -> None:
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        map_node = _map_node(flow)

        unfed = [t for t in declared_input_titles(map_node) if t not in fed_inputs(flow, map_node.name)]
        assert not unfed, (
            f"{cell_id}: MapNode {map_node.name!r} declares input(s) {unfed} that NO DataFlowEdge "
            "feeds. The collection to iterate lives only in metadata['neograph/each_spec']['over'], "
            "which a metadata-blind runtime never reads. "
            f"Wired edges: {wired_edges(flow)}"
        )
