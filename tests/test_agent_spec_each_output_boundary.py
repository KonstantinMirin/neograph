"""An ``Each``'s fan-out RESULT must leave the exported MapNode over a real data
path (neograph-qtfof.11).

Core Invariant (shared with neograph-qtfof.7/.9): the exported Flow must carry
every fact a METADATA-BLIND Agent Spec runtime needs to execute it, as real
topology (``DataFlowEdge`` + Property SHAPE). ``neograph/*`` metadata is the
round-trip channel only, never the sole carrier of a runtime-load-bearing value.

qtfof.7 closed the MapNode's INPUT side (what gets iterated). This is the OUTPUT
side. ``_lower_each`` built the sub-Flow's terminal boundary as a bare
``EndNode(name=...)`` with no declared outputs and no ``DataFlowEdge`` feeding
it, so:

  * the sub-Flow exposes NO outputs (``Flow._get_inferred_outputs`` reads its
    EndNodes),
  * therefore the MapNode infers NO outputs (``MapNode._get_inferred_outputs``
    reads ``subflow.outputs``, prefixing each with ``collected_`` under its
    default APPEND reducer),
  * therefore nothing downstream can source a ``DataFlowEdge`` from it -- and
    the outermost ``EndNode`` (the downstream consumer every export has) falls
    into neograph-qtfof.9's ``_WIRABLE_SHAPES`` fallback, declaring nothing.

The observable end state: a third-party runtime runs the whole fan-out and
``invoke()`` returns ``{}``. The body executed; the result was dropped on the
floor at the boundary.

**Why the assertion is the RESULT, not "an edge exists".** An edge whose
``source_output`` names a Property the MapNode does not expose is rejected by
pyagentspec's own validation, and an edge that IS accepted can still carry the
wrong shape (qtfof.7's broadcast trap, one level up). Only reading what the
third-party runtime actually returns grades the whole chain -- inner EndNode
declares + is fed, MapNode infers, outer EndNode declares + is fed -- at once.

**The collection shape is pyagentspec's, not neograph's.** Under the default
APPEND reducer a MapNode's output for inner Property ``ok`` is
``collected_ok: array<ok>``. These tests therefore assert one entry per
collection element, NOT neograph's own ``dict[str, X]`` keying -- the exported
artifact is an Agent Spec program and obeys Agent Spec's reduction semantics.

Gated on ``pyagentspec`` (the ``[agent-spec]`` extra keeps ``src/neograph``
dependency-light)::

    uv run pytest tests/test_agent_spec_each_output_boundary.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("pyagentspec")

from neograph._agent_spec import to_agent_spec  # noqa: E402
from tests.agent_spec_flow_walk import all_flows, wired_edges  # noqa: E402
from tests.agent_spec_loader_harness import (  # noqa: E402
    run_via_agent_spec_loader,
    server_tools,
)
from tests.test_agent_spec_matrix import CELLS, build_cell  # noqa: E402

#: The same scripted Each cells neograph-qtfof.7 closed the input side of, read
#: from the matrix's own table so these tests grade the SAME exported shapes the
#: EXECUTE tier tracks.
_EACH_CELLS: tuple[str, ...] = ("scripted-each-single", "scripted-each-dict", "scripted-each-context")

_PRODUCER = "prod"
_INNER = "target"

#: Three distinct elements -- enough that a per-element collection is
#: distinguishable from a single value that happened to survive.
_ELEMENTS: tuple[str, ...] = ("elem-a", "elem-b", "elem-c")

#: The Each body ECHOES the element it was handed (``Out``, build_cell's Each
#: output model, declares one string field ``ok``). A CONSTANT result would let
#: the whole suite pass with the body's INPUT path broken: the sub-Flow's
#: ``data_flow_connections`` flips from ``None`` to a list the moment this fix
#: adds its first explicit edge, and pyagentspec's loader synthesises same-title
#: edges ONLY while that field is ``None`` (all-or-nothing,
#: ``_langgraphconverter.py``) -- so the StartNode -> body edge must be emitted
#: in the same change, and an echo is what makes its absence VISIBLE.


def _echo_body(**kwargs: Any) -> dict[str, str]:
    received = sorted(leaf for leaf in _leaf_strings(kwargs) if leaf in _ELEMENTS)
    return {"ok": received[0] if len(received) == 1 else f"<{len(received)} elements: {received!r}>"}


def _leaf_strings(value: Any) -> Iterator[str]:
    """Every string leaf of a nested tool-call payload.

    The receiver's SHAPE is the lowering's private choice (an object-typed
    ``item`` Property arrives as an instance of a pyagentspec-generated model,
    not a dict), so the walk reads THROUGH the shape rather than pinning it --
    same reason ``test_agent_spec_fanout_data_flow.py`` does.
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


_CONTEXT_PAYLOADS: dict[str, dict[str, str]] = {
    "pb": {"b": "ctx-pb"},
    "source": {"c": "ctx-source"},
}


def _registry(flow: Any) -> dict[str, Any]:
    """A ``tool_registry`` feeding a 3-element collection in, the Each body
    echoing back whichever element it was handed.

    An unrecognised tool raises rather than falling through to a pass-through
    default: it would mean the cell's shape moved and these assertions no longer
    grade what they claim to.
    """
    registry: dict[str, Any] = {}
    for name in server_tools(flow):
        if name == _PRODUCER:
            registry[name] = lambda **_kwargs: {"groups": [{"v": v} for v in _ELEMENTS]}
        elif name == _INNER:
            registry[name] = _echo_body
        elif name in _CONTEXT_PAYLOADS:
            registry[name] = (lambda payload: lambda **_kwargs: dict(payload))(_CONTEXT_PAYLOADS[name])
        else:
            raise AssertionError(
                f"unrecognised exported tool {name!r} (known: {_PRODUCER}, {_INNER}, "
                f"{sorted(_CONTEXT_PAYLOADS)}) -- the cell's shape changed and these "
                "assertions no longer grade what they claim to"
            )
    return registry


def _run(cell_id: str) -> tuple[dict[str, Any], Exception | None]:
    """Export one Each cell and run it through the third-party runtime.

    The failure is RETURNED so each test leads with its own behavioral message
    and folds the loader error into it, rather than surfacing a bare traceback
    that says nothing about the boundary.
    """
    flow = to_agent_spec(build_cell(*CELLS[cell_id]))
    try:
        result = run_via_agent_spec_loader(flow, cell_id, _registry(flow))
    except Exception as exc:  # noqa: BLE001 -- the failure IS part of the finding
        return {}, exc
    return result.get("outputs", {}) if isinstance(result, dict) else {}, None


def _map_node(flow: Any) -> Any:
    nodes = [n for n in flow.nodes if type(n).__name__ == "MapNode"]
    assert len(nodes) == 1, f"expected exactly one MapNode, got {[n.name for n in nodes]}"
    return nodes[0]


def _sub_flow(flow: Any, name: str) -> Any:
    matches = [f for f in all_flows(flow) if f.name == name]
    assert len(matches) == 1, f"expected exactly one sub-flow named {name!r}, got {[f.name for f in all_flows(flow)]}"
    return matches[0]


class TestEachResultLeavesTheMapNode:
    """A metadata-blind runtime must be able to READ what the fan-out produced."""

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_invoke_surfaces_one_result_per_collection_element(self, cell_id: str) -> None:
        outputs, error = _run(cell_id)

        collected = [value for value in outputs.values() if isinstance(value, list)]
        assert len(collected) == 1, (
            f"{cell_id}: a third-party Agent Spec runtime's invoke() surfaced {outputs!r} for a "
            f"{len(_ELEMENTS)}-element fan-out. The Each sub-flow's EndNode declares no outputs, so "
            f"the MapNode infers none and nothing downstream can source a DataFlowEdge from it -- "
            f"the whole fan-out result is dropped at the boundary.\n"
            f"  loader outcome: {error!r}"
        )
        assert len(collected[0]) == len(_ELEMENTS), (
            f"{cell_id}: the collected fan-out result has {len(collected[0])} entries for a "
            f"{len(_ELEMENTS)}-element collection: {collected[0]!r}"
        )
        echoed = sorted(leaf for entry in collected[0] for leaf in _leaf_strings(entry))
        assert echoed == sorted(_ELEMENTS), (
            f"{cell_id}: the collected entries echo {echoed!r}, not one distinct element each. "
            f"The body ran on the wrong input -- adding the first explicit DataFlowEdge to the "
            f"sub-Flow switches the loader's same-title auto-wiring OFF for the WHOLE sub-Flow, "
            f"so the StartNode -> body input edge must be emitted in the same change.\n"
            f"  collected: {collected[0]!r}"
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_the_map_node_declares_the_collected_output(self, cell_id: str) -> None:
        """The structural half: the MapNode's own output Properties.

        Asserted alongside the behavioral test because the TITLE is what a
        downstream ``DataFlowEdge`` must name, and pyagentspec's reduction
        convention (``collected_`` + APPEND) is the SDK's, not neograph's -- a
        silent change to it is a real break in exported artifacts even while the
        end-to-end result still happens to arrive.
        """
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        titles = [p.title for p in (_map_node(flow).outputs or [])]

        assert titles == ["collected_ok"], (
            f"{cell_id}: the exported MapNode declares outputs {titles!r}; it infers them from its "
            f"sub-flow's EndNode, which declares none."
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_a_real_edge_feeds_the_sub_flow_end_node(self, cell_id: str) -> None:
        """Inside the sub-Flow: the inner terminal producer -> EndNode edge.

        Declaring the EndNode's outputs without feeding them is failure mode (b)
        from neograph-qtfof.9 -- the loader raises "Expected node to have a value
        for property X" -- so the declaration and the edge are one fix, and the
        edge is asserted here rather than inferred from the green end-to-end run.
        """
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        sub = _sub_flow(flow, f"{_INNER}__each_body")
        end_name = f"{_INNER}__each_end"

        into_end = [edge for edge in wired_edges(sub) if edge[2] == end_name]
        assert into_end == [(_INNER, "ok", end_name, "ok")], (
            f"{cell_id}: the Each sub-flow's EndNode is fed by {into_end!r} -- the terminal producer's "
            f"output reaches the boundary over no data path."
        )

    @pytest.mark.parametrize("cell_id", _EACH_CELLS)
    def test_the_outermost_end_node_sources_from_the_map_node(self, cell_id: str) -> None:
        """The downstream-consumer half of the acceptance: an outer ``DataFlowEdge``
        whose ``source_node`` is the MapNode and whose ``source_output`` is the
        collected Property, resolving under the real loader."""
        flow = to_agent_spec(build_cell(*CELLS[cell_id]))
        end_name = f"{cell_id}__end"

        into_end = [edge for edge in wired_edges(flow) if edge[2] == end_name]
        assert into_end == [(_INNER, "collected_ok", end_name, "collected_ok")], (
            f"{cell_id}: the outermost EndNode is fed by {into_end!r} -- an Each terminal falls into "
            f"neograph-qtfof.9's unwirable fallback, so invoke() returns nothing."
        )
