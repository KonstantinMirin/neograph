"""Shared AgentSpecLoader harness for the differential-execution grid (neograph-dgbqv.2).

The EXECUTE/COMPARE tiers of the Oracle differential-execution grid
(``tests/test_agent_spec_execute.py``) run the exported Flow through a
genuinely third-party, independently-authored runtime --
``pyagentspec.adapters.langgraph.AgentSpecLoader`` -- rather than a walker
neograph's own team wrote. This module holds the mechanics both tiers share:
building a ``tool_registry`` for a Flow, invoking the loader, and running
neograph's own side of a COMPARE cell.

**Independence discipline.** This module imports NOTHING from neograph's own
Agent Spec IMPORT path (``neograph.loader``, ``from_agent_spec``,
``neograph._spec_loader``, ``load_spec``) and constructs no LangGraph
``Command`` -- the Core Invariant ``tests/test_guards_agent_spec_execute_independence.py``
pins by scanning THIS MODULE'S OWN import statements (not the transitive
closure -- see that guard's module docstring for why a runtime
``sys.modules`` check would not work here).
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from pyagentspec.adapters.langgraph import AgentSpecLoader
from pydantic import BaseModel

from neograph import compile, run
from tests.agent_spec_flow_walk import all_flows


def server_tools(flow: Any) -> dict[str, Any]:
    """Every ``ServerTool`` reachable from ``flow``, keyed by the name the loader's
    ``tool_registry`` must supply -- the TOOL name, which is NOT always the node
    name (an Oracle's two variant nodes ``gen__variant_0``/``gen__variant_1`` both
    carry the single tool ``gen``, while the merge node ``gen`` carries
    ``m_combine``).

    ``all_flows`` is imported directly from ``tests.agent_spec_flow_walk``
    (neograph-dgbqv.9). It used to be an INJECTED parameter because the only
    implementation lived in ``tests.test_agent_spec_reachability``, which drags
    ``neograph.loader.from_agent_spec`` in transitively via
    ``tests.test_agent_spec_matrix``. That injection was independence HYGIENE,
    not guard evasion -- ``test_guards_agent_spec_execute_independence`` scans
    this module's OWN import statements and would not have fired on a transitive
    import anyway, and this module already imports ``neograph`` for compile/run.
    The walk module imports nothing from ``neograph``, so the parameter is gone.
    """
    tools: dict[str, Any] = {}
    for sub in all_flows(flow):
        for node in sub.nodes:
            tool = getattr(node, "tool", None)
            if tool is not None:
                tools[tool.name] = tool
    return tools


def stub_value(prop: Any) -> Any:
    """A shape-correct placeholder for one declared output property."""
    kind = getattr(prop, "type", None)
    if kind == "array":
        return [stub_value(prop.item_type)]
    if kind == "object":
        return {title: stub_value(p) for title, p in (prop.properties or {}).items()}
    return "x"


def stub_registry(flow: Any) -> dict[str, Any]:
    """A ``tool_registry`` whose every callable returns its tool's own declared
    output shape. Enough for the EXECUTE tier, which asks only whether the exported
    program RUNS -- COMPARE supplies real bodies instead."""

    def _stub(tool: Any) -> Any:
        value = {p.title: stub_value(p) for p in (tool.outputs or [])}
        return lambda **_kwargs: value

    return {name: _stub(tool) for name, tool in server_tools(flow).items()}


def run_via_agent_spec_loader(flow: Any, cell_id: str, registry: dict[str, Any]) -> Any:
    """Load and run one exported Flow through the third-party runtime.

    A ``MemorySaver`` + ``thread_id`` is supplied unconditionally: an Operator cell
    exports an ``InputMessageNode`` and needs one, and it is inert for cells that
    do not pause.
    """
    graph = AgentSpecLoader(tool_registry=registry, checkpointer=MemorySaver()).load_dict(flow.to_dict())
    return graph.invoke({}, config={"configurable": {"thread_id": cell_id}})


def compare_registry(flow: Any, bodies: dict[str, BaseModel]) -> dict[str, Any]:
    """The loader-side registry for a COMPARE cell: the shared bodies where one is
    declared, and a pass-through elsewhere.

    The pass-through is what an Oracle's ``m_combine`` merge becomes -- ``build_cell``
    registers it as ``lambda variants, config: variants[0]``, i.e. surface what the
    fan-in delivered -- so the two sides are given the same merge semantics and any
    divergence is the runtimes', not the fixtures'.
    """

    def _const(model: BaseModel) -> Any:
        dumped = model.model_dump()
        return lambda **_kwargs: dumped

    registry: dict[str, Any] = {}
    for name in server_tools(flow):
        registry[name] = _const(bodies[name]) if name in bodies else (lambda **kwargs: dict(kwargs))
    return registry


def run_via_neograph(construct: Any, bodies: dict[str, BaseModel]) -> dict[str, Any]:
    """neograph's own side of the comparison, fed the SAME per-cell bodies."""

    def _body(model: BaseModel) -> Any:
        return lambda _input_data, _config: model

    graph = compile(construct, scripted={name: _body(model) for name, model in bodies.items()})
    return run(graph, input={})


def classify_load_failure(exc: Exception) -> str | None:
    """Map an EXECUTE-tier failure to its tracked root-cause ticket, or ``None``
    if the failure does not match a known, already-filed export gap.

    ``None`` means: STOP -- this is a new finding, not a re-report of
    neograph-qtfof.6/.7/.8. Do not silently add it to EXEC_EXEMPT.
    """
    msg = str(exc)
    if "api_provider" in msg:
        return "neograph-qtfof.8: LlmConfig with api_provider=None is not supported by the loader"
    if "iterated_item" in msg:
        return "neograph-qtfof.7: MapNode's iterated_item has no DataFlowEdge"
    if "branching_mapping_key" in msg:
        return "neograph-qtfof.6: BranchingNode's branching_mapping_key has no DataFlowEdge"
    return None
