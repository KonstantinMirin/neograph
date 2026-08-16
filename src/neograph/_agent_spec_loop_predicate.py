"""Loop branch-predicate synthesis for the Agent Spec export (neograph-qtfof.6).

Extracted from ``_agent_spec_modifier_lowering.py`` (over its file-size ceiling)
as a focused peer module, mirroring the ``_validation_portal.py`` split-by-concern
precedent. Package-private: imported only by ``_agent_spec_modifier_lowering.py``.

A metadata-blind runtime falls silently to ``DEFAULT_BRANCH`` when a
``BranchingNode`` has no incoming ``DataFlowEdge`` (``BranchingNodeExecutor``
does ``inputs.get(title, DEFAULT_BRANCH)``). This module closes that for the
statically-determinable case: a Loop ``when`` that is an EXPRESSION over the
body's own output field. No Agent Spec primitive can COMPUTE the branch key (no
expression/transform node exists in the installed pyagentspec node registry), so
a predicate ``ToolNode`` is synthesized and DECLARED, not executed, in-Flow — a
foreign runtime must resolve the named ``ServerTool``. That is the documented,
non-``PORTABLE`` gain: loud-unresolved-tool beats silent-wrong-branch.

A registered-NAME ``when`` has no field to point at (its ``condition_fn`` reads
the whole state bus) and is left exactly as before — same treatment Operator
gets, which is out of this ticket's scope entirely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neograph._agent_spec_markers import _MARK_MODIFIER
from neograph.conditions import condition_field
from neograph.construct import Construct
from neograph.modifiers import Loop
from neograph.node import Node

if TYPE_CHECKING:
    from pyagentspec.flows.edges import ControlFlowEdge, DataFlowEdge
    from pyagentspec.flows.node import Node as SpecNode


def synthesize_loop_predicate(
    node: Node | Construct,
    loop: Loop,
    body: SpecNode,
    branch: SpecNode,
    flow_classes: Any,
) -> tuple[list[SpecNode], list[ControlFlowEdge], list[DataFlowEdge], SpecNode]:
    """Returns ``(extra_nodes, control_edges, data_edges, check_entry)``.

    ``extra_nodes``/edges are empty and ``check_entry`` is ``body`` unchanged
    when ``loop.when`` is not a statically-determinable expression. ``flow_classes``
    is the caller's already-resolved ``_import_agent_spec_flow_classes()`` tuple
    (nodes_mod, edges_mod, property_mod, tools_mod) -- never re-imported here.
    """
    nodes_mod, edges_mod, property_mod, tools_mod = flow_classes

    predicate_field = condition_field(loop.when) if isinstance(loop.when, str) else None
    body_outputs = {p.title: p for p in (body.outputs or [])}
    source_prop = body_outputs.get(predicate_field) if predicate_field else None
    if source_prop is None:
        return [], [], [], body

    assert branch.inputs, "BranchingNode always infers a branching_mapping_key input Property"
    branch_key_title = branch.inputs[0].title
    branch_key_prop = property_mod.StringProperty(json_schema={"title": branch_key_title, "type": "string"})
    predicate = nodes_mod.ToolNode(
        name=f"{node.name}__loop_predicate",
        inputs=[source_prop],
        outputs=[branch_key_prop],
        tool=tools_mod.ServerTool(
            name=f"{node.name}_loop_predicate",
            description=f"Evaluate Loop condition {loop.when!r} for {node.name!r}",
            inputs=[source_prop],
            outputs=[branch_key_prop],
        ),
        metadata={_MARK_MODIFIER: "loop"},
    )
    control_edges = [
        edges_mod.ControlFlowEdge(name=f"{node.name}__loop_body_to_predicate", from_node=body, to_node=predicate)
    ]
    data_edges = [
        edges_mod.DataFlowEdge(
            name=f"{node.name}__loop_predicate_source",
            source_node=body,
            source_output=source_prop.title,
            destination_node=predicate,
            destination_input=source_prop.title,
        ),
        edges_mod.DataFlowEdge(
            name=f"{node.name}__loop_predicate_result",
            source_node=predicate,
            source_output=branch_key_title,
            destination_node=branch,
            destination_input=branch_key_title,
        ),
    ]
    return [predicate], control_edges, data_edges, predicate
