"""Node-level lowering: neograph ``Node`` -> Agent Spec primitives.

Extracted from ``_agent_spec.py`` (neograph-3ffdg.3) as a pure file split — the
functions below are unchanged, only their home moved. A ``Construct`` item never
reaches this cluster; construct-level lowering stays in ``_agent_spec.py``.

Depends one-way on ``_agent_spec_markers`` and ``_agent_spec_placeholders``;
nothing here imports back into ``_agent_spec``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from neograph._agent_spec_markers import (
    _MARK_AGENT_SPEC,
    _MARK_MODE,
    _MARK_PROMPT_SPEC,
    _MARK_TOOL_SPEC,
    _import_agent_spec_flow_classes,
)
from neograph._agent_spec_placeholders import (
    _prompt_spec_marker,
    _properties_for,
    _translate_placeholders,
)
from neograph.errors import ConfigurationError
from neograph.node import Node
from neograph.tool import Tool

if TYPE_CHECKING:
    from pyagentspec.flows.node import Node as SpecNode
    from pyagentspec.property import Property


def _reject_unrepresentable_fields(node: Node) -> None:
    """Fail loud on any Node field that has no Agent Spec representation.

    Per the Core Invariant, ``to_agent_spec()`` must never silently drop a
    construct it cannot lower. Checked before any lowering attempt.
    """
    if node.raw_fn is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses raw_fn — a Python callable with no Agent Spec representation",
            expected="scripted/think/agent/act mode with a name-serializable body",
            found="raw_fn set",
            hint="raw_fn nodes cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.skip_when is not None or node.skip_value is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses skip_when/skip_value — Python callables with no Agent Spec representation",
            expected="a node without conditional-skip logic",
            found="skip_when and/or skip_value set",
            hint="skip_when/skip_value cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.renderer is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} uses a custom renderer — no Agent Spec representation",
            expected="the default rendering pipeline",
            found="renderer set",
            hint="a custom renderer cannot be exported to Agent Spec (callable-valued field, doc s6)",
        )
    if node.handoff_param is not None or node.handoff_channel is not None:
        raise ConfigurationError.build(
            f"node {node.name!r} is a Portal mesh member (handoff_param/handoff_channel set) — "
            "Agent Spec has no combinator for runtime peer-to-peer Command(goto) routing",
            expected="a node outside a Portal mesh",
            found="handoff_param and/or handoff_channel set",
            hint="Portal mesh members cannot be exported to Agent Spec — symmetric to the "
            "ratified Swarm import-reject (agent-spec-ratification-2026-07-13.md)",
        )
    if callable(node.gate_tools_when):
        raise ConfigurationError.build(
            f"node {node.name!r} uses a callable gate_tools_when — no Agent Spec representation",
            expected="a registered condition NAME (str) or no gate_tools_when",
            found="gate_tools_when is a callable",
            hint="only the string (registered-condition-name) form of gate_tools_when serializes",
        )


def _lower_generation_step(
    node: Node,
    *,
    name: str,
    outputs: list[Property],
    metadata: dict[str, Any],
    model_tier: str | None = None,
    tool_description: str | None = None,
) -> SpecNode:
    """The SINGLE per-node.mode generation dispatch (think / agent-act / scripted-raw).

    Shared by BOTH callers (neograph-2s2o6, retiring the two hand-written copies that
    were the 'one validator, not two' anti-pattern CLAUDE.md bans elsewhere):
    ``_lower_node`` passes ``name=node.name``/``metadata={}``; ``_lower_oracle``'s
    variant loop passes ``name=f'{node.name}__variant_{i}'``, the oracle group/variant
    markers as ``metadata``, and the per-variant ``model_tier`` (Oracle.models). A new
    mode branch or a fix now lands in ONE place for every caller -- if agent/act-mode
    Oracle variants get a lowering change, they get it for free, never a third copy.
    """
    nodes_mod, _flow_mod, _edges_mod, _property_mod, tools_mod = _import_agent_spec_flow_classes()

    inputs = _properties_for(node.inputs)

    if node.mode == "think":
        # Option F neograph-cbpyx: translate ${path} -> {{ flat }}; the LlmNode
        # declares ONLY the referenced flat Properties, and the neograph/prompt_spec
        # marker carries the untranslated text + full original inputs for round trip.
        rewritten, ref_props, flat_to_original = _translate_placeholders(node.prompt or "", inputs, name)
        return nodes_mod.LlmNode(
            name=name,
            inputs=ref_props or None,
            outputs=outputs or None,
            llm_config=_make_llm_config(Node(name=node.name, model=model_tier or node.model)),
            prompt_template=rewritten,
            metadata={**metadata, _MARK_PROMPT_SPEC: _prompt_spec_marker(node, flat_to_original)},
        )

    if node.mode in ("agent", "act"):
        # Lossless lowering (neograph-i3zsh.1): a real pyagentspec
        # AgentNode+Agent+ServerTool composite, never the ToolNode placeholder
        # that used to silently drop prompt/model/tools. The `neograph/agent_spec`
        # marker carries everything the from_agent_spec() importer needs to
        # reconstruct the node exactly -- the export->import round trip is now
        # implemented (neograph-aa5gq, loader._reconstruct_agent_node). Option F:
        # the Agent's system_prompt is placeholder-translated and the AgentNode
        # declares the referenced flat Properties; the original ${var} text rides
        # the existing neograph/agent_spec marker (marker["prompt"]). Per-variant
        # Oracle.models tier + a unique component name ride the model_copy
        # (agent_source); the neograph/agent_spec + prompt_spec markers keep the
        # ORIGINAL node (model=node.model), so _reconstruct_oracle_group recovers
        # base_node | Oracle(models=...).
        rewritten, ref_props, flat_to_original = _translate_placeholders(node.prompt or "", inputs, name)
        agent_source = node.model_copy(update={"name": name, "model": model_tier or node.model})
        agent = _make_agent(agent_source, tools_mod, ref_props, outputs, rewritten)
        return nodes_mod.AgentNode(
            name=name,
            inputs=ref_props or None,
            outputs=outputs or None,
            agent=agent,
            metadata={
                **metadata,
                _MARK_MODE: node.mode,
                _MARK_AGENT_SPEC: _agent_spec_marker(node),
                # The neograph/prompt_spec marker carries the FULL original inputs so
                # a translated agent/act node (whose declared inputs are the flat
                # placeholder names) reconstructs its true input TypeSpec -- e.g. an
                # Each fan-out receiver must round-trip to the SAME element type as
                # the producer's list element neograph-3lk2l. marker["prompt"] on
                # _MARK_AGENT_SPEC already carries the untranslated ${var} text.
                _MARK_PROMPT_SPEC: _prompt_spec_marker(node, flat_to_original),
            },
        )

    # scripted / raw already rejected raw_fn upstream; scripted_fn is name-only.
    return nodes_mod.ToolNode(
        name=name,
        inputs=inputs or None,
        outputs=outputs or None,
        tool=_make_server_tool(node, tools_mod, inputs, outputs, description=tool_description),
        metadata=metadata,
    )


def _lower_node(node: Node) -> SpecNode:
    """Dispatch a single neograph Node to its Agent Spec primitive by mode.

    Thin wrapper over the shared ``_lower_generation_step`` neograph-2s2o6: the
    per-mode dispatch lives in ONE place. ``_lower_node`` adds only the top-level
    ``_reject_unrepresentable_fields`` guard that the Oracle-variant path deliberately
    omits, and passes the node's own name + empty base metadata.
    """
    _reject_unrepresentable_fields(node)
    return _lower_generation_step(node, name=node.name, outputs=_properties_for(node.outputs), metadata={})


def _make_agent(node: Node, tools_mod: Any, inputs: list[Property], outputs: list[Property], system_prompt: str) -> Any:
    """Build the pyagentspec ``Agent`` for an agent/act node. ``inputs`` are the
    Option-F-translated referenced flat Properties and ``system_prompt`` is the
    ``{{ flat }}``-rewritten text (both computed once in ``_lower_node``); the
    original ``${var}`` text rides ``neograph/agent_spec`` marker["prompt"]."""
    from pyagentspec.agent import Agent

    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) normalizes any raw BaseTool to Tool at construction time -- same
    # cast precedent as _agent_cycle.py:235.
    tools = cast("list[Tool]", node.tools)
    return Agent(
        name=f"{node.name}-agent",
        llm_config=_make_llm_config(node),
        system_prompt=system_prompt,
        tools=[_tool_to_server_tool(tool, tools_mod) for tool in tools],
        inputs=inputs or None,
        outputs=outputs or None,
        human_in_the_loop=False,
    )


def _tool_to_server_tool(tool: Any, tools_mod: Any) -> Any:
    """Lower one neograph ``Tool`` to a ``ServerTool``, name-only.

    Mirrors ``_make_server_tool``'s ``ServerTool`` shape but is a standalone
    helper: this is an agent/act node's ``tools=[...]`` list attaching to its
    ``Agent``, not a scripted/think node's own ``ToolNode.tool=`` field (a
    different Agent Spec primitive entirely). ``ServerTool`` is used
    UNIFORMLY for every neograph ``Tool`` export (MCP-bound or not) --
    MCP-ness is a runtime factory-registration detail
    (``mcp_tool_factory``), never a wire-format distinction (pyagentspec has
    no ``MCPTool`` class; doc s7/s8's 'do not own the MCP gateway'
    positioning).

    Stamps a ``neograph/tool_spec`` marker on the ServerTool itself
    (budget/config/idempotent -- name-only fields with no plain-``ServerTool``
    equivalent), mirroring ``ToolSpec``'s shape (``_spec_schema.py:66``) and
    the established ``metadata['neograph/*_spec']`` marker convention. Per
    the Core Invariant, ``tool._bound_tool`` (a live callable) is NEVER
    referenced here -- factory binding is exclusively a runtime,
    post-deserialization concern.
    """
    return tools_mod.ServerTool(
        name=tool.name,
        description=f"neograph tool {tool.name!r}",
        inputs=None,
        outputs=None,
        metadata={
            _MARK_TOOL_SPEC: {
                "name": tool.name,
                "budget": tool.budget,
                "config": tool.config,
                "idempotent": tool.idempotent,
            }
        },
    )


def _agent_spec_marker(node: Node) -> dict[str, Any]:
    """Build the ``neograph/agent_spec`` reconstruction blob for an agent/act
    node — every field the plain ``Agent``/``ServerTool`` primitives cannot
    represent, so a future ``from_agent_spec()`` importer can rebuild the
    exact node. Callable ``gate_tools_when`` is already rejected by
    ``_reject_unrepresentable_fields`` before this runs; only the string form
    reaches here.
    """
    # node.tools is declared list[Tool | BaseTool], but _normalize_raw_base_tools
    # (node.py) normalizes any raw BaseTool to Tool at construction time -- same
    # cast precedent as _agent_cycle.py:235.
    tools = cast(list[Tool], node.tools)
    return {
        "mode": node.mode,
        "prompt": node.prompt,
        "model": node.model,
        "tools": [
            {"name": tool.name, "budget": tool.budget, "config": tool.config, "idempotent": tool.idempotent}
            for tool in tools
        ],
        "gate_tools_when": node.gate_tools_when,
        "context": node.context,
    }


def _make_llm_config(node: Node) -> Any:
    _nodes_mod, _flow_mod, _edges_mod, _property_mod, _tools_mod = _import_agent_spec_flow_classes()
    from pyagentspec.llms.llmconfig import LlmConfig as SpecLlmConfig

    return SpecLlmConfig(name=f"{node.name}-llm", model_id=node.model or "default")


def _make_server_tool(
    node: Node,
    tools_mod: Any,
    inputs: list[Property],
    outputs: list[Property],
    description: str | None = None,
) -> Any:
    return tools_mod.ServerTool(
        name=node.scripted_fn or node.name,
        description=description if description is not None else f"neograph node {node.name!r} (mode={node.mode})",
        inputs=inputs or None,
        outputs=outputs or None,
    )
