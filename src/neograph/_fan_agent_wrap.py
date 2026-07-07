"""Fan-over-agent auto-wrap — build the isolating sub-construct.

The compile pre-pass (called from ``compiler.compile`` before the state model is
built) that rewrites each SUPPORTED fan-over-agent Node into an isolated
single-node sub-construct, so the fan runs over isolated subgraph state — the only
correct mechanism (see ``_fan_agent`` + docs/design/fan-over-agent-node-2026-07-07.md
and neograph-m6d3.6).

This lives in its own module (not ``_fan_agent``) because building the wrapper
needs ``Construct``, and ``construct`` -> ``_construct_validation`` -> ``_fan_agent``
would cycle if ``_fan_agent`` imported ``Construct``. Nothing in the ``construct``/
``_construct_validation`` import chain imports this module, so importing ``Construct``
here is cycle-free; ``compiler`` imports it at module top.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, create_model

from neograph._fan_agent import is_supported_fan_over_agent
from neograph._ir_branch import iter_with_arms
from neograph._normalize import normalize_inputs
from neograph._state_keys import StateKeys
from neograph.construct import Construct
from neograph.errors import CompileError
from neograph.modifiers import ModifierSet
from neograph.naming import field_name_for
from neograph.node import Node


def wrap_fan_over_agents(construct: Construct) -> Construct:
    """Rewrite each supported fan-over-agent Node into an isolated single-node
    sub-construct. ``_fan_agent.is_supported_fan_over_agent`` is the single source
    of truth for which shapes qualify (Oracle over a self-contained agent/act
    node); unsupported shapes already failed loud at assembly. Returns the
    construct unchanged when there is nothing to wrap.

    Only TOP-LEVEL ``construct.nodes`` are rewritten. A supported fan-over-agent
    that appears inside a sub-construct is handled by that sub-construct's own
    recursive compile; one inside a branch ARM cannot be wrapped here (arms are
    added verbatim by ``_add_arm_nodes``), so it is rejected fail-loud below
    rather than silently dropping its Oracle modifier.
    """
    # Identity-preserving on no-op: only model_copy when a node is actually
    # rewritten, so a construct with nothing to wrap is returned unchanged (keeps
    # `compiled.construct is pipeline` for post-compile introspection). The raw
    # top-level `.nodes` walk is deliberate — arms are NOT rewritten here (they are
    # added verbatim by _add_arm_nodes) and are rejected fail-loud below instead.
    changed = False
    new_nodes: list[object] = []
    for item in construct.nodes:
        if is_supported_fan_over_agent(item):
            new_nodes.append(_wrap_agent_node(item))
            changed = True
        else:
            new_nodes.append(item)
    wrapped = construct.model_copy(update={"nodes": new_nodes}) if changed else construct

    # Anything still classifying as a supported fan-over-agent after the
    # top-level rewrite is necessarily in a branch arm (iter_with_arms flattens
    # _BranchNode arms but does NOT descend into sub-construct nodes). The base
    # case does not cover arms — fail loud instead of a broken graph.
    for item in iter_with_arms(wrapped):
        if is_supported_fan_over_agent(item):
            raise CompileError.build(
                "Oracle over an agent/act node inside a branch arm is not supported",
                found=f"Oracle over agent/act node '{item.name}' in a branch arm",
                hint=(
                    "The fan-over-agent auto-wrap only rewrites top-level nodes. "
                    "Move the Oracle-over-agent out of the branch arm, or wrap the "
                    "agent in an explicit sub-construct. Tracking neograph-qot6."
                ),
                node=item.name,
                construct=construct.name,
            )
    return wrapped


def _synthesize_port(node: Node) -> tuple[type[BaseModel], Any]:
    """Derive the sub-construct's ``input=`` boundary port and the bare agent's
    rewritten ``inputs`` from the agent's declared upstream inputs (see
    neograph-qot6).

    Three shapes reach here — the ONLY shapes ``is_supported_fan_over_agent``
    admits (self-contained, single-type, single-key dict-form):

    - **Self-contained** (``inputs`` is None): the port is an empty synthesized
      model and the bare agent keeps ``inputs=None``. (The proven base case.)
    - **Single-type** (``inputs=T``): the port IS ``T``. The parent's upstream
      ``T`` is found by the subgraph's type-based scan and delivered as
      ``neo_subgraph_input``; the bare agent keeps ``inputs=T`` and single-type
      extraction naturally reads it back from that field — no rewrite needed.
    - **Single-key dict-form** (``inputs={k: T}``): the port is ``T`` and the bare
      agent's read is rewritten to ``{neo_subgraph_input: T}`` — the exact
      convention the ``@node`` sub-construct port mechanism uses
      (``_construct_builder._cleanup_inputs_and_register``), so fan-in extraction
      reads the boundary field. The original prompt-var name ``k`` is not
      preserved (it becomes ``neo_subgraph_input``, matching manual
      ``construct_from_functions(input=...)`` wrapping).

    DI (``FromInput``/``FromConfig``) params never appear in ``inputs`` (the
    decorator strips them into ``_param_res``, preserved by ``model_copy``), so
    they ride the forwarded config into the sub-construct and need no port.

    Multiple distinct dict-form producers cannot map to the single-value boundary
    and are rejected fail-loud upstream (``_fan_agent._unsupported_reason``).
    """
    ni = normalize_inputs(node.inputs)
    if ni.is_none:
        empty = create_model(
            f"_NeoAgentPort_{field_name_for(node.name)}", __base__=BaseModel
        )
        return empty, node.inputs
    if ni.is_dict_form:
        # Single-key guaranteed by is_supported_fan_over_agent (>1 fails loud).
        assert len(ni.by_name) == 1, "multi-producer dict-form must fail loud upstream"
        ((_key, port_type),) = ni.by_name.items()
        return port_type, {StateKeys.SUBGRAPH_INPUT: port_type}
    # Single-type: port IS the input type; bare agent keeps single-type inputs.
    return ni.single_type, ni.single_type


def _wrap_agent_node(node: Node) -> Construct:
    """Wrap one supported fan-over-agent Node into an isolated single-node
    sub-construct that carries the fan (+ any Operator) modifier.

    The inner node becomes a BARE agent whose ReAct cycle runs in the
    sub-construct's isolated state (own ``neo_agent_*`` channels per invocation),
    so ``_add_subgraph`` + ``_wire_oracle`` fan over it with true per-variant
    isolation. The wrapper takes the node's name so ``field_name_for`` and the
    returned output field are unchanged for any downstream consumer.

    ``_synthesize_port`` derives the ``input=`` boundary and the bare agent's
    read side from the agent's upstream inputs (see neograph-qot6); a
    self-contained agent gets an empty port (the base case).
    """
    port, inner_inputs = _synthesize_port(node)
    # model_copy preserves the @node sidecar/_param_res PrivateAttrs (Pydantic v2
    # copies __pydantic_private__), so a decorator-built agent keeps its DI bindings.
    bare = node.model_copy(update={"modifier_set": ModifierSet(), "inputs": inner_inputs})
    return Construct(
        name=node.name,
        input=port,
        output=node.outputs,  # single-type; dict-form is rejected upstream
        nodes=[bare],
        modifier_set=node.modifier_set,
    )
