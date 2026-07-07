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

from pydantic import BaseModel, create_model

from neograph._fan_agent import is_supported_fan_over_agent
from neograph._ir_branch import iter_with_arms
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


def _wrap_agent_node(node: Node) -> Construct:
    """Wrap one self-contained fan-over-agent Node into an isolated single-node
    sub-construct that carries the fan (+ any Operator) modifier.

    The inner node becomes a BARE agent whose ReAct cycle runs in the
    sub-construct's isolated state (own ``neo_agent_*`` channels per invocation),
    so ``_add_subgraph`` + ``_wire_oracle`` fan over it with true per-variant
    isolation. The wrapper takes the node's name so ``field_name_for`` and the
    returned output field are unchanged for any downstream consumer.
    """
    # model_copy preserves the @node sidecar/_param_res PrivateAttrs (Pydantic v2
    # copies __pydantic_private__), so a decorator-built agent keeps its DI bindings.
    bare = node.model_copy(update={"modifier_set": ModifierSet()})
    # Self-contained (guaranteed by is_supported_fan_over_agent): no upstream
    # inputs, so the sub-construct's boundary port is an empty synthesized model.
    port = create_model(
        f"_NeoAgentPort_{field_name_for(node.name)}", __base__=BaseModel
    )
    return Construct(
        name=node.name,
        input=port,
        output=node.outputs,  # single-type; dict-form is rejected upstream
        nodes=[bare],
        modifier_set=node.modifier_set,
    )
